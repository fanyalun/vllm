# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure DSpark SSD verify windows and select an internal fan-out."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from benchmarks.replayssm.async_ssd_eagle3_matrix import (
        Cell,
        cell_is_complete,
        prepare_prompts,
        run_cell,
        sha256_file,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:
    from async_ssd_eagle3_matrix import (
        Cell,
        cell_is_complete,
        prepare_prompts,
        run_cell,
        sha256_file,
        utc_now,
        write_json,
    )

FAN_OUT_ENV = "ASYNC_DRAFT_DSPARK_FAN_OUT"
TIMING_CELL = Cell("performance", "async_cache", "eager", 1)


@dataclass(frozen=True)
class ModelSweep:
    key: str
    label: str
    target: str
    draft: str
    verify_width: int
    qwen_gdn_mode: str


def parse_fan_outs(value: str) -> tuple[int, ...]:
    fan_outs = tuple(sorted({int(item) for item in value.split(",")}))
    if not fan_outs or any(not 1 <= item <= 512 for item in fan_outs):
        raise ValueError("fan-outs must be comma-separated integers in [1, 512]")
    return fan_outs


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(quantile * len(ordered)) - 1, 0)
    return ordered[index]


def timing_summary(
    trace_path: Path,
    result_path: Path,
    verify_width: int,
    fan_out: int,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    formal = [record for record in records if "-formal-" in record["request_id"]]
    pairs = [record for record in formal if record.get("async_timing_pair_valid")]
    windows = [float(record["async_verify_window_seconds"]) for record in pairs]
    builds = [float(record["async_previous_branch_build_seconds"]) for record in pairs]
    exposed = [max(build - window, 0.0) for build, window in zip(builds, windows)]
    waits = [float(record["async_next_proposal_wait_seconds"]) for record in pairs]
    if not pairs:
        raise ValueError(f"No paired async timing records in {trace_path}")
    if any(int(record["async_fan_out"]) != fan_out for record in pairs):
        raise ValueError(f"Trace fan-out does not match F={fan_out}")
    if any(int(record["async_batch_num_reqs"]) != 1 for record in pairs):
        raise ValueError("Fan-out timing calibration requires B=1")

    by_request: dict[str, list[dict[str, Any]]] = {}
    for record in formal:
        by_request.setdefault(record["request_id"], []).append(record)
    eligible = [record for values in by_request.values() for record in values[1:]]
    hits = sum(bool(record.get("cache_hit")) for record in eligible)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["metrics_delta"]
    markov_branches = sum(
        value
        for name, value in metrics.items()
        if name.startswith("vllm:async_draft_dspark_markov_branches_total")
    )
    refreshes = sum(
        value
        for name, value in metrics.items()
        if name.startswith("vllm:async_draft_dspark_backbone_refreshes_total")
    )
    branches_per_round = (verify_width + 1) * fan_out
    fully_hidden_round_fraction = sum(
        build <= window for build, window in zip(builds, windows)
    ) / len(pairs)
    conservative_window_fit = percentile(builds, 0.95) <= percentile(windows, 0.05)
    return {
        "status": "complete",
        "verify_width": verify_width,
        "fan_out": fan_out,
        "branches_per_round": branches_per_round,
        "formal_trace_records": len(formal),
        "paired_timing_rounds": len(pairs),
        "verify_window_seconds": {
            "p05": percentile(windows, 0.05),
            "p50": percentile(windows, 0.50),
            "p95": percentile(windows, 0.95),
            "mean": statistics.mean(windows),
        },
        "branch_build_seconds": {
            "p05": percentile(builds, 0.05),
            "p50": percentile(builds, 0.50),
            "p95": percentile(builds, 0.95),
            "mean": statistics.mean(builds),
        },
        "branch_exposed_seconds": {
            "p50": percentile(exposed, 0.50),
            "p95": percentile(exposed, 0.95),
            "mean": statistics.mean(exposed),
            "positive_rounds": sum(value > 0.0 for value in exposed),
        },
        "next_proposal_wait_seconds": {
            "p50": percentile(waits, 0.50),
            "p95": percentile(waits, 0.95),
            "mean": statistics.mean(waits),
        },
        "fully_hidden_round_fraction": fully_hidden_round_fraction,
        "conservative_window_fit": conservative_window_fit,
        "sustainable_window_fit": (
            conservative_window_fit and fully_hidden_round_fraction >= 0.99
        ),
        "cache": {
            "eligible_rounds": len(eligible),
            "hits": hits,
            "hit_rate": hits / len(eligible) if eligible else None,
        },
        "acceptance": {
            "mean_accepted_draft_count": statistics.mean(
                int(record["accepted_draft_count"]) for record in formal
            ),
        },
        "runtime_branch_audit": {
            "markov_branches": markov_branches,
            "backbone_refreshes": refreshes,
            "expected_branches_per_refresh": branches_per_round,
            "ratio": markov_branches / refreshes if refreshes else None,
            "passed": bool(
                refreshes > 0
                and math.isclose(
                    markov_branches / refreshes,
                    branches_per_round,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ),
        },
        "completion_throughput_tok_s": result["summary"]["completion_throughput_tok_s"],
    }


def select_fan_out(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [
        row
        for row in rows
        if row.get("status") == "complete" and row["runtime_branch_audit"]["passed"]
    ]
    safe = [row for row in complete if row["sustainable_window_fit"]]
    candidates = safe or complete
    if not candidates:
        return {
            "status": "failed",
            "reason": "no_complete_fan_out_cells",
        }
    best_throughput = max(row["completion_throughput_tok_s"] for row in candidates)
    near_best = [
        row
        for row in candidates
        if row["completion_throughput_tok_s"] >= best_throughput * 0.99
    ]
    selected = min(near_best, key=lambda row: row["fan_out"])
    capacity_row = (
        max(safe, key=lambda row: row["branches_per_round"]) if safe else None
    )
    return {
        "status": "complete",
        "selection_rule": (
            "maximize measured throughput among sustainable-window-fit cells; "
            "choose the smallest F within 1 percent of the best"
        ),
        "selected_fan_out": selected["fan_out"],
        "selected_branches_per_round": selected["branches_per_round"],
        "selected_completion_throughput_tok_s": selected["completion_throughput_tok_s"],
        "largest_conservatively_hidden_fan_out": (
            capacity_row["fan_out"] if capacity_row else None
        ),
        "largest_conservatively_hidden_branches_per_round": (
            capacity_row["branches_per_round"] if capacity_row else None
        ),
        "used_conservative_window_gate": bool(safe),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-target", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--qwen-draft",
        default="/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
    )
    parser.add_argument(
        "--gemma-target",
        default="/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it",
    )
    parser.add_argument(
        "--gemma-draft",
        default="/home/fanya/data1/fanya/models/gemma4-26b-a4b-dspark",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/fanya/data1/fanya/hf_datasets_cache/processed_datasets",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--fan-outs", default="3,6,12,24,48")
    parser.add_argument("--num-prompts-per-dataset", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--replayssm-buffer-len", type=int, default=16)
    parser.add_argument("--target-device", type=int, default=0)
    parser.add_argument("--draft-device", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--port-base", type=int, default=47500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use-runtime-default",
        action="store_true",
        help="Do not set the internal fan-out override in child server processes",
    )
    parser.add_argument(
        "--phase", choices=("prepare", "run", "analyze", "all"), default="all"
    )
    parser.add_argument("--models", choices=("all", "qwen36", "gemma4"), default="all")
    args = parser.parse_args(argv)
    args.fan_out_values = parse_fan_outs(args.fan_outs)
    if args.num_prompts_per_dataset != 4:
        parser.error("fan-out calibration requires 4 prompts per dataset (16 total)")
    if args.output_length != 128:
        parser.error("fan-out calibration requires output_length=128")
    if args.warmup_seconds < 30.0:
        parser.error("fan-out calibration requires at least 30 seconds of warmup")
    return args


def default_output_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(__file__).resolve().parents[2]
        / "SSSD_results"
        / "benchmarks"
        / "dspark_fanout"
        / f"async_dspark_fanout_b1_16x128_{timestamp}"
    )


def model_sweeps(args: argparse.Namespace) -> tuple[ModelSweep, ...]:
    models = (
        ModelSweep(
            "qwen36_dspark",
            "Qwen3.6 + DSpark",
            args.qwen_target,
            args.qwen_draft,
            3,
            "replayssm",
        ),
        ModelSweep(
            "gemma4_dspark",
            "Gemma4 + DSpark",
            args.gemma_target,
            args.gemma_draft,
            2,
            "baseline",
        ),
    )
    if args.models == "all":
        return models
    return tuple(model for model in models if model.key.startswith(args.models))


def make_run_args(
    args: argparse.Namespace,
    model: ModelSweep,
    cell_root: Path,
    port: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        target=model.target,
        draft=model.draft,
        method="dspark",
        dtype="bfloat16",
        dataset_root=args.dataset_root,
        output_root=str(cell_root),
        target_device=args.target_device,
        draft_device=args.draft_device,
        target_tensor_parallel_size=1,
        draft_tensor_parallel_size=1,
        attention_backend=None,
        gdn_recurrent_reference=False,
        qwen_gdn_mode=model.qwen_gdn_mode,
        replayssm_buffer_len=args.replayssm_buffer_len,
        num_prompts_per_dataset=args.num_prompts_per_dataset,
        input_length=args.input_length,
        output_length=args.output_length,
        num_speculative_tokens=model.verify_width,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        warmup_seconds=args.warmup_seconds,
        tie_logprob_tolerance=0.1,
        draft_tie_logit_tolerance=0.1,
        acceptance_length_relative_tolerance=0.01,
        correctness_scope="b1-eager",
        target_output_policy="audited-path-numerical",
        startup_timeout=args.startup_timeout,
        resume=args.resume,
        port_base=port,
        trace_async_timing=True,
    )


def source_fingerprint() -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", ".", ":(exclude)TDO"],
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            ".",
            ":(exclude)TDO",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    digest = hashlib.sha256(diff)
    for relative_path in sorted(filter(None, untracked)):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(Path(relative_path).read_bytes())
        digest.update(b"\0")
    return {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_diff_sha256": digest.hexdigest(),
        "untracked_files_in_fingerprint": sorted(filter(None, untracked)),
    }


def prepare_sweep(args: argparse.Namespace, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    models = model_sweeps(args)
    manifest = {
        "artifact_kind": "async_dspark_fanout_calibration",
        "created_at_utc": utc_now(),
        "source": source_fingerprint(),
        "contract": {
            "batch_size": 1,
            "prompt_count": 16,
            "input_length_cap": args.input_length,
            "output_length": args.output_length,
            "dtype": "bfloat16",
            "engine": "eager",
            "warmup_seconds_minimum": args.warmup_seconds,
            "fan_out_values": list(args.fan_out_values),
            "timing_window": ("next propose start minus previous Draft response ready"),
            "branch_build": (
                "previous response publish through all branch completion events"
            ),
        },
        "models": [asdict(model) for model in models],
        "checkpoint_configs": {
            model.key: {
                "target_config": sha256_file(Path(model.target) / "config.json"),
                "draft_config": sha256_file(Path(model.draft) / "config.json"),
            }
            for model in models
        },
    }
    write_json(output_root / "manifest.json", manifest)
    cells = []
    for model in models:
        model_root = output_root / model.key
        run_args = make_run_args(args, model, model_root, args.port_base)
        prompts = prepare_prompts(run_args, model_root)
        for fan_out in args.fan_out_values:
            cell_root = model_root / f"fanout_{fan_out}"
            write_json(cell_root / "prompts.json", prompts)
            cells.append(
                {
                    "model": model.key,
                    "verify_width": model.verify_width,
                    "fan_out": fan_out,
                    "branches_per_round": (model.verify_width + 1) * fan_out,
                    "path": str(cell_root.relative_to(output_root)),
                }
            )
    write_json(
        output_root / "matrix.json",
        {
            "status": "expected",
            "expected_cell_count": len(cells),
            "cells": cells,
        },
    )


def run_sweep(args: argparse.Namespace, output_root: Path) -> None:
    failures = []
    for model_index, model in enumerate(model_sweeps(args)):
        model_root = output_root / model.key
        prompts = json.loads((model_root / "prompts.json").read_text(encoding="utf-8"))
        for fan_index, fan_out in enumerate(args.fan_out_values):
            cell_root = model_root / f"fanout_{fan_out}"
            port = args.port_base + model_index * 100 + fan_index
            run_args = make_run_args(args, model, cell_root, port)
            if args.resume and cell_is_complete(cell_root / "cells" / TIMING_CELL.name):
                continue
            previous = os.environ.get(FAN_OUT_ENV)
            if args.use_runtime_default:
                os.environ.pop(FAN_OUT_ENV, None)
            else:
                os.environ[FAN_OUT_ENV] = str(fan_out)
            try:
                run_cell(run_args, cell_root, prompts, TIMING_CELL, port)
            except BaseException as error:
                failures.append(
                    {
                        "model": model.key,
                        "fan_out": fan_out,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                if previous is None:
                    os.environ.pop(FAN_OUT_ENV, None)
                else:
                    os.environ[FAN_OUT_ENV] = previous
            write_json(
                output_root / "run_status.json",
                {
                    "status": "running" if not failures else "running_with_failures",
                    "updated_at_utc": utc_now(),
                    "failures": failures,
                },
            )
    write_json(
        output_root / "run_status.json",
        {
            "status": "complete" if not failures else "complete_with_failures",
            "updated_at_utc": utc_now(),
            "failures": failures,
        },
    )


def analyze_sweep(args: argparse.Namespace, output_root: Path) -> bool:
    all_rows = []
    selections = {}
    for model in model_sweeps(args):
        model_rows = []
        for fan_out in args.fan_out_values:
            cell_root = output_root / model.key / f"fanout_{fan_out}"
            cell_dir = cell_root / "cells" / TIMING_CELL.name
            if not cell_is_complete(cell_dir):
                row = {
                    "status": "failed",
                    "model": model.key,
                    "fan_out": fan_out,
                    "reason": "cell_incomplete",
                }
            else:
                row = timing_summary(
                    cell_dir / "proposals.jsonl",
                    cell_dir / "result.json",
                    model.verify_width,
                    fan_out,
                )
                row["model"] = model.key
            model_rows.append(row)
            all_rows.append(row)
        selections[model.key] = select_fan_out(model_rows)
    write_json(
        output_root / "fanout_analysis.json",
        {
            "status": (
                "complete"
                if all(value["status"] == "complete" for value in selections.values())
                else "failed"
            ),
            "created_at_utc": utc_now(),
            "selections": selections,
            "rows": all_rows,
        },
    )
    complete_rows = [row for row in all_rows if row["status"] == "complete"]
    if complete_rows:
        with (output_root / "fanout_analysis.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            fieldnames = (
                "model",
                "verify_width",
                "fan_out",
                "branches_per_round",
                "paired_timing_rounds",
                "completion_throughput_tok_s",
                "fully_hidden_round_fraction",
                "conservative_window_fit",
                "sustainable_window_fit",
                "verify_window_p05_ms",
                "verify_window_p50_ms",
                "branch_build_p50_ms",
                "branch_build_p95_ms",
                "cache_hit_rate",
                "mean_accepted_draft_count",
            )
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in complete_rows:
                writer.writerow(
                    {
                        "model": row["model"],
                        "verify_width": row["verify_width"],
                        "fan_out": row["fan_out"],
                        "branches_per_round": row["branches_per_round"],
                        "paired_timing_rounds": row["paired_timing_rounds"],
                        "completion_throughput_tok_s": row[
                            "completion_throughput_tok_s"
                        ],
                        "fully_hidden_round_fraction": row[
                            "fully_hidden_round_fraction"
                        ],
                        "conservative_window_fit": row["conservative_window_fit"],
                        "sustainable_window_fit": row["sustainable_window_fit"],
                        "verify_window_p05_ms": 1000
                        * row["verify_window_seconds"]["p05"],
                        "verify_window_p50_ms": 1000
                        * row["verify_window_seconds"]["p50"],
                        "branch_build_p50_ms": 1000
                        * row["branch_build_seconds"]["p50"],
                        "branch_build_p95_ms": 1000
                        * row["branch_build_seconds"]["p95"],
                        "cache_hit_rate": row["cache"]["hit_rate"],
                        "mean_accepted_draft_count": row["acceptance"][
                            "mean_accepted_draft_count"
                        ],
                    }
                )
    return all(value["status"] == "complete" for value in selections.values())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = (
        Path(args.output_root).resolve() if args.output_root else default_output_root()
    )
    if args.phase in ("prepare", "all"):
        prepare_sweep(args, output_root)
        if args.phase == "prepare":
            print(output_root)
            return 0
    if args.phase in ("run", "all"):
        run_sweep(args, output_root)
    if args.phase in ("analyze", "all"):
        complete = analyze_sweep(args, output_root)
        print(output_root)
        return 0 if complete else 2
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
