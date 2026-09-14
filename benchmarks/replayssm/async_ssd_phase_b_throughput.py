# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the fixed Phase-B D=3, B=1 throughput comparison matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from benchmarks.replayssm.async_ssd_eagle3_matrix import (
        Cell,
        cell_is_complete,
        checkpoint_manifest,
        metric_total,
        prepare_prompts,
        run_cell,
        server_command,
        sha256_json,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:
    from async_ssd_eagle3_matrix import (
        Cell,
        cell_is_complete,
        checkpoint_manifest,
        metric_total,
        prepare_prompts,
        run_cell,
        server_command,
        sha256_json,
        utc_now,
        write_json,
    )

DECODE_MODES = ("ar", "sync", "async_cache")
MODE_LABELS = {"ar": "AR", "sync": "Sync", "async_cache": "Async"}
MODE_COLORS = {
    "ar": "#4C78A8",
    "sync": "#F58518",
    "async_cache": "#54A24B",
}


@dataclass(frozen=True)
class ModelExperiment:
    key: str
    label: str
    method: str
    target: str
    draft: str | None
    target_family: str


@dataclass(frozen=True)
class MatrixCell:
    model_key: str
    model_label: str
    method: str
    mode: str
    cell_name: str
    port: int

    @property
    def key(self) -> str:
        return f"{self.model_key}/{self.cell_name}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-target",
        default="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--qwen-dspark",
        default="/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
    )
    parser.add_argument(
        "--gemma-target",
        default="/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it",
    )
    parser.add_argument(
        "--gemma-dspark",
        default="/home/fanya/data1/fanya/models/gemma4-26b-a4b-dspark",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/fanya/data1/fanya/hf_datasets_cache/processed_datasets",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--target-device", type=int, default=0)
    parser.add_argument("--draft-device", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--num-prompts-per-dataset", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--replayssm-buffer-len", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--port-base", type=int, default=46200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("prepare", "run", "audit", "plot", "all"),
        default="all",
    )
    args = parser.parse_args(argv)
    if args.num_speculative_tokens != 3:
        parser.error("this formal matrix requires D=3")
    if args.num_prompts_per_dataset != 4:
        parser.error("this formal matrix requires 4 prompts from each dataset")
    if args.output_length != 128:
        parser.error("this formal matrix requires output_length=128")
    if args.batch_size != 1:
        parser.error("this formal matrix requires batch_size=1")
    if args.warmup_seconds <= 0:
        parser.error("formal experiments require positive warmup_seconds")
    return args


def model_experiments(args: argparse.Namespace) -> tuple[ModelExperiment, ...]:
    return (
        ModelExperiment(
            key="qwen36_mtp",
            label="Qwen3.6 + MTP",
            method="mtp",
            target=args.qwen_target,
            draft=None,
            target_family="qwen36",
        ),
        ModelExperiment(
            key="qwen36_dspark",
            label="Qwen3.6 + DSpark",
            method="dspark",
            target=args.qwen_target,
            draft=args.qwen_dspark,
            target_family="qwen36",
        ),
        ModelExperiment(
            key="gemma4_dspark",
            label="Gemma4 + DSpark",
            method="dspark",
            target=args.gemma_target,
            draft=args.gemma_dspark,
            target_family="gemma4",
        ),
    )


def matrix_cells(
    experiments: tuple[ModelExperiment, ...], port_base: int
) -> list[MatrixCell]:
    cells: list[MatrixCell] = []
    for experiment_index, experiment in enumerate(experiments):
        for mode_index, mode in enumerate(DECODE_MODES):
            cell = Cell("performance", mode, "eager", 1)
            cells.append(
                MatrixCell(
                    model_key=experiment.key,
                    model_label=experiment.label,
                    method=experiment.method,
                    mode=mode,
                    cell_name=cell.name,
                    port=port_base + experiment_index * len(DECODE_MODES) + mode_index,
                )
            )
    return cells


def default_output_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(__file__).resolve().parents[2]
        / "SSSD_results"
        / "benchmarks"
        / "phase_b_throughput"
        / f"phase_b_d3_b1_16x128_{timestamp}"
    )


def model_output_root(output_root: Path, experiment: ModelExperiment) -> Path:
    return output_root / "models" / experiment.key


def make_run_args(
    args: argparse.Namespace,
    experiment: ModelExperiment,
    output_root: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        target=experiment.target,
        draft=experiment.draft,
        method=experiment.method,
        dtype=args.dtype,
        dataset_root=args.dataset_root,
        output_root=str(output_root),
        target_device=args.target_device,
        draft_device=args.draft_device,
        target_tensor_parallel_size=1,
        draft_tensor_parallel_size=1,
        attention_backend=None,
        gdn_recurrent_reference=False,
        qwen_gdn_mode="replayssm",
        replayssm_buffer_len=args.replayssm_buffer_len,
        num_prompts_per_dataset=args.num_prompts_per_dataset,
        input_length=args.input_length,
        output_length=args.output_length,
        num_speculative_tokens=args.num_speculative_tokens,
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
        port_base=args.port_base,
    )


def validate_paths(
    args: argparse.Namespace, experiments: tuple[ModelExperiment, ...]
) -> None:
    dataset_root = Path(args.dataset_root)
    required_datasets = (
        "humaneval/humaneval_data_10000.jsonl",
        "alpaca/alpaca_data_10000.jsonl",
        "gsm8k/gsm8k_data_10000.jsonl",
        "ultrafeedback/ultrafeedback_data_10000.jsonl",
    )
    missing = [
        str(dataset_root / path)
        for path in required_datasets
        if not (dataset_root / path).is_file()
    ]
    for experiment in experiments:
        for path in (experiment.target, experiment.draft):
            if path is not None and not Path(path).is_dir():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing formal experiment inputs: " + ", ".join(missing)
        )


def worktree_fingerprint() -> dict[str, object]:
    tracked_diff = subprocess.run(
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
    untracked = sorted(filter(None, untracked))
    digest = hashlib.sha256(tracked_diff)
    for relative_path in untracked:
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(Path(relative_path).read_bytes())
        digest.update(b"\0")
    return {
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "working_tree_diff_sha256": digest.hexdigest(),
        "untracked_files_in_fingerprint": untracked,
    }


def prepare_artifact(
    args: argparse.Namespace,
    output_root: Path,
    experiments: tuple[ModelExperiment, ...],
    cells: list[MatrixCell],
) -> None:
    import torch
    import transformers

    import vllm

    validate_paths(args, experiments)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_cache: dict[str, dict[str, object]] = {}
    prompt_records: dict[str, dict[str, object]] = {}
    for experiment in experiments:
        run_args = make_run_args(
            args, experiment, model_output_root(output_root, experiment)
        )
        prompts = prepare_prompts(run_args, Path(run_args.output_root))
        prompt_records[experiment.key] = {
            "count": len(prompts),
            "tokenized_prompt_sha256": sha256_json(prompts),
            "sample_ids": [
                [prompt["dataset"], prompt["dataset_index"]] for prompt in prompts
            ],
        }
        for path in (experiment.target, experiment.draft):
            if path is not None and path not in checkpoint_cache:
                checkpoint_cache[path] = checkpoint_manifest(Path(path))

    sample_ids = {
        tuple(map(tuple, record["sample_ids"])) for record in prompt_records.values()
    }
    if len(sample_ids) != 1:
        raise AssertionError("Model tokenizers did not use the same 16 dataset samples")

    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "artifact_kind": "async_ssd_phase_b_throughput_d3_b1_16x128",
        "created_at_utc": utc_now(),
        "git": worktree_fingerprint(),
        "source": {
            "python": sys.executable,
            "vllm": vllm.__file__,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "contract": {
            "dtype": args.dtype,
            "engine": "eager",
            "batch_size": args.batch_size,
            "prompt_count": 4 * args.num_prompts_per_dataset,
            "input_length_cap": args.input_length,
            "output_length": args.output_length,
            "num_speculative_tokens": args.num_speculative_tokens,
            "fan_out_by_model": {
                "qwen36_mtp": 96,
                "qwen36_dspark": 24,
                "gemma4_dspark": 48,
            },
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
            "warmup_seconds_minimum": args.warmup_seconds,
            "prefix_caching": False,
            "scheduler_async": False,
            "runner": "V2",
            "async_mode": "real_cache",
            "measurement_repeats": 1,
            "user_assumption": "Phase-B adapters are ready for throughput measurement",
        },
        "topology": {
            "target_device": args.target_device,
            "draft_device": args.draft_device,
            "gpu_inventory": gpu_inventory,
            "nvidia_smi_topology": topology,
        },
        "replayssm_contract": {
            "qwen36_ar": "use_replayssm",
            "qwen36_sync": "use_replayssm_spec",
            "qwen36_async": "use_replayssm_spec",
            "qwen36_buffer_len": args.replayssm_buffer_len,
            "gemma4_all_modes": "not_applicable",
        },
        "experiments": [asdict(experiment) for experiment in experiments],
        "checkpoints": checkpoint_cache,
        "prompts": prompt_records,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(
        output_root / "matrix.json",
        {
            "status": "expected",
            "created_at_utc": utc_now(),
            "expected_cell_count": len(cells),
            "cells": [asdict(cell) | {"key": cell.key} for cell in cells],
        },
    )
    write_json(
        output_root / "run_status.json",
        {
            "status": "prepared",
            "updated_at_utc": utc_now(),
            "completed_cells": [],
            "failed_cells": [],
        },
    )


def run_matrix(
    args: argparse.Namespace,
    output_root: Path,
    experiments: tuple[ModelExperiment, ...],
    cells: list[MatrixCell],
) -> bool:
    experiment_by_key = {experiment.key: experiment for experiment in experiments}
    completed: list[str] = []
    failures: list[dict[str, str]] = []
    write_json(
        output_root / "run_started.json",
        {"status": "running", "started_at_utc": utc_now()},
    )
    for matrix_cell in cells:
        experiment = experiment_by_key[matrix_cell.model_key]
        experiment_root = model_output_root(output_root, experiment)
        cell = Cell("performance", matrix_cell.mode, "eager", 1)
        run_args = make_run_args(args, experiment, experiment_root)
        prompts = prepare_prompts(run_args, experiment_root)
        try:
            run_cell(run_args, experiment_root, prompts, cell, matrix_cell.port)
            completed.append(matrix_cell.key)
        except BaseException as error:
            failures.append(
                {
                    "cell": matrix_cell.key,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        write_json(
            output_root / "run_status.json",
            {
                "status": "running"
                if len(completed) + len(failures) < len(cells)
                else "finished",
                "updated_at_utc": utc_now(),
                "completed_cells": completed,
                "failed_cells": failures,
            },
        )
    return not failures


def _load_cell_result(
    output_root: Path, experiment: ModelExperiment, mode: str
) -> tuple[Path, dict[str, Any]]:
    cell = Cell("performance", mode, "eager", 1)
    cell_dir = model_output_root(output_root, experiment) / "cells" / cell.name
    result = json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))
    return cell_dir, result


def audit_matrix(
    args: argparse.Namespace,
    output_root: Path,
    experiments: tuple[ModelExperiment, ...],
    cells: list[MatrixCell],
) -> bool:
    failures: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    expected_tokens = 4 * args.num_prompts_per_dataset * args.output_length
    experiment_by_key = {experiment.key: experiment for experiment in experiments}
    for matrix_cell in cells:
        experiment = experiment_by_key[matrix_cell.model_key]
        cell = Cell("performance", matrix_cell.mode, "eager", 1)
        cell_dir = model_output_root(output_root, experiment) / "cells" / cell.name
        cell_failures: list[str] = []
        if not cell_is_complete(cell_dir):
            cell_failures.append("missing or inconsistent completion marker")
            result: dict[str, Any] = {}
        else:
            result = json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))
            summary = result.get("summary", {})
            warmup = result.get("warmup", {})
            if summary.get("completion_tokens") != expected_tokens:
                cell_failures.append("formal completion token count mismatch")
            if summary.get("completed_request_count") != 16:
                cell_failures.append("formal completed request count mismatch")
            if warmup.get("seconds", 0) < args.warmup_seconds:
                cell_failures.append("formal warmup duration was too short")
            if warmup.get("requests", 0) <= 0:
                cell_failures.append("formal warmup did not complete a request")
            shutdown = json.loads(
                (cell_dir / "shutdown.json").read_text(encoding="utf-8")
            )
            if shutdown.get("exit_code") != 0 or shutdown.get("forced_kill"):
                cell_failures.append("server did not shut down cleanly")
            command = json.loads(
                (cell_dir / "command.json").read_text(encoding="utf-8")
            )
            expected_command = server_command(
                make_run_args(
                    args, experiment, model_output_root(output_root, experiment)
                ),
                cell,
                matrix_cell.port,
            )
            if command != expected_command:
                cell_failures.append("recorded command differs from matrix contract")
            if experiment.target_family == "qwen36":
                flag = (
                    "--use-replayssm"
                    if matrix_cell.mode == "ar"
                    else "--use-replayssm-spec"
                )
                if flag not in command:
                    cell_failures.append("Qwen ReplaySSM route flag is missing")
            elif "--use-replayssm" in command or "--use-replayssm-spec" in command:
                cell_failures.append("Gemma command unexpectedly enables ReplaySSM")
            if matrix_cell.mode == "async_cache":
                metrics = result.get("metrics_delta", {})
                hits = metric_total(metrics, "vllm:async_draft_cache_hits_total")
                misses = metric_total(metrics, "vllm:async_draft_cache_misses_total")
                branch_seconds = metric_total(
                    metrics, "vllm:async_draft_branch_build_seconds_total"
                )
                if hits <= 0 or misses <= 0 or branch_seconds <= 0:
                    cell_failures.append(
                        "Async cell did not exercise real cache hit/miss/build"
                    )
        record = {
            "cell": matrix_cell.key,
            "status": "passed" if not cell_failures else "failed",
            "failures": cell_failures,
        }
        records.append(record)
        if cell_failures:
            failures.append(record)

    status = "passed" if not failures and len(records) == 9 else "failed"
    audit = {
        "status": status,
        "completed_at_utc": utc_now(),
        "expected_cell_count": 9,
        "observed_cell_count": len(records),
        "expected_completion_tokens_per_cell": expected_tokens,
        "failures": failures,
        "records": records,
    }
    write_json(output_root / "artifact_audit.json", audit)
    if status == "passed":
        completed_at = utc_now()
        write_json(
            output_root / "matrix_complete.json",
            {
                "status": "complete",
                "completed_at_utc": completed_at,
                "cell_count": len(records),
            },
        )
        write_json(
            output_root / "performance_measurement_complete.json",
            {
                "status": "complete",
                "completed_at_utc": completed_at,
                "cell_count": len(records),
                "measurement_repeats": 1,
            },
        )
    else:
        write_json(
            output_root / "matrix_incomplete.json",
            {
                "status": "incomplete",
                "completed_at_utc": utc_now(),
                "failures": failures,
            },
        )
    return status == "passed"


def collect_rows(
    output_root: Path, experiments: tuple[ModelExperiment, ...]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment in experiments:
        for mode in DECODE_MODES:
            _, result = _load_cell_result(output_root, experiment, mode)
            summary = result["summary"]
            metrics = result.get("metrics_delta", {})
            rows.append(
                {
                    "model_key": experiment.key,
                    "model_config": experiment.label,
                    "method": experiment.method,
                    "decode_mode": MODE_LABELS[mode],
                    "mode_key": mode,
                    "completion_throughput_tok_s": summary[
                        "completion_throughput_tok_s"
                    ],
                    "tokens_per_gpu_second": summary["tokens_per_gpu_second"],
                    "ttft_p50_seconds": summary["ttft_p50_seconds"],
                    "ttft_p95_seconds": summary["ttft_p95_seconds"],
                    "tpot_p50_seconds": summary["tpot_p50_seconds"],
                    "tpot_p95_seconds": summary["tpot_p95_seconds"],
                    "cache_hits": metric_total(
                        metrics, "vllm:async_draft_cache_hits_total"
                    ),
                    "cache_misses": metric_total(
                        metrics, "vllm:async_draft_cache_misses_total"
                    ),
                }
            )
    return rows


def render_outputs(output_root: Path, experiments: tuple[ModelExperiment, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = collect_rows(output_root, experiments)
    csv_path = output_root / "throughput.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["model_key"], row["mode_key"]): row for row in rows}
    gate_records = []
    for experiment in experiments:
        sync_throughput = float(
            by_key[(experiment.key, "sync")]["completion_throughput_tok_s"]
        )
        async_throughput = float(
            by_key[(experiment.key, "async_cache")]["completion_throughput_tok_s"]
        )
        gate_records.append(
            {
                "model_key": experiment.key,
                "model_config": experiment.label,
                "sync_throughput_tok_s": sync_throughput,
                "async_throughput_tok_s": async_throughput,
                "async_vs_sync_percent": (async_throughput / sync_throughput - 1) * 100,
                "passed": async_throughput > sync_throughput,
            }
        )
    gate_passed = all(record["passed"] for record in gate_records)
    gate_filename = (
        "performance_gate_passed.json"
        if gate_passed
        else "performance_gate_failed.json"
    )
    write_json(
        output_root / gate_filename,
        {
            "status": "passed" if gate_passed else "failed",
            "criterion": "async_completion_throughput_gt_sync_completion_throughput",
            "records": gate_records,
            "completed_at_utc": utc_now(),
        },
    )
    x_values = list(range(len(experiments)))
    width = 0.24
    offsets = {"ar": -width, "sync": 0.0, "async_cache": width}
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for mode in DECODE_MODES:
        values = [
            float(by_key[(experiment.key, mode)]["completion_throughput_tok_s"])
            for experiment in experiments
        ]
        positions = [value + offsets[mode] for value in x_values]
        bars = axis.bar(
            positions,
            values,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(x_values, [experiment.label for experiment in experiments])
    axis.set_ylabel("Completion throughput (tokens/s)")
    axis.set_xlabel("Model configuration")
    axis.set_title("Phase-B decoding throughput (D=3, B=1, 16 x 128 tokens)")
    axis.legend(title="Decoding mode")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_root / "throughput.png", dpi=180)
    figure.savefig(output_root / "throughput.svg")
    plt.close(figure)

    table_lines = [
        "| Model configuration | AR tok/s | Sync tok/s | Async tok/s |",
        "|---|---:|---:|---:|",
    ]
    for experiment in experiments:
        values = {
            mode: float(by_key[(experiment.key, mode)]["completion_throughput_tok_s"])
            for mode in DECODE_MODES
        }
        table_lines.append(
            f"| {experiment.label} | {values['ar']:.4f} | "
            f"{values['sync']:.4f} | {values['async_cache']:.4f} |"
        )
    readme = "\n".join(
        [
            "# Phase-B D=3 throughput matrix",
            "",
            "Formal contract: BF16, eager, B=1, 16 prompts, input cap 128, "
            "output 128, D=3, 30-second minimum warmup, seed 0, temperature 0.",
            "Async denotes the real-cache SSD path on Target GPU0 + Draft GPU1.",
            "Sync and AR use GPU0 only; tokens/GPU-second is retained in the CSV.",
            "Performance gate: " + ("passed." if gate_passed else "failed."),
            "",
            *table_lines,
            "",
            "See `manifest.json`, `matrix.json`, `artifact_audit.json`, and each "
            "cell directory for commands, logs, metrics, GPU samples, and shutdown.",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = (
        Path(args.output_root).resolve() if args.output_root else default_output_root()
    )
    experiments = model_experiments(args)
    cells = matrix_cells(experiments, args.port_base)
    if args.phase in ("prepare", "all"):
        prepare_artifact(args, output_root, experiments, cells)
        if args.phase == "prepare":
            print(output_root, flush=True)
            return 0
    elif not (output_root / "manifest.json").is_file():
        raise FileNotFoundError("Run --phase prepare before this phase")

    run_ok = True
    if args.phase in ("run", "all"):
        run_ok = run_matrix(args, output_root, experiments, cells)
        if args.phase == "run":
            print(output_root, flush=True)
            return 0 if run_ok else 2

    audit_ok = True
    if args.phase in ("audit", "all"):
        audit_ok = audit_matrix(args, output_root, experiments, cells)
        if args.phase == "audit":
            print(output_root, flush=True)
            return 0 if audit_ok else 3

    if args.phase in ("plot", "all"):
        if not audit_ok and args.phase == "all":
            print(output_root, flush=True)
            return 3
        if not (output_root / "matrix_complete.json").is_file():
            raise RuntimeError("Plotting requires a complete audited matrix")
        render_outputs(output_root, experiments)
    print(output_root, flush=True)
    return 0 if run_ok and audit_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
