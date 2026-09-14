# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import csv
import importlib.metadata
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from run_experiment import is_allowed_near_tie, load_near_tie_allowance

DRAFT_LENGTHS = (4, 8, 16, 32)
MODES = ("eager", "graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-near-tie-audit")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def first_mismatch(expected: list[int], actual: list[int]) -> dict | None:
    sentinel = object()
    for position, (ar_token, spec_token) in enumerate(
        itertools.zip_longest(expected, actual, fillvalue=sentinel)
    ):
        if ar_token != spec_token:
            return {
                "position": position,
                "ar_token": None if ar_token is sentinel else ar_token,
                "spec_token": None if spec_token is sentinel else spec_token,
            }
    return None


def collect_correctness(
    run_dir: Path, cells: dict[str, dict], allowance: dict | None
) -> dict:
    checks = []
    failures = []
    allowed_near_ties = []
    for name, cell in cells.items():
        if cell["method"] != "moe_skip":
            continue
        baseline = cells[f"ar_{cell['mode']}"]
        for ar_output, spec_output in zip(
            baseline["outputs"], cell["outputs"], strict=True
        ):
            mismatch = first_mismatch(ar_output["token_ids"], spec_output["token_ids"])
            is_allowed = mismatch is not None and is_allowed_near_tie(
                allowance,
                spec_output,
                mismatch["position"],
                mismatch["ar_token"],
                mismatch["spec_token"],
            )
            check = {
                "cell": name,
                "mode": cell["mode"],
                "draft_length": cell["draft_length"],
                "top_h": cell["top_h"],
                "sample_index": spec_output["sample_index"],
                "prompt_sha256": spec_output["prompt_sha256"],
                "exact_match": mismatch is None,
                "allowed_near_tie": is_allowed,
                "first_mismatch": mismatch,
            }
            checks.append(check)
            if is_allowed:
                allowed_near_ties.append(check)
            elif mismatch is not None:
                failures.append(check)
    if failures:
        status = "failed"
    elif allowed_near_ties:
        assert allowance is not None
        status = allowance["final_correctness_label"]
    else:
        status = "passed"
    audit = {
        "status": status,
        "required_modes": list(MODES),
        "required_draft_lengths": list(DRAFT_LENGTHS),
        "required_top_h": 4,
        "identity_diagnostics": {"top_h": 8, "draft_lengths": [4, 32]},
        "num_checks": len(checks),
        "num_failures": len(failures),
        "num_allowed_near_ties": len(allowed_near_ties),
        "near_tie_evidence": allowance,
        "checks": checks,
    }
    (run_dir / "correctness_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def combine_traces(run_dir: Path, cells: dict[str, dict]) -> list[dict]:
    combined = []
    for draft_length in DRAFT_LENGTHS:
        name = f"moe_skip_top4_graph_d{draft_length}"
        cell = cells[name]
        request_to_output = {
            str(output["request_id"]): output for output in cell["outputs"]
        }
        trace_path = run_dir / "cells" / name / "trace" / "raw_trace.jsonl"
        seen = set()
        with trace_path.open(encoding="utf-8") as trace_file:
            for line in trace_file:
                row = json.loads(line)
                key = (
                    str(row["request_id"]),
                    row["verify_step"],
                    row["draft_position"],
                )
                if key in seen:
                    raise RuntimeError(f"Duplicate trace record in {trace_path}: {key}")
                seen.add(key)
                output = request_to_output[str(row["request_id"])]
                row["sample_index"] = output["sample_index"]
                metrics = output["spec_decode_metrics"]
                assert metrics is not None
                generated_before_step = 1
                step = row["verify_step"]
                for accepted in metrics["per_step_accepted"][:step]:
                    generated_before_step += accepted + 1
                output_position = generated_before_step + row["draft_position"] - 1
                row["output_position"] = output_position
                row["valid_mask"] = bool(
                    row["valid_mask"] and output_position < len(output["token_ids"])
                )
                combined.append(row)
    output_path = run_dir / "raw_trace.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in combined:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
    return combined


def write_acceptance_summary(run_dir: Path, cells: dict[str, dict]) -> list[dict]:
    rows = []
    for draft_length in DRAFT_LENGTHS:
        cell = cells[f"moe_skip_top4_graph_d{draft_length}"]
        metrics = [output["spec_decode_metrics"] for output in cell["outputs"]]
        if any(metric is None for metric in metrics):
            raise RuntimeError(f"Missing spec-decode metrics for D={draft_length}")
        verify_steps = sum(metric["num_spec_steps"] for metric in metrics)
        accepted = sum(metric["num_accepted_draft_tokens"] for metric in metrics)
        drafted = sum(metric["num_draft_tokens"] for metric in metrics)
        rows.append(
            {
                "draft_length": draft_length,
                "num_samples": len(metrics),
                "verify_steps": verify_steps,
                "drafted_tokens": drafted,
                "accepted_draft_tokens": accepted,
                "mean_acceptance_length": 1 + accepted / verify_steps,
            }
        )
    with (run_dir / "acceptance_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_position_metrics(run_dir: Path, trace: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in trace:
        if not row["valid_mask"]:
            continue
        grouped[(row["draft_length"], row["draft_position"])].append(row)
    rows = []
    for draft_length in DRAFT_LENGTHS:
        for position in range(1, draft_length + 1):
            values = grouped[(draft_length, position)]
            if not values:
                raise RuntimeError(
                    f"No valid trace rows for D={draft_length}, position={position}"
                )
            n = len(values)
            rows.append(
                {
                    "draft_length": draft_length,
                    "draft_position": position,
                    "n": n,
                    "top1_precision": sum(
                        row["target_top1_token_id"]
                        == row["draft_argmax_ordered_top8_token_ids"][0]
                        for row in values
                    )
                    / n,
                    "top2_recall": sum(
                        row["target_top1_token_id"]
                        in row["draft_argmax_ordered_top8_token_ids"][:2]
                        for row in values
                    )
                    / n,
                    "top3_recall": sum(
                        row["target_top1_token_id"]
                        in row["draft_argmax_ordered_top8_token_ids"][:3]
                        for row in values
                    )
                    / n,
                    "top8_recall": sum(
                        row["target_top1_token_id"]
                        in row["draft_argmax_ordered_top8_token_ids"]
                        for row in values
                    )
                    / n,
                }
            )
    with (run_dir / "position_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_rank_recall_summary(run_dir: Path, trace: list[dict]) -> list[dict]:
    valid_rows = [row for row in trace if row["valid_mask"]]
    groups: list[tuple[int | str, list[dict]]] = [
        (
            draft_length,
            [row for row in valid_rows if row["draft_length"] == draft_length],
        )
        for draft_length in DRAFT_LENGTHS
    ]
    groups.append(("all", valid_rows))
    rows = []
    for draft_length, values in groups:
        n = len(values)
        hits = {
            rank: sum(
                row["target_top1_token_id"]
                in row["draft_argmax_ordered_top8_token_ids"][:rank]
                for row in values
            )
            for rank in (1, 2, 3, 8)
        }
        rows.append(
            {
                "draft_length": draft_length,
                "n": n,
                "top1_hits": hits[1],
                "top1_precision": hits[1] / n,
                "top2_hits": hits[2],
                "top2_recall": hits[2] / n,
                "top3_hits": hits[3],
                "top3_recall": hits[3] / n,
                "top8_hits": hits[8],
                "top8_recall": hits[8] / n,
                "top2_additional_hits_vs_top1": hits[2] - hits[1],
                "top3_additional_hits_vs_top2": hits[3] - hits[2],
            }
        )
    with (run_dir / "rank_recall_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_plots(
    run_dir: Path, acceptance: list[dict], position_metrics: list[dict]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draft_lengths = [row["draft_length"] for row in acceptance]
    mean_acceptance_lengths = [row["mean_acceptance_length"] for row in acceptance]
    ax.plot(draft_lengths, mean_acceptance_lengths, marker="o")
    for draft_length, mean_acceptance_length in zip(
        draft_lengths, mean_acceptance_lengths, strict=True
    ):
        ax.annotate(
            f"{mean_acceptance_length:.2f}",
            (draft_length, mean_acceptance_length),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlabel("Draft length D")
    ax.set_ylabel("Mean acceptance length (including bonus)")
    ax.set_xticks(DRAFT_LENGTHS)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "acceptance_length_vs_draft_length.png", dpi=180)
    plt.close(fig)

    metric_names = (
        ("top1_precision", "Top-1 precision"),
        ("top2_recall", "Top-2 recall"),
        ("top3_recall", "Top-3 recall"),
        ("top8_recall", "Top-8 recall"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), constrained_layout=True)
    max_position = max(DRAFT_LENGTHS)
    labeled_positions = (1, 4, 8, 12, 16, 20, 24, 28, 32)
    for axis, (metric, title) in zip(axes, metric_names, strict=True):
        matrix = np.full((len(DRAFT_LENGTHS), max_position), np.nan)
        for row in position_metrics:
            d_idx = DRAFT_LENGTHS.index(row["draft_length"])
            matrix[d_idx, row["draft_position"] - 1] = row[metric]
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title)
        axis.set_yticks(range(len(DRAFT_LENGTHS)), DRAFT_LENGTHS)
        axis.set_ylabel("D")
        axis.set_xlim(-0.5, max_position - 0.5)
        axis.set_xticks(
            [position - 1 for position in labeled_positions], labeled_positions
        )
    axes[-1].set_xlabel("Draft position")
    fig.colorbar(image, ax=axes, label="Fraction", location="right")
    fig.savefig(run_dir / "token_quality_by_position.png", dpi=180)
    plt.close(fig)


def environment_info() -> dict:
    import torch

    gpu_query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": importlib.metadata.version("vllm"),
        "flashinfer-python": importlib.metadata.version("flashinfer-python"),
        "flashinfer-cubin": importlib.metadata.version("flashinfer-cubin"),
        "gpus": gpu_query,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    allowance = load_near_tie_allowance(args.allow_near_tie_audit)
    cell_files = sorted((run_dir / "cells").glob("*/cell_output.json"))
    cells = {path.parent.name: load_json(path) for path in cell_files}
    expected = {f"ar_{mode}" for mode in MODES}
    expected.update(
        f"moe_skip_top4_{mode}_d{draft_length}"
        for mode in MODES
        for draft_length in DRAFT_LENGTHS
    )
    expected.update(
        f"moe_skip_top8_{mode}_d{draft_length}"
        for mode in MODES
        for draft_length in (4, 32)
    )
    missing = expected - cells.keys()
    if missing:
        raise RuntimeError(f"Missing completed cells: {sorted(missing)}")

    audit = collect_correctness(run_dir, cells, allowance)
    (run_dir / "outputs.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    if audit["status"] == "failed":
        raise RuntimeError(f"AR parity failed in {audit['num_failures']} sample checks")
    trace = combine_traces(run_dir, cells)
    acceptance = write_acceptance_summary(run_dir, cells)
    position_metrics = write_position_metrics(run_dir, trace)
    write_rank_recall_summary(run_dir, trace)
    make_plots(run_dir, acceptance, position_metrics)
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "RUN_COMPLETE").write_text(audit["status"] + "\n", encoding="utf-8")
    print(f"RUN_COMPLETE {run_dir}")


if __name__ == "__main__":
    main()
