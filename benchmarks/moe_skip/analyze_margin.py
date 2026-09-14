# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import bisect
import csv
import importlib.metadata
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

DRAFT_LENGTHS = (4, 8, 16, 32)
GROUPS = (
    "top1_correct",
    "top1_wrong_target_rank2",
    "top1_wrong_target_rank3_plus",
)
MARGIN_FIELDS = (
    "draft_top1_minus_top2",
    "target_top1_minus_top2",
    "target_draft_top1_minus_draft_top2",
)
THRESHOLDS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
REQUIRED_TRACE_FIELDS = {
    "draft_length",
    "request_id",
    "verify_step",
    "draft_position",
    "draft_top8_token_ids",
    "draft_argmax_ordered_top8_token_ids",
    "draft_argmax_token_id",
    "draft_runner_up_token_id",
    "draft_top1_logit",
    "draft_top2_logit",
    "draft_top1_minus_top2",
    "target_top1_token_id",
    "target_top2_token_ids",
    "target_top1_logit",
    "target_top2_logit",
    "target_top1_minus_top2",
    "target_logit_for_draft_top1",
    "target_logit_for_draft_top2",
    "target_draft_top1_minus_draft_top2",
    "accepted_draft_tokens",
    "valid_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-trace")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def target_rank(row: dict) -> int:
    target = row["target_top1_token_id"]
    ranked_tokens = row.get(
        "draft_argmax_ordered_top8_token_ids", row["draft_top8_token_ids"]
    )
    try:
        return ranked_tokens.index(target) + 1
    except ValueError:
        return 9


def outcome_for_rank(rank: int) -> str:
    if rank == 1:
        return "top1_correct"
    if rank == 2:
        return "top1_wrong_target_rank2"
    return "top1_wrong_target_rank3_plus"


def output_position(output: dict, row: dict) -> int:
    metrics = output["spec_decode_metrics"]
    generated_before_step = 1 + sum(
        accepted + 1 for accepted in metrics["per_step_accepted"][: row["verify_step"]]
    )
    return generated_before_step + row["draft_position"] - 1


def combine_traces(run_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    combined = []
    cells = {}
    for draft_length in DRAFT_LENGTHS:
        name = f"moe_skip_top4_graph_d{draft_length}"
        cell_dir = run_dir / "cells" / name
        cell = load_json(cell_dir / "cell_output.json")
        cells[name] = cell
        request_to_output = {
            str(output["request_id"]): output for output in cell["outputs"]
        }
        trace_path = cell_dir / "trace" / "raw_trace.jsonl"
        with trace_path.open(encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                row = json.loads(line)
                missing = REQUIRED_TRACE_FIELDS - row.keys()
                if missing:
                    raise RuntimeError(
                        f"{trace_path}:{line_number}: missing {sorted(missing)}"
                    )
                output = request_to_output[str(row["request_id"])]
                position = output_position(output, row)
                row["sample_index"] = output["sample_index"]
                row["prompt_sha256"] = output["prompt_sha256"]
                row["output_position"] = position
                row["valid_mask"] = bool(
                    row["valid_mask"] and position < len(output["token_ids"])
                )
                rank = target_rank(row)
                row["target_rank_in_draft_top8"] = rank
                row["outcome"] = outcome_for_rank(rank)
                combined.append(row)
    output_path = run_dir / "raw_margin_trace.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in combined:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
    return combined, cells


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def descriptive_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def common_language_probability(larger: list[float], smaller: list[float]) -> float:
    ordered = sorted(smaller)
    favorable = 0.0
    for value in larger:
        below = bisect.bisect_left(ordered, value)
        at_most = bisect.bisect_right(ordered, value)
        favorable += below + 0.5 * (at_most - below)
    return favorable / (len(larger) * len(smaller))


def threshold_fractions(values: list[float]) -> dict[str, dict[str, float | int]]:
    return {
        str(threshold): {
            "count": sum(value <= threshold for value in values),
            "fraction": sum(value <= threshold for value in values) / len(values),
        }
        for threshold in THRESHOLDS
    }


def sign_counts(values: list[float]) -> dict[str, int]:
    return {
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
    }


def write_group_summary(run_dir: Path, valid: list[dict]) -> list[dict]:
    rows = []
    scopes: list[tuple[int | str, list[dict]]] = [("all", valid)]
    scopes.extend(
        (draft_length, [r for r in valid if r["draft_length"] == draft_length])
        for draft_length in DRAFT_LENGTHS
    )
    for draft_length, scoped_rows in scopes:
        for group in GROUPS:
            group_rows = [row for row in scoped_rows if row["outcome"] == group]
            if not group_rows:
                continue
            for field in MARGIN_FIELDS:
                values = [float(row[field]) for row in group_rows]
                rows.append(
                    {
                        "draft_length": draft_length,
                        "outcome": group,
                        "margin": field,
                        **descriptive_stats(values),
                    }
                )
    with (run_dir / "margin_group_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_threshold_summary(run_dir: Path, valid: list[dict]) -> list[dict]:
    rows = []
    scopes: list[tuple[int | str, list[dict]]] = [("all", valid)]
    scopes.extend(
        (draft_length, [r for r in valid if r["draft_length"] == draft_length])
        for draft_length in DRAFT_LENGTHS
    )
    for draft_length, scoped_rows in scopes:
        for group in GROUPS:
            values = [
                float(row["draft_top1_minus_top2"])
                for row in scoped_rows
                if row["outcome"] == group
            ]
            if not values:
                continue
            for threshold in THRESHOLDS:
                count = sum(value <= threshold for value in values)
                rows.append(
                    {
                        "draft_length": draft_length,
                        "outcome": group,
                        "threshold": threshold,
                        "n": len(values),
                        "count_at_or_below": count,
                        "fraction_at_or_below": count / len(values),
                    }
                )
    with (run_dir / "margin_threshold_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def rank_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    result = {}
    for draft_length in (*DRAFT_LENGTHS, "all"):
        scoped = (
            rows
            if draft_length == "all"
            else [row for row in rows if row["draft_length"] == draft_length]
        )
        counts = Counter(target_rank(row) for row in scoped)
        result[str(draft_length)] = {
            "n": len(scoped),
            **{f"rank_{rank}": counts[rank] for rank in range(1, 9)},
            "outside_top8": counts[9],
        }
    return result


def load_reference_rank_counts(path: Path) -> dict[str, dict[str, int]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rank_counts([row for row in rows if row["valid_mask"]])


def write_rank_distribution(run_dir: Path, counts: dict[str, dict]) -> None:
    rows = [
        {"draft_length": draft_length, **values}
        for draft_length, values in counts.items()
    ]
    with (run_dir / "rank_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(run_dir: Path, valid: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = {
        "top1_correct": "Top-1 correct",
        "top1_wrong_target_rank2": "Wrong: target at draft rank 2",
        "top1_wrong_target_rank3_plus": "Wrong: target at draft rank >=3",
    }
    colors = {
        "top1_correct": "#2E7D32",
        "top1_wrong_target_rank2": "#D55E00",
        "top1_wrong_target_rank3_plus": "#7B1FA2",
    }
    grouped = {
        group: [
            float(row["draft_top1_minus_top2"])
            for row in valid
            if row["outcome"] == group
        ]
        for group in GROUPS
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)

    for group in GROUPS:
        values = np.sort(np.asarray(grouped[group]))
        axes[0, 0].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=f"{labels[group]} (n={len(values)})",
            color=colors[group],
        )
    axes[0, 0].set_xscale("symlog", linthresh=0.125)
    axes[0, 0].set_xlabel("Draft Top-1 minus Top-2 logit")
    axes[0, 0].set_ylabel("Empirical cumulative fraction")
    axes[0, 0].set_title("Draft-margin ECDF")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    box_values = [grouped[group] for group in GROUPS]
    axes[0, 1].boxplot(
        box_values,
        tick_labels=[labels[group] for group in GROUPS],
        showfliers=False,
        whis=(5, 95),
    )
    axes[0, 1].set_yscale("symlog", linthresh=0.125)
    axes[0, 1].set_ylabel("Draft Top-1 minus Top-2 logit")
    axes[0, 1].set_title("Median, IQR, and 5th-95th percentiles")
    axes[0, 1].tick_params(axis="x", labelrotation=12)
    axes[0, 1].grid(axis="y", alpha=0.25)

    for group in GROUPS:
        values = grouped[group]
        fractions = [
            sum(value <= threshold for value in values) / len(values)
            for threshold in THRESHOLDS
        ]
        axes[1, 0].plot(
            THRESHOLDS,
            fractions,
            marker="o",
            label=labels[group],
            color=colors[group],
        )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_xlabel("Draft-margin threshold")
    axes[1, 0].set_ylabel("Fraction at or below threshold")
    axes[1, 0].set_title("Near-tie concentration")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    target_pair_values = [
        [
            float(row["target_draft_top1_minus_draft_top2"])
            for row in valid
            if row["outcome"] == group
        ]
        for group in GROUPS
    ]
    axes[1, 1].boxplot(
        target_pair_values,
        tick_labels=[labels[group] for group in GROUPS],
        showfliers=False,
        whis=(5, 95),
    )
    axes[1, 1].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[1, 1].set_yscale("symlog", linthresh=0.125)
    axes[1, 1].set_ylabel("Target logit(draft Top-1) - logit(draft Top-2)")
    axes[1, 1].set_title("Target re-ranking of the same draft pair")
    axes[1, 1].tick_params(axis="x", labelrotation=12)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("MoE-Skip Top-2 logit-margin analysis", fontsize=15)
    fig.savefig(run_dir / "top2_logit_margin_by_outcome.png", dpi=180)
    plt.close(fig)


def environment_info() -> dict:
    import torch

    gpu_query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
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


def write_results(run_dir: Path, analysis: dict) -> None:
    comparison = analysis["overall_comparison"]
    correct = comparison["top1_correct"]
    wrong = comparison["top1_wrong_all"]
    rank2 = comparison["top1_wrong_target_rank2"]
    thresholds = analysis["near_tie_fractions"]
    rank_counts_all = analysis["rank_counts"]["all"]
    wrong_count = analysis["valid_trace_rows"] - rank_counts_all["rank_1"]
    lines = [
        "# MoE-Skip Top-2 logit-margin analysis",
        "",
        "The primary margin is the draft Top-1 logit minus its Top-2 logit. "
        "Rows are grouped by whether the full target Top-1 matches the draft "
        "ranking on the same actual draft prefix.",
        "",
        f"- Valid proposal positions: {analysis['valid_trace_rows']}",
        f"- Top-1 correct: n={correct['n']}, median={correct['median']:.6g}, "
        f"mean={correct['mean']:.6g}",
        f"- Top-1 wrong: n={wrong['n']}, median={wrong['median']:.6g}, "
        f"mean={wrong['mean']:.6g}",
        f"- Wrong with target at draft rank 2: n={rank2['n']}, "
        f"median={rank2['median']:.6g}",
        "- Target at the actual draft runner-up among Top-1 errors: "
        f"{rank_counts_all['rank_2']}/{wrong_count} "
        f"({rank_counts_all['rank_2'] / wrong_count:.2%})",
        "- Margin <= 1.0: "
        f"{thresholds['top1_correct']['1.0']['fraction']:.2%} correct versus "
        f"{thresholds['top1_wrong_all']['1.0']['fraction']:.2%} wrong",
        "- Margin <= 2.0: "
        f"{thresholds['top1_correct']['2.0']['fraction']:.2%} correct versus "
        f"{thresholds['top1_wrong_all']['2.0']['fraction']:.2%} wrong",
        "- P(correct margin > wrong margin), with half credit for ties: "
        f"{analysis['effect']['common_language_probability']:.6f}",
        "",
        "This is a conditional association, not proof that a small margin is "
        "the sole cause of every mismatch. The full target is evaluated on the "
        "actual draft prefix, including rows after an earlier rejection.",
        "",
    ]
    (run_dir / "MARGIN_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    trace, cells = combine_traces(run_dir)
    valid = [row for row in trace if row["valid_mask"]]
    invalid = [row for row in trace if not row["valid_mask"]]
    for row in valid:
        ordered_tokens = row["draft_argmax_ordered_top8_token_ids"]
        if ordered_tokens[0] != row["draft_argmax_token_id"]:
            raise RuntimeError(f"Draft argmax ordering is inconsistent: {row}")
        if ordered_tokens[1] != row["draft_runner_up_token_id"]:
            raise RuntimeError(f"Draft runner-up ordering is inconsistent: {row}")
        for field in MARGIN_FIELDS:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Non-finite {field}: {row}")
        recomputed = float(row["draft_top1_logit"]) - float(row["draft_top2_logit"])
        if not math.isclose(
            recomputed,
            float(row["draft_top1_minus_top2"]),
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"Draft margin does not match logits: {row}")
        if recomputed < -1e-6:
            raise RuntimeError(f"Negative draft Top-1/Top-2 margin: {row}")
        target_pair_recomputed = float(row["target_logit_for_draft_top1"]) - float(
            row["target_logit_for_draft_top2"]
        )
        if not math.isclose(
            target_pair_recomputed,
            float(row["target_draft_top1_minus_draft_top2"]),
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"Target draft-pair margin mismatch: {row}")

    group_summary = write_group_summary(run_dir, valid)
    threshold_summary = write_threshold_summary(run_dir, valid)
    counts = rank_counts(valid)
    write_rank_distribution(run_dir, counts)

    by_group = {
        group: [
            float(row["draft_top1_minus_top2"])
            for row in valid
            if row["outcome"] == group
        ]
        for group in GROUPS
    }
    wrong_all = (
        by_group["top1_wrong_target_rank2"] + by_group["top1_wrong_target_rank3_plus"]
    )
    comparison = {
        "top1_correct": descriptive_stats(by_group["top1_correct"]),
        "top1_wrong_all": descriptive_stats(wrong_all),
        "top1_wrong_target_rank2": descriptive_stats(
            by_group["top1_wrong_target_rank2"]
        ),
        "top1_wrong_target_rank3_plus": descriptive_stats(
            by_group["top1_wrong_target_rank3_plus"]
        ),
    }
    effect = {
        "common_language_probability": common_language_probability(
            by_group["top1_correct"], wrong_all
        ),
        "median_difference_correct_minus_wrong": (
            comparison["top1_correct"]["median"]
            - comparison["top1_wrong_all"]["median"]
        ),
    }
    near_tie_fractions = {
        "top1_correct": threshold_fractions(by_group["top1_correct"]),
        "top1_wrong_all": threshold_fractions(wrong_all),
        "top1_wrong_target_rank2": threshold_fractions(
            by_group["top1_wrong_target_rank2"]
        ),
        "top1_wrong_target_rank3_plus": threshold_fractions(
            by_group["top1_wrong_target_rank3_plus"]
        ),
    }
    target_pair_sign_counts = {
        group: sign_counts(
            [
                float(row["target_draft_top1_minus_draft_top2"])
                for row in valid
                if row["outcome"] == group
            ]
        )
        for group in GROUPS
    }
    exact_draft_tie_counts = {
        group: sum(
            row["outcome"] == group and float(row["draft_top1_minus_top2"]) == 0
            for row in valid
        )
        for group in GROUPS
    }
    reference = None
    if args.reference_trace:
        reference_path = Path(args.reference_trace).resolve()
        reference_counts = load_reference_rank_counts(reference_path)
        reference = {
            "path": str(reference_path),
            "rank_counts": reference_counts,
            "exact_rank_count_match": reference_counts == counts,
        }
    analysis = {
        "status": "completed_margin_analysis",
        "scope": {
            "draft_lengths": list(DRAFT_LENGTHS),
            "top_h": 4,
            "mode": "graph",
            "num_samples": 4,
            "max_tokens": 128,
            "batch_size": 1,
            "denominator": "all valid verifier-computed proposal positions",
        },
        "raw_trace_rows": len(trace),
        "valid_trace_rows": len(valid),
        "boundary_filtered_trace_rows": len(invalid),
        "rank_counts": counts,
        "overall_comparison": comparison,
        "effect": effect,
        "near_tie_fractions": near_tie_fractions,
        "target_pair_sign_counts": target_pair_sign_counts,
        "exact_draft_tie_counts": exact_draft_tie_counts,
        "reference": reference,
    }
    (run_dir / "margin_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2) + "\n", encoding="utf-8"
    )
    make_plot(run_dir, valid)
    write_results(run_dir, analysis)

    errors = []
    if len(cells) != len(DRAFT_LENGTHS):
        errors.append("did not load four cells")
    if any(len(cell["outputs"]) != 4 for cell in cells.values()):
        errors.append("a cell does not contain four outputs")
    if any(
        len(output["token_ids"]) != 128
        for cell in cells.values()
        for output in cell["outputs"]
    ):
        errors.append("an output does not contain 128 tokens")
    if any(not 0 <= row["output_position"] < 128 for row in valid):
        errors.append("valid trace contains an out-of-bound output position")
    if sum(len(values) for values in by_group.values()) != len(valid):
        errors.append("outcome groups do not partition the valid trace")
    if target_pair_sign_counts["top1_correct"]["negative"]:
        errors.append("target ranks a runner-up above a supposedly correct argmax")
    if target_pair_sign_counts["top1_wrong_target_rank2"]["positive"]:
        errors.append("target did not re-rank a selected runner-up above draft argmax")
    summary_overall_count = sum(
        int(row["n"])
        for row in group_summary
        if row["draft_length"] == "all" and row["margin"] == "draft_top1_minus_top2"
    )
    if summary_overall_count != len(valid):
        errors.append("group-summary denominators do not sum to valid rows")
    if not threshold_summary:
        errors.append("threshold summary is empty")
    required_files = (
        "commands.json",
        "raw_margin_trace.jsonl",
        "margin_group_summary.csv",
        "margin_threshold_summary.csv",
        "rank_distribution.csv",
        "margin_analysis.json",
        "environment.json",
        "MARGIN_RESULTS.md",
        "top2_logit_margin_by_outcome.png",
    )
    for name in required_files:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {name}")
    audit = {
        "status": "passed" if not errors else "failed",
        "completed_cells": sorted(cells),
        "output_count": sum(len(cell["outputs"]) for cell in cells.values()),
        "output_tokens_per_sample": 128,
        "raw_trace_rows": len(trace),
        "valid_trace_rows": len(valid),
        "boundary_filtered_trace_rows": len(invalid),
        "outcome_counts": {group: len(values) for group, values in by_group.items()},
        "errors": errors,
    }
    (run_dir / "margin_analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Margin analysis audit failed: {errors}")
    (run_dir / "RUN_COMPLETE").write_text(
        "completed_margin_analysis\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
