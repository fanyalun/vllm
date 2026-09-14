# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from analyze_margin import (
    GROUPS,
    MARGIN_FIELDS,
    REQUIRED_TRACE_FIELDS,
    THRESHOLDS,
    common_language_probability,
    descriptive_stats,
    environment_info,
    outcome_for_rank,
    sign_counts,
    target_rank,
    threshold_fractions,
)
from analyze_margin import (
    make_plot as make_margin_plot,
)
from benchmark_integrity import validate_target

CATEGORIES = ("human_eval", "alpaca", "gsm8k", "ultra_feedback")
NUM_SAMPLES = 128
SAMPLES_PER_CATEGORY = 32
MAX_TOKENS = 512
MAX_MODEL_LEN = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_position(output: dict, row: dict) -> int:
    metrics = output["spec_decode_metrics"]
    generated_before_step = 1 + sum(
        accepted + 1 for accepted in metrics["per_step_accepted"][: row["verify_step"]]
    )
    return generated_before_step + row["draft_position"] - 1


def cell_name(method: str, draft_length: int) -> str:
    if method == "moe_skip":
        return f"moe_skip_top4_graph_d{draft_length}"
    return f"{method}_graph_d{draft_length}"


def combine_traces(
    run_dir: Path, method: str, draft_lengths: tuple[int, ...]
) -> tuple[list[dict], dict[str, dict]]:
    combined = []
    cells = {}
    contract = load_json(run_dir / "EXPERIMENT_CONTRACT.json")
    for draft_length in draft_lengths:
        name = cell_name(method, draft_length)
        cell_dir = run_dir / "cells" / name
        cell = load_json(cell_dir / "cell_output.json")
        validate_target(cell, contract["model"])
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
                output = request_to_output.get(str(row["request_id"]))
                if output is None:
                    raise RuntimeError(
                        f"{trace_path}:{line_number}: unknown request "
                        f"{row['request_id']!r}"
                    )
                position = output_position(output, row)
                row.update(
                    {
                        "sample_index": output["sample_index"],
                        "source_index": output["source_index"],
                        "category": output["category"],
                        "prompt_sha256": output["prompt_sha256"],
                        "output_position": position,
                        "valid_mask": bool(
                            row["valid_mask"] and position < len(output["token_ids"])
                        ),
                    }
                )
                rank = target_rank(row)
                row["target_rank_in_draft_top8"] = rank
                row["outcome"] = outcome_for_rank(rank)
                combined.append(row)
    with (run_dir / "raw_trace.jsonl").open("w", encoding="utf-8") as output_file:
        for row in combined:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
    return combined, cells


def rank_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    hits = {
        rank: sum(
            row["target_top1_token_id"]
            in row["draft_argmax_ordered_top8_token_ids"][:rank]
            for row in rows
        )
        for rank in (1, 2, 3, 8)
    }
    errors = n - hits[1]
    return {
        "n": n,
        "top1_hits": hits[1],
        "top1_precision": hits[1] / n,
        "top2_hits": hits[2],
        "top2_recall": hits[2] / n,
        "top3_hits": hits[3],
        "top3_recall": hits[3] / n,
        "top8_hits": hits[8],
        "top8_recall": hits[8] / n,
        "top1_errors": errors,
        "target_at_runner_up": hits[2] - hits[1],
        "runner_up_share_of_top1_errors": (
            (hits[2] - hits[1]) / errors if errors else None
        ),
    }


def make_scopes(
    rows: list[dict], draft_lengths: tuple[int, ...]
) -> list[tuple[int | str, str, list[dict]]]:
    scopes = [("all", "all", rows)]
    scopes.extend(
        (
            draft_length,
            "all",
            [row for row in rows if row["draft_length"] == draft_length],
        )
        for draft_length in draft_lengths
    )
    scopes.extend(
        (
            "all",
            category,
            [row for row in rows if row["category"] == category],
        )
        for category in CATEGORIES
    )
    scopes.extend(
        (
            draft_length,
            category,
            [
                row
                for row in rows
                if row["draft_length"] == draft_length and row["category"] == category
            ],
        )
        for draft_length in draft_lengths
        for category in CATEGORIES
    )
    return scopes


def write_quality_summary(
    run_dir: Path, valid: list[dict], draft_lengths: tuple[int, ...]
) -> list[dict]:
    rows = [
        {
            "draft_length": draft_length,
            "category": category,
            **rank_metrics(scoped),
        }
        for draft_length, category, scoped in make_scopes(valid, draft_lengths)
    ]
    write_csv(run_dir / "quality_summary.csv", rows)
    return rows


def write_acceptance_summary(
    run_dir: Path,
    cells: dict[str, dict],
    method: str,
    draft_lengths: tuple[int, ...],
) -> list[dict]:
    rows = []
    for draft_length in draft_lengths:
        outputs = cells[cell_name(method, draft_length)]["outputs"]
        for category in ("all", *CATEGORIES):
            selected = [
                output
                for output in outputs
                if category == "all" or output["category"] == category
            ]
            metrics = [output["spec_decode_metrics"] for output in selected]
            verify_steps = sum(metric["num_spec_steps"] for metric in metrics)
            accepted = sum(metric["num_accepted_draft_tokens"] for metric in metrics)
            drafted = sum(metric["num_draft_tokens"] for metric in metrics)
            rows.append(
                {
                    "draft_length": draft_length,
                    "category": category,
                    "num_samples": len(selected),
                    "verify_steps": verify_steps,
                    "drafted_tokens": drafted,
                    "accepted_draft_tokens": accepted,
                    "draft_acceptance_fraction": accepted / drafted,
                    "mean_acceptance_length": 1 + accepted / verify_steps,
                }
            )
    write_csv(run_dir / "acceptance_summary.csv", rows)
    return rows


def write_position_metrics(
    run_dir: Path, valid: list[dict], draft_lengths: tuple[int, ...]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row["draft_length"], row["draft_position"])].append(row)
    rows = []
    for draft_length in draft_lengths:
        for position in range(1, draft_length + 1):
            scoped = grouped[(draft_length, position)]
            if not scoped:
                raise RuntimeError(
                    f"No valid rows for D={draft_length}, position={position}"
                )
            rows.append(
                {
                    "draft_length": draft_length,
                    "draft_position": position,
                    **rank_metrics(scoped),
                }
            )
    write_csv(run_dir / "position_metrics.csv", rows)
    return rows


def write_margin_summaries(
    run_dir: Path,
    valid: list[dict],
    draft_lengths: tuple[int, ...],
) -> tuple[list[dict], list[dict], list[dict]]:
    group_rows = []
    threshold_rows = []
    effect_rows = []
    for draft_length, category, scoped in make_scopes(valid, draft_lengths):
        by_group = {
            group: [row for row in scoped if row["outcome"] == group]
            for group in GROUPS
        }
        correct = [
            float(row["draft_top1_minus_top2"]) for row in by_group["top1_correct"]
        ]
        wrong = [
            float(row["draft_top1_minus_top2"])
            for row in scoped
            if row["outcome"] != "top1_correct"
        ]
        if correct and wrong:
            effect_rows.append(
                {
                    "draft_length": draft_length,
                    "category": category,
                    "correct_n": len(correct),
                    "wrong_n": len(wrong),
                    "correct_median": descriptive_stats(correct)["median"],
                    "wrong_median": descriptive_stats(wrong)["median"],
                    "median_difference": (
                        descriptive_stats(correct)["median"]
                        - descriptive_stats(wrong)["median"]
                    ),
                    "probability_correct_margin_greater": (
                        common_language_probability(correct, wrong)
                    ),
                }
            )
        for group, rows in by_group.items():
            if not rows:
                continue
            for margin_field in MARGIN_FIELDS:
                values = [float(row[margin_field]) for row in rows]
                group_rows.append(
                    {
                        "draft_length": draft_length,
                        "category": category,
                        "outcome": group,
                        "margin": margin_field,
                        **descriptive_stats(values),
                    }
                )
            values = [float(row["draft_top1_minus_top2"]) for row in rows]
            for threshold in THRESHOLDS:
                count = sum(value <= threshold for value in values)
                threshold_rows.append(
                    {
                        "draft_length": draft_length,
                        "category": category,
                        "outcome": group,
                        "threshold": threshold,
                        "n": len(values),
                        "count_at_or_below": count,
                        "fraction_at_or_below": count / len(values),
                    }
                )
    write_csv(run_dir / "margin_group_summary.csv", group_rows)
    write_csv(run_dir / "margin_threshold_summary.csv", threshold_rows)
    write_csv(run_dir / "margin_effect_summary.csv", effect_rows)
    return group_rows, threshold_rows, effect_rows


def make_acceptance_plot(
    run_dir: Path, rows: list[dict], draft_lengths: tuple[int, ...]
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for category in ("all", *CATEGORIES):
        selected = [row for row in rows if row["category"] == category]
        axis.plot(
            [row["draft_length"] for row in selected],
            [row["mean_acceptance_length"] for row in selected],
            marker="o",
            linewidth=2 if category == "all" else 1.4,
            label="overall" if category == "all" else category,
        )
    axis.set_xlabel("Draft length D")
    axis.set_ylabel("Mean acceptance length (including target bonus)")
    axis.set_xticks(draft_lengths)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "acceptance_length_by_category.png", dpi=180)
    plt.close(fig)


def make_quality_plot(
    run_dir: Path, rows: list[dict], draft_lengths: tuple[int, ...]
) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("top1_precision", "Top-1 precision"),
        ("top2_recall", "Top-2 recall"),
        ("top3_recall", "Top-3 recall"),
        ("top8_recall", "Top-8 recall"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        for category in ("all", *CATEGORIES):
            selected = [
                row
                for row in rows
                if row["category"] == category and row["draft_length"] != "all"
            ]
            axis.plot(
                [row["draft_length"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                linewidth=2 if category == "all" else 1.3,
                label="overall" if category == "all" else category,
            )
        axis.set_title(title)
        axis.set_xlabel("Draft length D")
        axis.set_ylabel("Fraction")
        axis.set_xticks(draft_lengths)
        axis.set_ylim(0.75 if metric == "top1_precision" else 0.9, 1.005)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(run_dir / "token_quality_by_category.png", dpi=180)
    plt.close(fig)


def make_position_plot(
    run_dir: Path, rows: list[dict], draft_lengths: tuple[int, ...]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = (
        ("top1_precision", "Top-1 precision"),
        ("top2_recall", "Top-2 recall"),
        ("top3_recall", "Top-3 recall"),
        ("top8_recall", "Top-8 recall"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), constrained_layout=True)
    max_position = max(draft_lengths)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.full((len(draft_lengths), max_position), np.nan)
        for row in rows:
            d_index = draft_lengths.index(row["draft_length"])
            matrix[d_index, row["draft_position"] - 1] = row[metric]
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title)
        axis.set_yticks(range(len(draft_lengths)), draft_lengths)
        axis.set_ylabel("D")
        axis.set_xlim(-0.5, max_position - 0.5)
        displayed_positions = tuple(
            dict.fromkeys((1, *range(8, max_position + 1, 8), max_position))
        )
        axis.set_xticks(
            [position - 1 for position in displayed_positions],
            displayed_positions,
        )
    axes[-1].set_xlabel("Draft position")
    fig.colorbar(image, ax=axes, label="Fraction", location="right")
    fig.savefig(run_dir / "token_quality_by_position.png", dpi=180)
    plt.close(fig)


def make_category_margin_plot(
    run_dir: Path, group_rows: list[dict], valid: list[dict]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    medians = {}
    for category in CATEGORIES:
        correct = next(
            row
            for row in group_rows
            if row["draft_length"] == "all"
            and row["category"] == category
            and row["outcome"] == "top1_correct"
            and row["margin"] == "draft_top1_minus_top2"
        )
        wrong_rows = [
            row
            for row in group_rows
            if row["draft_length"] == "all"
            and row["category"] == category
            and row["outcome"] != "top1_correct"
            and row["margin"] == "draft_top1_minus_top2"
        ]
        wrong_values = []
        for row in wrong_rows:
            source_rows = [
                trace_row
                for trace_row in valid
                if trace_row["category"] == category
                and trace_row["outcome"] == row["outcome"]
            ]
            wrong_values.extend(
                float(trace_row["draft_top1_minus_top2"]) for trace_row in source_rows
            )
        medians[category] = (
            correct["median"],
            descriptive_stats(wrong_values)["median"],
        )
    x = np.arange(len(CATEGORIES))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(
        x - width / 2,
        [medians[category][0] for category in CATEGORIES],
        width,
        label="Top-1 correct",
    )
    axis.bar(
        x + width / 2,
        [medians[category][1] for category in CATEGORIES],
        width,
        label="Top-1 wrong",
    )
    axis.set_xticks(x, CATEGORIES)
    axis.set_ylabel("Median draft Top-1 minus Top-2 logit")
    axis.set_title("Logit-margin separation by category")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "margin_median_by_category.png", dpi=180)
    plt.close(fig)


def write_results(
    run_dir: Path,
    quality: list[dict],
    acceptance: list[dict],
    effects: list[dict],
    draft_lengths: tuple[int, ...],
    model: str,
    method: str,
) -> None:
    overall_quality = {
        row["draft_length"]: row
        for row in quality
        if row["category"] == "all" and row["draft_length"] != "all"
    }
    overall_acceptance = {
        row["draft_length"]: row for row in acceptance if row["category"] == "all"
    }
    overall_effect = next(
        row
        for row in effects
        if row["category"] == "all" and row["draft_length"] == "all"
    )
    lines = [
        f"# {Path(model).name} {method} 128x512 multi-category evaluation",
        "",
        "128 raw-text prompts: 32 each from HumanEval, Alpaca, GSM8K, and "
        "UltraFeedback. Each request generated exactly 512 tokens with greedy "
        "decoding and ignore_eos=True.",
        "",
        "| D | Valid proposal rows | Top-1 | Top-2 | Top-3 | Top-8 | "
        "Mean acceptance length |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for draft_length in draft_lengths:
        q = overall_quality[draft_length]
        a = overall_acceptance[draft_length]
        lines.append(
            f"| {draft_length} | {q['n']} | {q['top1_precision']:.2%} | "
            f"{q['top2_recall']:.2%} | {q['top3_recall']:.2%} | "
            f"{q['top8_recall']:.2%} | {a['mean_acceptance_length']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Across all D values, the empirical probability that a Top-1-correct "
            "position has a larger draft Top-1/Top-2 margin than a Top-1-wrong "
            "position is "
            f"{overall_effect['probability_correct_margin_greater']:.2%}. "
            f"Median margins are {overall_effect['correct_median']:.4g} versus "
            f"{overall_effect['wrong_median']:.4g}.",
            "",
            "All quality denominators are valid target-verifier proposal rows on "
            "the actual draft prefix. Boundary-filtered, padding, and unverified "
            "positions are excluded.",
            "",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    experiment_contract = load_json(run_dir / "EXPERIMENT_CONTRACT.json")
    method = experiment_contract.get("method", "moe_skip")
    draft_lengths = tuple(experiment_contract["draft_lengths"])
    if not draft_lengths or any(
        not isinstance(length, int) or length <= 0 for length in draft_lengths
    ):
        raise RuntimeError("Experiment contract contains invalid draft lengths")
    expected_contract: dict[str, object] = {
        "num_samples": NUM_SAMPLES,
        "category_counts": {category: SAMPLES_PER_CATEGORY for category in CATEGORIES},
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "mode": "graph",
        "batch_size": 1,
    }
    if method == "moe_skip":
        expected_contract.update({"top_h": 4, "target_top_k": 8})
    for key, value in expected_contract.items():
        if experiment_contract.get(key) != value:
            raise RuntimeError(
                f"Experiment contract mismatch: expected {key}={value!r}, "
                f"got {experiment_contract.get(key)!r}"
            )
    dataset_manifest = load_json(Path(args.dataset_manifest).resolve())
    trace, cells = combine_traces(run_dir, method, draft_lengths)
    valid = [row for row in trace if row["valid_mask"]]
    invalid = [row for row in trace if not row["valid_mask"]]

    for row in valid:
        ordered = row["draft_argmax_ordered_top8_token_ids"]
        if ordered[0] != row["draft_argmax_token_id"]:
            raise RuntimeError(f"Draft argmax ordering mismatch: {row}")
        if ordered[1] != row["draft_runner_up_token_id"]:
            raise RuntimeError(f"Draft runner-up ordering mismatch: {row}")
        for field in MARGIN_FIELDS:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Non-finite margin: {row}")
        draft_margin = float(row["draft_top1_logit"]) - float(row["draft_top2_logit"])
        if not math.isclose(
            draft_margin, float(row["draft_top1_minus_top2"]), abs_tol=1e-6
        ):
            raise RuntimeError(f"Draft margin mismatch: {row}")
        if draft_margin < -1e-6:
            raise RuntimeError(f"Negative draft margin: {row}")

    quality = write_quality_summary(run_dir, valid, draft_lengths)
    acceptance = write_acceptance_summary(run_dir, cells, method, draft_lengths)
    positions = write_position_metrics(run_dir, valid, draft_lengths)
    group_rows, threshold_rows, effect_rows = write_margin_summaries(
        run_dir, valid, draft_lengths
    )
    (run_dir / "outputs.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2) + "\n", encoding="utf-8"
    )

    make_margin_plot(run_dir, valid)
    make_acceptance_plot(run_dir, acceptance, draft_lengths)
    make_quality_plot(run_dir, quality, draft_lengths)
    make_position_plot(run_dir, positions, draft_lengths)
    make_category_margin_plot(run_dir, group_rows, valid)
    write_results(
        run_dir,
        quality,
        acceptance,
        effect_rows,
        draft_lengths,
        experiment_contract["model"],
        method,
    )

    overall_quality = next(
        row
        for row in quality
        if row["draft_length"] == "all" and row["category"] == "all"
    )
    overall_effect = next(
        row
        for row in effect_rows
        if row["draft_length"] == "all" and row["category"] == "all"
    )
    category_quality = {
        category: next(
            row
            for row in quality
            if row["draft_length"] == "all" and row["category"] == category
        )
        for category in CATEGORIES
    }
    analysis = {
        "status": "completed_large_multicategory_analysis",
        "scope": {
            "num_samples": NUM_SAMPLES,
            "samples_per_category": SAMPLES_PER_CATEGORY,
            "categories": list(CATEGORIES),
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "draft_lengths": list(draft_lengths),
            "method": method,
            "spec_model": experiment_contract.get("spec_model"),
            "mode": "graph",
            "batch_size": 1,
            "prompt_format": "raw_text_no_chat_template",
            "denominator": "all valid verifier-computed proposal positions",
        },
        "dataset_manifest": dataset_manifest,
        "raw_trace_rows": len(trace),
        "valid_trace_rows": len(valid),
        "boundary_filtered_trace_rows": len(invalid),
        "overall_quality": overall_quality,
        "category_quality": category_quality,
        "overall_margin_effect": overall_effect,
        "target_pair_sign_counts": {
            group: sign_counts(
                [
                    float(row["target_draft_top1_minus_draft_top2"])
                    for row in valid
                    if row["outcome"] == group
                ]
            )
            for group in GROUPS
        },
        "near_tie_fractions": {
            "top1_correct": threshold_fractions(
                [
                    float(row["draft_top1_minus_top2"])
                    for row in valid
                    if row["outcome"] == "top1_correct"
                ]
            ),
            "top1_wrong": threshold_fractions(
                [
                    float(row["draft_top1_minus_top2"])
                    for row in valid
                    if row["outcome"] != "top1_correct"
                ]
            ),
        },
    }
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )

    errors = []
    expected_cells = {cell_name(method, draft_length) for draft_length in draft_lengths}
    if set(cells) != expected_cells:
        errors.append("cell set is incomplete")
    reference_hashes = None
    for name, cell in cells.items():
        outputs = cell["outputs"]
        if len(outputs) != NUM_SAMPLES:
            errors.append(f"{name}: expected {NUM_SAMPLES} outputs")
        if any(len(output["token_ids"]) != MAX_TOKENS for output in outputs):
            errors.append(f"{name}: an output is not {MAX_TOKENS} tokens")
        counts = Counter(output["category"] for output in outputs)
        if counts != Counter(
            {category: SAMPLES_PER_CATEGORY for category in CATEGORIES}
        ):
            errors.append(f"{name}: category counts mismatch")
        hashes = [output["prompt_sha256"] for output in outputs]
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            errors.append(f"{name}: prompt set or ordering differs")
        log_path = run_dir / "cells" / name / "run.log"
        log = log_path.read_text(encoding="utf-8")
        if log.count("CELL_COMPLETE") != 1:
            errors.append(f"{name}: CELL_COMPLETE count is not one")
        if log.count("Loading model from scratch") < 1:
            errors.append(f"{name}: no model-load marker found")
        if "Traceback (most recent call last)" in log:
            errors.append(f"{name}: traceback found in successful log")
    if any(not 0 <= row["output_position"] < MAX_TOKENS for row in valid):
        errors.append("valid trace contains out-of-bound output positions")
    if analysis["target_pair_sign_counts"]["top1_correct"]["negative"]:
        errors.append("negative target pair gap in Top-1-correct group")
    if analysis["target_pair_sign_counts"]["top1_wrong_target_rank2"]["positive"]:
        errors.append("positive target pair gap in runner-up mismatch group")
    if sum(row["n"] for row in positions) != len(valid):
        errors.append("position-metric denominators do not sum to valid trace")
    for row in acceptance:
        expected_mean = 1 + row["accepted_draft_tokens"] / row["verify_steps"]
        if not math.isclose(row["mean_acceptance_length"], expected_mean):
            errors.append(f"acceptance formula mismatch: {row}")
    if not group_rows or not threshold_rows or not effect_rows:
        errors.append("one or more summary tables are empty")
    commands = load_json(run_dir / "commands.json")
    if len(commands) != len(draft_lengths):
        errors.append(f"commands.json does not contain {len(draft_lengths)} cells")
    required_files = (
        "EXPERIMENT_CONTRACT.json",
        "commands.json",
        "outputs.json",
        "raw_trace.jsonl",
        "quality_summary.csv",
        "acceptance_summary.csv",
        "position_metrics.csv",
        "margin_group_summary.csv",
        "margin_threshold_summary.csv",
        "margin_effect_summary.csv",
        "analysis.json",
        "environment.json",
        "RESULTS.md",
        "acceptance_length_by_category.png",
        "token_quality_by_category.png",
        "token_quality_by_position.png",
        "top2_logit_margin_by_outcome.png",
        "margin_median_by_category.png",
    )
    for name in required_files:
        path = run_dir / name
        if not path.is_file() or not path.stat().st_size:
            errors.append(f"missing or empty artifact: {name}")
    if (run_dir / "RUN_FAILED.json").exists():
        errors.append("RUN_FAILED.json is present")

    audit = {
        "status": "passed" if not errors else "failed",
        "completed_cells": sorted(cells),
        "num_samples_per_cell": NUM_SAMPLES,
        "output_tokens_per_sample": MAX_TOKENS,
        "total_outputs": sum(len(cell["outputs"]) for cell in cells.values()),
        "total_generated_tokens": sum(
            len(output["token_ids"])
            for cell in cells.values()
            for output in cell["outputs"]
        ),
        "category_counts_per_cell": {
            category: SAMPLES_PER_CATEGORY for category in CATEGORIES
        },
        "raw_trace_rows": len(trace),
        "valid_trace_rows": len(valid),
        "boundary_filtered_trace_rows": len(invalid),
        "errors": errors,
    }
    (run_dir / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    correctness_audit = {
        **audit,
        "ar_exact_match": {
            "status": "not_rerun_by_user_scope",
            "reason": (
                "Single-GPU batch-1 correctness was previously accepted; this "
                "expanded run audits generation and trace integrity only."
            ),
        },
    }
    (run_dir / "correctness_audit.json").write_text(
        json.dumps(correctness_audit, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Artifact audit failed: {errors}")
    (run_dir / "RUN_COMPLETE").write_text(
        "completed_large_multicategory_analysis\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
