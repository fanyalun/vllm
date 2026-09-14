# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the Phase-B throughput figure with tuned Async fan-out results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_KEYS = ("qwen36_mtp", "qwen36_dspark", "gemma4_dspark")
MODEL_LABELS = {
    "qwen36_mtp": "Qwen3.6 + MTP",
    "qwen36_dspark": "Qwen3.6 + DSpark",
    "gemma4_dspark": "Gemma4 + DSpark",
}
SERIES = ("AR", "Sync", "Async before", "Async tuned")
COLORS = {
    "AR": "#4C78A8",
    "Sync": "#F58518",
    "Async before": "#9D9DA1",
    "Async tuned": "#54A24B",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--mtp-tuned-root", type=Path)
    parser.add_argument("--qwen-tuned-root", type=Path, required=True)
    parser.add_argument("--gemma-tuned-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--output-stem",
        default="throughput_fanout_optimized",
    )
    parser.add_argument(
        "--metrics-output-stem",
        default="throughput_cache_acceptance_tuned",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_baseline(root: Path) -> dict[tuple[str, str], float]:
    required_markers = (
        "matrix_complete.json",
        "performance_measurement_complete.json",
    )
    for marker in required_markers:
        value = read_json(root / marker)
        if value.get("status") not in ("complete", "passed"):
            raise ValueError(f"baseline marker is not complete: {root / marker}")

    rows: dict[tuple[str, str], float] = {}
    with (root / "throughput.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            model_key = row["model_key"]
            mode = row["decode_mode"]
            if model_key in MODEL_KEYS and mode in ("AR", "Sync", "Async"):
                rows[(model_key, mode)] = float(row["completion_throughput_tok_s"])
    expected = {
        (model, mode) for model in MODEL_KEYS for mode in ("AR", "Sync", "Async")
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise ValueError(f"baseline throughput rows are incomplete: {missing}")
    return rows


def metric_total(metrics: dict[str, Any], prefix: str) -> float:
    return sum(
        float(value) for name, value in metrics.items() if name.startswith(prefix)
    )


def load_sync_acceptance(root: Path) -> dict[str, float]:
    values = {}
    for model_key in MODEL_KEYS:
        result_path = (
            root
            / "models"
            / model_key
            / "cells"
            / "performance_sync_eager_b1"
            / "result.json"
        )
        result = read_json(result_path)
        if result.get("status") != "complete":
            raise ValueError(f"Sync baseline cell is not complete: {result_path}")
        metrics = result.get("metrics_delta") or {}
        accepted = metric_total(metrics, "vllm:spec_decode_num_accepted_tokens_total")
        rounds = metric_total(metrics, "vllm:spec_decode_num_drafts_total")
        if accepted < 0.0 or rounds <= 0.0:
            raise ValueError(f"Sync acceptance metrics are invalid: {result_path}")
        values[model_key] = accepted / rounds
    return values


def load_async_quality(analysis_row: dict[str, Any], root: Path) -> dict[str, Any]:
    cache = analysis_row.get("cache") or {}
    hits = int(cache.get("hits", -1))
    eligible_rounds = int(cache.get("eligible_rounds", -1))
    hit_rate = float(cache.get("hit_rate", -1.0))
    if hits < 0 or eligible_rounds <= 0 or hits > eligible_rounds:
        raise ValueError(f"Async cache metrics are invalid: {root}")
    expected_rate = hits / eligible_rounds
    if not math.isclose(hit_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Async cache hit rate is inconsistent: {root}")

    acceptance = analysis_row.get("acceptance") or {}
    mean_accepted = float(acceptance.get("mean_accepted_draft_count", -1.0))
    verify_width = int(analysis_row["verify_width"])
    if not 0.0 <= mean_accepted <= verify_width:
        raise ValueError(f"Async acceptance metrics are invalid: {root}")
    return {
        "cache_hits": hits,
        "cache_eligible_rounds": eligible_rounds,
        "cache_hit_rate": hit_rate,
        "mean_accepted_draft_count": mean_accepted,
    }


def load_tuned(root: Path, expected_model: str) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    models = manifest.get("models") or []
    if len(models) != 1 or models[0].get("key") != expected_model:
        raise ValueError(f"unexpected tuned model in {root}")

    contract = manifest.get("contract") or {}
    if contract.get("batch_size") != 1:
        raise ValueError(f"tuned artifact is not B=1: {root}")
    if contract.get("prompt_count") != 16 or contract.get("output_length") != 128:
        raise ValueError(f"tuned artifact is not 16x128: {root}")
    if float(contract.get("warmup_seconds_minimum", 0.0)) < 30.0:
        raise ValueError(f"tuned artifact lacks the 30-second warmup contract: {root}")

    complete_markers = list(root.rglob("cell_complete.json"))
    if len(complete_markers) != 1:
        raise ValueError(f"expected one complete tuned cell in {root}")
    cell_dir = complete_markers[0].parent
    complete = read_json(complete_markers[0])
    result = read_json(cell_dir / "result.json")
    if complete.get("status") != "complete" or result.get("status") != "complete":
        raise ValueError(f"tuned cell is not complete: {cell_dir}")
    if result.get("expected_completion_tokens") != 2048:
        raise ValueError(f"unexpected tuned token contract: {cell_dir}")
    summary = result.get("summary") or {}
    if summary.get("completion_tokens") != 2048:
        raise ValueError(f"tuned cell did not finish all tokens: {cell_dir}")
    if summary.get("completed_request_count") != 16:
        raise ValueError(f"tuned cell did not finish all requests: {cell_dir}")
    if float((result.get("warmup") or {}).get("seconds", 0.0)) < 30.0:
        raise ValueError(f"tuned cell warmup was too short: {cell_dir}")

    analysis = read_json(root / "fanout_analysis.json")
    analysis_rows = analysis.get("rows") or []
    if len(analysis_rows) != 1 or analysis_rows[0].get("model") != expected_model:
        raise ValueError(f"unexpected fan-out analysis in {root}")
    analysis_row = analysis_rows[0]
    if analysis_row.get("status") != "complete":
        raise ValueError(f"fan-out analysis is not complete: {root}")
    if not analysis_row.get("runtime_branch_audit", {}).get("passed"):
        raise ValueError(f"fan-out branch audit did not pass: {root}")

    return {
        "artifact": str(root.resolve()),
        "throughput_tok_s": float(summary["completion_throughput_tok_s"]),
        "verify_width": int(analysis_row["verify_width"]),
        "fan_out": int(analysis_row["fan_out"]),
        "branches_per_round": int(analysis_row["branches_per_round"]),
        "warmup_seconds": float(result["warmup"]["seconds"]),
        **load_async_quality(analysis_row, root),
    }


def load_mtp_tuned(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if manifest.get("artifact_kind") != "async_mtp_fanout_calibration":
        raise ValueError(f"unexpected MTP artifact kind in {root}")
    contract = manifest.get("contract") or {}
    if (
        contract.get("batch_size") != 1
        or contract.get("prompt_count") != 16
        or contract.get("output_length") != 128
        or contract.get("verify_width") != 3
    ):
        raise ValueError(f"MTP tuned artifact is not B1/16x128/D3: {root}")

    complete_markers = list(root.rglob("cell_complete.json"))
    if len(complete_markers) != 1:
        raise ValueError(f"expected one complete MTP tuned cell in {root}")
    cell_dir = complete_markers[0].parent
    complete = read_json(complete_markers[0])
    result = read_json(cell_dir / "result.json")
    if complete.get("status") != "complete" or result.get("status") != "complete":
        raise ValueError(f"MTP tuned cell is not complete: {cell_dir}")
    summary = result.get("summary") or {}
    if (
        result.get("expected_completion_tokens") != 2048
        or summary.get("completion_tokens") != 2048
        or summary.get("completed_request_count") != 16
    ):
        raise ValueError(f"MTP tuned cell did not finish 16x128: {cell_dir}")
    if float((result.get("warmup") or {}).get("seconds", 0.0)) < 30.0:
        raise ValueError(f"MTP tuned cell warmup was too short: {cell_dir}")

    analysis = read_json(root / "fanout_analysis.json")
    analysis_rows = analysis.get("rows") or []
    if len(analysis_rows) != 1 or analysis_rows[0].get("status") != "complete":
        raise ValueError(f"unexpected MTP fan-out analysis in {root}")
    analysis_row = analysis_rows[0]
    if not analysis_row.get("runtime_branch_audit", {}).get("passed"):
        raise ValueError(f"MTP fan-out branch audit did not pass: {root}")
    return {
        "artifact": str(root.resolve()),
        "throughput_tok_s": float(summary["completion_throughput_tok_s"]),
        "verify_width": int(analysis_row["verify_width"]),
        "fan_out": int(analysis_row["fan_out"]),
        "branches_per_round": int(analysis_row["branches_per_round"]),
        "warmup_seconds": float(result["warmup"]["seconds"]),
        **load_async_quality(analysis_row, root),
    }


def comparison_rows(
    baseline: dict[tuple[str, str], float],
    sync_acceptance: dict[str, float],
    tuned: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for model_key in MODEL_KEYS:
        before = baseline[(model_key, "Async")]
        tuned_row = tuned.get(model_key)
        rows.append(
            {
                "model_key": model_key,
                "model_config": MODEL_LABELS[model_key],
                "ar_tok_s": baseline[(model_key, "AR")],
                "sync_tok_s": baseline[(model_key, "Sync")],
                "async_before_tok_s": before,
                "async_tuned_tok_s": (
                    tuned_row["throughput_tok_s"] if tuned_row else None
                ),
                "async_improvement_percent": (
                    (tuned_row["throughput_tok_s"] / before - 1.0) * 100.0
                    if tuned_row
                    else None
                ),
                "tuned_verify_width": (
                    tuned_row["verify_width"] if tuned_row else None
                ),
                "tuned_fan_out": tuned_row["fan_out"] if tuned_row else None,
                "tuned_branches_per_round": (
                    tuned_row["branches_per_round"] if tuned_row else None
                ),
                "ar_cache_hit_rate": None,
                "sync_cache_hit_rate": None,
                "async_tuned_cache_hits": (
                    tuned_row["cache_hits"] if tuned_row else None
                ),
                "async_tuned_cache_eligible_rounds": (
                    tuned_row["cache_eligible_rounds"] if tuned_row else None
                ),
                "async_tuned_cache_hit_rate": (
                    tuned_row["cache_hit_rate"] if tuned_row else None
                ),
                "ar_mean_accepted_draft_count": None,
                "sync_mean_accepted_draft_count": sync_acceptance[model_key],
                "async_tuned_mean_accepted_draft_count": (
                    tuned_row["mean_accepted_draft_count"] if tuned_row else None
                ),
            }
        )
    return rows


def render_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = list(range(len(rows)))
    width = 0.19
    offsets = {
        "AR": -1.5 * width,
        "Sync": -0.5 * width,
        "Async before": 0.5 * width,
        "Async tuned": 1.5 * width,
    }
    keys = {
        "AR": "ar_tok_s",
        "Sync": "sync_tok_s",
        "Async before": "async_before_tok_s",
        "Async tuned": "async_tuned_tok_s",
    }

    figure, axis = plt.subplots(figsize=(12, 7.1))
    for series in SERIES:
        positions = [x_value + offsets[series] for x_value in x_values]
        values = [row[keys[series]] for row in rows]
        plotted_positions = [
            position for position, value in zip(positions, values) if value is not None
        ]
        plotted_values = [value for value in values if value is not None]
        bars = axis.bar(
            plotted_positions,
            plotted_values,
            width,
            label=series,
            color=COLORS[series],
        )
        if series != "Async tuned":
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            continue
        tuned_rows = [row for row in rows if row["async_tuned_tok_s"] is not None]
        labels = [
            f"{row['async_tuned_tok_s']:.2f}\n"
            f"({row['async_improvement_percent']:+.1f}%)"
            for row in tuned_rows
        ]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=9)

    if rows[0]["async_tuned_tok_s"] is None:
        mtp_tuned_position = x_values[0] + offsets["Async tuned"]
        axis.text(
            mtp_tuned_position,
            1.0,
            "not\nretuned",
            ha="center",
            va="bottom",
            color="#555555",
            fontsize=8,
        )
    axis.set_xticks(x_values, [row["model_config"] for row in rows])
    axis.set_ylabel("Completion throughput (tokens/s)")
    axis.set_xlabel("Model configuration")
    axis.set_ylim(0, max(row["sync_tok_s"] for row in rows) * 1.18)
    axis.set_title("Phase-B throughput: historical baseline vs tuned Async fan-out")
    axis.legend(title="Decoding mode", ncols=2, loc="upper left")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    figure.text(
        0.5,
        0.015,
        "B=1, 16 x 128, BF16, eager. Historical matrix: D=3.\n"
        "Tuned: Qwen MTP D=3/F=96; Qwen DSpark D=3/F=24; "
        "Gemma DSpark D=2/F=48.\n"
        "Gemma before/after is directional, not a strict same-D comparison.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    figure.savefig(output_path.with_suffix(".png"), dpi=180)
    figure.savefig(output_path.with_suffix(".svg"))
    plt.close(figure)


def render_mode_metrics_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["model_config"] for row in rows]
    x_values = list(range(len(rows)))
    figure = plt.figure(figsize=(13.5, 9.2))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.2, 1.0))
    throughput_axis = figure.add_subplot(grid[0, :])
    cache_axis = figure.add_subplot(grid[1, 0])
    acceptance_axis = figure.add_subplot(grid[1, 1])

    throughput_series = (
        ("AR", "ar_tok_s"),
        ("Sync", "sync_tok_s"),
        ("Async tuned", "async_tuned_tok_s"),
    )
    width = 0.24
    for index, (series, key) in enumerate(throughput_series):
        positions = [value + (index - 1) * width for value in x_values]
        raw_values = [row[key] for row in rows]
        values = [value if value is not None else 0.0 for value in raw_values]
        bars = throughput_axis.bar(
            positions,
            values,
            width,
            label=series,
            color=COLORS[series],
        )
        bar_labels = [
            f"{value:.2f}" if value is not None else "N/A" for value in raw_values
        ]
        throughput_axis.bar_label(bars, labels=bar_labels, padding=3, fontsize=9)
    throughput_axis.set_xticks(x_values, labels)
    throughput_axis.set_ylabel("Completion throughput (tokens/s)")
    throughput_axis.set_title("Throughput by decoding mode")
    throughput_axis.legend(title="Decoding mode", ncols=3, loc="upper left")
    throughput_axis.set_ylim(0, max(row["sync_tok_s"] for row in rows) * 1.2)

    cache_rates = [row["async_tuned_cache_hit_rate"] for row in rows]
    cache_values = [
        value * 100.0 if value is not None else 0.0 for value in cache_rates
    ]
    cache_bars = cache_axis.bar(
        x_values,
        cache_values,
        0.55,
        color=COLORS["Async tuned"],
    )
    cache_labels = [
        f"{value:.2%}" if value is not None else "N/A" for value in cache_rates
    ]
    cache_axis.bar_label(cache_bars, labels=cache_labels, padding=3, fontsize=9)
    cache_axis.set_xticks(x_values, labels, rotation=12, ha="right")
    cache_axis.set_ylabel("Cache hit rate (%)")
    cache_axis.set_ylim(0, 105)
    cache_axis.set_title("Async tuned branch-cache hit rate")

    acceptance_width = 0.34
    for index, (series, key) in enumerate(
        (
            ("Sync", "sync_mean_accepted_draft_count"),
            ("Async tuned", "async_tuned_mean_accepted_draft_count"),
        )
    ):
        positions = [value + (index - 0.5) * acceptance_width for value in x_values]
        raw_values = [row[key] for row in rows]
        values = [value if value is not None else 0.0 for value in raw_values]
        bars = acceptance_axis.bar(
            positions,
            values,
            acceptance_width,
            label=series,
            color=COLORS[series],
        )
        bar_labels = [
            f"{value:.2f}" if value is not None else "N/A" for value in raw_values
        ]
        acceptance_axis.bar_label(bars, labels=bar_labels, padding=3, fontsize=9)
    acceptance_axis.set_xticks(x_values, labels, rotation=12, ha="right")
    acceptance_axis.set_ylabel("Mean accepted draft tokens / verify round")
    acceptance_axis.set_title("Speculative acceptance length")
    acceptance_axis.legend(title="Decoding mode", ncols=2, loc="upper right")
    acceptance_axis.set_ylim(
        0,
        max(row["sync_mean_accepted_draft_count"] for row in rows) * 1.25,
    )

    for axis in (throughput_axis, cache_axis, acceptance_axis):
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Phase-B tuned Async: throughput, cache hit rate, and acceptance",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.012,
        "B=1, 16 x 128, BF16, eager. Cache eligibility excludes each request's "
        "compulsory first JIT round.\n"
        "AR has no speculative acceptance or branch cache; Sync has no Async "
        "branch cache. Qwen uses D=3.\n"
        "Gemma Sync uses D=3 while tuned Async uses the approved D=2 configuration.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.1, 1.0, 0.97))
    figure.savefig(output_path.with_suffix(".png"), dpi=180)
    figure.savefig(output_path.with_suffix(".svg"))
    plt.close(figure)


def write_mode_metrics_outputs(
    rows: list[dict[str, Any]], output_path: Path, baseline_root: Path
) -> None:
    fieldnames = (
        "model_key",
        "model_config",
        "ar_tok_s",
        "sync_tok_s",
        "async_tuned_tok_s",
        "ar_cache_hit_rate",
        "sync_cache_hit_rate",
        "async_tuned_cache_hits",
        "async_tuned_cache_eligible_rounds",
        "async_tuned_cache_hit_rate",
        "ar_mean_accepted_draft_count",
        "sync_mean_accepted_draft_count",
        "async_tuned_mean_accepted_draft_count",
        "tuned_verify_width",
        "tuned_fan_out",
        "tuned_branches_per_round",
    )
    with output_path.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    write_json(
        output_path.with_suffix(".json"),
        {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_artifact": str(baseline_root),
            "rows": rows,
            "metric_definitions": {
                "cache_hit_rate": (
                    "cache hits divided by eligible rounds; each request's first "
                    "compulsory JIT round is excluded"
                ),
                "mean_accepted_draft_count": (
                    "mean accepted draft tokens per formal Target verification round; "
                    "the Target bonus token is excluded"
                ),
            },
            "not_applicable": {
                "AR": ["cache_hit_rate", "mean_accepted_draft_count"],
                "Sync": ["cache_hit_rate"],
            },
            "caveat": (
                "Qwen Sync and tuned Async use D=3. Gemma Sync uses D=3 while "
                "tuned Async uses D=2, so its acceptance lengths are not a strict "
                "same-width comparison."
            ),
            "correctness_status": "not_claimed_by_performance_plot",
        },
    )
    render_mode_metrics_plot(rows, output_path)

    markdown = [
        "# Phase-B tuned Async mode metrics",
        "",
        "| Model | AR tok/s | Sync tok/s | Async tok/s | Async cache hit | "
        "Sync accepted | Async accepted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        async_throughput = row["async_tuned_tok_s"]
        cache_hit_rate = row["async_tuned_cache_hit_rate"]
        async_accepted = row["async_tuned_mean_accepted_draft_count"]
        async_throughput_text = (
            f"{async_throughput:.4f}" if async_throughput is not None else "N/A"
        )
        cache_text = (
            f"{cache_hit_rate:.2%} ({row['async_tuned_cache_hits']}/"
            f"{row['async_tuned_cache_eligible_rounds']})"
            if cache_hit_rate is not None
            else "N/A"
        )
        async_accepted_text = (
            f"{async_accepted:.4f}" if async_accepted is not None else "N/A"
        )
        markdown.append(
            f"| {row['model_config']} | {row['ar_tok_s']:.4f} | "
            f"{row['sync_tok_s']:.4f} | {async_throughput_text} | "
            f"{cache_text} | "
            f"{row['sync_mean_accepted_draft_count']:.4f} | "
            f"{async_accepted_text} |"
        )
    markdown.extend(
        [
            "",
            "Cache eligibility excludes each request's compulsory first JIT round. "
            "AR has neither speculative acceptance nor an Async branch cache; Sync "
            "has no Async branch cache.",
            "",
            "Qwen uses D=3 for Sync and tuned Async. Gemma Sync is D=3 while its "
            "tuned Async result uses the approved D=2 configuration, so those two "
            "acceptance lengths are not a strict same-width comparison.",
            "",
            "This plot reports performance behavior and does not establish the "
            "outstanding correctness gate.",
            "",
        ]
    )
    output_path.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")


def render_outputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline_root = args.baseline_root.resolve()
    qwen_root = args.qwen_tuned_root.resolve()
    gemma_root = args.gemma_tuned_root.resolve()
    output_root = (args.output_root or baseline_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(baseline_root)
    sync_acceptance = load_sync_acceptance(baseline_root)
    tuned = {
        "qwen36_dspark": load_tuned(qwen_root, "qwen36_dspark"),
        "gemma4_dspark": load_tuned(gemma_root, "gemma4_dspark"),
    }
    if args.mtp_tuned_root is not None:
        tuned["qwen36_mtp"] = load_mtp_tuned(args.mtp_tuned_root.resolve())
    rows = comparison_rows(baseline, sync_acceptance, tuned)
    output_path = output_root / args.output_stem
    metrics_output_path = output_root / args.metrics_output_stem

    with output_path.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        output_path.with_suffix(".json"),
        {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_artifact": str(baseline_root),
            "tuned_artifacts": tuned,
            "rows": rows,
            "caveat": (
                "Qwen MTP and Qwen DSpark are D=3 before and after. Gemma "
                "historical Async is D=3 while tuned Async is the approved D=2 "
                "scheme-2 configuration; its improvement is directional and cannot "
                "be attributed to fan-out alone."
            ),
            "correctness_status": "not_claimed_by_performance_plot",
        },
    )
    render_plot(rows, output_path)
    write_mode_metrics_outputs(rows, metrics_output_path, baseline_root)

    mtp = next(row for row in rows if row["model_key"] == "qwen36_mtp")
    qwen = next(row for row in rows if row["model_key"] == "qwen36_dspark")
    gemma = next(row for row in rows if row["model_key"] == "gemma4_dspark")
    markdown = "\n".join(
        [
            "# Phase-B throughput after Async fan-out tuning",
            "",
            "The original `throughput.png` is preserved. This figure adds the tuned "
            "Async cells to the historical AR/Sync/Async matrix.",
            "",
            "| Model | AR | Sync | Async before | Async tuned | Change |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| Qwen3.6 + MTP | {mtp['ar_tok_s']:.4f} | "
                f"{mtp['sync_tok_s']:.4f} | {mtp['async_before_tok_s']:.4f} | "
                f"{mtp['async_tuned_tok_s']:.4f} | "
                f"{mtp['async_improvement_percent']:+.2f}% |"
                if mtp["async_tuned_tok_s"] is not None
                else (
                    f"| Qwen3.6 + MTP | {mtp['ar_tok_s']:.4f} | "
                    f"{mtp['sync_tok_s']:.4f} | {mtp['async_before_tok_s']:.4f} | "
                    "not retuned | -- |"
                )
            ),
            (
                f"| Qwen3.6 + DSpark | {qwen['ar_tok_s']:.4f} | "
                f"{qwen['sync_tok_s']:.4f} | {qwen['async_before_tok_s']:.4f} | "
                f"{qwen['async_tuned_tok_s']:.4f} | "
                f"{qwen['async_improvement_percent']:+.2f}% |"
            ),
            (
                f"| Gemma4 + DSpark | {gemma['ar_tok_s']:.4f} | "
                f"{gemma['sync_tok_s']:.4f} | {gemma['async_before_tok_s']:.4f} | "
                f"{gemma['async_tuned_tok_s']:.4f} | "
                f"{gemma['async_improvement_percent']:+.2f}% |"
            ),
            "",
            "Qwen MTP and Qwen DSpark use D=3 both before and after. Gemma's "
            "historical cell used D=3; the tuned scheme-2 cell uses its approved "
            "D=2 maximum, so that change is directional rather than a strict "
            "same-D comparison.",
            "",
            "This performance visualization does not close the outstanding correctness "
            "or acceptance-quality gate.",
            "",
        ]
    )
    output_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    render_outputs(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
