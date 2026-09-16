# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize fixed-width probes and the historical MTP comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (args.input / "MEASUREMENT_COMPLETE").exists()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = json.loads((args.input / "result.json").read_text())
    rows = []
    for case in raw:
        assert case["control"] == case["observed"]
        for row in case["rows"]:
            times = np.asarray(row["milliseconds"])
            experts = [r["unique_experts"] for r in row["routes"]]
            rows.append(
                dict(
                    batch=row["batch"],
                    width=row["width"],
                    top_h=row["top_h"],
                    median_ms=float(np.median(times)),
                    p25_ms=float(np.quantile(times, 0.25)),
                    p75_ms=float(np.quantile(times, 0.75)),
                    mean_unique_experts=float(np.mean(experts)),
                    min_unique_experts=min(experts),
                    max_unique_experts=max(experts),
                    query_tokens_per_ms=row["batch"] * row["width"] / np.median(times),
                )
            )
    rows.sort(key=lambda r: (r["batch"], -r["top_h"], r["width"]))
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 13,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#4C78A8", "#F2B447", "#59A14F", "#E45756"]

    def export(fig, name, note):
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(
                folder / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04
            )
        (folder / f"{name}.md").write_text(note + "\n")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for b, color in zip((1, 4, 8, 16), colors):
        subset = [r for r in rows if r["batch"] == b and r["top_h"] == 8]
        x = [r["width"] - 1 for r in subset]
        axes[0].plot(
            x, [r["median_ms"] for r in subset], "o-", color=color, label=f"B={b}"
        )
        axes[1].plot(x, [r["mean_unique_experts"] for r in subset], "o-", color=color)
    axes[0].set_ylabel("Forward latency (ms)")
    axes[1].set_ylabel("Distinct experts / layer")
    axes[0].legend(ncol=2, fontsize=11)
    for ax, label in zip(axes, ("(a) Full-model forward", "(b) Expert union")):
        ax.set_xticks([4, 10, 20, 30])
        ax.set_xlabel("Candidate tokens\n" + label)
        ax.grid(alpha=0.2)
    export(
        fig,
        "verify_width",
        "# Verification width\n\n"
        "Source: ../summary.csv and raw fixed-prefix result.json. "
        "Gemma4 BF16, TP1, GPU0 A100 80GB; prefix = prompt + 64 AR tokens. "
        "Width includes one anchor. h8 private verification forward proxy "
        "includes logits, top2 and argmax; not native Target execute_model. "
        "Median of 20 warmed CUDA graph replays, cold L2; CUDA event fallback "
        "because cupti-python is absent. One nested prompt group per B. "
        "Expert counts are untimed means over 30 layers, not per-token top-k. "
        "The two curves show association, not isolated expert-count causality. "
        "Reproduce with benchmarks/hierarchical/analyze_verify_width.py.",
    )

    base = Path(__file__).parent
    matrix = json.loads(
        (base / "policy_matrix_16x512_20260915/summary.json").read_text()
    )
    rounds = json.loads((base / "round_sweep_16x512_20260916/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    x = np.arange(4)
    for i, policy in enumerate(("low_error", "balanced", "aggressive")):
        for r, style in ((4, "-"), (6, "--")):
            subset = sorted(
                (v for v in rounds if v["rounds"] == r and v["mode"] == policy),
                key=lambda v: v["batch"],
            )
            axes[0].plot(
                x,
                [v["speedup_mtp"] for v in subset],
                style,
                marker="o",
                color=colors[i],
                label=f"{policy}, R{r}",
            )
    axes[0].axhline(1, color="black", linewidth=1)
    axes[0].set_ylabel("Throughput / MTP D4")
    axes[0].legend(fontsize=9, ncol=2, loc="upper right")
    for r, offset, color in ((4, -0.17, colors[0]), (6, 0.17, colors[1])):
        subset = sorted(
            (v for v in rounds if v["rounds"] == r and v["mode"] == "low_error"),
            key=lambda v: v["batch"],
        )
        denominators = [
            next(
                v["target_engine_steps"]
                for v in matrix
                if v["batch"] == b and v["mode"] == "mtp"
            )
            for b in (1, 4, 8, 16)
        ]
        target = np.array([v["target_calls"] for v in subset]) / denominators
        pv = np.array([v["batch_inner_calls"] for v in subset]) / denominators
        axes[1].bar(x + offset, target, 0.32, color=color, label=f"R{r} Target")
        axes[1].bar(
            x + offset,
            pv,
            0.32,
            bottom=target,
            color=color,
            hatch="///",
            alpha=0.55,
            label=f"R{r} Pre-Verify",
        )
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set_ylabel("Calls / MTP Target calls")
    axes[1].legend(fontsize=10)
    for ax, label in zip(
        axes, ("(a) All stopping policies", "(b) Default policy call counts")
    ):
        ax.set_xticks(x, [1, 4, 8, 16])
        ax.set_xlabel("Batch size\n" + label)
    export(
        fig,
        "mtp_cost_comparison",
        "# MTP comparison\n\n"
        "Historical 16 prompts x 512 output tokens; h4 D4, R4/R6. "
        "Sources: policy_matrix_16x512_20260915 and "
        "round_sweep_16x512_20260916 summary.json. "
        "Left: all three policies normalized to native MTP D4. "
        "Right: default low_error batch-level Target and Pre-Verify calls, "
        "normalized to MTP Target calls. Call counts are NOT time fractions; "
        "Target counters include prefill. No error bars: one valid E2E run "
        "per cell, different outputs; strict AR equivalence not established. "
        "Reproduce with benchmarks/hierarchical/analyze_verify_width.py.",
    )


if __name__ == "__main__":
    main()
