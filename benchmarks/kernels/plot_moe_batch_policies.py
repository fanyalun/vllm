# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize isolated MoE policy measurements and plot their batch scaling."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.input / "timings.json").read_text())
    complete = json.loads((args.input / "complete.json").read_text())
    batches, layers = complete["batches"], complete["layers"]
    assert len(rows) == len(batches) * len(layers) * 3 * 2 == complete["rows"]
    result = []
    for boundary in ["routing_and_experts", "moe_with_shared"]:
        for batch in batches:
            for case in ["h8", "h4", "p0125"]:
                selected = [
                    r
                    for r in rows
                    if (r["boundary"], r["batch"], r["case"]) == (boundary, batch, case)
                ]
                assert sorted(r["layer"] for r in selected) == sorted(layers)
                times = sorted(t for r in selected for t in r["samples_us"])
                result.append(
                    dict(
                        boundary=boundary,
                        batch=batch,
                        case=case,
                        mean_median_us=statistics.mean(
                            r["median_us"] for r in selected
                        ),
                        pooled_p95_us=times[int(0.95 * (len(times) - 1))],
                        mean_selected=statistics.mean(
                            r["selected_mean"] for r in selected
                        ),
                        mean_unique_experts=statistics.mean(
                            r["unique_experts"] for r in selected
                        ),
                    )
                )
    (args.input / "summary.json").write_text(json.dumps(result, indent=2))
    table = {
        (r["batch"], r["case"]): r for r in result if r["boundary"] == "moe_with_shared"
    }
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 15,
            "axes.labelsize": 16,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for case, label, color, marker in zip(
        ["h8", "h4", "p0125"],
        ["Native top-8", "Top-4", "p = 0.125"],
        ["#4C78A8", "#F2B447", "#59A14F"],
        ["o", "s", "^"],
        strict=True,
    ):
        y = [table[b, case]["mean_median_us"] for b in batches]
        ratio = [
            table[b, "h8"]["mean_median_us"] / v
            for b, v in zip(batches, y, strict=True)
        ]
        axes[0].plot(
            range(len(batches)), y, label=label, color=color, marker=marker, lw=2, ms=7
        )
        axes[1].plot(range(len(batches)), ratio, color=color, marker=marker, lw=2, ms=7)
    axes[0].set_ylabel("MoE latency (μs)")
    axes[1].set_ylabel("Speedup over top-8 (×)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0.9)
    for ax, label in zip(
        axes, ["(a) Including shared expert", "(b) Relative speedup"], strict=True
    ):
        ax.set_xticks(range(len(batches)), [str(b) for b in batches])
        ax.set_xlabel("Batch size\n" + label)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "moe_batch_policies"
    for suffix in ["png", "pdf"]:
        fig.savefig(
            stem.with_suffix("." + suffix),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    notes = [
        "# MoE batch policy latency",
        "",
        f"Source: `{args.input}`. A100 80GB PCIe, BF16, T5, layers {layers}.",
        "Real Qwen3.6 checkpoint weights with independent seeded Gaussian inputs.",
        "Top-8 baseline; top-4 and p=0.125 preserve retained native weights.",
        "Graph replay with explicit cold L2 flush, 30 warmups and 100 samples.",
        (
            "The plot includes router projection/selection, expert "
            "dispatch/GEMMs, shared expert and sum."
        ),
        (
            "It is an isolated functional composition, not the compiled "
            "full-model wrapper."
        ),
        (
            "Three layer medians are averaged. P95 pools 300 samples, "
            "including layer variation."
        ),
        (
            "Horizontal positions denote sampled batch categories. No "
            "confidence interval is claimed."
        ),
        "",
        (
            "| B | Policy | Routed path μs | With shared μs | P95 μs | "
            "Speedup | Experts/token |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    routed = {
        (r["batch"], r["case"]): r
        for r in result
        if r["boundary"] == "routing_and_experts"
    }
    for b in batches:
        for c in ["h8", "h4", "p0125"]:
            r = table[b, c]
            speed = table[b, "h8"]["mean_median_us"] / r["mean_median_us"]
            notes.append(
                f"| {b} | {c} | {routed[b, c]['mean_median_us']:.2f} | "
                f"{r['mean_median_us']:.2f} | {r['pooled_p95_us']:.2f} | "
                f"{speed:.3f}× | {r['mean_selected']:.3f} |"
            )
    notes += [
        "",
        (
            "Uses existing Triton fused_experts and default MoE launch "
            "config; the device-specific tuned config file is absent. "
            "Retained assignments are genuinely skipped/compacted. Shared "
            "expert uses the same unchanged BF16 weights in all cells."
        ),
        (
            "All 54 cells pass graph/eager equality and each policy passes "
            "the zero-weight native dispatch reference. Synthetic input "
            "routing is not a quality or acceptance measurement."
        ),
        "",
        "Reproduce:",
        "```bash",
        (
            "CUDA_VISIBLE_DEVICES=0 .venv/bin/python "
            "benchmarks/kernels/benchmark_moe_batch_policies.py --output "
            "benchmark_results/moe_batch_reproduction/run"
        ),
        ".venv/bin/python benchmarks/kernels/plot_moe_batch_policies.py "
        f"--input {args.input} --output {args.output}",
        "```",
    ]
    stem.with_suffix(".md").write_text("\n".join(notes) + "\n")
    print("\n".join(notes[11:]))


if __name__ == "__main__":
    main()
