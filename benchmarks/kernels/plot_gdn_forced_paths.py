# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot forced V2 path latency and paired native speedup."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.input / "complete.json").read_text())
    rows = json.loads((args.input / "timings.json").read_text())
    batches = complete["args"]["batches"]
    cases = ["v0", "v2_full", "v2_decay", "v2_skip"]
    layers = complete["args"]["layers"]
    assert len(rows) == len(batches) * len(cases) * len(layers)
    table = {}
    for batch in batches:
        for case in cases:
            selected = [r for r in rows if r["batch"] == batch and r["case"] == case]
            assert sorted(r["layer"] for r in selected) == sorted(layers)
            samples = sorted(t for r in selected for t in r["samples_us"])
            table[batch, case] = dict(
                median_mean=statistics.mean(r["median_us"] for r in selected),
                p95=samples[int(0.95 * (len(samples) - 1))],
            )
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 15,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    x = np.arange(len(batches))
    colors = ["#4C78A8", "#F2B447", "#59A14F", "#B279A2"]
    labels = ["V0 native", "V2 all Full", "V2 all Decay", "V2 all Skip"]
    for case, color, label, marker in zip(
        cases, colors, labels, ["o", "s", "^", "D"], strict=True
    ):
        y = [table[b, case]["median_mean"] for b in batches]
        speed = [
            table[b, "v0"]["median_mean"] / v for b, v in zip(batches, y, strict=True)
        ]
        axes[0].plot(x, y, marker=marker, color=color, label=label, lw=2, ms=7)
        axes[1].plot(x, speed, marker=marker, color=color, lw=2, ms=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Latency (μs, log scale)")
    axes[1].set_ylabel("Speedup over V0 (×)")
    axes[1].set_ylim(bottom=0)
    for ax, title in zip(
        axes, ["(a) Post-conv latency", "(b) Relative speedup"], strict=True
    ):
        ax.set_xticks(x, [str(b) for b in batches])
        ax.set_xlabel("Batch size\n" + title)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        frameon=False,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "gdn_forced_paths"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    notes = [
        "# Forced V2 paths versus native V0",
        "",
        f"Source: `{args.input}`. A100 80GB, B={batches}, T=5, "
        f"captured layers {layers}.",
        (
            "BF16 inputs, FP32 SSM; captured QKV/BA/state replicated per "
            "request; synthetic Z and unit norm weights."
        ),
        (
            "Runtime GPU thresholds force all Full/Decay/Skip while "
            "preserving V2 classification and original gate values."
        ),
        (
            "All paths include post-conv recurrence and gated norm, including "
            "required V2 layout copy."
        ),
        (
            "V0 writes candidate snapshots; V2 writes one tail, and all Skip "
            "writes no state."
        ),
        "No input/output projections, Conv, MoE or decoding are timed.",
        (
            "30 warmups, 100 samples per cell. State restore and 64MiB L2 "
            "flush occur outside each CUDA-event interval."
        ),
        (
            "Panel (a): average of three layer medians, per layer for the "
            "entire batch, log y axis."
        ),
        (
            "Panel (b): ratio of those mean medians; equal horizontal spacing "
            "denotes sampled batch categories."
        ),
        (
            "P95 below pools 300 samples including layer variation; no "
            "confidence intervals are claimed."
        ),
        "",
        "| Batch | Method | Latency μs | P95 μs | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for b in batches:
        for c in cases:
            v = table[b, c]
            speed = table[b, "v0"]["median_mean"] / v["median_mean"]
            notes.append(
                f"| {b} | {c} | {v['median_mean']:.2f} | "
                f"{v['p95']:.2f} | {speed:.3f}× |"
            )
    notes += [
        "",
        (
            "All cells passed graph/eager and replicated-request equality. "
            "Full is checked against native; Skip state is bitwise unchanged; "
            "Decay state matches FP32 stepwise decay. Action counters confirm "
            "every requested branch."
        ),
        "",
        "Reproduce:",
        "```bash",
        (
            "CUDA_VISIBLE_DEVICES=0 .venv/bin/python "
            "benchmarks/kernels/benchmark_gdn_native_batch.py \\"
        ),
        (
            "  --inputs benchmark_results/.sources/"
            "three_level_p50_20260916/raw_inputs.pt \\"
        ),
        "  --output benchmark_results/gdn_forced_paths_reproduction/run --forced-paths",
        ".venv/bin/python benchmarks/kernels/plot_gdn_forced_paths.py \\",
        f"  --input {args.input} --output {args.output}",
        "```",
    ]
    stem.with_suffix(".md").write_text("\n".join(notes) + "\n")
    print("\n".join(notes[13:]))


if __name__ == "__main__":
    main()
