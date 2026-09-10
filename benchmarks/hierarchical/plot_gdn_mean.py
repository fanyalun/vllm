# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot complete-cycle mean-GDN diagnostics from audited CSV rows."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_gdn_components(source, output):
    rows = list(csv.DictReader(source.open()))
    data = {
        (r["mode"], r["component"]): float(r["kernel_ms"])
        for r in rows
        if r["top_h"] == "4"
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(3)
    bottom = np.zeros(3)
    modes = ["none", "ssm_mean", "input_mean"]
    for label, fields, color in [
        ("Projections", ["projection"], "#4C78A8"),
        ("Convolution", ["convolution"], "#59A14F"),
        ("SSM (+ fused norm)", ["ssm_and_norm_fused", "pooled_ssm"], "#F2B447"),
        ("Other GDN work", ["other", "normalization"], "#B279A2"),
    ]:
        values = np.array(
            [sum(data.get((mode, field), 0) for field in fields) for mode in modes]
        )
        ax.bar(x, values, bottom=bottom, width=0.62, color=color, label=label)
        bottom += values
    ax.set_xticks(x, ["MoE-Skip", "+ SSM mean", "+ Input mean"])
    ax.set_ylabel("GDN kernel work (ms)")
    ax.set_ylim(0, max(bottom) * 1.05)
    ax.yaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / f"{output.name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    (output / f"{output.name}.md").write_text(
        "# GDN kernel work\n\n"
        f"Source: `{source}`. Same five-token candidate block and initial state, "
        "Qwen3.6, TP=1/B=1, top-4 Pre-Verify, BF16 activations and FP32 SSM. "
        "One CUDA graph trace of the third fixed-prefix probe after all 20-replay "
        "timing passes. All kernels match the annotated eager invocation by name "
        "and order. No per-layer CUDA events are present in the traced graph.\n\n"
        "The baseline recurrence kernel includes normalization; mean modes use "
        "a separate normalization kernel included in Other GDN work, along with "
        "pooling, casts, copies and initialization. Projections include GEMV "
        "reduction kernels. Bars sum active GDN kernel durations over 30 layers. "
        "This is an eager-captured diagnostic, not the compiled end-to-end graph "
        "timing. It does not include state reset or metadata CPU overhead.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open()))
    data = {(r["inner"], r["mode"]): r for r in rows}
    assert len(data) == 6
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    metrics = [
        ("e2e_after_tps", "Token/s", "(a) End-to-end throughput", 1),
        ("cycle_ms", "ms", "(b) Complete cycle", 1),
        ("returned_per_cycle", "Tokens/cycle", "(c) Final returned tokens", 1),
        ("target_acceptance", "Acceptance (%)", "(d) Target acceptance", 100),
    ]
    modes = [
        ("none", "MoE-Skip", "#4C78A8"),
        ("ssm_mean", "+ SSM mean", "#F2B447"),
        ("input_mean", "+ Input mean", "#59A14F"),
    ]
    for ax, (metric, unit, label, scale) in zip(axes.flat, metrics, strict=True):
        x = np.arange(2)
        for i, (mode, name, color) in enumerate(modes):
            values = [
                float(data[method, mode][metric]) * scale
                for method in ("mtp", "dspark")
            ]
            ax.bar(x + (i - 1) * 0.25, values, width=0.24, label=name, color=color)
        ax.set_xticks(x, ["MTP", "DSpark"])
        ax.set_ylabel(unit)
        ax.set_xlabel(label, labelpad=8)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0, w_pad=1.2)
    args.output.mkdir(parents=True, exist_ok=True)
    name = args.output.name
    fig.savefig(
        args.output / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04
    )
    fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    (args.output / f"{name}.md").write_text(
        "# Mean GDN comparison\n\n"
        f"Source: `{args.input}`. Qwen3.6-35B-A3B, TP=1/B=1, inner D=4/N=4, "
        "top-8 Target and top-4 Pre-Verify, four fixed prompts, 256 output tokens "
        "each, greedy sampling. MTP modes share GPU 1; DSpark modes share GPU 0. "
        "Compare modes within each inner method.\n\n"
        "(a) uses the second uninstrumented pass after all prompts were warmed. "
        "(b-d) use instrumented proposal-to-next-Target cycles, excluding the "
        "prefill-associated cycle and proposals with no subsequent verification. "
        "(c) clips the final cycle to tokens actually returned under the output "
        "budget. (d) is accepted/proposed at Target, before output-budget clipping. "
        "Bars aggregate four prompts; no confidence intervals or significance "
        "claim. AR output equivalence is not certified; see results.md.\n\n"
        "Reproduce with `benchmarks/hierarchical/plot_gdn_mean.py --input "
        f"{args.input} --output {args.output}` using `.venv/bin/python`.\n"
    )
    plot_gdn_components(
        args.input.parent / "gdn_components.csv",
        args.output.parent / "gdn_kernel_breakdown",
    )


if __name__ == "__main__":
    main()
