# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot audited throughput and Target acceptance for the h4 policy matrix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root, output):
    assert (root / "MATRIX_AUDIT_COMPLETE").exists()
    rows = json.loads((root / "summary.json").read_text())
    assert len(rows) == 20
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    styles = [
        ("ar", "AR", "#777777", "o"),
        ("mtp", "MTP D4", "#F2B447", "s"),
        ("low_error", "Low-error", "#4C78A8", "^"),
        ("balanced", "Balanced", "#59A14F", "D"),
        ("aggressive", "Aggressive", "#E45756", "v"),
    ]
    for mode, label, color, marker in styles:
        selected = sorted(
            (r for r in rows if r["mode"] == mode), key=lambda r: r["batch"]
        )
        x = [r["batch"] for r in selected]
        axes[0].plot(
            x,
            [r["tokens_per_second"] for r in selected],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
        )
        if mode != "ar":
            axes[1].plot(
                x,
                [100 * r["outer_acceptance_rate"] for r in selected],
                color=color,
                marker=marker,
                linewidth=1.8,
            )
    for index, ax in enumerate(axes):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16], labels=["1", "4", "8", "16"])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Batch size")
        ax.text(
            0.5,
            -0.31,
            ["(a) Throughput", "(b) Target acceptance"][index],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )
    axes[0].set_ylabel("Output tokens / s")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Accepted candidates (%)")
    axes[1].set_ylim(0, 102)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.77, bottom=0.23, left=0.1, right=0.99, wspace=0.32)
    folder = output / "policy_scaling"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            folder / f"policy_scaling.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    (folder / "policy_scaling.md").write_text(
        "# h4 policy scaling\n\n"
        f"Source: `{root}/summary.json`, audited 20-cell matrix.\n\n"
        "Gemma-4-26B-A4B-it + assistant; h4, D4, at most four inner rounds; "
        "16 raw prompts, 512 output tokens each, TP1, greedy, ignore_eos, "
        "synchronous scheduling, prefix caching disabled. "
        "Panel (a) includes prefill and decode in generate wall time, excluding "
        "startup and warmup. Panel (b) divides accepted candidate tokens by "
        "verified candidates, excluding Target bonus; AR has no candidate rate. "
        "One warm measurement per cell; no error bars or significance claim. "
        "Output equivalence is reported separately in output_comparisons.csv.\n\n"
        "Reproduce with `.venv/bin/python -m "
        "benchmarks.hierarchical.plot_policy_matrix "
        "<run_directory> <output_directory>`.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.root, args.output)
