# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the audited 4/6/8-round sweep, one four-panel figure per batch size."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root):
    assert (root / "MATRIX_AUDIT_COMPLETE").exists()
    rows = json.loads((root / "summary.json").read_text())
    assert len(rows) == 36
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
        }
    )
    fields = (
        ("tokens_per_second", "Output tokens / s", "Throughput", 1),
        ("outer_acceptance_rate", "Accepted candidates (%)", "Acceptance", 100),
        ("mean_accepted", "Accepted candidates / step", "Accepted length", 1),
        (
            "all_accepted_probability",
            "Fully accepted steps (%)",
            "Full acceptance",
            100,
        ),
    )
    styles = (
        ("low_error", "Low-error", "#4C78A8", "^"),
        ("balanced", "Balanced", "#59A14F", "D"),
        ("aggressive", "Aggressive", "#E45756", "v"),
    )
    for batch in (1, 4, 8, 16):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
        for mode, label, color, marker in styles:
            data = sorted(
                (r for r in rows if r["batch"] == batch and r["mode"] == mode),
                key=lambda r: r["rounds"],
            )
            for ax, (field, _, _, scale) in zip(axes.flat, fields, strict=True):
                ax.plot(
                    [r["rounds"] for r in data],
                    [r[field] * scale for r in data],
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=1.8,
                )
        for i, (ax, (_, ylabel, panel, _)) in enumerate(
            zip(axes.flat, fields, strict=True)
        ):
            ax.set_xlabel("Maximum inner rounds")
            ax.set_ylabel(ylabel)
            ax.set_xticks([4, 6, 8])
            ax.set_ylim(bottom=0)
            if i in (1, 3):
                ax.set_ylim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.text(
                0.5,
                -0.34,
                f"({chr(97 + i)}) {panel}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=13,
            )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(
            left=0.1, right=0.99, top=0.91, bottom=0.13, wspace=0.38, hspace=0.75
        )
        name = f"rounds_b{batch}"
        folder = root / name
        folder.mkdir(exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(
                folder / f"{name}.{extension}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.03,
            )
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# Round limit comparison, batch size {batch}\n\n"
            "Source: ../summary.json. Gemma h4, MTP D4, 16 identical raw prompts, "
            "512 output tokens/request, TP1, greedy, ignore_eos, synchronous. "
            "R4 reuses the historical 2026-09-15 baseline; R6/R8 are new runs. "
            "One warmed measurement per configuration; no error bars.\n\n"
            "(a) Total output throughput including prefill, excluding loading/warmup. "
            "(b) Accepted candidate tokens / submitted candidate tokens. "
            "(c) Accepted candidates per nonempty request verification, no bonus. "
            "(d) Fraction of nonempty request verifications accepting every "
            "submitted candidate; proposal lengths can differ. Final verification "
            "is included. Output equivalence remains a separate check.\n\n"
            "Reproduce: `.venv/bin/python -m "
            "benchmarks.hierarchical.plot_round_sweep <report_directory>`.\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
