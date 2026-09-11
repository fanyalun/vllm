# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the audited small-sample long-context acceptance matrix."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    if not (run / "RUN_COMPLETE").exists():
        raise RuntimeError("The complete audited matrix is required")
    with (run / "acceptance_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for axis, length, label in zip(
        axes, (16384, 32768), ("(a) 16K context", "(b) 32K context"), strict=True
    ):
        for method, title, color, marker in (
            ("mtp", "MTP", "#4C78A8", "o"),
            ("moe_skip", "MoE-Skip (top-4)", "#F2B447", "s"),
        ):
            values = [
                r
                for r in rows
                if r["method"] == method and int(r["context_tokens"]) == length
            ]
            axis.plot(
                [int(r["d"]) for r in values],
                [float(r["mean_acceptance_length"]) for r in values],
                label=title,
                color=color,
                marker=marker,
                linewidth=1.8,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        axis.set_xlabel("Draft length D\n" + label)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean acceptance length")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=0.4, w_pad=1)
    directory = run / "long_context_acceptance"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"long_context_acceptance.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    notes = """# Qwen3.6 long-context acceptance pilot

Qwen3.6-35B-A3B; MoE-Skip top-h=4 versus native MTP; D=4/8/16/32.
Input lengths are exactly 16,384 and 32,768 tokens. Each cell has four requests
and 256 generated tokens per request; B=1, TP=1, A100 80GB, CUDA graphs,
prefix caching disabled, temperature=1, top_p=0.95, ignore_eos=True.
Seeds are 20260911 + sample index. The methods run on separate GPUs;
this experiment does not measure throughput.

Four disjoint concatenated C4 blocks are supplied as raw token continuations,
without a chat template. Each 16K input is the suffix of its paired 32K input.
This is a context-length pilot, not a long-document QA quality evaluation.
The same input tokens and request seeds are used across methods and widths.
Stochastic methods may consume random numbers differently and need not produce
identical token trajectories. No distributional correctness claim is made.

Mean acceptance length = 1 + sum(accepted draft tokens) / sum(spec verify steps).
This is weighted by verification steps, with the conventional +1 recovery/bonus
term; it is not the exact emitted-token yield at the generation boundary.
The CSV also reports accepted draft tokens per step and the actual output token
count divided by speculative steps (which includes any non-spec output).
No confidence intervals are estimated from this four-sample pilot.
Panels (a)/(b) show 16K/32K respectively. Raw data: ../acceptance_summary.csv,
../acceptance_by_request.csv, ../cells/*/result.json; checks: ../audit.json.

AI assistance was used to prepare the benchmark and report.

Reproduce from the repository root:

```bash
.venv/bin/python benchmarks/moe_skip/run_long_context_acceptance.py --run-dir RUN_DIR
export MPLCONFIGDIR=/tmp/moe_skip_mpl
.venv/bin/python benchmarks/moe_skip/plot_long_context_acceptance.py --run-dir RUN_DIR
```
"""
    (directory / "long_context_acceptance.md").write_text(notes)
    table = ["| Context | D | MTP | MoE-Skip |", "| --- | ---: | ---: | ---: |"]
    for length in (16384, 32768):
        for width in (4, 8, 16, 32):
            selected = {
                r["method"]: float(r["mean_acceptance_length"])
                for r in rows
                if int(r["context_tokens"]) == length and int(r["d"]) == width
            }
            table.append(
                f"| {length} | {width} | {selected['mtp']:.4f} | "
                f"{selected['moe_skip']:.4f} |"
            )
    (run / "RESULTS.md").write_text(notes + "\n" + "\n".join(table) + "\n")


if __name__ == "__main__":
    main()
