# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"Audit and plot the h=16 three-method GDN experiment."

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("standard", "replay", "parallel_last")
LABELS = ("Baseline SD", "ReplaySSM (h=16)", "Parallel, last state only")
COLORS = ("#E39738", "#4C78A8", "#59A14F")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    rows = []
    for b in (1, 4, 8, 16):
        for d in (4, 8, 16, 32):
            data = json.loads((root / f"b{b}_d{d}_s0.json").read_text())
            assert len(data) == 2 and {r["layers"] for r in data} == {1, 30}
            for r in data:
                assert r["history"] == 16 and r["correctness"]
                assert r["batch"] == b and r["draft"] == d
                assert set(r["us"]) == set(METHODS)
                assert all(
                    len(v) == 21 and all(x > 0 for x in v) for v in r["us"].values()
                )
            rows.extend(data)
    summary = []
    for r in rows:
        s = {k: r[k] for k in ("batch", "draft", "history", "layers")}
        for key in METHODS:
            s[key + "_us"] = statistics.median(r["us"][key])
            s[key + "_speedup"] = (
                statistics.median(r["us"]["standard"]) / s[key + "_us"]
            )
        summary.append(s)
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 17,
            "axes.labelsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for metric in ("latency", "speedup"):
        name = "h16_" + metric
        folder = root / name
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
        for index, (ax, d) in enumerate(zip(axes.flat, (4, 8, 16, 32))):
            selected = [
                next(
                    r
                    for r in rows
                    if r["batch"] == b and r["draft"] == d and r["layers"] == 30
                )
                for b in (1, 4, 8, 16)
            ]
            for j, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
                if metric == "latency":
                    vals = [np.median(r["us"][method]) for r in selected]
                    low = [min(r["us"][method]) for r in selected]
                    high = [max(r["us"][method]) for r in selected]
                else:
                    vals = [
                        np.median(r["us"]["standard"]) / np.median(r["us"][method])
                        for r in selected
                    ]
                    low = [
                        min(np.array(r["us"]["standard"]) / r["us"][method])
                        for r in selected
                    ]
                    high = [
                        max(np.array(r["us"]["standard"]) / r["us"][method])
                        for r in selected
                    ]
                errs = [
                    np.maximum(0, np.array(vals) - low),
                    np.maximum(0, np.array(high) - vals),
                ]
                ax.bar(
                    np.arange(4) + (j - 1) * 0.25,
                    vals,
                    width=0.23,
                    color=color,
                    label=label,
                    yerr=errs,
                    error_kw={"elinewidth": 0.8, "capsize": 2},
                )
            ax.set_xticks(range(4), ["1", "4", "8", "16"])
            ax.set_xlabel(f"Batch size\n({chr(97 + index)}) Draft = {d}")
            ax.set_ylabel(
                "GDN latency (µs / layer)"
                if metric == "latency"
                else "Kernel speedup over baseline"
            )
            if metric == "latency":
                ax.set_yscale("log")
            else:
                ax.axhline(1, color="0.45", linestyle=":", linewidth=1)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# Fixed history: {metric}\n\n"
            "One A100 80GB per experiment; Qwen3.6 GDN dimensions, synthetic "
            "inputs. Thirty independent layer buffers, total latency divided "
            "by 30. D=4/8/16/32; verify width T=D+1; batch=1/4/8/16; history "
            "h=16. Replay uses R=64, ring capacity=128, history tile=64, and "
            "no flush. Baseline writes every candidate state; parallel-last "
            "writes only the final state, with no history-ring writes. FP32 "
            "state, BF16 inputs, FP16 replay d/k ring.\n\nBars: median latency "
            "or ratio of median baseline/method latency over 21 paired "
            "rounds. Whiskers: min/max latency or paired speedup, not "
            "confidence intervals. Latency uses a log axis. Speedup is "
            "kernel-only, not end-to-end. Reset, compilation, warmup and GPU "
            "prelude are outside timing. Final candidate state is scratch; no"
            " acceptance or scheduler integration is measured.\n\nSource: "
            "../b*_d*_s0.json and ../summary.csv. Reproduce: `"
            ".venv/bin/python "
            "benchmarks/replayssm/plot_qwen36_fixed_history.py --output "
            "benchmark_results/qwen36_a100_h16_three_methods_20260914`.\n"
        )
    lines = [
        "# Qwen3.6 A100: three GDN methods at h=16",
        "",
        "User-requested fixed-distance rerun. All 16 batch/draft "
        "combinations complete. Both single-layer and 30-layer working "
        "sets were freshly measured; 21 rounds per method. All output "
        "checks and parallel-last final-state checks passed (rtol=0.04, "
        "atol=0.01).",
        "",
        "## Methods",
        "",
        "- Baseline SD: read the current state, recurrent verification, "
        "write all candidate states.",
        "- ReplaySSM: read a checkpoint 16 committed positions earlier "
        "plus history, parallel verification, write d/k/g. No flush.",
        "- Parallel-last: read the same current state as baseline, use "
        "the ReplaySSM within-window solve, write only the final "
        "candidate state. No d/k/g ring writes.",
        "",
        "The benchmark-only parallel-last kernel retains the existing "
        "solve and verify launch configuration. Final state is formed "
        "from the solved deltas and normalized keys. Small windows pad "
        "only the final-state dot to a reduction width of 16. The "
        "final-state epilogue reloads starting-state tiles and keys; "
        "actual read traffic is not asserted to equal one full-state "
        "read. It has one kernel launch; the unmodified ReplaySSM wrapper"
        " has two launches with device-side routing.",
        "",
        "## Configuration and limits",
        "",
        "Actual checkpoint distance h=16 is fixed by the user. R=64, "
        "physical ring=128, history tile=64 are retained from the "
        "previous study. The source default R=16 is not being used. "
        "T=D+1. HQ=16, HV=32, K=V=128. The input checkpoint is read-only "
        "for parallel-last; the final state is written to a separate "
        "scratch buffer. A canary checks that the unused slot stays "
        "untouched. No intermediate full-state tensor is allocated by the"
        " new kernel.",
        "",
        "Each repetition restores inputs outside timing. Each point uses "
        "identical current state and verification inputs across methods. "
        "CUDA graphs exclude Python dispatch, reset and warmup. A GPU "
        "prelude excludes CPU submission gaps. Only GDN core computation "
        "is measured: no projection, convolution, full attention, MoE, "
        "acceptance, prefill or end-to-end execution. One seed (0); "
        "min/max error bars describe these 21 rounds, not cross-input "
        "uncertainty. Source snapshots and per-cell launch commands "
        "accompany the raw results.",
        "",
        "## Thirty-layer results",
        "",
        "| Batch | Draft | Baseline µs | Replay µs | Parallel-last µs | "
        "Replay speedup | Parallel-last speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary:
        if s["layers"] == 30:
            lines.append(
                f"| {s['batch']} | {s['draft']} | {s['standard_us']:.2f} | "
                f"{s['replay_us']:.2f} | {s['parallel_last_us']:.2f} | "
                f"{s['replay_speedup']:.3f} | {s['parallel_last_speedup']:.3f} |"
            )
    lines += [
        "",
        "## Figures",
        "",
        "![Speedup](h16_speedup/h16_speedup.png)",
        "",
        "[Speedup PDF](h16_speedup/h16_speedup.pdf)",
        "",
        "![Latency](h16_latency/h16_latency.png)",
        "",
        "[Latency PDF](h16_latency/h16_latency.pdf)",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python benchmarks/replayssm/qwen36_fixed_history.py "
        f"--queue --gpu 0 --output {root}",
        ".venv/bin/python benchmarks/replayssm/qwen36_fixed_history.py "
        f"--queue --gpu 1 --output {root}",
        ".venv/bin/python "
        "benchmarks/replayssm/plot_qwen36_fixed_history.py "
        f"--output {root}",
        "```",
        "",
        "Code and report prepared with AI assistance; no upstream PR.",
        "",
    ]
    (root / "readme.md").write_text("\n".join(lines))
    (root / "measurement_complete.json").write_text(
        json.dumps(
            dict(
                cells=16,
                working_sets=[1, 30],
                history=16,
                repeats=21,
                timed_values=32 * 3 * 21,
                correctness=True,
                figures_visually_validated=False,
            ),
            indent=2,
        )
        + "\n"
    )
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("b*_d*_s0.json")
    }
    (root / "raw_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
