# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot D3 ReplaySSM versus dual-checkpoint acceptance timing."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCHES = (1, 8, 64)
ACCEPTED = (3, 2)
MODES = (("replayssm", "ReplaySSM", "#4C78A8", "o"), ("dual", "Dual", "#F2B447", "s"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output)
    transition = json.loads((root / "raw.json").read_text())
    steady = json.loads((root / "steady_raw.json").read_text())
    assert len(transition) == 12 and len(steady) == 36
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.9, 6.2), layout="constrained")
    for row_index in range(2):
        for column, batch in enumerate(BATCHES):
            ax = axes[row_index, column]
            for mode, label, color, marker in MODES:
                medians, lower, upper = [], [], []
                for accepted in ACCEPTED:
                    if row_index == 0:
                        point = next(
                            r
                            for r in transition
                            if (r["batch"], r["accepted_drafts"], r["mode"])
                            == (batch, accepted, mode)
                        )
                        samples = point["components_us"]["pair"]
                    else:
                        trajectory = "all" if accepted == 3 else "penultimate"
                        samples = [
                            r["cycle_us"]
                            for r in steady
                            if (r["batch"], r["trajectory"], r["mode"])
                            == (
                                batch,
                                trajectory,
                                "original" if mode == "replayssm" else "dual",
                            )
                        ]
                    median = statistics.median(samples)
                    medians.append(median)
                    lower.append(median - min(samples))
                    upper.append(max(samples) - median)
                ax.errorbar(
                    (0, 1),
                    medians,
                    yerr=(lower, upper),
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=6,
                    capsize=2,
                    label=label,
                )
            ax.set_xticks((0, 1), ("3 (full)", "2 (partial)"))
            panel = chr(ord("a") + row_index * 3 + column)
            scope = "two-window pair" if row_index == 0 else "steady cycle"
            ax.set_xlabel(f"Accepted drafts\n({panel}) bs={batch}, {scope}")
            ax.set_ylabel("Latency (µs / layer / batch)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    folder = root / "acceptance_latency"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "acceptance_latency.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / "acceptance_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    (folder / "acceptance_latency.md").write_text(
        "# D=3 的 ReplaySSM 与 Dual-checkpoint GDN 延迟\n\n"
        "数据：`../raw.json` 与 `../steady_raw.json`；中位数见 `../summary.csv` "
        "和 `../steady_summary.csv`。横轴是接受草稿数，3 为全部接受，2 为最后一个"
        "草稿未接受。上排测上一轮 verify、按接受结果 commit 和下一轮 decode "
        "的连续两窗口 GPU 时间：每点 21 次 CUDA Graph 计时的中位数，误差线为 "
        "min/max。下排测连续全接受或连续只接受两个草稿的稳态平均周期时间："
        "每点三次独立测量的中位数，误差线为 min/max；每次测量包含 "
        "20×32 个连续周期。\n\n"
        "单张 A100 80GB PCIe、单个 Qwen3.6 形状 GDN 层，BF16 输入、FP32 "
        "State、FP16 d/k 历史。原 ReplaySSM 使用逻辑 cap=20/物理 ring=32，"
        "Dual 使用 hard cap=16/物理 ring=16。数值为整个 batch 而非每请求。"
        "这些结果不代表端到端吞吐。\n\n"
        "复现：`.venv/bin/python benchmarks/replayssm/"
        "plot_dual_checkpoint_acceptance_d3.py --output "
        "benchmark_results/qwen36_dual_vs_replayssm_acceptance_d3_20260928`。\n"
    )


if __name__ == "__main__":
    main()
