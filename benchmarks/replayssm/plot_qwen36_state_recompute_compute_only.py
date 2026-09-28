# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the checkpoint-free State recomputation timing matrix."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output)
    raw = json.loads((root / "raw.json").read_text())
    assert {(r["batch"], r["history"], r["cache"]) for r in raw} == {
        (b, h, c)
        for b in (1, 8, 64)
        for h in (1, 4, 8, 16)
        for c in ("warm", "evicted")
    }
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), layout="constrained")
    colors = {"warm": "#4C78A8", "evicted": "#E45756"}
    for ax, batch, panel in zip(axes, (1, 8, 64), ("a", "b", "c")):
        for cache, marker in (("warm", "o"), ("evicted", "s")):
            points = [
                next(
                    r
                    for r in raw
                    if r["batch"] == batch and r["history"] == h and r["cache"] == cache
                )
                for h in (1, 4, 8, 16)
            ]
            median = [statistics.median(r["us"]["recompute"]) for r in points]
            ax.plot(
                (1, 4, 8, 16),
                median,
                marker=marker,
                linewidth=2,
                markersize=6,
                color=colors[cache],
                label=f"Recompute ({cache})",
            )
            store = statistics.median(
                value for r in points for value in r["us"]["store"]
            )
            ax.axhline(
                store,
                linestyle="--" if cache == "warm" else ":",
                linewidth=1.8,
                color=colors[cache],
                label=f"State store ({cache})",
            )
        ax.set_xticks((1, 4, 8, 16))
        ax.set_xlabel(f"h\n({panel}) bs={batch}")
        ax.set_ylabel("Latency (µs / layer / batch)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    folder = root / "state_recompute_cost"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "state_recompute_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / "state_recompute_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    (folder / "state_recompute_cost.md").write_text(
        "# SSM State 写回与重计算\n\n"
        "数据：`../raw.json`，汇总：`../summary.csv`。三幅面板分别为 "
        "bs=1、8、64；横轴为从 h 个位置前重算，纵轴是整批单个 GDN 层的 "
        "CUDA event 延迟。点为 21 轮中位数，水平线为同一批量的完整 State "
        "纯写入中位数。\n\n"
        "Qwen3.6 GDN 形状：每请求 32×128×128 FP32 State (2 MiB)；"
        "历史 d/k 为 FP16、g 为 FP32，矩阵乘为 TF32x3。重算不读取起点 "
        "State，也不写出完整终点 State；起点在寄存器构造，历史读取和每 tile "
        "一个校验值的写入计入时间。两种 cache 条件分别是 3 次预热后测量和 "
        "256 MiB 清扫后测量，清扫不计时。\n\n"
        "复现：`.venv/bin/python benchmarks/replayssm/"
        "plot_qwen36_state_recompute_compute_only.py --output "
        "benchmark_results/qwen36_state_recompute_compute_only_20260928`。\n"
    )


if __name__ == "__main__":
    main()
