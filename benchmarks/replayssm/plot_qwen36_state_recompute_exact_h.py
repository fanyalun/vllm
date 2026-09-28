# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot exact-h recurrence against State store and fixed-K16 dot."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCHES = (1, 8, 64)
HISTORIES = (1, 4, 8, 16)
CACHES = ("warm", "evicted")
METHODS = (
    ("store", "State store", "#59A14F", "--", "s"),
    ("fixed16", "Fixed K=16 dot", "#B279A2", ":", "D"),
    ("exact_h", "h-step recurrence", "#4C78A8", "-", "o"),
    ("tile_h", "K=h software tile", "#F2B447", "-.", "^"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output)
    raw = json.loads((root / "raw.json").read_text())
    assert {(r["batch"], r["history"], r["cache"]) for r in raw} == {
        (b, h, c) for b in BATCHES for h in HISTORIES for c in CACHES
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
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), layout="constrained")
    for row_index, cache in enumerate(CACHES):
        for column, batch in enumerate(BATCHES):
            ax = axes[row_index, column]
            points = [
                next(
                    row
                    for row in raw
                    if row["batch"] == batch
                    and row["history"] == h
                    and row["cache"] == cache
                )
                for h in HISTORIES
            ]
            for name, label, color, style, marker in METHODS:
                y = [statistics.median(row["us"][name]) for row in points]
                ax.plot(
                    HISTORIES,
                    y,
                    label=label,
                    color=color,
                    linestyle=style,
                    marker=marker,
                    linewidth=2,
                    markersize=5,
                )
            ax.set_xticks(HISTORIES)
            panel = chr(ord("a") + row_index * 3 + column)
            ax.set_xlabel(f"h\n({panel}) bs={batch}, {cache}")
            ax.set_ylabel("Latency (µs / layer / batch)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    folder = root / "h_tile_latency"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "h_tile_latency.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / "h_tile_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    (folder / "h_tile_latency.md").write_text(
        "# K=h 的 GDN State 重算\n\n"
        "数据：`../raw.json`；汇总：`../summary.csv`。六个面板按 batch size "
        "(1/8/64) 与缓存条件 (warm/evicted) 排列。每点是 21 轮 CUDA "
        "event 测量的中位数，单位为 µs/单层/整批。\n\n"
        "K=h software tile 加载 32×h 的 d 与 h×32 的 k，并累加恰好 h "
        "个外积；h-step recurrence 逐步更新 State；两者都无 h 方向填充。"
        "Fixed K=16 是上一版 TF32x3 Tensor Core 矩阵重建。"
        "State store 是每请求 2 MiB FP32 State 的纯写近似。重算计时不读取 "
        "起点 State、不写出完整终点 State，但计入 d/k/g 历史读取和每 tile "
        "一个校验值的写入。warm 在清扫 256 MiB 后额外执行 3 次待测 graph，"
        "evicted 则直接测量；清扫不计时。\n\n"
        "复现：`.venv/bin/python benchmarks/replayssm/"
        "plot_qwen36_state_recompute_exact_h.py --output "
        "benchmark_results/qwen36_state_recompute_h_tile_20260928`。\n"
    )


if __name__ == "__main__":
    main()
