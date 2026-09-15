# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the non-spec GDN buffer-size comparison."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCHES = (1, 4, 8, 16, 32)
HISTORIES = (4, 8, 16, 32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output)
    raw = json.loads((root / "raw.json").read_text())
    assert len(raw) == 40
    assert {(r["batch"], r["buffer_size"], r["cache"]) for r in raw} == {
        (b, h, c) for b in BATCHES for h in HISTORIES for c in ("warm", "evicted")
    }
    assert all(r["correctness"] for r in raw)
    assert all(len(v) == 7 and min(v) > 0 for r in raw for v in r["us"].values())
    rows = []
    for cache in ("warm", "evicted"):
        for b in BATCHES:
            points = [r for r in raw if r["cache"] == cache and r["batch"] == b]
            row = dict(batch=b, cache=cache, state_mib=b * 2)
            for name in ("baseline",):
                row[name + "_us"] = statistics.median(
                    v for r in points for v in r["us"][name]
                )
            for h in HISTORIES:
                point = next(r for r in points if r["buffer_size"] == h)
                row[f"replayssm_h{h}_us"] = statistics.median(point["us"]["replayssm"])
            rows.append(row)
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 17,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), layout="constrained")
    curves = [
        ("baseline", "Baseline GDN", "#E39738", "s", "--"),
        (4, "ReplaySSM buffer=4", "#4C78A8", "o", "-"),
        (8, "ReplaySSM buffer=8", "#59A14F", "^", "-"),
        (16, "ReplaySSM buffer=16", "#B279A2", "v", "-"),
        (32, "ReplaySSM buffer=32", "#E45756", "P", "-"),
    ]
    for ax, cache, panel in zip(axes, ("warm", "evicted"), ("a", "b")):
        for key, label, color, marker, style in curves:
            values = []
            for b in BATCHES:
                points = [r for r in raw if r["batch"] == b and r["cache"] == cache]
                if isinstance(key, str):
                    values.append([v for r in points for v in r["us"][key]])
                else:
                    values.append(
                        next(r for r in points if r["buffer_size"] == key)["us"][
                            "replayssm"
                        ]
                    )
            med = np.array([statistics.median(v) for v in values])
            lo = np.array([min(v) for v in values])
            hi = np.array([max(v) for v in values])
            ax.errorbar(
                range(5),
                med,
                yerr=[med - lo, hi - med],
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=2,
                capsize=2,
                markersize=7,
                label=label,
            )
        ax.set_xticks(range(5), list(map(str, BATCHES)))
        ax.set_ylabel("Latency (µs / layer / batch)")
        name = "Warm working set" if cache == "warm" else "After L2 eviction sweep"
        ax.set_xlabel(f"Batch size\n({panel}) {name}")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    folder = root / "ar_buffer_cost"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "ar_buffer_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / "ar_buffer_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    notes = [
        "# 非投机 GDN：ReplaySSM 不写回完整 State 的延迟",
        "",
        "单张 A100 80GB，Qwen3.6 GDN 维度：HQ=16、HV=32、K=V=128。"
        "单层、单个当前位置，每个请求的完整 FP32 State 为 2 MiB。"
        "输入 BF16，历史 d/k FP16，g FP32。现有生产 kernel 和默认 launch "
        "配置，未修改精度、计算逻辑或刷新策略。",
        "",
        "Baseline：fused_recurrent_gated_delta_rule_packed_decode，读取上一位置 "
        "State，计算当前一个 token 的输出与 State，写回完整 State。",
        "",
        "ReplaySSM：fused_recurrent_gated_delta_rule_replayssm，实际分配 "
        "h 个缓存位置，write_pos=h-1；基准副本将 b_is_flush 固定为 False。"
        "从 checkpoint 重建前 h-1 个历史位置，再处理同一个当前 token，"
        "计算当前输出并追加 d/k/g，不写回完整 State。"
        "checkpoint 到当前结果恰好跨 h 个位置。",
        "",
        "固定当前位置为 33，baseline 初始 State 为 S32；ReplaySSM "
        "checkpoint 为 S(33-h)，缓存位置为 33-h 到 31（零基更新编号）。"
        "例如 h=8：从 S25 开始，重放得到 S32，再计算 S33。"
        "这不是从 S24 重放 8 条历史后再计算 S33 的 h+1 路径。",
        "",
        "本次不使用投机解码，不使用合成写入或独立重建核。两个 kernel 都包含 "
        "当前 token 的 gate、Q/K 归一化、GDN 更新和输出计算；"
        "不包含线性投影、Conv、普通 Attention、MoE 和端到端调度。"
        "仅 baseline 写回完整 State。ReplaySSM 保留原生非 flush 的 d/k/g 写入。"
        "ReplaySSM 不在全局内存物化当前完整 State，而是计算其对当前输出的作用。"
        "本轮因此不是两边都输出完整 State tensor 的 materialization 比较。",
        "",
        "batch=1/4/8/16/32，buffer=4/8/16/32；每点7轮，seed=0。"
        "ReplaySSM 曲线是7轮中位数；不依赖 buffer 的 baseline 合并四种 "
        "buffer 对照的28轮。误差线为 min/max，不是置信区间。"
        "纵轴为整个 batch、单个 GDN 层的耗时。",
        "",
        "每轮先恢复原始 State。冷条件随后清扫256 MiB 无关 buffer；"
        "热条件在清扫后执行3次待测 graph，再次恢复 State。"
        "状态恢复、清扫、预热和编译均在计时外。没有硬件计数器证明 "
        "冷条件的全部访问都来自 HBM，也不保证大 batch 的热数据全部驻留 L2。",
        "",
        "全部20组的当前输出通过 baseline 检查 (rtol=0.02, atol=0.002)。"
        "计时外利用更新后的 d/k/g 重建 State，与 baseline 完整 State 对照也通过。"
        "ReplaySSM 的整个 checkpoint 逐元素完全不变。"
        "关闭边界处的自动 flush 仅发生在基准副本；生产文件未修改。"
        "每点测一次，不推进游标或运行满周期，不能据此关闭生产中的刷新。",
        "",
        "数据来源：../raw.json 与 ../summary.csv。",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python "
        "benchmarks/replayssm/qwen36_ar_buffer_cost.py "
        f"--output {root}",
        ".venv/bin/python benchmarks/replayssm/plot_qwen36_ar_buffer_cost.py "
        f"--output {root}",
        "```",
        "",
    ]
    (folder / "ar_buffer_cost.md").write_text("\n".join(notes))
    report = notes + ["## 测量结果", ""]
    for cache in ("warm", "evicted"):
        report += [
            f"### {cache}",
            "",
            "| Batch | Baseline µs | buffer=4 µs | buffer=8 µs | "
            "buffer=16 µs | buffer=32 µs |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            if r["cache"] == cache:
                vals = [r["baseline_us"]] + [r[f"replayssm_h{h}_us"] for h in HISTORIES]
                report.append(
                    f"| {r['batch']} | " + " | ".join(f"{v:.2f}" for v in vals) + " |"
                )
        report.append("")
    report += [
        "![Latency](ar_buffer_cost/ar_buffer_cost.png)",
        "",
        "[PDF](ar_buffer_cost/ar_buffer_cost.pdf)",
        "",
        "代码与报告由 AI 辅助生成；无上游 PR。",
        "",
    ]
    (root / "readme.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
