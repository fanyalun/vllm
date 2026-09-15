# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the small-sample state materialization experiment."""

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
    assert {(r["batch"], r["history"], r["cache"]) for r in raw} == {
        (b, h, c) for b in BATCHES for h in HISTORIES for c in ("warm", "evicted")
    }
    assert all(r["correctness"] for r in raw)
    assert all(len(v) == 7 and min(v) > 0 for r in raw for v in r["us"].values())
    rows = []
    for cache in ("warm", "evicted"):
        for b in BATCHES:
            points = [r for r in raw if r["cache"] == cache and r["batch"] == b]
            row = dict(batch=b, cache=cache, state_mib=b * 2)
            for name in ("store_only", "copy"):
                row[name + "_us"] = statistics.median(
                    v for r in points for v in r["us"][name]
                )
            for h in HISTORIES:
                point = next(r for r in points if r["history"] == h)
                row[f"reconstruct_h{h}_us"] = statistics.median(
                    point["us"]["reconstruct"]
                )
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
        ("store_only", "Store only (synthetic)", "#E39738", "s", "--"),
        ("copy", "Copy (read + write)", "#777777", "D", "--"),
        (4, "Reconstruct h=4", "#4C78A8", "o", "-"),
        (8, "Reconstruct h=8", "#59A14F", "^", "-"),
        (16, "Reconstruct h=16", "#B279A2", "v", "-"),
        (32, "Reconstruct h=32", "#E45756", "P", "-"),
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
                        next(r for r in points if r["history"] == key)["us"][
                            "reconstruct"
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
    folder = root / "state_cost"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "state_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / "state_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    note = [
        "# SSM State 写回与重建的小样本比较",
        "",
        "Qwen3.6 GDN 维度；单张 A100 80GB；单层，每个请求的 State 为 "
        "32×128×128 FP32，即 2 MiB。横轴为 batch size，纵轴为整个 batch "
        "处理一层的耗时，不是每个请求的耗时。",
        "",
        "Store only：由寄存器中的简单整数运算生成非零 FP32 模式并写入 State "
        "大小的 buffer，不读取源 State。它是纯 store 成本的近似，包含生成模式的 "
        "少量计算和 kernel 启动成本，不是实际模型 flush kernel。",
        "",
        "Copy：读取已经物化的当前 State，写入另一个 buffer，包含完整的读和写。",
        "",
        "Reconstruct：读取 h 个位置之前的 checkpoint 和 h 条 d/k/g 历史更新，"
        "重建并写出当前完整 State。计时包含 checkpoint/历史读取、重建计算和结果写入；"
        "不包含投影、草稿验证、历史生成、接受/拒绝和端到端调度。",
        "",
        "所有 h 固定同一个当前位置 32，分别从位置 32-h 的 checkpoint 开始。"
        "State FP32，d/k 缓存 FP16，g FP32。矩阵乘使用 TF32x3；历史 tile "
        "为 max(16,next_pow2(h))，不是上一轮的固定 64 tile。",
        "",
        "每个 batch/h/cache 组合 7 轮，随机交错三种方法；seed=0。"
        "重建曲线取 7 轮中位数；与 h 无关的 store/copy 合并四个 h 的共 28 轮。"
        "误差线为 min/max，不是置信区间。",
        "",
        "每次测量前写入 256 MiB 无关 buffer，操作位于计时外。左图随后先执行 "
        "3 次待测 kernel，右图直接测量。清扫操作未通过硬件计数器验证 L2 命中率；"
        "热工作集也不保证大 batch 的全部数据能够驻留 L2。CUDA event 包围一次 "
        "CUDA graph 中的 kernel，预热、编译、输入构造不计时。",
        "",
        "图中重建时间已经包含一次完整 State 写出，因此不能把它全部解释为纯算术成本，"
        "也不能把两条曲线相减当作严格隔离的重计算时间。",
        "",
        "数据来源：../raw.json、../summary.csv。复现：",
        "",
        "```bash",
        ".venv/bin/python benchmarks/replayssm/plot_qwen36_state_cost.py "
        f"--output {root}",
        "```",
        "",
    ]
    (folder / "state_cost.md").write_text("\n".join(note))
    report = note[:1] + [
        "",
        "20 个 batch/h 组合，2 种缓存条件，3 种操作，"
        "每项 7 轮，共 840 个计时值；测量与数值检查均完成。",
        "",
    ]
    report += note[2:]
    report += ["## 耗时结果", "", "单位为 µs，完整 batch、单个 GDN 层。", ""]
    for cache in ("warm", "evicted"):
        report += [
            f"### {cache}",
            "",
            "| Batch | State MiB | 仅写入 | 读＋写拷贝 | h=4 重建 | "
            "h=8 重建 | h=16 重建 | h=32 重建 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            if r["cache"] == cache:
                vals = [r["store_only_us"], r["copy_us"]] + [
                    r[f"reconstruct_h{h}_us"] for h in HISTORIES
                ]
                report.append(
                    f"| {r['batch']} | {r['state_mib']} | "
                    + " | ".join(f"{v:.2f}" for v in vals)
                    + " |"
                )
        report.append("")
    report += [
        "## 数值检查与限制",
        "",
        "重建输出同时对照原始逐步 FP32 GDN 递推结果 "
        "(rtol=0.02, atol=0.002)，以及根据实际缓存向量计算的独立 "
        "FP64 重建结果 (rtol=0.002, atol=0.0002)。全部通过。"
        "拷贝与原 State 逐元素完全一致；store 输出模式检查通过。",
        "",
        "初始普通 TF32 版本在 FP64 对照中有一个元素超过门槛，"
        "失败记录保留在 initial_tf32_gate_failure/；正式结果全部采用 "
        "TF32x3 重新测量，没有放宽门槛，也没有混用初测耗时。"
        "这不是对生产 ReplaySSM 完整 flush 的测量。",
        "",
        "![State cost](state_cost/state_cost.png)",
        "",
        "[PDF](state_cost/state_cost.pdf)",
        "",
        "代码与报告由 AI 辅助生成。本次没有上游 PR。",
        "",
    ]
    (root / "readme.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
