# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and render the fixed Qwen3.6 A100 performance matrix."""

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import regex as re


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def profile_summary(cell, mode):
    aggregates = defaultdict(lambda: [0, 0.0])
    traces = sorted((cell / "profile").glob("*.pt.trace.json.gz"))
    cached_path = cell / "kernel_summary.json"
    if not traces and cached_path.exists():
        cached = json.loads(cached_path.read_text())
        steps = cached["inferred_target_steps"]
        return dict(
            profile_steps=steps,
            gdn_layer_us=cached["recurrent_us"] / (steps * 30),
            gdn_total_step_us=(cached["recurrent_us"] + cached["cursor_us"]) / steps,
        )
    for path in traces:
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        for event in data["traceEvents"]:
            if event.get("cat") != "kernel":
                continue
            name = event["name"]
            aggregates[name][0] += 1
            aggregates[name][1] += event.get("dur", 0)
    if mode == "ar":
        anchor = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
        kernels_per_layer = 1
    elif mode == "standard":
        anchor = "fused_sigmoid_gating_delta_rule_update_kernel"
        kernels_per_layer = 1
    else:
        anchor = "gdn_replayssm_spec_circular_kernel"
        kernels_per_layer = 2
    count = sum(v[0] for k, v in aggregates.items() if anchor in k)
    recurrent_us = sum(v[1] for k, v in aggregates.items() if anchor in k)
    cursor_us = sum(
        v[1] for k, v in aggregates.items() if "_advance_gdn_spec_cursors_kernel" in k
    )
    steps = count / (30 * kernels_per_layer)
    dump(
        cell / "kernel_summary.json",
        dict(
            traces=[str(p) for p in traces],
            anchor=anchor,
            anchor_calls=count,
            inferred_target_steps=steps,
            recurrent_us=recurrent_us,
            cursor_us=cursor_us,
            all_kernels=dict(aggregates),
        ),
    )
    return dict(
        profile_steps=steps,
        gdn_layer_us=recurrent_us / (steps * 30) if steps else None,
        gdn_total_step_us=(recurrent_us + cursor_us) / steps if steps else None,
    )


def compare(left, right):
    exact = 0
    prefix_lengths = []
    divergences = []
    for index, (a, b) in enumerate(zip(left, right)):
        first = next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
        )
        prefix_lengths.append(first)
        exact += a == b
        if a != b:
            divergences.append(
                dict(
                    sample=index,
                    first_difference=first,
                    left=a[first : first + 8],
                    right=b[first : first + 8],
                )
            )
    return dict(
        exact_samples=exact,
        total_samples=len(left),
        common_prefix_tokens=sum(prefix_lengths),
        divergences=divergences,
    )


def report(root, rows, audit):
    lookup = {(r["batch"], r["draft"], r["mode"]): r for r in rows}
    unstable = sum(not r["repeated_output_exact"] for r in rows)
    comparisons = [r["comparison"] for r in audit if "comparison" in r]
    differing = sum(r["exact_samples"] < r["total_samples"] for r in comparisons)
    controlled_path = root / "controlled_kernels.json"
    controlled = (
        {(r["batch"], r["draft"]): r for r in json.loads(controlled_path.read_text())}
        if controlled_path.exists()
        else {}
    )
    text = [
        "# Qwen3.6 单 A100：AR / baseline SD / ReplaySSM\n",
        f"已完成 {len(rows)}/36 个配置的计时和 CUDA profile。\n",
        "完整矩阵为 4 个 AR 对照、16 个 baseline SD、16 个 ReplaySSM；"
        "同一 B 的四个草稿长度共用该 B 的 AR 对照。\n",
        "模型为本地 Qwen3.6-35B-A3B，BF16 权重/激活、FP32 SSM state。"
        "每组单卡 A100 80GB PCIe，TP=1，EP 关闭；两卡分别跑不同配置。"
        "B 为每次提交的请求数及 max_num_seqs；实际执行 batch 由调度器决定。"
        "GSM8K test 前 16 条，greedy，thinking 关闭、ignore_eos=True，"
        "每条严格输出 128 tokens。"
        "原生 MTP 草稿长度 D=4/8/16/32；验证宽度 D+1。"
        "ReplaySSM history block 固定 64。CUDA Graph 开启，"
        "prefix cache 关闭，max_model_len=1024，KV cache 固定 7 GiB，"
        "显式 cache 预算覆盖 gpu_memory_utilization 设置。\n",
        "端到端吞吐为 2048 / 16 条请求的总执行秒数，含 prefill、draft、"
        "verify、调度及输出处理，不含模型加载、预热或 profiler。"
        "表中为三轮中位数；图中误差线为三轮 min/max，不是置信区间。"
        "各 batch 的 AR 对照位于同一张物理 GPU。先预热两轮完整数据；"
        "计时轮若有 worker JIT 事件则归档为额外预热，补足三轮无 JIT 测量。\n",
        "summary 的 accepted_length 为引擎计数器的 "
        "1 + accepted_draft_tokens / draft_rounds，三轮取平均；"
        "它不是最终截断后每轮实际输出 token 数。吞吐始终按"
        "实际返回的 16×128 tokens 除以执行耗时计算。\n",
        "注意：带 * 的 SD 配置，其引擎报告的 1024-token 请求缓存并发容量"
        "小于请求 B；这不是实际短请求并发数的直接测量。"
        "实际准入还应结合 summary 中的 Running/Waiting 日志采样值判断，"
        "端到端结果包含引擎实际调度行为，缓存不足点不是固定实际 batch=B 的对照。"
        "主图与下表的 kernel speedup 使用另行测量的固定 B 单层 kernel："
        "真实 Qwen GDN 维度、合成输入、每步固定提交 4 tokens，"
        "包含 ReplaySSM verify/flush 和 cursor commit，不含 Conv、投影、MoE。"
        "单层 CUDA Graph 重复使用同一组 buffers，缓存局部性与完整模型不同。"
        "补充 kernel latency 图使用完整模型的独立 profile，"
        "它会受准入限制及自然结束降批影响。两种 kernel 口径分别保存。\n",
        f"已完成配置中，{unstable}/{len(rows)} 个配置的三轮输出不完全一致；"
        f"已完成的跨方法对照中，{differing}/{len(comparisons)} 组存在逐 token 差异。"
        "有输出差异的对照只能解释为当前运行条件下的实测性能，不能作为严格"
        "输出等价下的纯 kernel 因果加速结论。"
        "详细分歧与重复稳定性见 output_consistency.json。\n",
        "| B | D | AR tok/s | SD tok/s | Replay tok/s | SD/AR | Replay/AR | "
        "Replay/SD | Kernel SD/Replay |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in (1, 4, 8, 16):
        for draft in (4, 8, 16, 32):
            keys = [
                (batch, 0, "ar"),
                (batch, draft, "standard"),
                (batch, draft, "replayssm"),
            ]
            if not all(k in lookup for k in keys):
                continue
            ar, base, replay = (lookup[k] for k in keys)
            b, s, a = base["throughput"], replay["throughput"], ar["throughput"]
            point = controlled.get((batch, draft))
            kernel = f"{point['speedup']:.3f}×" if point else "pending"
            marker = "*" if base["capacity_below_batch"] else ""
            text.append(
                f"| {batch} | {draft} | {a:.2f} | {b:.2f}{marker} | "
                f"{s:.2f} | {b / a:.3f}× | {s / a:.3f}× | {s / b:.3f}× | "
                f"{kernel} |"
            )
    if len(rows) == 36 and len(controlled) == 16:
        text.extend(
            [
                "\n## 图\n",
                "![Kernel and E2E speedup]"
                "(qwen36_a100_figure7/qwen36_a100_figure7.png)\n",
                "[主图 PDF](qwen36_a100_figure7/qwen36_a100_figure7.pdf)\n",
                "- 绝对吞吐：[PNG]"
                "(qwen36_a100_throughput/qwen36_a100_throughput.png) / "
                "[PDF](qwen36_a100_throughput/qwen36_a100_throughput.pdf)",
                "- 整模型 profile 中的 GDN 延迟：[PNG]"
                "(qwen36_a100_kernel_latency/qwen36_a100_kernel_latency.png) / "
                "[PDF](qwen36_a100_kernel_latency/qwen36_a100_kernel_latency.pdf)",
            ]
        )
    text.extend(
        [
            "\n## 运行稳定性\n",
            "| 配置 | 三轮输出完全一致 | 吞吐 min–max (tok/s) | 缓存并发上限 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for r in rows:
        text.append(
            f"| B={r['batch']}, D={r['draft']}, {r['mode']} | "
            f"{r['repeated_output_exact']} | {r['throughput_min']:.2f}–"
            f"{r['throughput_max']:.2f} | {r['max_context_concurrency']} |"
        )
    text.extend(
        [
            "\n## 文件与复现\n",
            "- contract.json：环境、模型/数据指纹与实验口径。",
            "- model_weights_sha256.json：26 个权重分片的完整 SHA-256；"
            "environment.json 记录 GPU、驱动、Python 与拓扑。",
            "- source/：实际测量源码的逐字副本，使用 .py.txt 后缀保留原始内容；"
            "逐字复现时在独立目录恢复 .py 文件名，从仓库根目录运行，"
            "CLI 参数与下方命令相同。",
            "- summary.csv / summary.json：完整标量结果。",
            "- controlled_kernels.csv / controlled_kernels.json：固定 B 单层 "
            "kernel 实测及七轮 min/max；主图 kernel speedup 的数据来源。",
            "- b*_d*_*：每个配置的参数、命令、日志、逐轮 token 与原始 profile。",
            "- kernel_correctness.json：真实 Qwen GDN 维度的四种 D 数值检查；"
            "每种含两个单步历史位置和 80 步随机接受/回滚检查。",
            "- 图目录各含同名 PNG、PDF、Markdown。",
            "\n以下在相同源码提交、模型和运行环境下新建复测目录。"
            "两条模型矩阵命令可在独立终端并行；kernel 测量须等模型进程退出。\n",
            "```bash\n"
            f"result_root={root}\n"
            f"run_root={root}_rerun\n"
            "frozen_dir=/tmp/qwen36_a100_measured\n"
            'mkdir -p "$frozen_dir" "$run_root/source"\n'
            'cp "$result_root/contract.json" "$run_root/contract.json"\n'
            'cp "$result_root/model_weights_sha256.json" "$run_root/"\n'
            'cp "$result_root/environment.json" "$run_root/"\n'
            'cp "$result_root"/source/*.py.txt "$run_root/source/"\n'
            "for name in qwen36_a100_matrix qwen36_replayssm_gate "
            "qwen36_gdn_kernel_matrix; do\n"
            '  cp "$result_root/source/$name.py.txt" "$frozen_dir/$name.py"\n'
            "done\n"
            '.venv/bin/python "$frozen_dir/qwen36_a100_matrix.py" '
            '--output "$run_root" --gpu 0 --batches 1,4\n'
            '.venv/bin/python "$frozen_dir/qwen36_a100_matrix.py" '
            '--output "$run_root" --gpu 1 --batches 8,16\n'
            "CUDA_VISIBLE_DEVICES=1 .venv/bin/python "
            '"$frozen_dir/qwen36_gdn_kernel_matrix.py" --output "$run_root"\n'
            ".venv/bin/python benchmarks/replayssm/analyze_qwen36_a100_matrix.py "
            '"$run_root"\n```\n',
            "展示参考 [ReplaySSM Figure 7](https://dao-lab.ai/blog/2026/replayssm/)。"
            "原图扫 buffer 容量；本实验按用户指定扫草稿长度，二者含义不同。\n",
            "实验和分析脚本使用了 AI 辅助。\n",
        ]
    )
    (root / "readme.md").write_text("\n".join(text))


def figures(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 18,
            "axes.labelsize": 20,
            "pdf.fonttype": 42,
        }
    )
    lookup = {(r["batch"], r["draft"], r["mode"]): r for r in rows}
    controlled = {
        (r["batch"], r["draft"]): r
        for r in json.loads((root / "controlled_kernels.json").read_text())
    }
    assert len(controlled) == 16
    drafts = [4, 8, 16, 32]
    colors = {"standard": "#F2B447", "replayssm": "#4C78A8"}
    labels = {"standard": "Baseline SD", "replayssm": "ReplaySSM"}
    notes = (
        "# Qwen3.6 A100 performance figure\n\n"
        "Qwen3.6-35B-A3B BF16, one A100 80GB PCIe per cell; TP1, EP off. "
        "GSM8K first 16 test samples, 128 output tokens each, greedy, MTP. "
        "D is number of proposed draft tokens; target width is D+1. "
        "B is submitted request batch and max_num_seqs, not a guarantee of "
        "constant execution batch under scheduler admission and completion. "
        "ReplaySSM history block=64. Three timing repeats, median throughput. "
        "All methods have a fixed 7 GiB KV cache budget. "
        "E2E includes prefill and all decode work, excludes load/warmup/profile. "
        "CUDA Graph enabled. Input data: ../summary.csv and ../contract.json.\n\n"
        "Figure7 panel (a) uses controlled fixed-B single-layer kernels with "
        "four committed tokens per verify step and real Qwen GDN dimensions. "
        "It includes ReplaySSM verify, flush and cursor commit, with 64 calls "
        "per CUDA graph, 20 graph replays per timing and seven timing repeats. "
        "Repeated single-layer buffers have different cache locality from a "
        "complete model. "
        "The standalone call includes one cursor commit; the model shares "
        "metadata commits across layers and its profile counts actual calls. "
        "Inputs are synthetic; E2E uses natural MTP acceptance on GSM8K. "
        "See ../controlled_kernels.json. The supplementary kernel latency "
        "figure instead uses a separate full-model CUDA profile, summing actual "
        "GDN verify and flush kernels plus cursor commits; normalized by "
        "inferred target steps (30 GDN layers, two ReplaySSM launches/layer). "
        "This is recurrent-kernel time, excluding convolution/projections/MoE. "
        "Kernel ratio compares SD and ReplaySSM at the same B,D; E2E ratio "
        "uses AR at the same B on the same physical GPU. Their denominators differ.\n\n"
        "Layout inspired by [ReplaySSM Figure 7]"
        "(https://dao-lab.ai/blog/2026/replayssm/); "
        "that figure sweeps buffer capacity, whereas these figures sweep D. "
        "See ../output_consistency.json for numerical/token parity limitations.\n"
        "Hatching denotes engine-reported cache capacity below requested B; "
        "This capacity is estimated for 1024-token requests, not a direct "
        "measurement of active short requests. E2E includes actual scheduling. "
        "Error bars are min/max of three timing repeats "
        "(not confidence intervals), with AR median held fixed for speedup. "
        "Profiles include naturally shrinking decode batches near completion.\n"
        "The supplementary kernel-latency figure uses a logarithmic y-axis "
        "to show both short-window timings and the large D=32 slowdown.\n"
    )
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.8))
    shades = ["#C2CFDF", "#93AAC7", "#6E8DB3", "#446589"]
    batches = (1, 4, 8, 16)
    for panel, ax in enumerate(axes):
        for index, draft in enumerate(drafts):
            values = []
            limited = []
            errors = []
            for batch in batches:
                ar = lookup[batch, 0, "ar"]
                base = lookup[batch, draft, "standard"]
                replay = lookup[batch, draft, "replayssm"]
                row = base if panel == 1 else replay
                capped = False if panel == 0 else row["capacity_below_batch"]
                limited.append(capped)
                values.append(
                    controlled[batch, draft]["speedup"]
                    if panel == 0
                    else row["throughput"] / ar["throughput"]
                )
                errors.append(
                    [
                        (row["throughput"] - row["throughput_min"]) / ar["throughput"],
                        (row["throughput_max"] - row["throughput"]) / ar["throughput"],
                    ]
                )
            bars = ax.bar(
                np.arange(4) + (index - 1.5) * 0.19,
                values,
                0.18,
                color=shades[index],
                label=f"D={draft}",
                yerr=np.array(errors).T if panel else None,
                error_kw={"elinewidth": 0.8, "capsize": 1.5},
            )
            for patch, capped in zip(bars, limited):
                if capped and panel:
                    patch.set_hatch("//")
                elif capped:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        0.03,
                        "N/A",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )
        ax.axhline(1, color="0.3", linestyle="--", linewidth=1.2)
        ax.set_xticks(np.arange(4), [f"B={b}" for b in batches])
        ax.set_xlabel(
            [
                "(a) GDN kernel: SD / ReplaySSM",
                "(b) E2E: baseline SD / AR",
                "(c) E2E: ReplaySSM / AR",
            ][panel]
        )
        ax.set_ylabel("Speedup (×)")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="//"))
    names.append("Cache capacity < B")
    fig.legend(handles, names, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1)
    name = "qwen36_a100_figure7"
    folder = root / name
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    (folder / f"{name}.md").write_text(notes)
    plt.close(fig)
    for kind in ("throughput", "kernel_latency"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for col, batch in enumerate((1, 4, 8, 16)):
            ar = lookup[batch, 0, "ar"]
            ax = axes.flat[col]
            x = np.arange(4)
            for index, mode in enumerate(("standard", "replayssm")):
                cells = [lookup[batch, d, mode] for d in drafts]
                if kind == "kernel_latency":
                    values = [r["gdn_total_step_us"] / 30 for r in cells]
                else:
                    values = [r["throughput"] for r in cells]
                bars = ax.bar(
                    x + (index - 0.5) * 0.36,
                    values,
                    0.36,
                    color=colors[mode],
                    label=labels[mode],
                    yerr=np.array(
                        [
                            [r["throughput"] - r["throughput_min"] for r in cells],
                            [r["throughput_max"] - r["throughput"] for r in cells],
                        ]
                    )
                    if kind == "throughput"
                    else None,
                    error_kw={"elinewidth": 0.8, "capsize": 1.5},
                )
                for patch, row in zip(bars, cells):
                    if row["capacity_below_batch"]:
                        patch.set_hatch("//")
            baseline = (
                ar["throughput"]
                if kind == "throughput"
                else ar["gdn_total_step_us"] / 30
            )
            ax.axhline(baseline, color="0.35", linestyle="--", label="AR")
            if kind == "kernel_latency":
                ax.set_yscale("log")
            ax.set_xticks(x, drafts)
            letter = chr(97 + col)
            ax.set_xlabel(f"Draft length D\n({letter}) B={batch}")
            ax.set_ylabel(
                {
                    "throughput": "Output tokens/s",
                    "kernel_latency": "GDN µs / layer / step",
                }[kind]
            )
        for ax in axes.flat:
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        handles, names = axes.flat[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.1, w_pad=1)
        name = f"qwen36_a100_{kind}"
        folder = root / name
        folder.mkdir(exist_ok=True)
        fig.savefig(
            folder / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03
        )
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
        (folder / f"{name}.md").write_text(notes)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    outputs = {}
    prompt_reference = None
    audit = []
    source_hash = hashlib.sha256(
        (root / "source" / "qwen36_a100_matrix.py.txt").read_bytes()
    ).hexdigest()
    for batch in (1, 4, 8, 16):
        for mode, draft in [("ar", 0)] + [
            (m, d) for d in (4, 8, 16, 32) for m in ("standard", "replayssm")
        ]:
            cell = root / f"b{batch}_d{draft}_{mode}"
            path = cell / "timing_complete.json"
            if not path.exists() or not (cell / "complete.json").exists():
                continue
            data = json.loads(path.read_text())
            config = json.loads((cell / "config.json").read_text())
            launch = json.loads((cell / "launch.json").read_text())
            assert launch["script_sha256"] == source_hash
            assert (data["batch"], data["draft"], data["mode"]) == (batch, draft, mode)
            assert config["tensor_parallel_size"] == 1
            assert config["enable_expert_parallel"] is False
            assert config["kv_cache_memory_bytes"] == 7 * 1024**3
            assert config["max_num_seqs"] == batch
            assert config["enforce_eager"] is False
            assert config.get("use_replayssm_spec", False) == (mode == "replayssm")
            if mode != "ar":
                assert config["speculative_config"] == {
                    "method": "mtp",
                    "num_speculative_tokens": draft,
                }
            if mode == "replayssm":
                assert config["replayssm_buffer_len"] == 64
            repeats = data["measurements"]
            assert len(repeats) == 3
            for repeat in repeats:
                assert len(repeat["token_ids"]) == 16
                assert all(len(t) == 128 for t in repeat["token_ids"])
                assert len(repeat["batch_seconds"]) == 16 // batch
                assert repeat["elapsed_s"] > 0
                assert abs(repeat["elapsed_s"] - sum(repeat["batch_seconds"])) < 1e-6
                assert abs(repeat["throughput"] * repeat["elapsed_s"] - 2048) < 1e-6
                if prompt_reference is None:
                    prompt_reference = repeat["prompt_token_ids"]
                assert repeat["prompt_token_ids"] == prompt_reference
            reference = repeats[0]
            outputs[batch, draft, mode] = reference["token_ids"]
            audit.append(
                dict(
                    cell=cell.name,
                    repeat_consistency=[
                        compare(reference["token_ids"], r["token_ids"]) for r in repeats
                    ],
                )
            )
            throughput = [r["throughput"] for r in repeats]
            log = (cell / "run.log").read_text()
            assert "Kernel JIT monitor activated" in log
            capacity_match = re.search(
                r"Maximum concurrency for .*?tokens per request: ([\d.]+)x", log
            )
            capacity = float(capacity_match[1]) if capacity_match else None
            concurrency_samples = re.findall(
                r"Running: (\d+) reqs, Waiting: (\d+) reqs", log
            )
            acc = []
            for repeat in repeats:
                c = repeat["counters"]
                n = c.get("vllm:spec_decode_num_drafts", 0)
                if n:
                    acc.append(1 + c["vllm:spec_decode_num_accepted_tokens"] / n)
            rows.append(
                dict(
                    batch=batch,
                    draft=draft,
                    mode=mode,
                    gpu=data["gpu"],
                    elapsed_s=statistics.median(r["elapsed_s"] for r in repeats),
                    throughput=statistics.median(throughput),
                    throughput_min=min(throughput),
                    throughput_max=max(throughput),
                    throughput_range_pct=(max(throughput) / min(throughput) - 1) * 100,
                    repeated_output_exact=all(
                        r["token_ids"] == reference["token_ids"] for r in repeats
                    ),
                    jit_free_verified=all(
                        "jit_events" in r and not r["jit_events"] for r in repeats
                    ),
                    max_context_concurrency=capacity,
                    capacity_below_batch=capacity < batch if capacity else None,
                    max_logged_running=max(
                        (int(a) for a, _ in concurrency_samples), default=None
                    ),
                    max_logged_waiting=max(
                        (int(b) for _, b in concurrency_samples), default=None
                    ),
                    accepted_length=statistics.mean(acc) if acc else None,
                    **profile_summary(cell, mode),
                )
            )
    for batch in (1, 4, 8, 16):
        for draft in (4, 8, 16, 32):
            for left, right in [
                ((batch, 0, "ar"), (batch, draft, "standard")),
                ((batch, draft, "standard"), (batch, draft, "replayssm")),
            ]:
                if left in outputs and right in outputs:
                    audit.append(
                        dict(
                            left=left,
                            right=right,
                            comparison=compare(outputs[left], outputs[right]),
                        )
                    )
    lookup = {(r["batch"], r["draft"], r["mode"]): r for r in rows}
    controlled_path = root / "controlled_kernels.json"
    controlled = (
        {(r["batch"], r["draft"]): r for r in json.loads(controlled_path.read_text())}
        if controlled_path.exists()
        else {}
    )
    for row in rows:
        ar = lookup.get((row["batch"], 0, "ar"))
        base = lookup.get((row["batch"], row["draft"], "standard"))
        row["speedup_vs_ar"] = row["throughput"] / ar["throughput"] if ar else None
        row["speedup_vs_sd"] = row["throughput"] / base["throughput"] if base else None
        row["profile_gdn_speedup_vs_sd"] = (
            base["gdn_total_step_us"] / row["gdn_total_step_us"]
            if base
            and not base["capacity_below_batch"]
            and not row["capacity_below_batch"]
            else None
        )
        point = controlled.get((row["batch"], row["draft"]))
        row["controlled_gdn_speedup_vs_sd"] = (
            (point["speedup"] if row["mode"] == "replayssm" else 1.0) if point else None
        )
    dump(root / "output_consistency.json", audit)
    if prompt_reference is not None:
        dump(
            root / "prompt_manifest.json",
            {
                "sample_count": len(prompt_reference),
                "token_lengths": [len(p) for p in prompt_reference],
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(prompt_reference, separators=(",", ":")).encode()
                ).hexdigest(),
                "all_formal_prompts_identical": True,
            },
        )
    dump(root / "summary.json", rows)
    report(root, rows, audit)
    if rows:
        with (root / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    complete = len(rows) == 36 and all(
        r["profile_steps"] > 0 and r["jit_free_verified"] for r in rows
    )
    kernel_gates = json.loads((root / "kernel_correctness.json").read_text())
    assert {r["draft"] for r in kernel_gates if r["passed"]} == {4, 8, 16, 32}
    dump(
        root / "coverage.json",
        {
            "completed_cells": len(rows),
            "expected_cells": 36,
            "timing_and_profile_complete": complete,
            "formal_output_tokens": len(rows) * 3 * 16 * 128,
            "kernel_correctness_passed": True,
            "worker_source_sha256": source_hash,
            "repeated_output_exact_cells": sum(
                r["repeated_output_exact"] for r in rows
            ),
            "strict_cross_method_output_equivalence": all(
                r["comparison"]["exact_samples"] == 16
                for r in audit
                if "comparison" in r
            ),
        },
    )
    controlled_path = root / "controlled_kernels.json"
    if (
        complete
        and controlled_path.exists()
        and len(json.loads(controlled_path.read_text())) == 16
    ):
        controlled_rows = json.loads(controlled_path.read_text())
        assert {(r["batch"], r["draft"]) for r in controlled_rows} == {
            (b, d) for b in (1, 4, 8, 16) for d in (4, 8, 16, 32)
        }
        assert all(r["correctness"] for r in controlled_rows)
        assert all(
            r["accepted_tokens"] == 4
            and r["standard_us"] > 0
            and r["replayssm_us"] > 0
            and abs(r["speedup"] * r["replayssm_us"] - r["standard_us"]) < 1e-6
            for r in controlled_rows
        )
        figures(root, rows)
        dump(
            root / "matrix_complete.json",
            {
                "measurement_complete": True,
                "e2e_and_profile_cells": 36,
                "controlled_kernel_cells": 16,
                "formal_output_tokens": 36 * 3 * 16 * 128,
                "jit_free_repeats_per_cell": 3,
                "kernel_correctness_passed": True,
                "strict_cross_method_output_equivalence": all(
                    r["comparison"]["exact_samples"] == 16
                    for r in audit
                    if "comparison" in r
                ),
                "worker_source_sha256": source_hash,
                "e2e_batch_semantics": "submitted requests and max_num_seqs",
                "constant_execution_batch_guaranteed": False,
            },
        )
    print(json.dumps({"cells": len(rows), "complete": complete}))


if __name__ == "__main__":
    main()
