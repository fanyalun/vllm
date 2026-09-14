# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Estimate, select, and report flush thresholds without hiding failed gates."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import regex as re
from qwen36_flush_crossover import save, trace_acceptance


def interval95(values):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(123)
    samples = rng.choice(data, size=(10000, len(data)), replace=True)
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def simulate(curves, track, threshold, width):
    histories = [0] * len(track[0])
    counts = {}
    elapsed = 0.0
    flushes = 0
    for accepted in track:
        step = []
        for i, history in enumerate(histories):
            flush = history + 2 * width > 64 + width
            flush |= threshold is not None and history >= threshold
            route = "flush" if flush else "replay"
            step.append(curves[history][route])
            counts[history] = counts.get(history, 0) + 1
            flushes += int(flush)
            histories[i] = accepted[i] if flush else history + accepted[i]
            assert histories[i] <= 64
        elapsed += 30 * statistics.mean(step)
    return dict(
        estimated_us_per_token=elapsed / sum(map(sum, track)),
        estimated_flush_count=flushes,
        simulated_history_counts=counts,
    )


def estimate(root):
    candidates = {}
    for path in sorted((root / "kernels").glob("distance_*_s0.json")):
        rows = json.loads(path.read_text())
        rotating = [r for r in rows if r["layers"] == 30]
        if len(rotating) != 65:
            continue
        batch, draft = rows[0]["batch"], rows[0]["draft"]
        key = f"b{batch}_d{draft}"
        trace = root / f"{key}_replayssm" / "flush_trace.json"
        if not trace.exists():
            continue
        track = trace_acceptance(trace, batch, draft + 1, 0)
        curves = {
            r["history"]: {k: statistics.median(v) for k, v in r["us"].items()}
            for r in rotating
        }
        estimates = [
            dict(interval=h, **simulate(curves, track, h, draft + 1))
            for h in [None, *range(1, 64 - draft)]
        ]
        ordered = sorted(estimates, key=lambda r: r["estimated_us_per_token"])
        best = ordered[0]["interval"]
        shortlist = {None, best}
        shortlist.update(r["interval"] for r in ordered[:3])
        if best is not None:
            shortlist.update(h for h in [best - 1, best + 1] if 1 <= h < 64 - draft)
        cross = {}
        for layers in (1, 30):
            subset = sorted(
                (r for r in rows if r["layers"] == layers), key=lambda r: r["history"]
            )
            significant = []
            for row in subset:
                ratios = np.asarray(row["us"]["replay"]) / row["us"]["standard"]
                significant.append(interval95(ratios)[0] > 1.02)
            points = [i for i in range(63) if all(significant[i : i + 3])]
            cross[str(layers)] = dict(
                first_stable_slow_history=points[0] if points else None,
                already_slower_at_zero=bool(points and points[0] == 0),
                significant_slow_histories=[
                    i for i, slow in enumerate(significant) if slow
                ],
            )
        candidates[key] = dict(
            batch=batch,
            draft=draft,
            crossovers=cross,
            estimates=estimates,
            candidate_intervals=sorted(shortlist, key=lambda x: 0 if x is None else x),
            estimate_warning=(
                "Mean of homogeneous-batch latency lookups; not measured cycle time."
            ),
        )
    save(root / "estimated_intervals.json", candidates)
    print("Estimated cells:", len(candidates))


def select(root):
    choices = {}
    for batch in (1, 4, 8, 16):
        for draft in (4, 8, 16, 32):
            markers = [
                root / "jobs" / f"policy_b{batch}_d{draft}_s{s}" / "complete.json"
                for s in (0, 1)
            ]
            if not all(path.exists() for path in markers):
                continue
            paths = [
                root / "kernels" / f"policy_b{batch}_d{draft}_s{s}.json" for s in (0, 1)
            ]
            if not all(path.exists() for path in paths):
                continue
            data = [
                {
                    r["interval"]: r
                    for r in json.loads(path.read_text())
                    if r["track"] == "natural"
                }
                for path in paths
            ]
            if None not in data[0] or None not in data[1]:
                continue
            assert data[0].keys() == data[1].keys()
            assert all(
                len(row["us_per_token"]) == 21 for seed in data for row in seed.values()
            )
            medians = {
                h: statistics.median(r["us_per_token"]) for h, r in data[0].items()
            }
            minimum = min(medians.values())
            near = [h for h, value in medians.items() if value <= minimum * 1.02]
            candidate = max(near, key=lambda h: 65 if h is None else h)
            if candidate not in data[1]:
                continue
            reference = data[1][candidate]["paired_default_us_per_token"]
            optimized = data[1][candidate]["us_per_token"]
            speedup = np.asarray(reference) / optimized
            ci = interval95(speedup)
            chosen = candidate if ci[0] > 1.02 else None
            choices[f"b{batch}_d{draft}"] = dict(
                interval=chosen,
                training_candidate=candidate,
                validation_speedup=float(np.median(speedup)),
                validation_ci95=ci,
                measured_intervals=sorted(medians, key=lambda h: 0 if h is None else h),
                decision="validated_improvement"
                if chosen is not None
                else "retain_default",
            )
    save(root / "selected_intervals.json", choices)
    print("Selected cells:", len(choices))


def figures(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    for batch in (1, 4, 8, 16):
        paths = [
            root / "kernels" / f"distance_b{batch}_d{d}_s0.json" for d in (4, 8, 16, 32)
        ]
        if not all(p.exists() for p in paths):
            continue
        data = [
            [r for r in json.loads(p.read_text()) if r["layers"] == 30] for p in paths
        ]
        if not all(len(rows) == 65 for rows in data):
            continue
        name = f"flush_distance_b{batch}"
        folder = root / name
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
        for index, (ax, rows, draft) in enumerate(zip(axes.flat, data, (4, 8, 16, 32))):
            rows = sorted(rows, key=lambda r: r["history"])
            h = [r["history"] for r in rows]
            for key, label, color in [
                ("standard", "Baseline SD", "#E39738"),
                ("replay", "Replay: no flush", "#4C78A8"),
                ("flush", "Replay: force flush", "#59A14F"),
            ]:
                median = [statistics.median(r["us"][key]) for r in rows]
                low = [min(r["us"][key]) for r in rows]
                high = [max(r["us"][key]) for r in rows]
                ax.plot(h, median, label=label, color=color, linewidth=2)
                ax.fill_between(h, low, high, color=color, alpha=0.12)
            ax.axvline(64 - draft, color="0.5", linestyle=":", linewidth=1.5)
            ax.set(
                xlabel=f"Checkpoint distance h\n({chr(97 + index)}) Draft = {draft}",
                ylabel="GDN latency (µs / layer)",
                xlim=(0, 64),
            )
            ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# GDN history distance, B={batch}\n\n"
            "Qwen3.6 dimensions on one A100; BF16 input, FP32 state, FP16 d/k ring. "
            "Thirty independent layer buffers rotate. Curves show medians of 21 "
            "repetitions; shaded bands show min/max, not confidence intervals. "
            "State restoration and GPU timing prelude are outside timed regions. "
            "The dotted line marks the default flush trigger. No-flush measurements "
            "beyond it are controlled counterfactuals. Ordinary attention and MoE "
            "are excluded. These measurements are not DRAM byte counters.\n\n"
            "Source: kernels/distance_b*_s0.json. Regenerate with "
            "`analyze_qwen36_flush_study.py --mode figures --output <root>`.\n"
        )
    scalar = []
    for path in sorted((root / "kernels").glob("distance_*json")):
        for row in json.loads(path.read_text()):
            scalar.append(
                {k: row[k] for k in ("batch", "draft", "history", "layers", "seed")}
                | {f"{k}_us": statistics.median(v) for k, v in row["us"].items()}
            )
    if scalar:
        with (root / "distance_summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scalar[0]))
            writer.writeheader()
            writer.writerows(scalar)
    for batch in (1, 4, 8, 16):
        paths = [
            root / "kernels" / f"policy_b{batch}_d{d}_s1.json" for d in (4, 8, 16, 32)
        ]
        if not all(p.exists() for p in paths):
            continue
        name = f"flush_policy_b{batch}"
        folder = root / name
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
        for index, (ax, path, draft) in enumerate(
            zip(axes.flat, paths, (4, 8, 16, 32))
        ):
            rows = json.loads(path.read_text())
            for track, label, color in [
                ("natural", "Observed acceptance", "#4C78A8"),
                ("fixed_1", "Accept 1", "#E39738"),
                ("fixed_4", "Accept 4", "#59A14F"),
                (f"fixed_{draft + 1}", "Accept all", "#B279A2"),
            ]:
                points = sorted(
                    (r for r in rows if r["track"] == track),
                    key=lambda r: 64 - draft
                    if r["interval"] is None
                    else r["interval"],
                )
                x = [
                    64 - draft if r["interval"] is None else r["interval"]
                    for r in points
                ]
                y = [statistics.median(r["us_per_token"]) for r in points]
                ax.plot(x, y, ".-", label=label, color=color, linewidth=1.8)
            ax.axvline(64 - draft, color="0.5", linestyle=":", linewidth=1.5)
            ax.set(
                xlabel=f"Flush interval\n({chr(97 + index)}) Draft = {draft}",
                ylabel="GDN µs / committed token",
                xlim=(0, 64),
            )
            ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# Flush policy, B={batch}\n\n"
            "Measured seed-1 cycle medians, including verify, flush and three shared "
            "cursor updates over 30 independent GDN layers. Every cycle contains "
            "at least two default flushes per request. Each point has 21 repetitions. "
            "Default policy is plotted at its equivalent trigger, 64-D. "
            "Lines connect measured thresholds; they do not imply unmeasured "
            "thresholds were timed. Observed acceptance is a controlled replay of "
            "the model's acceptance counts, not a model activation replay.\n"
        )
    comparison_path = root / "e2e_comparisons.json"
    if comparison_path.exists():
        data = json.loads(comparison_path.read_text())
        name = "flush_e2e"
        folder = root / name
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
        for index, (ax, batch) in enumerate(zip(axes.flat, (1, 4, 8, 16))):
            rows = sorted(
                (r for r in data if r["batch"] == batch), key=lambda r: r["draft"]
            )
            x = np.arange(4)
            for offset, key, label, color in [
                (-0.26, "sd_tok_s", "Baseline SD", "#E39738"),
                (0, "default_tok_s", "Replay: default", "#4C78A8"),
                (0.26, "tuned_tok_s", "Replay: selected", "#59A14F"),
            ]:
                bars = ax.bar(
                    x + offset,
                    [r[key] / r["ar_tok_s"] for r in rows],
                    width=0.25,
                    label=label,
                    color=color,
                )
                if key == "sd_tok_s":
                    for bar, row in zip(bars, rows):
                        if row["sd_capacity_limited"]:
                            bar.set_hatch("///")
            ax.axhline(1, color="0.4", linestyle="--")
            ax.set(
                xticks=x,
                xticklabels=[4, 8, 16, 32],
                xlabel=f"Draft length\n({chr(97 + index)}) Batch = {batch}",
                ylabel="E2E throughput / AR",
            )
            ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            "# Flush policy end-to-end validation\n\n"
            "Single A100, Qwen3.6, 16 GSM8K prompts, exactly 128 generated tokens "
            "each, three JIT-free repeats per configuration. Bars use median "
            "throughput normalized by the same batch's AR median. Hatching marks "
            "baseline cache capacity below the requested batch. Selected thresholds "
            "were frozen before the E2E runs. A retained default is explicitly "
            "recorded as null in selected_intervals.json; run-to-run differences "
            "at those points are not optimization gains. Output parity and raw "
            "throughput are reported in e2e_comparisons.json.\n"
        )


def report(root):
    selected = json.loads((root / "selected_intervals.json").read_text())
    assert len(selected) == 16, "Threshold validation is incomplete"
    rows = []
    tokens_by_cell = {}
    prompts = None
    for batch in (1, 4, 8, 16):
        cells = [(0, "ar")]
        cells += [
            (d, m) for d in (4, 8, 16, 32) for m in ("standard", "control", "tuned")
        ]
        for draft, mode in cells:
            path = root / f"b{batch}_d{draft}_{mode}"
            assert (path / "complete.json").exists(), path
            config = json.loads((path / "config.json").read_text())
            assert config["tensor_parallel_size"] == 1
            assert config["enable_expert_parallel"] is False
            assert config["kv_cache_memory_bytes"] == 7 * 1024**3
            repetitions = [
                json.loads((path / f"repeat_{i}.json").read_text()) for i in range(3)
            ]
            for rep in repetitions:
                assert not rep["jit_events"]
                assert len(rep["token_ids"]) == 16
                assert all(len(ids) == 128 for ids in rep["token_ids"])
                assert len(rep["batch_seconds"]) == 16 // batch
                if prompts is None:
                    prompts = rep["prompt_token_ids"]
                assert prompts == rep["prompt_token_ids"]
                assert abs(rep["throughput"] * sum(rep["batch_seconds"]) - 2048) < 1e-6
            log = (path / "run.log").read_text()
            cap = re.search(
                r"Maximum concurrency for .*?tokens per request: ([\d.]+)x", log
            )
            waiting = re.findall(r"Running: (\d+) reqs, Waiting: (\d+) reqs", log)
            rates = [rep["throughput"] for rep in repetitions]
            row = dict(
                batch=batch,
                draft=draft,
                mode=mode,
                throughput=statistics.median(rates),
                min_throughput=min(rates),
                max_throughput=max(rates),
                interval=config.get("replayssm_spec_flush_interval"),
                capacity_below_batch=float(cap[1]) < batch if cap else None,
                max_waiting=max([int(x[1]) for x in waiting], default=0),
                repeated_tokens_exact=all(
                    rep["token_ids"] == repetitions[0]["token_ids"]
                    for rep in repetitions
                ),
            )
            if mode == "tuned":
                assert row["interval"] == selected[f"b{batch}_d{draft}"]["interval"]
            tokens_by_cell[(batch, draft, mode)] = repetitions[0]["token_ids"]
            rows.append(row)
    assert len(rows) == 52
    comparisons = []
    lookup = {(r["batch"], r["draft"], r["mode"]): r for r in rows}
    for batch in (1, 4, 8, 16):
        for draft in (4, 8, 16, 32):
            control = lookup[batch, draft, "control"]
            tuned = lookup[batch, draft, "tuned"]
            standard = lookup[batch, draft, "standard"]
            ar = lookup[batch, 0, "ar"]
            exact = sum(
                a == b
                for a, b in zip(
                    tokens_by_cell[batch, draft, "control"],
                    tokens_by_cell[batch, draft, "tuned"],
                )
            )
            comparisons.append(
                dict(
                    batch=batch,
                    draft=draft,
                    interval=tuned["interval"],
                    ar_tok_s=ar["throughput"],
                    sd_tok_s=standard["throughput"],
                    default_tok_s=control["throughput"],
                    tuned_tok_s=tuned["throughput"],
                    tuned_vs_default=tuned["throughput"] / control["throughput"],
                    tuned_vs_sd=tuned["throughput"] / standard["throughput"],
                    tuned_vs_ar=tuned["throughput"] / ar["throughput"],
                    exact_requests_vs_default=exact,
                    sd_capacity_limited=standard["capacity_below_batch"],
                )
            )
    save(root / "e2e_summary.json", rows)
    save(root / "e2e_comparisons.json", comparisons)
    with (root / "e2e_comparisons.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    save(
        root / "measurement_complete.json",
        dict(
            e2e_cells=52,
            formal_output_tokens=52 * 3 * 16 * 128,
            selected_cells=16,
            cross_method_exact=all(
                r["exact_requests_vs_default"] == 16 for r in comparisons
            ),
            figures_validated=False,
        ),
    )


def stage1(root):
    summaries = []
    all_rows = []
    for batch in (1, 4, 8, 16):
        for draft in (4, 8, 16, 32):
            path = root / "kernels" / f"distance_b{batch}_d{draft}_s0.json"
            rows = json.loads(path.read_text())
            assert len(rows) == 130
            assert {(r["layers"], r["history"]) for r in rows} == {
                (layers, h) for layers in (1, 30) for h in range(65)
            }
            for row in rows:
                assert row["correctness"]
                assert set(row["us"]) == {"standard", "replay", "flush"}
                assert all(len(v) == 21 and min(v) > 0 for v in row["us"].values())
            all_rows.extend(rows)
            for layers in (1, 30):
                curve = sorted(
                    (r for r in rows if r["layers"] == layers),
                    key=lambda r: r["history"],
                )
                values = [
                    {k: statistics.median(v) for k, v in r["us"].items()} for r in curve
                ]
                median_slow = [v["replay"] > v["standard"] for v in values]
                confidence = [
                    interval95(np.asarray(r["us"]["replay"]) / r["us"]["standard"])
                    for r in curve
                ]
                significant = [ci[0] > 1.02 for ci in confidence]
                median_cross = next(
                    (h for h in range(63) if all(median_slow[h : h + 3])), None
                )
                stable_cross = next(
                    (h for h in range(63) if all(significant[h : h + 3])), None
                )
                summaries.append(
                    dict(
                        batch=batch,
                        draft=draft,
                        layers=layers,
                        sd_h0_us=values[0]["standard"],
                        replay_h0_us=values[0]["replay"],
                        replay_h64_us=values[64]["replay"],
                        flush_h0_us=values[0]["flush"],
                        flush_h64_us=values[64]["flush"],
                        first_three_point_median_slow=median_cross,
                        first_three_point_significant_slow=stable_cross,
                        replay_slower_all_h=all(median_slow),
                        replay_faster_all_h=not any(median_slow),
                        ci95_replay_over_sd=confidence,
                    )
                )
    save(root / "stage1_summary.json", summaries)
    with (root / "stage1_summary.csv").open("w") as stream:
        keys = [k for k in summaries[0] if k != "ci95_replay_over_sd"]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Qwen3.6 / A100：历史距离扫描结果",
        "",
        "第一阶段已完成；用户要求暂停后续阈值选择和端到端实验。",
        "",
        "共 16 个 B/D 组合 × 65 个历史距离 × 2 种工作集 = 2,080 个点。"
        "每点比较 baseline SD、ReplaySSM 非 flush、ReplaySSM 强制 flush，"
        "各 21 轮，共 131,040 个正式计时值。所有点的两条 ReplaySSM 路径"
        "均通过对 baseline 输出的数值检查，rtol=0.04、atol=0.01。",
        "",
        "B=1/4/8/16；D=4/8/16/32，实际验证宽度 T=D+1。"
        "Qwen GDN 维度为 HQ=16、HV=32、K=V=128；输入 BF16，checkpoint FP32，"
        "d/k ring FP16、g ring FP32。物理 ring=128，历史计算 tile=64。"
        "每个点使用相互对应的 checkpoint、历史和当前位置 state。",
        "",
        "主图使用 30 个独立 GDN 层 buffer 轮换，计时除以 30；"
        "同时保存单层工作集数据。它们都是合成输入的 GDN 核心 kernel 测量，"
        "不包含投影、Conv、普通 Attention、MoE 或整模型调度。"
        "重置、编译、预热和用于排除 CPU 提交间隙的 GPU prelude 均不计时。",
        "",
        "## 30 层轮换结果",
        "",
        "下表单位为 µs/层；h=0 与 h=64 均为 21 轮中位数。",
        "",
        "| B | D | SD h=0 | 非 flush h=0 | 非 flush h=64 | "
        "flush h=0 | flush h=64 | 中位数关系 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        if row["layers"] != 30:
            continue
        relation = (
            "全区间慢于 SD"
            if row["replay_slower_all_h"]
            else (
                "全区间快于 SD"
                if row["replay_faster_all_h"]
                else f"约 h={row['first_three_point_median_slow']} 起连续三点慢于 SD"
            )
        )
        lines.append(
            f"| {row['batch']} | {row['draft']} | {row['sd_h0_us']:.2f} | "
            f"{row['replay_h0_us']:.2f} | {row['replay_h64_us']:.2f} | "
            f"{row['flush_h0_us']:.2f} | {row['flush_h64_us']:.2f} | {relation} |"
        )
    lines += [
        "",
        "## 图与交点解释",
        "",
        "30 层轮换下，10 组非 flush 从 h=0 起即慢于 SD，5 组全区间快于 SD。"
        "仅 B=8、D=4 在 h≈47 出现连续三点的中位数交叉；"
        "它未达到下述稳定变慢判据。没有观察到通用的 h=8 拐点。"
        "单层结果有所不同，例如 B=4、D=8 的稳定变慢点为 h=45；"
        "不能将一种工作集下的交点直接推广到另一种工作集。",
        "",
        "D=32 时，强制 flush 在 h=0 就比非 flush 更快。"
        "历史 tile 固定为 64，h 的变化不会缩小矩阵计算的 tile 维度。"
        "这些结果提示编译分支和固定计算成本也有影响，"
        "不能把大 D 的性能问题全部归因于 checkpoint 距离。",
        "",
        "曲线阴影为 min/max，不是置信区间。JSON 另给出配对 bootstrap "
        "95% 区间：只有 Replay/SD 的区间下界 >1.02 且连续三点满足，"
        "才记为稳定变慢点。h=0 已经较慢的配置不应解释为存在正距离拐点。",
        "",
        "默认策略在 h≥64-D 时触发 flush；主图虚线标记该位置。"
        "线右侧的非 flush 曲线是受控反事实，不表示运行时关闭了溢出保护。",
        "",
        "直接读取 state 的 baseline 参考线不计此前物化该 state 的成本。"
        "当前步强制 flush 的延迟也不能单独决定长期最优刷新周期。"
        "因此本阶段不输出最优 interval 或新的端到端加速结论。",
        "",
    ]
    for batch in (1, 4, 8, 16):
        name = f"flush_distance_b{batch}"
        lines += [
            f"### B={batch}",
            "",
            f"![B={batch}]({name}/{name}.png)",
            "",
            f"[PDF]({name}/{name}.pdf)",
            "",
        ]
    lines += [
        "## 资源与验证记录",
        "",
        "Nsight Compute 直接探测返回 ERR_NVGPUCTRPERM；未采集硬件 DRAM/L2 "
        "计数器，也未修改驱动权限。resources/ 保存 CUDA 函数属性：寄存器数、"
        "每线程 local-memory 分配及 shared memory。它们不是实际 IO 字节数。",
        "",
        "21 项随机回滚/提前 flush 检查、5 项配置检查通过；"
        "新增请求重用 reset 检查另行通过。后续模型校准数据和阈值估算"
        "不纳入本阶段的结果与完成判定。",
        "",
        "kernel 原始数据见 kernels/distance_*.json；"
        "汇总见 stage1_summary.csv 与 distance_summary.csv；"
        "测量命令及冻结源码见 jobs/distance_*/launch.json 与 source/。",
        "",
        "重新汇总及绘图：",
        "",
        "```bash",
        ".venv/bin/python benchmarks/replayssm/analyze_qwen36_flush_study.py \\",
        f"  --output {root} --mode stage1",
        ".venv/bin/python benchmarks/replayssm/analyze_qwen36_flush_study.py \\",
        f"  --output {root} --mode figures",
        "```",
        "",
        "代码与报告使用 AI 辅助生成。本次未创建上游 PR。",
        "",
    ]
    (root / "readme.md").write_text("\n".join(lines))
    save(
        root / "stage1_complete.json",
        dict(
            measurement_complete=True,
            batch_draft_cells=16,
            history_points=65,
            working_sets=[1, 30],
            total_points=len(all_rows),
            repetitions=21,
            timed_values=len(all_rows) * 3 * 21,
            output_checks_passed=True,
            later_stages_stopped_by_user=True,
            figures_validated=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["estimate", "select", "figures", "report", "stage1"],
        required=True,
    )
    args = parser.parse_args()
    {
        "estimate": estimate,
        "select": select,
        "figures": figures,
        "report": report,
        "stage1": stage1,
    }[args.mode](Path(args.output))


if __name__ == "__main__":
    main()
