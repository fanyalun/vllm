# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and plot Gemma block-level expert-pool measurements."""

import argparse
import csv
import json
from pathlib import Path

from run_token_importance import digest, write_json


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--acceptance-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    contract = json.loads((root / "contract.json").read_text())
    if args.acceptance_only:
        assert contract["acceptance_audit"]
    assert digest(root / "dataset.jsonl") == contract["dataset_sha256"]
    assert (
        digest(Path(contract["model_path"]) / "config.json")
        == contract["model_config_sha256"]
    )
    assert (
        digest(Path(contract["assistant_path"]) / "config.json")
        == contract["assistant_config_sha256"]
    )
    for name, fingerprint in contract["source_sha256"].items():
        assert digest(root / name) == fingerprint
    samples = [json.loads(s) for s in (root / "dataset.jsonl").read_text().splitlines()]
    results, rows, layers, timing, parity = {}, [], [], [], []
    for cell in contract["cells"]:
        directory = root / cell["name"]
        assert (directory / "CELL_COMPLETE").exists(), directory
        result = json.loads((directory / "result.json").read_text())
        assert all(result[k] == v for k, v in cell.items())
        assert result["model_path"] == contract["model_path"]
        assert result["source_sha256"] == contract["source_sha256"]
        assert result["dataset_sha256"] == contract["dataset_sha256"]
        assert len(result["outputs"]) == len(samples) == 4
        for output, sample in zip(result["outputs"], samples, strict=True):
            assert output["prompt_sha256"] == sample["prompt_sha256"]
            assert len(output["token_ids"]) == 128
        log = sorted(directory.glob("run_*.log"))[-1].read_text()
        parts = log.split("WARMUP_COMPLETE", 1)
        assert len(parts) == 2
        jit = [
            s for s in parts[1].splitlines() if "JIT compilation during inference" in s
        ]
        timing.append({"method": cell["name"], "post_warmup_jit_warnings": jit})
        results[cell["name"]] = result
    ar = (results["ar_start"]["e2e_seconds"] + results["ar_end"]["e2e_seconds"]) / 2
    for name, result in results.items():
        seconds = sum(o["e2e_seconds"] for o in result["outputs"])
        assert abs(seconds - result["e2e_seconds"]) < 1e-8
        accepted = steps = emitted = drafted = 0
        for output in result["outputs"]:
            if not result["h"]:
                continue
            metrics = output["spec_decode_metrics"]
            a, n = metrics["per_step_accepted"], metrics["per_step_drafted"]
            assert len(a) == len(n) == metrics["num_spec_steps"]
            assert all(0 <= x <= y <= 20 for x, y in zip(a, n, strict=True))
            assert sum(a) == metrics["num_accepted_draft_tokens"]
            assert sum(n) == metrics["num_draft_tokens"]
            assert metrics["acceptance_histogram"] == [a.count(i) for i in range(21)]
            remaining = 128 - (1 + sum(x + 1 for x in a[:-1]))
            assert 1 <= remaining <= a[-1] + 1
            emitted += sum(a[:-1]) + min(a[-1], remaining)
            accepted += sum(a)
            drafted += sum(n)
            steps += len(a)
        inner = [
            sum(w["inner_counts"][i] for w in result["measurement"]) for i in range(3)
        ]
        budgets = [0] * 6
        for worker in result["measurement"]:
            for layer, count in worker["layers"].items():
                if count[-1] == 0:
                    continue
                assert count[-1] == inner[2]
                assert count[4] == count[-1] * 5
                if result["pool"] != "none":
                    assert 0.6 * count[0] <= count[1] < 0.6 * count[0] + count[-1]
                else:
                    assert count[0] == count[1] == 0
                    assert count[2] == result["h"] * count[4]
                    assert count[3] == 0
                budgets = [a + b for a, b in zip(budgets, count, strict=True)]
                layers.append(
                    {
                        "method": name,
                        "layer": layer,
                        "union_sum": count[0],
                        "selected_sum": count[1],
                        "retained_edges": count[2],
                        "zero_rows": count[3],
                        "token_rows": count[4],
                        "calls": count[5],
                    }
                )
        mean_h = budgets[2] / budgets[4] if budgets[4] else result["h"]
        if args.acceptance_only and result["h"]:
            effective = result["effective_speculative_config"]
            assert effective["hierarchical_stop_policy"] == "none"
            assert effective["moe_skip_weight_mode"] == "renormalize"
            assert effective["preverify_gdn_mode"] == "none"
            assert effective["inner_num_speculative_tokens"] == 4
            assert effective["inner_num_rounds"] == 4
            assert len(result["measurement"]) == 1
            assert (
                len([c for c in result["measurement"][0]["layers"].values() if c[-1]])
                == 30
            )
            assert inner[0] == 4 * inner[2]
            assert inner[2] % 4 == 0
            assert 0 <= inner[1] <= inner[0]
            assert budgets[4] == 30 * 5 * inner[2]
        row = {
            "method": name,
            "ms_per_output_token": seconds * 1000 / 512,
            "ms_per_accepted_token": seconds * 1000 / emitted if emitted else None,
            "outer_accepted_per_step": accepted / steps if steps else None,
            "inner_accepted_per_round": inner[1] / inner[2] if inner[2] else None,
            "emitted_accepted_tokens": emitted,
            "outer_drafted": drafted,
            "outer_steps": steps,
            "inner_rounds": inner[2],
            "mean_h": mean_h,
            "mean_union": budgets[0] / budgets[5] if budgets[5] else None,
            "mean_selected": budgets[1] / budgets[5] if budgets[5] else None,
            "zero_expert_row_fraction": budgets[3] / budgets[4] if budgets[4] else 0,
            "speedup_vs_ar_async": ar / seconds,
        }
        if args.acceptance_only:
            row = {
                "method": name,
                "accepted": accepted,
                "proposed": drafted,
                "verify_steps": steps,
                "acceptance_rate": accepted / drafted if drafted else None,
                "mean_acceptance_length": 1 + accepted / steps if steps else None,
                "inner_accepted": inner[1],
                "inner_proposed": inner[0],
                "inner_rounds": inner[2],
                "inner_acceptance_rate": inner[1] / inner[0] if inner[0] else None,
                "inner_mean_acceptance_length": 1 + inner[1] / inner[2]
                if inner[2]
                else None,
                "retained_expert_edges": budgets[2],
                "native_expert_edges": 8 * budgets[4],
                "token_layer_rows": budgets[4],
                "mean_kept_experts": mean_h if result["h"] else None,
                "mean_skipped_experts": 8 - mean_h if result["h"] else None,
                "expert_skip_fraction": 1 - mean_h / 8 if result["h"] else None,
                "zero_expert_rows": budgets[3],
                "zero_expert_row_fraction": budgets[3] / budgets[4]
                if budgets[4]
                else None,
            }
        rows.append(row)
        for reference in ("ar_start", "h8"):
            matches = sum(
                a["token_ids"] == b["token_ids"]
                for a, b in zip(
                    result["outputs"], results[reference]["outputs"], strict=True
                )
            )
            parity.append(
                {
                    "method": name,
                    "reference": reference,
                    "exact_matches": matches,
                    "requests": 4,
                }
            )
    write_csv(root / "summary.csv", rows)
    write_csv(root / "layer_budgets.csv", layers)
    write_csv(root / "output_parity.csv", parity)
    write_json(
        root / "audit.json",
        {
            "measurement_contract": "passed",
            "cells": len(rows),
            "timing": timing,
            "parity": parity,
            "strict_output_parity": "passed"
            if all(p["exact_matches"] == p["requests"] for p in parity)
            else "failed",
            "jit_gate": "failed"
            if any(t["post_warmup_jit_warnings"] for t in timing)
            else "passed",
        },
    )
    if args.acceptance_only:
        acceptance_report(root, rows, parity)
    else:
        plot(root, rows)
        report(root, rows, parity)
    (root / "ANALYSIS_COMPLETE").write_text(
        "7 x 4 x 128 audited; output parity reported separately\n"
    )
    print(json.dumps(rows, indent=2))


def acceptance_report(root, rows, parity):
    lines = [
        "# m10：token importance 专家池接受与跳过指标",
        "",
        "Gemma4，TP1/B1，greedy，4 个原始 prompt × 128 输出 token；"
        "MTP D4 × 固定 4 轮（stop_policy=none），外层容量 20，"
        "weight_mode=renormalize。attention60/routing60 与固定 h4/h6/h8 "
        "均在当前代码上重新测量；每配置 4 个完整请求 warmup 不计入结果。",
        "",
        "主表与 m06 使用同一接受指标定义：接受率 = accepted/proposed；"
        "平均接受长度 = 1 + accepted/verify_steps，包含 bonus token。"
        "这里主表是外层 Pre-Verify→Target；m06 是独立 Draft→Target，"
        "两者协议不同，不直接比较数值高低。",
        "",
        "专家跳过率 = 1 − retained_edges/(8 × token_layer_rows)，"
        "分母为原生 top8；不是从全模型专家总数计算。"
        "attention60 的 60% 是候选专家 union 的保留预算，"
        "并不意味着每个 token 固定跳过 40%。所有五个 Pre-Verify 位置"
        "（含 anchor 和末尾 draft）及 30 个 MoE 层均计入。",
        "",
        "| 方法 | 接受/提出 | 接受率 | 平均接受长度 | 平均保留专家 | "
        "平均跳过专家 | 专家跳过率 | 零专家率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if not r["proposed"]:
            continue
        lines.append(
            f"| {r['method']} | {r['accepted']}/{r['proposed']} | "
            f"{r['acceptance_rate']:.2%} | {r['mean_acceptance_length']:.3f} | "
            f"{r['mean_kept_experts']:.3f} | {r['mean_skipped_experts']:.3f} | "
            f"{r['expert_skip_fraction']:.2%} | {r['zero_expert_row_fraction']:.2%} |"
        )
    lines += [
        "",
        "| 方法 | 内层接受/提出 | 内层接受率 | 内层平均接受长度 | 内层轮数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["inner_proposed"]:
            lines.append(
                f"| {r['method']} | {r['inner_accepted']}/{r['inner_proposed']} | "
                f"{r['inner_acceptance_rate']:.2%} | "
                f"{r['inner_mean_acceptance_length']:.3f} | {r['inner_rounds']} |"
            )
    lines += [
        "",
        "内层 MTP→Pre-Verify 的计数和专家预算包含测量请求发起的所有 proposal，"
        "包括最后未被 Target 消费的 proposal；外层接受计数来自实际 Target verify，"
        "按 m06 口径保留最后一步截断前的验证接受计数。",
        "",
        "attention60/routing60 保持原来的候选相关专家池算法与权重归一化；"
        "Target 路由不变。输出一致性单独列出，接受率不代表质量或 lossless 保证。",
        "",
        "| 方法 | 对照 | 完整输出相同请求数 |",
        "| --- | --- | ---: |",
    ]
    lines += [
        f"| {p['method']} | {p['reference']} | {p['exact_matches']}/{p['requests']} |"
        for p in parity
    ]
    lines += [
        "",
        "[完整指标](summary.csv) · [逐层预算整数计数](layer_budgets.csv) · "
        "[审计](audit.json) · [实验契约](contract.json)",
        "",
        "本报告不展示耗时、吞吐或加速比。结果仅覆盖原实验的 4 个样本。",
    ]
    (root / "readme.md").write_text("\n".join(lines) + "\n")


def report(root, rows, parity):
    lines = [
        "# Gemma token-importance expert-pool pilot",
        "",
        "Gemma4, TP=1, B=1, greedy, four prompts x 128 output tokens. "
        "MTP D=4, four inner rounds, outer capacity=20. Pre-Verify and "
        "Target share the same model instance and weights. "
        "Each configuration has four excluded full-request warmups. "
        "All configurations are measured afresh on the same GPU.",
        "",
        "Attention60 scores each expert using the sum of normalized "
        "native-top8 gate probability times mean-head attention from "
        "the last draft position to earlier draft positions. Attention "
        "uses the actual paged KV cache and full-context softmax, including "
        "GQA, shared KV, sliding masks, scaling and soft-cap. The anchor "
        "and last draft have zero scoring weight. Routing60 uses unit "
        "weights on the same scoring positions. The candidate expert union "
        "includes all four drafts. Retain ceil(0.6 * union size) per layer; "
        "break score ties by ascending expert ID. Apply the selected pool "
        "to all five Pre-Verify rows, including the anchor and last draft. "
        "Renormalize retained gate weights and preserve expert scales. "
        "Rows without surviving experts have zero routed contribution; "
        "there is no budget-expanding fallback.",
        "",
        "This pool is specific to the current Pre-Verify candidate block. "
        "Its dependence on the last position makes it a candidate-dependent "
        "approximate filter, not a causal AR model. Target routing is untouched.",
        "",
        "ms/output and ms/accepted both use complete llm.generate wall time, "
        "including scoring, ranking, MTP, Pre-Verify, final Target, API "
        "and metric collection. ms/accepted divides by final emitted "
        "accepted draft tokens after last-step clipping. Inner metrics "
        "and budgets include every proposal initiated by measured requests, "
        "including the final unused proposal. No kernel-only timings "
        "are presented as end-to-end speedups.",
        "",
        "| Method | Mean h | Inner accepted/round | Outer accepted/step | "
        "ms/output | ms/accepted | AR speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:

        def show(key, row=r):
            return "-" if row[key] is None else f"{row[key]:.3f}"

        lines.append(
            f"| {r['method']} | {show('mean_h')} | "
            f"{show('inner_accepted_per_round')} | {show('outer_accepted_per_step')} | "
            f"{show('ms_per_output_token')} | {show('ms_per_accepted_token')} | "
            f"{show('speedup_vs_ar_async')}x |"
        )
    ar_repeat = next(
        p for p in parity if p["method"] == "ar_end" and p["reference"] == "ar_start"
    )
    speculative_parity = ", ".join(
        f"{p['method']}={p['exact_matches']}/4"
        for p in parity
        if p["reference"] == "ar_start" and not p["method"].startswith("ar_")
    )
    lines += [
        "",
        f"AR repeat exact-output agreement: {ar_repeat['exact_matches']}/4. "
        f"Speculative exact matches against ar_start: {speculative_parity}. "
        "The speedup column is an observed wall-time ratio, "
        "not a validated same-output acceleration. Different generated "
        "continuations can also change acceptance and expert routing.",
        "",
        "This is a small exploratory sample without confidence intervals. "
        "The existing Gemma/hierarchical path has unresolved exact-output "
        "parity limitations. See output_parity.csv; no lossless or production "
        "readiness claim is made. The new scoring and expert masking primitives "
        "are checked against GPU references in gpu_reference.log. "
        "Aligned assignments keep original token slots and mask expert blocks "
        "after assignment, avoiding the earlier invalid-slot limitation.",
        "",
        "Reproduce with `.venv/bin/python "
        "benchmarks/hierarchical/run_token_importance.py "
        "--run-dir <fresh_directory> --gpu 1`, followed by "
        "`.venv/bin/python benchmarks/hierarchical/analyze_token_importance.py "
        "<fresh_directory>`. Source snapshots and hashes are in contract.json. "
        "AI assistance was used.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")


def plot(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 13,
            "axes.labelsize": 14,
            "pdf.fonttype": 42,
        }
    )
    data = {r["method"]: r for r in rows}
    methods = ["h4", "h6", "h8", "routing60", "attention60"]
    labels = ["h=4", "h=6", "h=8", "Routing\n60%", "Attention\n60%"]
    groups = {
        "token_importance_time": [
            ("ms_per_output_token", "ms / output token"),
            ("ms_per_accepted_token", "ms / accepted token"),
        ],
        "token_importance_acceptance": [
            ("inner_accepted_per_round", "MTP accepted / round"),
            ("outer_accepted_per_step", "Target accepted / step"),
        ],
        "token_importance_budget": [
            ("mean_h", "Mean experts / token / layer"),
            ("zero_expert_row_fraction", "Fraction with no experts"),
        ],
    }
    for name, panels in groups.items():
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), layout="constrained")
        for index, (ax, (metric, label)) in enumerate(zip(axes, panels, strict=True)):
            ax.bar(
                labels,
                [data[m][metric] for m in methods],
                color=["#4C78A8"] * 3 + ["#59A14F", "#F2B447"],
                width=0.7,
            )
            ax.set_ylabel(label)
            ax.set_xlabel(f"({chr(97 + index)}) Gemma4")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
            if metric == "ms_per_output_token":
                ar = (data["ar_start"][metric] + data["ar_end"][metric]) / 2
                ax.axhline(ar, ls="--", color="#E45756", label="AR (Async)")
        if name == "token_importance_time":
            fig.legend(
                *axes[0].get_legend_handles_labels(),
                loc="outside upper right",
                frameon=False,
            )
        folder = root / "figures" / name
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            fig.savefig(
                folder / f"{name}.{suffix}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.03,
            )
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# {name}\n\nSource: ../../summary.csv; "
            "protocol and limitations: ../../README.md. "
            "Gemma4, A100 80GB PCIe, TP=1/B=1, greedy, 4x128; MTP D=4, "
            "four rounds, Target capacity=20. Aggregate wall-time ratios, "
            "no error bars. Full scoring/selection overhead is included. "
            "AR is the mean of two fresh async controls. Actual retained edge "
            "counts determine mean h; the expert-pool fraction is rounded up.\n\n"
            "Reproduce with `.venv/bin/python "
            "benchmarks/hierarchical/analyze_token_importance.py "
            f"{root}`.\n"
        )


if __name__ == "__main__":
    main()
