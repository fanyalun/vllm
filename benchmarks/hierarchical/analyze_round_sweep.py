# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit h4 D4 round limits 4/6/8 and compare all three stopping policies."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

POLICIES = ("low_error", "balanced", "aggressive")
BATCHES = (1, 4, 8, 16)


def acceptance_stats(metrics, capacity):
    pairs = []
    for metric in metrics:
        accepted = metric["per_step_accepted"]
        drafted = metric["per_step_drafted"]
        assert len(accepted) == len(drafted) == metric["num_spec_steps"]
        assert sum(accepted) == metric["num_accepted_draft_tokens"]
        assert sum(drafted) == metric["num_draft_tokens"]
        assert all(0 <= a <= d <= capacity for a, d in zip(accepted, drafted))
        pairs.extend((a, d) for a, d in zip(accepted, drafted) if d > 0)
    assert pairs
    accepted = sum(a for a, _ in pairs)
    drafted = sum(d for _, d in pairs)
    at_cap = sum(d == capacity for _, d in pairs)
    full_at_cap = sum(a == d == capacity for a, d in pairs)
    return {
        "outer_steps": len(pairs),
        "outer_proposed": drafted,
        "outer_accepted": accepted,
        "outer_acceptance_rate": accepted / drafted,
        "mean_proposed": drafted / len(pairs),
        "mean_accepted": accepted / len(pairs),
        "all_accepted_steps": sum(a == d for a, d in pairs),
        "all_accepted_probability": sum(a == d for a, d in pairs) / len(pairs),
        "capacity_steps": at_cap,
        "all_accepted_at_capacity": full_at_cap,
        "all_accepted_given_capacity": full_at_cap / at_cap if at_cap else None,
        "tail_proposed_after20": sum(max(0, d - 20) for _, d in pairs),
        "tail_accepted_after20": sum(max(0, a - 20) for a, _ in pairs),
    }


def audit(root, rounds, reference=None):
    contract = json.loads((root / "contract.json").read_text())
    assert contract["samples"] == 16 and contract["output_length"] == 512
    modes = ("ar", "mtp", *POLICIES) if rounds == 4 else POLICIES
    expected = {(b, p) for b in BATCHES for p in modes}
    assert len(contract["cells"]) == len(expected)
    assert {tuple(c) for c in contract["cells"]} == expected
    assert contract.get("inner_rounds", 4) == rounds
    assert contract.get("inner_depth", 4) == 4
    assert contract.get("candidate_capacity", 20) == rounds * 5
    dataset = (root / "dataset.jsonl").read_bytes()
    assert hashlib.sha256(dataset).hexdigest() == contract["dataset_sha256"]
    for name, digest in contract["source_sha256"].items():
        assert (
            hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
        )
    if reference:
        for key in (
            "dataset_sha256",
            "model",
            "assistant",
            "model_config_sha256",
            "assistant_config_sha256",
            "protocol",
        ):
            assert contract[key] == reference[key], key
        for name, digest in reference["source_sha256"].items():
            if name.startswith("vllm/"):
                assert contract["source_sha256"][name] == digest, name
    samples = [json.loads(line) for line in dataset.splitlines()]
    results = {}
    for batch, mode in contract["cells"]:
        folder = root / f"b{batch}_{mode}"
        assert (folder / "CELL_COMPLETE").exists(), str(folder)
        result = json.loads((folder / "result.json").read_text())
        assert result["source_sha256"] == contract["source_sha256"]
        assert result["batch_size"] == batch and result["mode"] == mode
        assert result.get("inner_rounds", 4) == rounds
        assert result["output_tokens"] == 8192 and len(result["outputs"]) == 16
        assert len(result["batches"]) == 16 // batch
        assert all(r["size"] == batch for r in result["batches"])
        assert result["seconds"] == sum(r["seconds"] for r in result["batches"])
        assert result["counters"].get("preverify_graphs", 0) == result["warmup_graphs"]
        for sample, output in zip(samples, result["outputs"], strict=True):
            assert sample["prompt_sha256"] == output["prompt_sha256"]
            assert len(output["token_ids"]) == 512
        results[batch, mode] = result
    return contract, results


def summarize(baseline, round6, round8, output):
    reference, old = audit(baseline, 4)
    runs = {4: old}
    for rounds, root in ((6, round6), (8, round8)):
        _, runs[rounds] = audit(root, rounds, reference)
    rows, comparisons = [], []
    for rounds, results in runs.items():
        for batch in BATCHES:
            for mode in POLICIES:
                result = results[batch, mode]
                counters = result["counters"]
                stats = acceptance_stats(
                    [o["spec_decode_metrics"] for o in result["outputs"]], rounds * 5
                )
                equal_ar = equal_r4 = 0
                for i, (out, ar, r4) in enumerate(
                    zip(
                        result["outputs"],
                        old[batch, "ar"]["outputs"],
                        old[batch, mode]["outputs"],
                        strict=True,
                    )
                ):
                    entry = {
                        "rounds": rounds,
                        "batch": batch,
                        "mode": mode,
                        "sample": i,
                    }
                    for name, ref in (("ar", ar), ("r4", r4)):
                        first = next(
                            (
                                j
                                for j, (a, b) in enumerate(
                                    zip(out["token_ids"], ref["token_ids"], strict=True)
                                )
                                if a != b
                            ),
                            None,
                        )
                        entry[f"first_difference_{name}"] = first
                        if name == "ar":
                            equal_ar += first is None
                        else:
                            equal_r4 += first is None
                    comparisons.append(entry)
                rows.append(
                    {
                        "rounds": rounds,
                        "batch": batch,
                        "mode": mode,
                        "candidate_capacity": 5 * rounds,
                        "gpu": result["gpu"],
                        "seconds": result["seconds"],
                        "tokens_per_second": 8192 / result["seconds"],
                        "speedup_r4": old[batch, mode]["seconds"] / result["seconds"],
                        "speedup_mtp": old[batch, "mtp"]["seconds"] / result["seconds"],
                        **stats,
                        "target_calls": counters["engine_steps"],
                        "batch_inner_calls": counters["batch_round_calls"],
                        "request_inner_rounds": counters["inner_rounds"],
                        "early_stops": counters["early_stops"],
                        "skipped_request_rounds": counters["skipped_rounds"],
                        "exact_ar_requests": equal_ar,
                        "exact_r4_requests": equal_r4,
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", rows), ("output_comparisons", comparisons)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        with (output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(data[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(data)
    lines = [
        "# h4 D4 三档停止信号：4/6/8 轮比较",
        "",
        "24 个新增配置完整，合并此前 12 个四轮配置。16 条相同样本，"
        "每条输出 512，B1/4/8/16。四轮是历史测量，六/八轮是本次新测量；"
        "每配置一次热图覆盖稳定的测量，无误差条或显著性结论。",
        "",
        "| B | Policy | R | tok/s | vs R4 | 接受率 | 平均提交 | 平均接受 |"
        " 全接受概率 | 满容量全接受概率 | 满容量次数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (r["batch"], r["mode"], r["rounds"])):
        cap = row["all_accepted_given_capacity"]
        lines.append(
            f"| {row['batch']} | {row['mode']} | {row['rounds']} | "
            f"{row['tokens_per_second']:.2f} | {row['speedup_r4']:.3f} | "
            f"{row['outer_acceptance_rate']:.2%} | {row['mean_proposed']:.2f} | "
            f"{row['mean_accepted']:.2f} | {row['all_accepted_probability']:.2%} | "
            f"{format(cap, '.2%') if cap is not None else 'N/A'} | "
            f"{row['capacity_steps']} |"
        )
    lines += [
        "",
        "## 指标与边界",
        "",
        "- 接受率为接受候选总数/提交候选总数；平均接受长度不含 Target token。",
        "- 全接受概率为 A=D 的非空请求验证次数/非空请求验证总次数；"
        "满容量概率条件为 D=5R。均为经验频率，包含最后一次验证。",
        "- 输出长度边界可能截断最后一次已接受候选。逐步计数保留原始定义。",
        "- 跳过请求内轮不是 GPU 时间节省；各策略生成轨迹可能不同。"
        "与同 batch AR 和同策略四轮输出的逐 token 一致性单独报告。",
        "- 固定四轮无早停对照不在本次范围；不能推导真实反事实误停率。",
        "- AI assistance was used for benchmark implementation and analysis.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    (output / "MATRIX_AUDIT_COMPLETE").write_text(
        "24 new cells + 12 historical R4 cells; counters and identities audited; "
        "output equality reported separately\n"
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("round6", type=Path)
    parser.add_argument("round8", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait:
        args.output.mkdir(parents=True, exist_ok=True)
        while True:
            states = []
            for root in (args.round6, args.round8):
                status = root / "status.json"
                states.append(json.loads(status.read_text()) if status.exists() else {})
            if any(s.get("state") == "failed" for s in states):
                raise RuntimeError(f"Round sweep worker failed: {states}")
            completed = sum(
                (root / f"b{b}_{p}" / "CELL_COMPLETE").exists()
                for root in (args.round6, args.round8)
                for b in BATCHES
                for p in POLICIES
            )
            (args.output / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_cells": completed,
                        "expected_cells": 24,
                        "workers": states,
                    },
                    indent=2,
                )
                + "\n"
            )
            if completed == 24:
                break
            print(f"ROUND_SWEEP_WAIT {completed}/24", flush=True)
            time.sleep(30)
    summarize(args.baseline, args.round6, args.round8, args.output)
    from benchmarks.hierarchical.plot_round_sweep import plot

    plot(args.output)
    (args.output / "ANALYSIS_COMPLETE").write_text(
        "36 comparisons audited and plotted; visual review pending\n"
    )
