# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create the audited report and evidence archive after all policy cells finish."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from benchmarks.hierarchical.plot_policy_matrix import plot
from benchmarks.hierarchical.summarize_policy_matrix import summarize


def finish(root):
    rows = summarize(root)
    output = root / "report"
    output.mkdir(exist_ok=True)
    plot(root, output)
    lines = [
        "# Gemma h4 三档停止策略：16×512 性能矩阵",
        "",
        "20 个配置已完成并通过样本、计数及预验证热图覆盖审计。"
        "输出与 AR 的逐 token 一致性另列，审计完成不等于严格等价通过。",
        "",
        "同一份 16 请求，输出 512，B1/4/8/16，TP1，greedy，ignore_eos，"
        "raw prompt，同步调度，关闭 prefix cache。每配置先 warmup 16×64；"
        "正式阶段若新增预验证图则重测，保留全部 attempt。"
        "每配置仅一轮有效测量，不作统计显著性判断。",
        "",
        "## 吞吐与接受情况",
        "",
        "| B | 方案 | tok/s | 相对 AR | 相对 MTP | 外层接受率 "
        "| 平均接受 Draft | 与 AR 完全相同请求 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rate = row["outer_acceptance_rate"]
        length = row["mean_accepted_draft"]
        lines.append(
            f"| {row['batch']} | {row['mode']} | {row['tokens_per_second']:.2f} "
            f"| {row['speedup_ar']:.3f}× | {row['speedup_mtp']:.3f}× "
            f"| {format(rate, '.1%') if rate is not None else '—'} "
            f"| {format(length, '.2f') if length is not None else '—'} "
            f"| {row['exact_ar_requests']}/16 |"
        )
    lines += [
        "",
        "## 内层计算",
        "",
        "| B | 方案 | Target 调用 | 请求内轮 | 批量内轮调用 "
        "| 提前停止 | 跳过请求内轮 | 内层接受率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["inner_rounds"] is None:
            continue
        lines.append(
            f"| {row['batch']} | {row['mode']} | {row['target_engine_steps']} "
            f"| {row['inner_rounds']} | {row['batch_round_calls']} "
            f"| {row['early_stops']} | {row['skipped_request_rounds']} "
            f"| {row['inner_acceptance_rate']:.1%} |"
        )
    lines += [
        "",
        "## 解释与证据",
        "",
        "low_error：L=0 时 M<2，否则 M<0.25；"
        "balanced：M<1；aggressive：M<2。"
        "L 为本轮接受长度，M 为 correction 的 Pre-Verify Top-1/Top-2 margin。",
        "",
        "吞吐包含 generate 内的 prefill/decode，排除模型加载与 warmup。"
        "外层接受率不含 Target bonus；Target 调用包含 prefill。"
        "跳过请求内轮不是 GPU 耗时节省比例，也不能直接当作实际误停率。"
        "多请求路径压缩已停止请求，"
        "后续 MTP prefill 使用 fresh eager metadata；"
        "MTP decode 与 Pre-Verify 保留 CUDA Graph。",
        "",
        "完整 summary.csv、output_comparisons.csv、各配置 outputs/counters/attempts、"
        "源码快照与运行配置在 raw_evidence.tar.gz 中。"
        "图表由审计数据自动生成，需在交付时另行目视检查。",
        "",
        "AI assistance was used for implementation, tests, and analysis.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    with tarfile.open(output / "raw_evidence.tar.gz", "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and relative.parts[0] not in {"report", "compiler_cache"}:
                archive.add(path, arcname=str(relative), recursive=False)
    (output / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(output)}\n"
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "checksums.sha256"
        )
    )
    (root / "MEASUREMENT_AND_ANALYSIS_COMPLETE").write_text(
        json.dumps({"cells": 20, "status": "complete", "visual_review": "pending"})
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    finish(parser.parse_args().root)
