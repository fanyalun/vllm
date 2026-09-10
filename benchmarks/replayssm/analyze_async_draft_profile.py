# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize host ranges and their CUDA work without treating waits as copies."""

import argparse
import bisect
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def distribution(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values) / 1e6,
        "median_ms": statistics.median(values) / 1e6,
        "total_ms": sum(values) / 1e6,
    }


def analyze(path: Path) -> dict:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    process_ids = dict(connection.execute("SELECT globalPid, pid FROM PROCESSES"))
    ranges = []
    for start, end, tid, text, text_id in connection.execute(
        "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL ORDER BY start"
    ):
        name = text or strings.get(text_id, "")
        if name.startswith(("async_draft:", "target:", "verify:", "accept:")) or (
            name == "eagle3: propose"
        ):
            ranges.append((start, end, tid, name))
    apis = defaultdict(list)
    for start, end, tid, corr, name_id in connection.execute(
        "SELECT start, end, globalTid, correlationId, nameId "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
    ):
        apis[tid].append((start, end, corr, strings[name_id]))
    api_starts = {tid: [row[0] for row in rows] for tid, rows in apis.items()}
    gpu = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for start, end, pid, corr, device in connection.execute(
            f"SELECT start, end, globalPid, correlationId, deviceId FROM {table}"
        ):
            gpu[(pid, corr)].append((start, end, device))
    summaries = defaultdict(list)
    api_summaries = defaultdict(list)
    timeline = []
    for start, end, tid, name in ranges:
        summaries[name].append(end - start)
        pid = tid & ~((1 << 24) - 1)
        operations = []
        graph_launches = 0
        first = bisect.bisect_left(api_starts.get(tid, []), start)
        for api_start, api_end, corr, api_name in apis[tid][first:]:
            if api_start >= end:
                break
            api_summaries[(name, api_name)].append(api_end - api_start)
            graph_launches += api_name.startswith("cudaGraphLaunch")
            operations.extend(gpu.get((pid, corr), []))
        row = {
            "stage": name,
            "pid": process_ids.get(pid, (tid >> 24) & ((1 << 24) - 1)),
            "host_start_ns": start,
            "host_end_ns": end,
            "host_ms": (end - start) / 1e6,
            "gpu_start_ns": min((op[0] for op in operations), default=None),
            "gpu_end_ns": max((op[1] for op in operations), default=None),
            "gpu_operations": len(operations),
            "graph_launches": graph_launches,
            "gpu_work_ms": sum(op[1] - op[0] for op in operations) / 1e6,
        }
        timeline.append(row)
    connection.close()
    output = path.parent / "stage_timeline.csv"
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timeline[0]))
        writer.writeheader()
        writer.writerows(timeline)
    summary = {
        "source": str(path),
        "limitations": [
            "Profiled run; not an uninstrumented throughput measurement.",
            "Host API duration includes waits; GPU work sums may overlap.",
            "GPU operations use process and launch correlation IDs; missing "
            "graph correlations must not be interpreted as zero GPU work.",
        ],
        "host_stages": {name: distribution(v) for name, v in summaries.items()},
        "host_cuda_apis": [
            {"stage": stage, "api": api, **distribution(values)}
            for (stage, api), values in sorted(
                api_summaries.items(), key=lambda item: -sum(item[1])
            )
        ],
        "critical_path": critical_path(
            timeline,
            [
                (start, end)
                for operations in gpu.values()
                for start, end, device in operations
                if device == 0
            ],
        ),
    }
    (path.parent / "stage_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def critical_path(timeline: list[dict], target_activity=()) -> dict:
    proposals = [r for r in timeline if r["stage"] == "eagle3: propose"]
    outcomes = [r for r in timeline if r["stage"] == "accept: state_postprocess"]
    targets = [r for r in timeline if r["stage"] == "target: verify_forward"]
    if not (len(proposals) == len(outcomes) == len(targets)):
        raise ValueError("Incomplete capture: Target/outcome/proposal counts differ")
    grouped = defaultdict(lambda: defaultdict(list))
    async_run = any(r["stage"] == "async_draft: send_request" for r in timeline)
    for index, (proposal, outcome, target) in enumerate(
        zip(proposals, outcomes, targets)
    ):
        if not target["graph_launches"]:
            continue
        start, end = proposal["host_start_ns"], proposal["host_end_ns"]
        children = [r for r in timeline if start <= r["host_start_ns"] < end]
        stages = {r["stage"] for r in children}
        if "async_draft: local_candidate" in stages:
            kind = "local_hit"
        elif "async_draft: remote_miss" in stages:
            kind = "miss"
        else:
            hit = bool(stages & {"async_draft: cache_hit", "async_draft: remote_hit"})
            kind = ("hit" if hit else "miss") if async_run else "sync"
        if proposal["gpu_end_ns"] is None or outcome["gpu_end_ns"] is None:
            raise ValueError("Missing GPU correlation for a decode proposal")
        grouped[kind]["outcome_to_proposal"].append(
            proposal["gpu_end_ns"] - outcome["gpu_end_ns"]
        )
        grouped[kind]["target_forward_span"].append(
            target["gpu_end_ns"] - target["gpu_start_ns"]
        )
        if index + 1 < len(targets) and targets[index + 1]["graph_launches"]:
            gap_start = proposal["gpu_end_ns"]
            gap_end = targets[index + 1]["gpu_start_ns"]
            grouped[kind]["proposal_to_next_target"].append(gap_end - gap_start)
            if target_activity:
                active = interval_coverage(target_activity, gap_start, gap_end)
                grouped[kind]["next_target_gap_gpu_active"].append(active)
                grouped[kind]["next_target_gap_gpu_idle"].append(
                    gap_end - gap_start - active
                )
        for row in children:
            if row["stage"].startswith("async_draft:"):
                grouped[kind][row["stage"] + "_host"].append(
                    row["host_end_ns"] - row["host_start_ns"]
                )
    return {
        group: {stage: distribution(v) for stage, v in stages.items()}
        for group, stages in grouped.items()
    }


def interval_coverage(intervals, start, end):
    clipped = sorted(
        (max(a, start), min(b, end)) for a, b in intervals if b > start and a < end
    )
    covered = 0
    cursor = start
    for left, right in clipped:
        covered += max(0, right - max(cursor, left))
        cursor = max(cursor, right)
    return covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    summary = analyze(args.sqlite)
    print(json.dumps(summary["host_stages"], indent=2))


if __name__ == "__main__":
    main()
