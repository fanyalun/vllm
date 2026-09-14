# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate uninstrumented request cost from diagnostic stream/kernel intervals."""

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def union_duration(intervals):
    end = None
    total = 0.0
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(end, right) if end is not None else right
    return total


def summarize(root):
    costs, phases, kernels = [], [], []
    for path in sorted(root.glob("*/*/complete.json")):
        cell = path.parent
        model, mode = cell.parent.name, cell.name
        e2e = json.loads((cell / "e2e.json").read_text())
        diagnostics = json.loads((cell / "diagnostics.json").read_text())
        elapsed = sum(r["seconds"] for r in e2e)
        tokens = sum(len(r["token_ids"]) for r in e2e)
        original = [r for r in e2e if r["sample_index"] == e2e[0]["sample_index"]]
        timed = next(d for d in diagnostics if d["mode"] == "events")
        costs.append(
            dict(
                model=model,
                mode=mode,
                ms_per_output=1000 * elapsed / tokens,
                requests=len(e2e),
                tokens=tokens,
                sample0_repeat_exact=all(
                    r["token_ids"] == original[0]["token_ids"] for r in original
                ),
                event_output_exact=timed["token_ids"] == original[0]["token_ids"],
                event_time_ratio=timed["seconds"]
                / statistics.mean(r["seconds"] for r in original),
            )
        )
        groups = defaultdict(list)
        for row in timed["spans"]:
            if row["step"] > 0:
                groups[row["phase"]].append(row)
        for phase, rows in groups.items():
            phases.append(
                dict(
                    model=model,
                    mode=mode,
                    phase=phase,
                    calls=len(rows),
                    mean_stream_ms=statistics.mean(r["stream_ms"] for r in rows),
                    median_stream_ms=statistics.median(r["stream_ms"] for r in rows),
                    mean_cpu_ms=statistics.mean(r["cpu_ms"] for r in rows),
                    total_stream_ms=sum(r["stream_ms"] for r in rows),
                )
            )
        with gzip.open(cell / "trace.json.gz", "rt") as file:
            trace = json.load(file)
        events = [
            e for e in trace["traceEvents"] if e.get("cat") == "kernel" and "dur" in e
        ]
        if not events:
            raise RuntimeError(f"No CUDA kernels captured: {cell}")
        intervals = [(e["ts"], e["ts"] + e["dur"]) for e in events]
        span = max(right for _, right in intervals) - min(left for left, _ in intervals)
        busy = union_duration(intervals)
        kernels.append(
            dict(
                model=model,
                mode=mode,
                kernel_calls=len(events),
                kernel_union_ms=busy / 1000,
                first_to_last_kernel_ms=span / 1000,
                gap_ms=(span - busy) / 1000,
            )
        )
    for name, rows in [
        ("costs", costs),
        ("phases", phases),
        ("kernel_activity", kernels),
    ]:
        if rows:
            with (root / f"{name}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (root / "analysis.json").write_text(
        json.dumps(
            dict(
                costs=costs,
                phases=phases,
                kernel_activity=kernels,
                scope=(
                    "E2E is uninstrumented. Phase intervals include CPU-induced gaps "
                    "and overlap with containing spans; do not sum phases. "
                    "Profiler timing is diagnostic only. Kernel busy time is the "
                    "union across streams, including prefill and terminal proposals."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    return costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output), indent=2))
