# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute GPU work to the innermost NVTX phase via CUDA launch correlation."""

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    ranges = [
        dict(r)
        for r in connection.execute(
            "SELECT start, end, globalTid, text FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND text IS NOT NULL"
        )
    ]
    runtimes = {}
    api = defaultdict(Counter)
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        parents = [
            r
            for r in ranges
            if r["globalTid"] == row["globalTid"]
            and r["start"] <= row["start"] <= r["end"]
        ]
        phase = (
            min(parents, key=lambda r: r["end"] - r["start"])["text"]
            if parents
            else "outside"
        )
        assert row["correlationId"] not in runtimes
        runtimes[row["correlationId"]] = phase
        key = (phase, strings[row["nameId"]])
        api[key]["calls"] += 1
        api[key]["cpu_ms"] += (row["end"] - row["start"]) / 1e6
    kernels = defaultdict(Counter)
    phase_totals = defaultdict(Counter)
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        phase = runtimes.get(row["correlationId"], "unmatched")
        duration = (row["end"] - row["start"]) / 1e6
        key = (phase, strings[row["shortName"]])
        kernels[key]["calls"] += 1
        kernels[key]["gpu_ms"] += duration
        phase_totals[phase]["kernel_ms"] += duration
        phase_totals[phase]["kernel_calls"] += 1
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        phase = runtimes.get(row["correlationId"], "unmatched")
        phase_totals[phase]["memcpy_ms"] += (row["end"] - row["start"]) / 1e6
        phase_totals[phase]["memcpy_calls"] += 1
        phase_totals[phase]["memcpy_bytes"] += row["bytes"]
    for row in ranges:
        phase_totals[row["text"]]["nvtx_calls"] += 1
        phase_totals[row["text"]]["nvtx_cpu_ms"] += (row["end"] - row["start"]) / 1e6
    return (
        dict(phase_totals),
        [{"phase": p, "kernel": k, **v} for (p, k), v in kernels.items()],
        [{"phase": p, "api": k, **v} for (p, k), v in api.items()],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    phases, kernels, api = analyze(args.sqlite)
    (args.output / "nsys_phases.json").write_text(json.dumps(phases, indent=2) + "\n")
    for name, rows in (("nsys_kernels.csv", kernels), ("nsys_api.csv", api)):
        with (args.output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
