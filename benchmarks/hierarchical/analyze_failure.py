# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate inner rejection, outer rejection, and complete-cycle costs."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def round_survival(trace, accepted):
    """Count retained candidate tokens in each inner round, excluding T bonus."""
    assert 0 <= accepted <= sum(r["emitted"] for r in trace)
    offset = 0
    retained = []
    for row in trace:
        assert row["offset"] == offset
        assert row["emitted"] == row["accepted"] + 1
        retained.append(max(0, min(row["emitted"], accepted - offset)))
        offset += row["emitted"]
    return retained


def analyze(directory):
    assert (directory / "MEASUREMENT_COMPLETE").exists()
    requests = json.loads((directory / "result.json").read_text())
    counts = Counter()
    phases = Counter()
    first_rejections = Counter()
    inner_by_round = [Counter() for _ in range(4)]
    rows = []
    for request in requests:
        if request["phase"] != "profile":
            continue
        proposals = {s["step"]: s for s in request["spans"] if s["phase"] == "proposal"}
        emitted = 0
        for cycle in request["cycles"]:
            if emitted >= len(request["token_ids"]) - 1:
                continue
            emitted += cycle["emitted"]
            if cycle["proposal_step"] == 0:
                continue
            trace = proposals[cycle["proposal_step"]]["inner_trace"]
            assert sum(r["emitted"] for r in trace) == cycle["scheduled"]
            retained = round_survival(trace, cycle["accepted"])
            for index, inner in enumerate(trace):
                inner_by_round[index]["proposed"] += inner["proposed"]
                inner_by_round[index]["accepted"] += inner["accepted"]
            first = next(
                (
                    i
                    for i, (r, k) in enumerate(zip(trace, retained))
                    if k < r["emitted"]
                ),
                len(trace),
            )
            first_rejections[str(first)] += 1
            row = {
                "sample_index": request["sample_index"],
                **cycle,
                "inner_proposed": sum(r["proposed"] for r in trace),
                "inner_accepted": sum(r["accepted"] for r in trace),
                "inner_rounds": len(trace),
                "zero_retained_rounds": sum(k == 0 for k in retained),
                "rounds_after_first_rejection": max(0, len(trace) - first - 1),
                "retained_by_round": retained,
            }
            rows.append(row)
            for key in (
                "scheduled",
                "accepted",
                "emitted",
                "inner_proposed",
                "inner_accepted",
                "inner_rounds",
                "zero_retained_rounds",
                "rounds_after_first_rejection",
                "cycle_stream_ms",
            ):
                counts[key] += row[key]
            for span in request["spans"]:
                step = (
                    cycle["step"]
                    if span["phase"].startswith("target_")
                    else cycle["proposal_step"]
                )
                if span["step"] == step:
                    phases[span["phase"]] += span["stream_ms"]
    n = len(rows)
    per_cycle = {key: value / n for key, value in phases.items()}
    graph = sum(v for k, v in per_cycle.items() if k.startswith("preverify_graph"))
    small = per_cycle["small_draft"]
    target = per_cycle["target_execute"] + per_cycle["target_sample"]
    cycle_ms = counts["cycle_stream_ms"] / n
    implementation = cycle_ms - small - graph - target
    result = {
        "case": directory.name,
        "cycles": n,
        "counts": dict(counts),
        "inner_by_round": {
            str(i): {**c, "acceptance": c["accepted"] / c["proposed"]}
            for i, c in enumerate(inner_by_round)
        },
        "inner_acceptance": counts["inner_accepted"] / counts["inner_proposed"],
        "outer_acceptance": counts["accepted"] / counts["scheduled"],
        "zero_retained_round_fraction": counts["zero_retained_rounds"]
        / counts["inner_rounds"],
        "trailing_round_fraction": counts["rounds_after_first_rejection"]
        / counts["inner_rounds"],
        "first_rejection_round_zero_based_or_4_if_all_accepted": dict(first_rejections),
        "scheduled_per_cycle": counts["scheduled"] / n,
        "emitted_per_cycle": counts["emitted"] / n,
        "cycle_ms": cycle_ms,
        "small_ms": small,
        "preverify_graph_ms": graph,
        "target_ms": target,
        "remaining_stream_ms": implementation,
        "phase_ms_per_cycle": per_cycle,
        "observed_ms_per_token": counts["cycle_stream_ms"] / counts["emitted"],
        "zero_outer_rejection_fixed_cost_ms_per_token": counts["cycle_stream_ms"]
        / (counts["scheduled"] + n),
        "zero_overhead_fixed_yield_ms_per_token": (small + graph + target)
        / (counts["emitted"] / n),
        "zero_both_fixed_trajectory_ms_per_token": (small + graph + target)
        / (counts["scheduled"] / n + 1),
    }
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    hashes = {}
    for directory in args.directories:
        summary, rows = analyze(directory)
        model = "gemma" if "gemma" in str(directory) else "qwen"
        summary["model"] = model
        summaries.append(summary)
        name = f"{model}_{directory.name}_cycles.json"
        (args.output / name).write_text(json.dumps(rows, indent=2) + "\n")
        for name in ("config.json", "result.json", "MEASUREMENT_COMPLETE"):
            path = directory / name
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "diagnosis.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    flat = [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries]
    with (args.output / "diagnosis.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
