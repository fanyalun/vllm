# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit complete-cycle measurements without mixing tokens from different passes."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(directory):
    config = json.loads((directory / "config.json").read_text())
    results = json.loads((directory / "result.json").read_text())
    assert (directory / "MEASUREMENT_COMPLETE").exists()
    plain = {r["sample_index"]: r for r in results if r["phase"] == "e2e"}
    profiled = [r for r in results if r["phase"] == "profile"]
    after = [r for r in results if r["phase"] == "e2e_after"]
    assert len(plain) == len(profiled) == len(config["samples"])
    cycles, phases = [], defaultdict(list)
    discarded_cycles = 0
    for request in profiled:
        assert len(request["token_ids"]) == 512
        assert len(plain[request["sample_index"]]["token_ids"]) == 512
        all_cycles = []
        emitted = 0
        for cycle in request["cycles"]:
            # Async scheduling can verify an additional in-flight batch after
            # the output limit has already been reached.
            if emitted >= 511:
                discarded_cycles += 1
                continue
            all_cycles.append(cycle)
            emitted += cycle["emitted"]
        assert sum(c["emitted"] for c in all_cycles) >= 511
        assert sum(c["emitted"] for c in all_cycles) <= 511 + max(
            c["scheduled"] for c in all_cycles
        )
        assert all(0 <= c["accepted"] <= c["scheduled"] for c in all_cycles)
        # Exclude the initial proposal attached to prompt prefill and the
        # terminal proposal that never reaches another Target verification.
        steady = [c for c in all_cycles if c["proposal_step"] > 0]
        cycles.extend(steady)
        proposals = {c["proposal_step"] for c in steady}
        targets = {c["step"] for c in steady}
        for span in request["spans"]:
            steps = targets if span["phase"].startswith("target_") else proposals
            if span["step"] in steps:
                phases[span["phase"]].append(span)
    spec = config["llm"]["speculative_config"]
    rounds = spec.get("inner_num_rounds", 1)
    result = {
        "case": directory.name,
        "device": config["cuda_visible_devices"],
        "rounds": rounds,
        "e2e_tps": len(plain) * 512 / sum(r["e2e_seconds"] for r in plain.values()),
        "profile_tps": len(profiled) * 512 / sum(r["e2e_seconds"] for r in profiled),
        "e2e_after_tps": (
            len(after) * 512 / sum(r["e2e_seconds"] for r in after) if after else None
        ),
        "profile_equal_requests": sum(
            r["token_ids"] == plain[r["sample_index"]]["token_ids"] for r in profiled
        ),
        "after_equal_requests": (
            sum(r["token_ids"] == plain[r["sample_index"]]["token_ids"] for r in after)
            if after
            else None
        ),
        "requests": len(profiled),
        "steady_cycles": len(cycles),
        "discarded_after_limit_cycles": discarded_cycles,
        "cycle_stream_ms": mean(c["cycle_stream_ms"] for c in cycles),
        "cycle_wall_ms": mean(c["cycle_wall_ms"] for c in cycles),
        "emitted_per_cycle": mean(c["emitted"] for c in cycles),
        "scheduled_per_cycle": mean(c["scheduled"] for c in cycles),
        "cycle_ms_per_emitted": sum(c["cycle_stream_ms"] for c in cycles)
        / sum(c["emitted"] for c in cycles),
        "cycle_ms_per_inner_round": mean(c["cycle_stream_ms"] for c in cycles) / rounds,
    }
    detail = []
    for phase, spans in phases.items():
        detail.append(
            {
                "case": directory.name,
                "phase": phase,
                "calls": len(spans),
                "stream_ms_per_call": mean(s["stream_ms"] for s in spans),
                "cpu_ms_per_call": mean(s["cpu_ms"] for s in spans),
                "stream_ms_per_cycle": sum(s["stream_ms"] for s in spans) / len(cycles),
            }
        )
    return result, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for directory in args.directories:
        row, detail = summarize(directory)
        rows.append(row)
        details.extend(detail)
    for name, values in (("summary.csv", rows), ("phases.csv", details)):
        with (args.output / name).open("w") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(values[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(values)
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in args.directories
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
