# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit mean-GDN comparisons, including returned-token budget clipping."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def steady_cycles(request, token_limit):
    remaining = token_limit - 1
    result = []
    for cycle in request["cycles"]:
        if remaining <= 0:
            break
        returned = min(cycle["emitted"], remaining)
        remaining -= returned
        assert 0 <= cycle["accepted"] <= cycle["scheduled"]
        if cycle["proposal_step"] > 0:
            result.append({**cycle, "returned": returned})
    assert remaining == 0
    return result


def analyze(directory, ar):
    config = json.loads((directory / "config.json").read_text())
    results = json.loads((directory / "result.json").read_text())
    assert (directory / "MEASUREMENT_COMPLETE").exists()
    limit = config["max_tokens"]
    phases = defaultdict(dict)
    for request in results:
        assert len(request["token_ids"]) == limit
        phases[request["phase"]][request["sample_index"]] = request
    expected = {sample["sample_index"] for sample in config["samples"]}
    assert all(set(phases[p]) == expected for p in ("e2e", "profile", "e2e_after"))
    cycles, spans = [], defaultdict(list)
    inner_accepted = inner_proposed = 0
    for request in phases["profile"].values():
        current = steady_cycles(request, limit)
        cycles.extend(current)
        proposals = {c["proposal_step"] for c in current}
        targets = {c["step"] for c in current}
        for span in request["spans"]:
            steps = targets if span["phase"].startswith("target_") else proposals
            if span["step"] not in steps:
                continue
            spans[span["phase"]].append(span)
            if span["phase"] == "proposal":
                for inner in span["inner_trace"]:
                    inner_accepted += inner["accepted"]
                    inner_proposed += inner["proposed"]
    spec = config["llm"]["speculative_config"]
    row = {
        "case": directory.name,
        "inner": spec["inner_method"],
        "mode": spec.get("preverify_gdn_mode", "none"),
        "requests_per_phase": len(expected),
        "tokens_per_request": limit,
        "e2e_tps": len(expected)
        * limit
        / sum(r["e2e_seconds"] for r in phases["e2e"].values()),
        "e2e_after_tps": len(expected)
        * limit
        / sum(r["e2e_seconds"] for r in phases["e2e_after"].values()),
        "cycle_ms": mean(c["cycle_stream_ms"] for c in cycles),
        "returned_per_cycle": mean(c["returned"] for c in cycles),
        "cycle_ms_per_returned": sum(c["cycle_stream_ms"] for c in cycles)
        / sum(c["returned"] for c in cycles),
        "target_accepted": sum(c["accepted"] for c in cycles),
        "target_proposed": sum(c["scheduled"] for c in cycles),
        "inner_accepted": inner_accepted,
        "inner_proposed": inner_proposed,
        "cycles": len(cycles),
        "ar_equal_requests": sum(
            phases["e2e"][i]["token_ids"] == ar[i]["token_ids"] for i in expected
        ),
        "profile_equal_requests": sum(
            phases["e2e"][i]["token_ids"] == phases["profile"][i]["token_ids"]
            for i in expected
        ),
        "repeat_equal_requests": sum(
            phases["e2e"][i]["token_ids"] == phases["e2e_after"][i]["token_ids"]
            for i in expected
        ),
    }
    row["target_acceptance"] = row["target_accepted"] / row["target_proposed"]
    row["inner_acceptance"] = inner_accepted / inner_proposed
    details = []
    for phase, values in spans.items():
        details.append(
            {
                "case": directory.name,
                "phase": phase,
                "calls": len(values),
                "stream_ms_per_call": mean(v["stream_ms"] for v in values),
                "stream_ms_per_cycle": sum(v["stream_ms"] for v in values)
                / len(cycles),
            }
        )
    return row, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--ar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (args.ar / "MEASUREMENT_COMPLETE").exists()
    ar_config = json.loads((args.ar / "config.json").read_text())
    ar = {
        r["sample_index"]: r
        for r in json.loads((args.ar / "result.json").read_text())
        if r["phase"] == "e2e"
    }
    rows, details = [], []
    comparisons = []
    for directory in args.directories:
        config = json.loads((directory / "config.json").read_text())
        assert config["samples"] == ar_config["samples"]
        assert config["max_tokens"] == ar_config["max_tokens"]
        for key in ("model", "seed", "tensor_parallel_size", "max_model_len"):
            assert config["llm"][key] == ar_config["llm"][key]
        row, detail = analyze(directory, ar)
        rows.append(row)
        details.extend(detail)
        for request in json.loads((directory / "result.json").read_text()):
            reference = ar[request["sample_index"]]["token_ids"]
            first_difference = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(reference, request["token_ids"], strict=True)
                    )
                    if a != b
                ),
                None,
            )
            comparisons.append(
                {
                    "case": directory.name,
                    "phase": request["phase"],
                    "sample_index": request["sample_index"],
                    "first_ar_difference": first_difference,
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("summary.csv", rows), ("phases.csv", details)):
        with (args.output / name).open("w") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(values[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(values)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "output_comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
