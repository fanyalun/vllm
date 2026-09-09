# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit cell outputs without treating throughput as proof of correctness."""

import argparse
import csv
import json
from pathlib import Path


def compare(reference, candidate):
    reference_rows = reference["outputs"]
    candidate_rows = candidate["outputs"]
    failures = []
    for key in ("model", "ssm_dtype", "max_tokens", "temperature"):
        if reference["args"][key] != candidate["args"][key]:
            failures.append({"reason": f"configuration mismatch: {key}"})
    expected = candidate["args"]["num_samples"] * candidate["args"]["repeats"]
    if len(candidate_rows) != expected or len(reference_rows) != expected:
        failures.append({"reason": "incomplete sample coverage"})
    for ref, row in zip(reference_rows, candidate_rows):
        identity = ("repeat", "sample_index", "prompt_sha256")
        if any(ref[key] != row[key] for key in identity):
            failures.append({"reason": "prompt or repetition mismatch"})
            continue
        if len(row["token_ids"]) != candidate["args"]["max_tokens"]:
            failures.append({"reason": "incorrect output length"})
        if ref["token_ids"] != row["token_ids"]:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(ref["token_ids"], row["token_ids"])
                    )
                    if left != right
                ),
                min(len(ref["token_ids"]), len(row["token_ids"])),
            )
            failures.append(
                {
                    "reason": "greedy token mismatch",
                    "sample_index": row["sample_index"],
                    "first_mismatch_zero_based": mismatch,
                }
            )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text())
    if reference["args"]["temperature"] != 0:
        raise ValueError("Token parity requires a greedy reference")
    audit = {"reference": str(args.reference), "cells": [], "status": "passed"}
    rows = []
    for path in args.cells:
        candidate = json.loads(path.read_text())
        if candidate["args"]["temperature"] != 0:
            raise ValueError("Do not compare stochastic trajectories by seed")
        failures = compare(reference, candidate)
        audit["cells"].append({"path": str(path), "failures": failures})
        if failures:
            audit["status"] = "failed"
        for row in candidate["outputs"]:
            rows.append(
                {
                    "cell": path.stem,
                    "sample_index": row["sample_index"],
                    "repeat": row["repeat"],
                    "output_tokens": len(row["token_ids"]),
                    "e2e_seconds": row["e2e_seconds"],
                    "tokens_per_second": (len(row["token_ids"]) / row["e2e_seconds"]),
                    "greedy_gate_passed": not failures,
                }
            )
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (args.output / "cells.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            lineterminator="\n",
            fieldnames=[
                "cell",
                "sample_index",
                "repeat",
                "output_tokens",
                "e2e_seconds",
                "tokens_per_second",
                "greedy_gate_passed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
