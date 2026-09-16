# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and summarize paired fixed-input pre-verifier timings."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(root):
    contract = json.loads((root / "contract.json").read_text())
    complete = json.loads((root / "measurement_complete.json").read_text())
    samples = json.loads((root / "results.json").read_text())
    cases = contract["cases"]
    boundaries = ("forward", "gdn", "advance", "combined")
    expected = {
        (trial, sample)
        for trial in range(contract["rounds"])
        for sample in range(len(contract["samples"]))
    }
    assert {(s["trial"], s["sample"]) for s in samples} == expected
    assert len(samples) == len(expected)
    assert complete["control_outputs_equal"]
    assert sum(len(s["rows"]) for s in samples) == complete["rows"]
    rows = []
    for sample in samples:
        identity = contract["samples"][sample["sample"]]["prompt_sha256"]
        assert sample["prompt_sha256"] == identity
        expected_cells = {
            (case, boundary, repeat)
            for case in cases
            for boundary in boundaries
            for repeat in range(contract["repeats"])
        }
        assert len(sample["rows"]) == len(expected_cells)
        assert {
            (r["case"], r["boundary"], r["repeat"]) for r in sample["rows"]
        } == expected_cells
        assert sample["checks"]["repeatable_predictions"]
        for boundary in boundaries:
            medians = {
                case: statistics.median(
                    r["ms"]
                    for r in sample["rows"]
                    if r["case"] == case and r["boundary"] == boundary
                )
                for case in cases
            }
            for case, ms in medians.items():
                rows.append(
                    dict(
                        trial=sample["trial"],
                        sample=sample["sample"],
                        boundary=boundary,
                        case=case,
                        median_ms=ms,
                        time_reduction_pct=100 * (1 - ms / medians["baseline"]),
                    )
                )
    summary = []
    for boundary in boundaries:
        for case in cases:
            selected = [
                r for r in rows if r["boundary"] == boundary and r["case"] == case
            ]
            summary.append(
                dict(
                    boundary=boundary,
                    case=case,
                    mean_median_ms=statistics.mean(r["median_ms"] for r in selected),
                    paired_mean_reduction_pct=statistics.mean(
                        r["time_reduction_pct"] for r in selected
                    ),
                    paired_min_reduction_pct=min(
                        r["time_reduction_pct"] for r in selected
                    ),
                    paired_max_reduction_pct=max(
                        r["time_reduction_pct"] for r in selected
                    ),
                )
            )
    result = dict(summary=summary, paired_rows=rows, measurement_complete=True)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "paired.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
