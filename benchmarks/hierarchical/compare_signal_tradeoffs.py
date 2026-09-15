# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare h4 stopping rules by saved rounds and original suffix cost."""

import argparse
import csv
import hashlib
from pathlib import Path

from benchmarks.hierarchical.analyze_confidence import load_events, write_json
from benchmarks.hierarchical.analyze_joint_confidence import trigger


def combined_trigger(rounds, length, margin, zero_length_margin):
    candidates = [trigger(rounds, length, margin)]
    if zero_length_margin is not None:
        candidates.append(trigger(rounds, 0, zero_length_margin))
    return min(
        (r for r in candidates if r is not None),
        key=lambda r: r["inner_round"],
        default=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, details, fingerprints = [], [], {}
    policies = [
        (f"L<={length},M<{margin}", length, margin, None)
        for length in (0, 1, 2, 3)
        for margin in (0.25, 0.5, 1, 2, None)
    ] + [
        (f"M<{base} OR (L=0 AND M<{wide})", 3, base, wide)
        for base, wide in (
            (0.25, 0.5),
            (0.25, 1),
            (0.25, 2),
            (0.5, 1),
            (0.5, 2),
            (1, 2),
        )
    ]
    for split, root in (("pilot", args.pilot), ("heldout", args.heldout)):
        _, cycles = load_events(root, split)
        fingerprints[split] = hashlib.sha256(
            (root / "h4/trace/distributions.jsonl").read_bytes()
        ).hexdigest()
        for name, length, margin, wide in policies:
            selected = []
            for cycle in cycles:
                row = combined_trigger(cycle["rounds"], length, margin, wide)
                if row is None:
                    continue
                boundary = row["offset"] + row["emitted"]
                cut = max(
                    0,
                    min(cycle["outer_accepted"], cycle["remaining_output"]) - boundary,
                )
                selected.append(
                    {
                        "split": split,
                        "policy": name,
                        "request_id": cycle["request_id"],
                        "category": cycle["category"],
                        "trace_line": cycle["trace_line"],
                        "round": row["inner_round"] + 1,
                        "saved": len(cycle["rounds"]) - row["inner_round"] - 1,
                        "cut": cut,
                    }
                )
            details.extend(selected)
            count = len(selected)
            saved = sum(r["saved"] for r in selected)
            harmful = sum(r["cut"] > 0 for r in selected)
            summaries.append(
                {
                    "split": split,
                    "policy": name,
                    "triggers": count,
                    "saved": saved,
                    "saved_fraction": saved / sum(len(c["rounds"]) for c in cycles),
                    "harmful": harmful,
                    "harmful_per_trigger": harmful / count if count else 0,
                    "harmful_per_cycle": harmful / len(cycles),
                    "cut": sum(r["cut"] for r in selected),
                    "max_cut": max((r["cut"] for r in selected), default=0),
                    "harmful_requests": len(
                        {r["request_id"] for r in selected if r["cut"] > 0}
                    ),
                }
            )
    for row in summaries:
        row["pareto_saved_cut"] = not any(
            other["split"] == row["split"]
            and other["saved"] >= row["saved"]
            and other["cut"] <= row["cut"]
            and (other["saved"] > row["saved"] or other["cut"] < row["cut"])
            for other in summaries
        )
    for filename, rows in (("policies.csv", summaries), ("decisions.csv", details)):
        with (args.output / filename).open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        args.output / "audit.json",
        {
            "trace_sha256": fingerprints,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rules": len(policies),
            "status": "exploratory reused h4 traces; no adaptive GPU timing",
            "utility": "saved - lambda * cut is sensitivity only; lambda not measured",
            "semantics": "current L, strict M, correction required, first nonfinal hit",
        },
    )


if __name__ == "__main__":
    main()
