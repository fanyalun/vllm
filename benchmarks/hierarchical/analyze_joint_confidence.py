# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare inner acceptance length AND correction margin on saved h4 traces."""

import argparse
import hashlib
from pathlib import Path

from benchmarks.hierarchical.analyze_confidence import (
    load_events,
    policy_trigger,
    write_csv,
    write_json,
)


def trigger(rounds, length, margin):
    for row in rounds[:-1]:
        if row["accepted"] >= row["proposed"]:
            continue
        if length is not None and row["accepted"] > length:
            continue
        value = (
            row["correction_margin"]
            if "correction_margin" in row
            else row["draft_preverify"]["right"]["margin"]
        )
        if margin is None or value < margin:
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, decisions, local = [], [], []
    fingerprints = {}
    for split, root in (("pilot", args.pilot), ("heldout", args.heldout)):
        events, cycles = load_events(root, split)
        fingerprints[split] = hashlib.sha256(
            (root / "h4/trace/distributions.jsonl").read_bytes()
        ).hexdigest()
        for length in (None, 0, 1, 2, 3):
            for margin in (None, 0.25, 0.5, 1, 2):
                name = f"accept_le{length}_margin_lt{margin}"
                selected = []
                for cycle in cycles:
                    row = trigger(cycle["rounds"], length, margin)
                    if length is None:
                        old = (
                            "any_correction"
                            if margin is None
                            else f"correction_margin_lt{margin}"
                        )
                        assert row == policy_trigger(cycle["rounds"], old)
                    if length == 1 and margin is None:
                        assert row == policy_trigger(
                            cycle["rounds"], "inner_accept_le1"
                        )
                    if row is None:
                        continue
                    boundary = row["offset"] + row["emitted"]
                    cut = max(
                        0,
                        min(cycle["outer_accepted"], cycle["remaining_output"])
                        - boundary,
                    )
                    selected.append(
                        {
                            "split": split,
                            "policy": name,
                            "request_id": cycle["request_id"],
                            "cycle": cycle["cycle"],
                            "trace_line": cycle["trace_line"],
                            "round": row["inner_round"] + 1,
                            "inner_accepted": row["accepted"],
                            "skipped_rounds": len(cycle["rounds"])
                            - row["inner_round"]
                            - 1,
                            "suffix_cut": cut,
                            "already_rejected": cycle["outer_accepted"] < boundary - 1,
                        }
                    )
                summaries.append(
                    {
                        "split": split,
                        "policy": name,
                        "length_le": length,
                        "margin_lt": margin,
                        "cycles": len(cycles),
                        "triggers": len(selected),
                        "requests": len({r["request_id"] for r in selected}),
                        "skipped_rounds": sum(r["skipped_rounds"] for r in selected),
                        "zero_suffix": sum(r["suffix_cut"] == 0 for r in selected),
                        "harmful_triggers": sum(r["suffix_cut"] > 0 for r in selected),
                        "suffix_cut": sum(r["suffix_cut"] for r in selected),
                        "already_rejected": sum(
                            r["already_rejected"] for r in selected
                        ),
                    }
                )
                decisions.extend(selected)
                reached = [
                    r
                    for r in events
                    if r["origin"] == "correction"
                    and r["reached"]
                    and (length is None or r["inner_accepted"] <= length)
                    and (margin is None or r["margin"] < margin)
                ]
                local.append(
                    {
                        "split": split,
                        "policy": name,
                        "reached": len(reached),
                        "rejected": sum(not r["accepted"] for r in reached),
                    }
                )
    for name, rows in (
        ("policies", summaries),
        ("decisions", decisions),
        ("local_corrections", local),
    ):
        write_csv(args.output / f"{name}.csv", rows)
        write_json(args.output / f"{name}.json", rows)
    write_json(
        args.output / "audit.json",
        {
            "trace_sha256": fingerprints,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "h4 only; reuse audited pilot4 and heldout16; no new GPU run",
            "selection": "exploratory joint grid after inspecting previous results",
            "semantics": "current round accepted length; correction required; AND",
            "stopping": "first eligible round, excluding final round; keep correction",
            "label": "original accepted suffix clipped to output128; not speedup",
            "checks": "load_events parity and counters; legacy marginal policies equal",
        },
    )


if __name__ == "__main__":
    main()
