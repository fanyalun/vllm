# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate completed windowed runs, retaining output divergence and cycle costs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def first_difference(left, right):
    return next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads((args.root / path).read_text())

    complete = read("final/measurement_complete.json")
    read("ar/complete.json")
    rows, ar = read("final/results.json"), read("ar/results.json")
    assert len(rows) == 880
    assert len(ar) == 80 and all(len(r["token_ids"]) == 256 for r in ar)
    ar_tokens = {r["sample"]: r["token_ids"] for r in ar if r["repeat"] == 0}
    ar_tps = sum(len(r["token_ids"]) for r in ar) / sum(r["seconds"] for r in ar)
    cases = list(dict.fromkeys(r["case"] for r in rows))
    native = {
        r["sample"]: r["token_ids"]
        for r in rows
        if r["case"] == "none:carry:balanced" and r["phase"] == "e2e"
    }
    summary = []
    for case in cases:
        timed = [r for r in rows if r["case"] == case and r["phase"] == "e2e"]
        assert len(timed) == 80 and all(len(r["token_ids"]) == 256 for r in timed)
        tokens = {r["sample"]: r["token_ids"] for r in timed}
        assert len(tokens) == 16 and all(
            r["token_ids"] == tokens[r["sample"]] for r in timed
        )
        divergences = [
            dict(
                sample=i,
                ar_first_difference=first_difference(tokens[i], ar_tokens[i]),
                native_first_difference=first_difference(tokens[i], native[i]),
            )
            for i in range(16)
        ]
        tps = sum(len(r["token_ids"]) for r in timed) / sum(r["seconds"] for r in timed)
        audit = [r for r in rows if r["case"] == case and r["phase"] == "audit"]
        assert len(audit) == 16
        phase_ms, phase_calls = defaultdict(float), Counter()
        policy_totals = Counter()
        accepted, proposed = 0, 0
        histogram, consecutive_rejections = Counter(), 0
        cycles = []
        target_ms, terminal_proposal_ms = 0.0, 0.0
        for row in audit:
            assert row["token_ids"] == tokens[row["sample"]]
            policy_totals.update(row["policy_metrics"])
            for span in row["spans"]:
                phase_ms[span["phase"]] += span["ms"]
                phase_calls[span["phase"]] += 1
            proposals = [s["ms"] for s in row["spans"] if s["phase"] == "proposal"]
            assert len(proposals) == len(row["cycles"]) + 1
            terminal_proposal_ms += proposals[-1]
            target_ms += sum(c["ms"] - p for c, p in zip(row["cycles"], proposals))
            for cycle in row["cycles"]:
                cycles.append(cycle)
                previous_reject = False
                for inner in cycle["inner"]:
                    accepted += inner["accepted"]
                    proposed += inner["proposed"]
                    histogram[inner["accepted"]] += 1
                    rejected = inner["accepted"] < inner["proposed"]
                    consecutive_rejections += previous_reject and rejected
                    previous_reject = rejected
        actions = [r for r in rows if r["case"] == case and r["phase"] == "actions"]
        if actions:
            assert len(actions) == 16
            assert all(r["token_ids"] == tokens[r["sample"]] for r in actions)
        action_totals = {}
        for key in ("forward_action_counts", "head_window_counts"):
            if actions and key in actions[0]:
                action_totals[key] = [sum(r[key][i] for r in actions) for i in range(3)]
        if actions and "total_heads" in actions[0]:
            action_totals["unchanged_heads"] = sum(
                r["unchanged_heads"] for r in actions
            )
            action_totals["total_heads"] = sum(r["total_heads"] for r in actions)
            assert action_totals["total_heads"] == policy_totals["inner_rounds"] * 960
        if actions:
            assert sum(action_totals["forward_action_counts"]) == (
                policy_totals["inner_rounds"] * 960 * 5
            )
        summary.append(
            dict(
                case=case,
                tokens_per_second=tps,
                milliseconds_per_returned_token=1000 / tps,
                speedup_over_ar=tps / ar_tps,
                repeat_tokens_per_second=[
                    4096 / sum(r["seconds"] for r in timed if r["repeat"] == i)
                    for i in range(5)
                ],
                identical_to_ar=sum(
                    d["ar_first_difference"] is None for d in divergences
                ),
                identical_to_native=sum(
                    d["native_first_difference"] is None for d in divergences
                ),
                divergences=divergences,
                audit_cycle_count=len(cycles),
                audit_target_and_between_stage_ms=target_ms,
                audit_terminal_proposal_ms=terminal_proposal_ms,
                audit_inner_accepted=accepted,
                audit_inner_proposed=proposed,
                audit_all_proposal_metrics=dict(policy_totals),
                audit_acceptance_histogram=dict(histogram),
                audit_consecutive_rejections=consecutive_rejections,
                audit_phase_total_ms=dict(phase_ms),
                audit_phase_calls=dict(phase_calls),
                actions=action_totals,
            )
        )
    result = dict(
        completion=complete,
        rows=len(rows),
        ar_tokens_per_second=ar_tps,
        cases=summary,
        private_state=read("final/private_state.json"),
        timing="Returned tokens / drained generation seconds; audit separate",
        audit_caveat="Events include CPU gaps; proposal overlaps inner phases",
    )
    (args.root / "online_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({r["case"]: r["tokens_per_second"] for r in summary}, indent=2))


if __name__ == "__main__":
    main()
