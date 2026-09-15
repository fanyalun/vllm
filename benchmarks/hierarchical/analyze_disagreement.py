# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit both directions of hierarchical first-rejection disagreement."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from run_token_importance import digest, write_json


def classify(pair, near_gap=1.0):
    if pair["same_top1"]:
        return "agreement"
    sides = [pair["left"], pair["right"]]
    if all(
        s["other_top1_rank"] <= 2 and s["other_top1_logit_gap"] <= near_gap
        for s in sides
    ):
        return "near"
    if any(s["other_top1_rank"] > 8 or s["other_top1_logit_gap"] > 2 for s in sides):
        return "far"
    return "middle"


def flatten(pair):
    result = {k: pair[k] for k in ("js_nats", "tv", "kl_left_right", "kl_right_left")}
    result["class"] = classify(pair)
    for side in ("left", "right"):
        for key in (
            "other_top1_rank",
            "other_top1_prob",
            "other_top1_logit_gap",
            "margin",
            "entropy",
        ):
            result[f"{side}_{key}"] = pair[side][key]
        result[f"{side}_top_tokens"] = pair[side]["top_tokens"]
        result[f"{side}_top_probs"] = pair[side]["top_probs"]
    return result


def write_csv(path, rows):
    if rows:
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(rows):
    reached = [r for r in rows if r["target_reached"]]
    suffix = [r for r in reached if r["suffix_scheduled"] > 0]
    corrected = [r for r in suffix if r["correction_accepted"]]
    denominator = sum(r["suffix_scheduled"] for r in suffix)
    return {
        "events": len(rows),
        "reached": len(reached),
        "unreached": len(rows) - len(reached),
        "correction_accepted": sum(r["correction_accepted"] for r in reached),
        "correction_acceptance": sum(r["correction_accepted"] for r in reached)
        / len(reached)
        if reached
        else None,
        "suffix_events": len(suffix),
        "suffix_zero": sum(r["suffix_accepted"] == 0 for r in suffix),
        "suffix_full": sum(
            r["suffix_accepted"] == r["suffix_scheduled"] for r in suffix
        ),
        "suffix_retention": sum(r["suffix_accepted"] for r in suffix) / denominator
        if denominator
        else None,
        "corrected_suffix_events": len(corrected),
        "corrected_suffix_full": sum(
            r["suffix_accepted"] == r["suffix_scheduled"] for r in corrected
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    contract = json.loads((root / "contract.json").read_text())
    assert digest(root / "dataset.jsonl") == contract["dataset_sha256"]
    for name, value in contract["source_sha256"].items():
        assert digest(root / name) == value
    dataset = [json.loads(s) for s in (root / "dataset.jsonl").read_text().splitlines()]
    parity = {}
    for method in ("h4",):
        control = root / f"{method}_control"
        assert (control / "CELL_COMPLETE").exists()
        measured = json.loads((root / method / "result.json").read_text())["outputs"]
        reference = json.loads((control / "result.json").read_text())["outputs"]
        parity[method] = [
            {
                "tokens_equal": a["token_ids"] == b["token_ids"],
                "acceptance_equal": a["spec_decode_metrics"]["per_step_accepted"]
                == b["spec_decode_metrics"]["per_step_accepted"],
                "scheduled_equal": a["spec_decode_metrics"]["per_step_drafted"]
                == b["spec_decode_metrics"]["per_step_drafted"],
            }
            for a, b in zip(measured, reference, strict=True)
        ]
    write_json(root / "instrumentation_parity.json", parity)
    if not all(all(row.values()) for rows in parity.values() for row in rows):
        (root / "INSTRUMENTATION_PARITY_FAILED").write_text(
            "Diagnostic outputs or cycle counters differ from controls\n"
        )
        raise AssertionError(
            "Instrumentation parity failed; exclude from baseline claims"
        )
    inner, outer, audits, positions = [], [], [], []
    for method in ("h4",):
        folder = root / method
        assert (folder / "CELL_COMPLETE").exists()
        result = json.loads((folder / "result.json").read_text())
        groups = defaultdict(list)
        for line, text in enumerate(
            (folder / "trace/distributions.jsonl").read_text().splitlines(), 1
        ):
            row = json.loads(text)
            row["line"] = line
            groups[row["request_id"]].append(row)
        assert len(groups) == len(result["outputs"]) == 4
        for sample, ((req, cycles), output) in enumerate(
            zip(groups.items(), result["outputs"], strict=True)
        ):
            assert int(req.split("-")[0]) == sample + 4
            assert output["prompt_sha256"] == dataset[sample]["prompt_sha256"]
            assert [r["outer_accepted"] for r in cycles] == output[
                "spec_decode_metrics"
            ]["per_step_accepted"]
            assert [r["outer_scheduled"] for r in cycles] == output[
                "spec_decode_metrics"
            ]["per_step_drafted"]
            produced = 1
            for step, cycle in enumerate(cycles, 1):
                a, scheduled = cycle["outer_accepted"], cycle["outer_scheduled"]
                assert 0 <= a <= scheduled
                common = {
                    "method": method,
                    "sample": sample,
                    "category": output["category"],
                    "request_id": req,
                    "cycle": step,
                    "trace_line": cycle["line"],
                    "outer_accepted": a,
                    "outer_scheduled": scheduled,
                    "last_cycle": step == len(cycles),
                }
                first = True
                for row in cycle["inner_rounds"]:
                    for position in range(
                        row["offset"], min(scheduled, row["offset"] + row["emitted"])
                    ):
                        origin = "accepted_draft"
                        if position == row["offset"] + row["accepted"]:
                            origin = (
                                "bonus"
                                if row["accepted"] == row["proposed"]
                                else "correction"
                            )
                        positions.append(
                            {
                                **common,
                                "position": position,
                                "origin": origin,
                                "reached": position <= a,
                                "accepted": position < a,
                                "first_rejected": position == a,
                            }
                        )
                    pair = row["draft_preverify"]
                    if pair is None:
                        continue
                    assert not pair["same_top1"]
                    inner.append(
                        {
                            **common,
                            "round": row["inner_round"] + 1,
                            "local_accepted": row["accepted"],
                            "first_rejecting_round": first,
                            **{
                                k: row[k]
                                for k in (
                                    "position",
                                    "target_reached",
                                    "correction_accepted",
                                    "suffix_scheduled",
                                    "suffix_accepted",
                                )
                            },
                            "returned_suffix_accepted": max(
                                0, min(a, 128 - produced) - row["position"] - 1
                            )
                            if row["target_reached"]
                            else None,
                            **flatten(pair),
                        }
                    )
                    first = False
                if a < scheduled:
                    pair = cycle["preverify_target"][a]
                    assert not pair["same_top1"]
                    origin = next(
                        r
                        for r in cycle["inner_rounds"]
                        if r["offset"] <= a < r["offset"] + r["emitted"]
                    )
                    outer.append(
                        {
                            **common,
                            "position": a,
                            "round": origin["inner_round"] + 1,
                            "origin": "correction_or_bonus"
                            if a == origin["offset"] + origin["accepted"]
                            else "accepted_draft",
                            **flatten(pair),
                        }
                    )
                produced += min(a + 1, 128 - produced)
            assert produced == len(output["token_ids"]) == 128
        audits.append(
            {
                "method": method,
                "requests": len(groups),
                "cycles": sum(map(len, groups.values())),
                "trace_sha256": digest(folder / "trace/distributions.jsonl"),
            }
        )
    summary = []
    for method in ("h4",):
        for scope in ("all_rounds", "first_rejecting_round"):
            for group in ("near", "middle", "far"):
                rows = [
                    r
                    for r in inner
                    if r["method"] == method
                    and r["class"] == group
                    and (scope == "all_rounds" or r["first_rejecting_round"])
                ]
                summary.append(
                    {
                        "method": method,
                        "scope": scope,
                        "class": group,
                        **summarize(rows),
                    }
                )
    sensitivity = []
    for threshold in (0.25, 0.5, 1.0):
        for method in ("h4",):
            rows = [
                r
                for r in inner
                if r["method"] == method
                and all(
                    r[f"{s}_other_top1_rank"] <= 2
                    and r[f"{s}_other_top1_logit_gap"] <= threshold
                    for s in ("left", "right")
                )
            ]
            sensitivity.append(
                {"method": method, "near_gap": threshold, **summarize(rows)}
            )
    rank_summary = []
    for mutual in (True, False):
        rows = [
            r
            for r in inner
            if (r["left_other_top1_rank"] <= 2 and r["right_other_top1_rank"] <= 2)
            == mutual
        ]
        rank_summary.append({"mutual_top2_rank": mutual, **summarize(rows)})
    position_summary = []
    for origin in ("accepted_draft", "correction", "bonus"):
        rows = [r for r in positions if r["origin"] == origin]
        position_summary.append(
            {
                "origin": origin,
                "scheduled": len(rows),
                "reached": sum(r["reached"] for r in rows),
                "accepted": sum(r["accepted"] for r in rows),
                "first_rejected": sum(r["first_rejected"] for r in rows),
            }
        )
    assert sum(r["first_rejected"] for r in positions) == len(outer)
    for name, rows in (
        ("inner_rejections", inner),
        ("outer_rejections", outer),
        ("inner_summary", summary),
        ("near_sensitivity", sensitivity),
        ("rank_summary", rank_summary),
        ("position_summary", position_summary),
    ):
        write_csv(root / f"{name}.csv", rows)
        write_json(root / f"{name}.json", rows)
    write_json(root / "audit.json", {"status": "passed", "cells": audits})
    print(
        json.dumps(
            {
                "summary": summary,
                "outer": {
                    m: {
                        c: sum(r["method"] == m and r["class"] == c for r in outer)
                        for c in ("near", "middle", "far")
                    }
                    for m in ("h4",)
                },
            },
            indent=2,
        )
    )
    (root / "ANALYSIS_COMPLETE").write_text(
        "paired distribution counters audited; see report for limitations\n"
    )


if __name__ == "__main__":
    main()
