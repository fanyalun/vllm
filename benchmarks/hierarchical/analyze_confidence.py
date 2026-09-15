# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate h4 correction-confidence hypotheses on independent prompts."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def margin_bin(value):
    for lower, upper in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)):
        if value < upper:
            return f"[{lower},{upper})"
    return "[4,inf)"


def auc(rows):
    positive = np.array([r["margin"] for r in rows if r["accepted"]])
    negative = np.array([r["margin"] for r in rows if not r["accepted"]])
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative
    return float(((differences > 0) + 0.5 * (differences == 0)).mean())


def rate_summary(rows):
    n = len(rows)
    k = sum(r["accepted"] for r in rows)
    if not n:
        return {"n": 0, "accepted": 0, "rate": None, "wilson95": None}
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "n": n,
        "accepted": k,
        "rate": p,
        "wilson95": [center - half, center + half],
    }


def cluster_interval(rows, requests, repetitions=2000):
    counts = np.array(
        [
            [
                sum(r["accepted"] for r in rows if r["request_id"] == request),
                sum(r["request_id"] == request for r in rows),
            ]
            for request in requests
        ]
    )
    rng = np.random.default_rng(0)
    sampled = counts[rng.integers(0, len(requests), (repetitions, len(requests)))].sum(
        1
    )
    valid = sampled[:, 1] > 0
    if not valid.any():
        return None
    return np.quantile(sampled[valid, 0] / sampled[valid, 1], [0.025, 0.975]).tolist()


def policy_trigger(rounds, policy):
    for row in rounds[:-1]:
        correction = row["accepted"] < row["proposed"]
        hit = policy == "inner_accept_le1" and row["accepted"] <= 1
        hit |= policy == "any_correction" and correction
        if policy.startswith("correction_margin_lt"):
            threshold = float(policy.removeprefix("correction_margin_lt"))
            if correction:
                margin = (
                    row["correction_margin"]
                    if "correction_margin" in row
                    else row["draft_preverify"]["right"]["margin"]
                )
                hit |= margin < threshold
        if hit:
            return row
    return None


def audit_capture(root):
    contract = json.loads((root / "contract.json").read_text())
    assert (
        hashlib.sha256((root / "dataset.jsonl").read_bytes()).hexdigest()
        == contract["dataset_sha256"]
    )
    for name, fingerprint in contract["source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == fingerprint
    for model in ("model", "assistant"):
        config_path = Path(contract[model]) / "config.json"
        assert (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            == contract[f"{model}_config_sha256"]
        )
    measured = json.loads((root / "h4/result.json").read_text())["outputs"]
    parity = {}
    for control_name in ("h4_control", "h4_control_previous"):
        if not (root / control_name).exists():
            continue
        reference = json.loads((root / control_name / "result.json").read_text())[
            "outputs"
        ]
        parity[control_name] = [
            {
                "tokens_equal": a["token_ids"] == b["token_ids"],
                "acceptance_equal": a["spec_decode_metrics"]["per_step_accepted"]
                == b["spec_decode_metrics"]["per_step_accepted"],
                "scheduled_equal": a["spec_decode_metrics"]["per_step_drafted"]
                == b["spec_decode_metrics"]["per_step_drafted"],
            }
            for a, b in zip(measured, reference, strict=True)
        ]
    write_json(root / "control_comparisons.json", parity)
    write_json(root / "instrumentation_parity.json", {"h4": parity["h4_control"]})
    if not all(all(row.values()) for row in parity["h4_control"]):
        (root / "INSTRUMENTATION_PARITY_FAILED").write_text(
            "Confidence capture differs from control\n"
        )
        raise AssertionError("Confidence capture parity failed")
    dataset = [
        json.loads(line) for line in (root / "dataset.jsonl").read_text().splitlines()
    ]
    assert len(dataset) == len(measured) == contract["measured_requests"]
    assert all(
        a["prompt_sha256"] == b["prompt_sha256"]
        for a, b in zip(dataset, measured, strict=True)
    )
    (root / "CONFIDENCE_TRACE_AUDIT_PASSED").write_text(
        "Input identity and uninstrumented parity passed\n"
    )


def load_events(root, split):
    contract = json.loads((root / "contract.json").read_text())
    assert (root / "ANALYSIS_COMPLETE").exists() or (
        root / "CONFIDENCE_TRACE_AUDIT_PASSED"
    ).exists()
    parity = json.loads((root / "instrumentation_parity.json").read_text())
    assert all(all(r.values()) for r in parity["h4"])
    outputs = json.loads((root / "h4/result.json").read_text())["outputs"]
    groups = defaultdict(list)
    for line_number, line in enumerate(
        (root / "h4/trace/distributions.jsonl").read_text().splitlines(), 1
    ):
        cycle = json.loads(line)
        cycle["trace_line"] = line_number
        groups[cycle["request_id"]].append(cycle)
    assert len(groups) == len(outputs)
    dataset = [
        json.loads(line) for line in (root / "dataset.jsonl").read_text().splitlines()
    ]
    events, cycles = [], []
    for sample, ((request, request_cycles), output) in enumerate(
        zip(groups.items(), outputs, strict=True)
    ):
        assert int(request.split("-")[0]) == sample + contract.get(
            "warmup_requests", len(dataset)
        )
        assert output["prompt_sha256"] == dataset[sample]["prompt_sha256"]
        assert [r["outer_accepted"] for r in request_cycles] == output[
            "spec_decode_metrics"
        ]["per_step_accepted"]
        assert [r["outer_scheduled"] for r in request_cycles] == output[
            "spec_decode_metrics"
        ]["per_step_drafted"]
        produced = 1
        for index, cycle in enumerate(request_cycles):
            a, length = cycle["outer_accepted"], cycle["outer_scheduled"]
            assert 0 <= a <= length == len(cycle["candidate_tokens"])
            offset = 0
            for row in cycle["inner_rounds"]:
                assert row["offset"] == offset
                assert row["emitted"] == row["accepted"] + 1
                offset += row["emitted"]
            assert length <= offset
            common = {
                "split": split,
                "request_id": request,
                "sample": sample,
                "category": output["category"],
                "cycle": index + 1,
                "trace_line": cycle["trace_line"],
                "outer_accepted": a,
                "outer_scheduled": length,
                "remaining_output": 128 - produced,
            }
            cycles.append({**common, "rounds": cycle["inner_rounds"]})
            for row in cycle["inner_rounds"]:
                correction_position = row["offset"] + row["accepted"]
                for position in range(
                    row["offset"], min(length, correction_position + 1)
                ):
                    origin = "accepted_draft"
                    if position == correction_position:
                        origin = (
                            "correction"
                            if row["accepted"] < row["proposed"]
                            else "bonus"
                        )
                    pair = cycle["preverify_target"][position]
                    pv, target = pair["left"], pair["right"]
                    reached = position <= a
                    selected = cycle["candidate_tokens"][position]
                    target_ties = [
                        token
                        for token, logit in zip(
                            target["top_tokens"], target["top_logits"], strict=True
                        )
                        if logit == target["top_logits"][0]
                    ]
                    target_token = min(target_ties) if len(target_ties) < 8 else None
                    suffix = max(0, length - position - 1)
                    events.append(
                        {
                            **common,
                            "round": row["inner_round"] + 1,
                            "position": position,
                            "origin": origin,
                            "reached": reached,
                            "accepted": position < a if reached else None,
                            "first_rejected": position == a,
                            "margin": pv["margin"],
                            "margin_bin": margin_bin(pv["margin"]),
                            "p1": pv["top_probs"][0],
                            "entropy": pv["entropy"],
                            "pv_token": selected,
                            "target_token": target_token,
                            "target_prob_of_candidate": target["other_top1_prob"],
                            "target_rank_of_candidate": target["other_top1_rank"],
                            "target_margin": target["margin"],
                            "suffix_scheduled": suffix,
                            "suffix_accepted": max(0, a - position - 1)
                            if reached
                            else None,
                            "returned_suffix_accepted": max(
                                0, min(a, 128 - produced) - position - 1
                            )
                            if reached
                            else None,
                            "inner_accepted": row["accepted"],
                        }
                    )
            produced += min(a + 1, 128 - produced)
        assert produced == len(output["token_ids"]) == 128
    return events, cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("heldout", type=Path)
    args = parser.parse_args()
    pilot_ids = {
        json.loads(line)["prompt_sha256"]
        for line in (args.pilot / "dataset.jsonl").read_text().splitlines()
    }
    heldout_ids = {
        json.loads(line)["prompt_sha256"]
        for line in (args.heldout / "dataset.jsonl").read_text().splitlines()
    }
    assert pilot_ids.isdisjoint(heldout_ids)
    hypotheses = json.loads((args.heldout / "hypotheses.json").read_text())
    assert (
        hypotheses["dataset_sha256"]
        == hashlib.sha256((args.heldout / "dataset.jsonl").read_bytes()).hexdigest()
    )
    assert (
        hypotheses["pilot_sha256"]
        == hashlib.sha256((args.pilot / "dataset.jsonl").read_bytes()).hexdigest()
    )
    assert hypotheses["low_margin_strictly_less_than"] == 0.5
    assert hypotheses["high_margin_at_least"] == 2
    audit_capture(args.heldout)
    all_events, groups, decisions, origins, scores, policies = [], [], [], [], [], []
    names = ["inner_accept_le1", "any_correction"] + [
        f"correction_margin_lt{x}" for x in (0.25, 0.5, 1, 2)
    ]
    for split, root in (("pilot", args.pilot), ("heldout", args.heldout)):
        events, cycles = load_events(root, split)
        all_events.extend(events)
        requests = sorted({r["request_id"] for r in events})
        reached = [r for r in events if r["reached"]]
        corrections = [r for r in reached if r["origin"] == "correction"]
        for origin in ("accepted_draft", "correction", "bonus"):
            rows = [r for r in reached if r["origin"] == origin]
            origins.append(
                {
                    "split": split,
                    "origin": origin,
                    **rate_summary(rows),
                    "first_rejected": sum(r["first_rejected"] for r in rows),
                    "request_bootstrap95": cluster_interval(rows, requests),
                }
            )
        for label in (
            "all",
            "low_lt0.5",
            "high_ge2",
            "[0,0.25)",
            "[0.25,0.5)",
            "[0.5,1)",
            "[1,2)",
            "[2,4)",
            "[4,inf)",
        ):
            rows = [
                r
                for r in corrections
                if label == "all"
                or label == "low_lt0.5"
                and r["margin"] < 0.5
                or label == "high_ge2"
                and r["margin"] >= 2
                or label == r["margin_bin"]
            ]
            suffix_rows = [r for r in rows if r["suffix_scheduled"] > 0]
            groups.append(
                {
                    "split": split,
                    "group": label,
                    **rate_summary(rows),
                    "request_bootstrap95": cluster_interval(rows, requests),
                    "suffix_events": len(suffix_rows),
                    "suffix_full": sum(
                        r["suffix_accepted"] == r["suffix_scheduled"]
                        for r in suffix_rows
                    ),
                    "suffix_zero": sum(r["suffix_accepted"] == 0 for r in suffix_rows),
                }
            )
        for category in sorted({r["category"] for r in reached}):
            for label in ("low_lt0.5", "high_ge2"):
                rows = [
                    r
                    for r in corrections
                    if r["category"] == category
                    and (
                        r["margin"] < 0.5 if label == "low_lt0.5" else r["margin"] >= 2
                    )
                ]
                decisions.append(
                    {
                        "split": split,
                        "category": category,
                        "group": label,
                        **rate_summary(rows),
                    }
                )
        scores.append(
            {
                "split": split,
                "requests": len(requests),
                "cycles": len(cycles),
                "correction_auc": auc(corrections),
                "p1_auc": auc([{**r, "margin": r["p1"]} for r in corrections]),
                "negative_entropy_auc": auc(
                    [{**r, "margin": -r["entropy"]} for r in corrections]
                ),
                "reached_corrections": len(corrections),
                "unreached_corrections_excluded": sum(
                    r["origin"] == "correction" and not r["reached"] for r in events
                ),
            }
        )
        for policy in names:
            triggered = []
            for cycle in cycles:
                row = policy_trigger(cycle["rounds"], policy)
                if row is None:
                    continue
                boundary = row["offset"] + row["emitted"]
                suffix = max(0, cycle["outer_accepted"] - boundary)
                returned = max(
                    0,
                    min(cycle["outer_accepted"], cycle["remaining_output"]) - boundary,
                )
                triggered.append(
                    {
                        "request_id": cycle["request_id"],
                        "cycle": cycle["cycle"],
                        "round": row["inner_round"] + 1,
                        "skipped_rounds": len(cycle["rounds"]) - row["inner_round"] - 1,
                        "accepted_suffix_cut": suffix,
                        "returned_suffix_cut": returned,
                        "correction_reached": cycle["outer_accepted"] >= boundary - 1,
                    }
                )
            policies.append(
                {
                    "split": split,
                    "policy": policy,
                    "cycles": len(cycles),
                    "triggers": len(triggered),
                    "skipped_rounds": sum(r["skipped_rounds"] for r in triggered),
                    "zero_suffix_triggers": sum(
                        r["accepted_suffix_cut"] == 0 for r in triggered
                    ),
                    "accepted_suffix_cut": sum(
                        r["accepted_suffix_cut"] for r in triggered
                    ),
                    "returned_suffix_cut": sum(
                        r["returned_suffix_cut"] for r in triggered
                    ),
                    "details": triggered,
                }
            )
    out = args.heldout / "confidence_analysis"
    out.mkdir(exist_ok=True)
    for name, rows in (
        ("events", all_events),
        ("margin_groups", groups),
        ("category_groups", decisions),
        ("origin_groups", origins),
        ("score_summary", scores),
        ("stop_policies", policies),
    ):
        write_json(out / f"{name}.json", rows)
        write_csv(
            out / f"{name}.csv",
            [{k: v for k, v in r.items() if k != "details"} for r in rows],
        )
    write_json(
        out / "audit.json",
        {
            "pilot_requests": len(pilot_ids),
            "heldout_requests": len(heldout_ids),
            "overlap": 0,
            "status": "passed",
            "thresholds_fixed_before_heldout": True,
            "bootstrap": "2000 request-cluster resamples, seed0; descriptive interval",
            "policy_limit": "Offline suffix cut; no speed or future trajectory claim",
        },
    )
    source = Path(__file__)
    (out / source.name).write_bytes(source.read_bytes())
    write_json(
        out / "source_sha256.json",
        {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
            "pilot_trace": hashlib.sha256(
                (args.pilot / "h4/trace/distributions.jsonl").read_bytes()
            ).hexdigest(),
            "heldout_trace": hashlib.sha256(
                (args.heldout / "h4/trace/distributions.jsonl").read_bytes()
            ).hexdigest(),
        },
    )
    print(
        json.dumps(
            {
                "scores": scores,
                "groups": [
                    r for r in groups if r["group"] in ("all", "low_lt0.5", "high_ge2")
                ],
                "origins": origins,
            },
            indent=2,
        )
    )
    (out / "ANALYSIS_COMPLETE").write_text(
        "Disjoint pilot/heldout inputs, paired outputs and counters audited\n"
    )


if __name__ == "__main__":
    main()
