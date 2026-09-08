# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict token and real-prefix audits for the DSpark branch-backbone smoke."""

import argparse
import json
import re
from pathlib import Path


def read_trace(path):
    result = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        match = re.search(r"formal-(\d+)", record["request_id"])
        if match:
            result.setdefault(int(match[1]), []).append(record)
    return result


def prefix_proposals(records):
    prefix = []
    proposals = {}
    for record in records:
        prefix.extend(record["accepted_draft_tokens"])
        prefix.append(record["recovery_token"])
        proposals[tuple(prefix)] = record
    return proposals


def audit_shadow(root):
    trace = read_trace(root / "cells/correctness_async_cache_eager_b1/proposals.jsonl")
    cache_rows = {
        (r["request_id"], r["async_generation"]): r
        for records in trace.values()
        for r in records
    }
    rows = [
        json.loads(line) for line in (root / "shadow.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if "formal-" in r["request_id"]]
    equal = positions = 0
    first = None
    for row in rows:
        cached = cache_rows[(row["request_id"], row["generation"])]
        assert cached["draft_tokens"] == row["cache_tokens"]
        equal += row["cache_tokens"] == row["fresh_jit_tokens"]
        positions += sum(
            a == b for a, b in zip(row["cache_tokens"], row["fresh_jit_tokens"])
        )
        if row["first_difference"] is not None and first is None:
            first = {**row, "cache_trace": cached}
    return {
        "scope": "separate provisional cache diagnostic on real prefixes",
        "paired_proposals": len(rows),
        "equal_proposals": equal,
        "equal_token_positions": positions,
        "total_token_positions": sum(len(r["cache_tokens"]) for r in rows),
        "first_difference": first,
    }


def audit(root):
    cells = root / "cells"
    prompt_path = root / "prompts.json"
    prompt_manifest = (
        json.loads(prompt_path.read_text()) if prompt_path.exists() else []
    )
    modes = ("ar", "sync", "async_jit", "async_cache")
    outputs = {}
    traces = {}
    report = {"status": "pending", "missing": [], "target_parity": {}, "acceptance": {}}
    for mode in modes:
        cell = cells / f"correctness_{mode}_eager_b1"
        if not (cell / "cell_complete.json").exists():
            report["missing"].append(mode)
            continue
        outputs[mode] = json.loads((cell / "requests.json").read_text())
        if mode != "ar":
            traces[mode] = read_trace(cell / "proposals.jsonl")
            rows = [r for records in traces[mode].values() for r in records[1:]]
            accepted = sum(r["accepted_draft_count"] for r in rows)
            report["acceptance"][mode] = {
                "scope": "smoke diagnostics, excludes each request bootstrap",
                "verify_rounds": len(rows),
                "accepted_draft_tokens": accepted,
                "accepted_draft_tokens_per_round": accepted / len(rows),
                "tokens_per_round_including_bonus": 1 + accepted / len(rows),
            }
    if "ar" in outputs:
        for mode, requests in outputs.items():
            if mode == "ar":
                continue
            differences = []
            for index, (ar, candidate) in enumerate(zip(outputs["ar"], requests)):
                left, right = ar["token_ids"], candidate["token_ids"]
                if left != right:
                    position = next(
                        (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                        min(len(left), len(right)),
                    )
                    differences.append(
                        {
                            "sample_order": index,
                            "prompt_index": (
                                prompt_manifest[index]["prompt_index"]
                                if prompt_manifest
                                else index
                            ),
                            "position": position,
                            "ar_token": left[position]
                            if position < len(left)
                            else None,
                            "candidate_token": right[position]
                            if position < len(right)
                            else None,
                            "ar_top_logprobs": ar["top_logprobs"][position],
                            "candidate_top_logprobs": candidate["top_logprobs"][
                                position
                            ],
                        }
                    )
            report["target_parity"][mode] = {
                "passed": not differences and len(requests) == 4,
                "differences": differences,
            }
    if "sync" in traces and "async_jit" in traces:
        paired = equal = 0
        first = None
        first_outcome = None
        outcomes_equal = True
        for index, records in traces["sync"].items():
            other = traces["async_jit"].get(index, [])
            fields = ("accepted_draft_count", "accepted_draft_tokens", "recovery_token")
            outcomes_equal &= [[r[k] for k in fields] for r in records] == [
                [r[k] for k in fields] for r in other
            ]
            if first_outcome is None:
                for step, (a, b) in enumerate(zip(records, other)):
                    if any(a[k] != b[k] for k in fields):
                        first_outcome = {
                            "prompt_index": index,
                            "round": step,
                            "sync": a,
                            "forced_jit": b,
                        }
                        break
            baseline = prefix_proposals(records)
            candidate = prefix_proposals(other)
            for prefix in baseline.keys() & candidate.keys():
                paired += 1
                a, b = baseline[prefix], candidate[prefix]
                if a["draft_tokens"] == b["draft_tokens"]:
                    equal += 1
                elif first is None or (index, len(prefix)) < (
                    first["prompt_index"],
                    first["prefix_length"],
                ):
                    first = {
                        "prompt_index": index,
                        "prefix_length": len(prefix),
                        "sync": a,
                        "forced_jit": b,
                    }
        report["forced_jit_vs_sync"] = {
            "paired_real_prefixes": paired,
            "equal_proposals": equal,
            "accepted_outcomes_equal": outcomes_equal,
            "first_proposal_difference": first,
            "first_accepted_outcome_difference": first_outcome,
            "passed": paired > 0 and equal == paired and outcomes_equal,
        }
    if any(not x["passed"] for x in report["target_parity"].values()):
        report["status"] = "failed"
    elif not report["missing"]:
        report["status"] = (
            "passed"
            if all(x["passed"] for x in report["target_parity"].values())
            and report["forced_jit_vs_sync"]["passed"]
            else "failed"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--shadow-root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.shadow_root:
        result["shadow"] = audit_shadow(args.shadow_root)
    (args.root / "correctness_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
