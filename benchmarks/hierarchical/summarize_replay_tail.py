# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit the requested replay-tail matrix, counts, and AR token comparisons."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def common_prefix(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b),
        len(left),
    )


def summarize(root):
    ar = json.loads((root / "ar.json").read_text())
    ar_tokens = {row["sample_index"]: row["token_ids"] for row in ar["outputs"]}
    summary, comparisons = [], []
    for method in ("mtp", "dspark"):
        folder = root / method
        marker = json.loads((folder / "measurement_complete.json").read_text())
        contract = json.loads((folder / "contract.json").read_text())
        memory = json.loads((folder / "private_state.json").read_text())
        rows = json.loads((folder / "results.json").read_text())
        n, length, repeats = (
            contract["samples"],
            contract["max_tokens"],
            contract["repeats"],
        )
        assert len(rows) == marker["expected"] == len(n) * (2 * repeats + 4)
        keys = {(r["case"], r["phase"], r["repeat"], r["sample"]) for r in rows}
        expected = {
            (c, "e2e", r, s)
            for c in ("none", "replay_tail")
            for r in range(repeats)
            for s in range(len(n))
        }
        expected |= {
            (c, "audit", 0, s) for c in contract["cases"] for s in range(len(n))
        }
        assert keys == expected and len(keys) == len(rows)
        assert all(len(r["token_ids"]) == length for r in rows)
        for case in contract["cases"]:
            audit = [r for r in rows if r["case"] == case and r["phase"] == "audit"]
            cycles = [c for r in audit for c in r["cycles"]]
            assert cycles
            effective = []
            for row in audit:
                remaining = length - 1
                for cycle in row["cycles"]:
                    emitted = min(remaining, cycle["emitted"])
                    effective.append(emitted)
                    remaining -= emitted
                    assert 0 <= cycle["emitted"] - 1 <= cycle["scheduled"]
                assert remaining == 0
                index = row["sample"]
                reference = ar_tokens[index]
                assert len(reference) == length
                comparisons.append(
                    dict(
                        method=method,
                        case=case,
                        sample=index,
                        ar_common_prefix=common_prefix(reference, row["token_ids"]),
                        ar_equal=reference == row["token_ids"],
                    )
                )
            inner = [i for c in cycles for i in c["inner"]]
            times = [
                sum(
                    r["seconds"]
                    for r in rows
                    if r["case"] == case
                    and r["phase"] == "e2e"
                    and r["repeat"] == repeat
                )
                for repeat in range(repeats)
            ]
            timed = case in ("none", "replay_tail")
            tps = [len(n) * length / seconds for seconds in times] if timed else []
            spans = [
                s["ms"] for r in audit for s in r["spans"] if s["phase"] == "preverify"
            ]
            summary.append(
                dict(
                    method=method,
                    case=case,
                    median_tokens_per_second=statistics.median(tps) if tps else None,
                    trials_tokens_per_second=tps,
                    cycles=len(cycles),
                    returned_tokens_per_cycle=sum(effective) / len(cycles),
                    outer_scheduled=sum(c["scheduled"] for c in cycles),
                    outer_accepted=sum(c["emitted"] - 1 for c in cycles),
                    inner_proposed=sum(i["proposed"] for i in inner),
                    inner_accepted=sum(i["accepted"] for i in inner),
                    mean_cycle_ms=statistics.mean(c["ms"] for c in cycles),
                    mean_preverify_ms=statistics.mean(spans),
                    ssm_bytes=memory[case]["ssm_bytes"],
                )
            )
    result = {
        "summary": summary,
        "ar_comparisons": comparisons,
        "all_ar_equal": all(c["ar_equal"] for c in comparisons),
        "measurement_complete": True,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
