# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure low-margin detection of first rejection and correction-token ranks."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()
    directory = args.root / "rejection_margin"
    directory.mkdir(exist_ok=True)
    detection, ranks, first_rows, audit = [], [], [], {}
    for model, trace_name in (("qwen36", "qwen_trace"), ("gemma4", "gemma_trace")):
        cell = args.root / model
        assert (cell / "RUN_COMPLETE").exists()
        result = json.loads((cell / "result.json").read_text())
        outputs = {str(x["sample_index"]): x for x in result["outputs"]}
        assert len(outputs) == 4
        assert all(len(x["token_ids"]) == 256 for x in outputs.values())
        path = args.root / trace_name / "raw_trace.jsonl"
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["valid_mask"]
            groups[row["request_id"], row["verify_step"]].append(row)
        counts = defaultdict(Counter)
        rank_counts = defaultdict(Counter)
        offsets = dict.fromkeys(outputs, 1)
        next_steps = dict.fromkeys(outputs, 0)
        excluded = 0
        for (req, step), rows in sorted(groups.items()):
            assert step == next_steps[req]
            next_steps[req] += 1
            rows.sort(key=lambda x: x["draft_position"])
            assert [x["draft_position"] for x in rows] == list(range(1, len(rows) + 1))
            accepted = rows[0]["accepted_draft_tokens"]
            assert 0 <= accepted <= len(rows)
            for row in rows:
                assert row["accepted_draft_tokens"] == accepted
                position = row["draft_position"]
                output_index = offsets[req] + position - 1
                if output_index >= 256:
                    excluded += 1
                    continue
                top1 = row["draft_argmax_token_id"]
                top2 = row["draft_runner_up_token_id"]
                assert top1 != top2
                target = row["target_top1_token_id"]
                mismatch = target != top1
                first = position == accepted + 1
                reached = position <= accepted + 1
                low = row["draft_top1_minus_top2"] < args.threshold
                scopes = [("all_positions_first_rejection", first)]
                scopes.append(("all_positions_local_mismatch", mismatch))
                if reached:
                    assert mismatch == first
                    assert outputs[req]["token_ids"][output_index] == target
                    scopes.append(("reached_positions_first_rejection", first))
                for scope, truth in scopes:
                    label = (
                        ("tp" if truth else "fp") if low else ("fn" if truth else "tn")
                    )
                    counts[scope][label] += 1
                if first:
                    rank = (
                        "top1"
                        if target == top1
                        else ("top2" if target == top2 else "neither")
                    )
                    rank_counts["all_first_rejections"][rank] += 1
                    if low:
                        rank_counts["low_margin_first_rejections"][rank] += 1
                    first_rows.append(
                        {
                            "model": model,
                            "sample_index": req,
                            "category": outputs[req]["category"],
                            "verify_step": step,
                            "draft_position": position,
                            "output_position": output_index + 1,
                            "margin": row["draft_top1_minus_top2"],
                            "low_margin": low,
                            "draft_top1": top1,
                            "draft_top2": top2,
                            "target_correction": target,
                            "correction_rank": rank,
                        }
                    )
            offsets[req] += accepted + 1
        for scope, count in counts.items():
            tp, fp, fn, tn = (count[k] for k in ("tp", "fp", "fn", "tn"))
            detection.append(
                {
                    "model": model,
                    "scope": scope,
                    "threshold": args.threshold,
                    "positions": tp + fp + fn + tn,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "predicted_positive": tp + fp,
                    "actual_positive": tp + fn,
                    "precision": ratio(tp, tp + fp),
                    "recall": ratio(tp, tp + fn),
                }
            )
        for scope, count in rank_counts.items():
            total = sum(count.values())
            ranks.append(
                {
                    "model": model,
                    "scope": scope,
                    "first_rejections": total,
                    **{rank: count[rank] for rank in ("top1", "top2", "neither")},
                    **{
                        f"{rank}_probability": ratio(count[rank], total)
                        for rank in ("top1", "top2", "neither")
                    },
                }
            )
        audit[model] = {
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trace_rows": sum(map(len, groups.values())),
            "excluded_beyond_output_budget": excluded,
            "verification_rounds": len(groups),
            "accepted_and_correction_tokens_match_actual_output": True,
            "first_rejection_equals_first_greedy_mismatch": True,
            "top1_uses_actual_argmax_not_unstable_topk_tie_order": True,
        }
    write_csv(directory / "detection.csv", detection)
    write_csv(directory / "correction_ranks.csv", ranks)
    write_csv(directory / "first_rejections.csv", first_rows)
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (directory / "RUN_COMPLETE").write_text(
        f"Two models; {len(first_rows)} first rejections audited against output\n"
    )
    print(json.dumps({"detection": detection, "ranks": ranks}, indent=2))


if __name__ == "__main__":
    main()
