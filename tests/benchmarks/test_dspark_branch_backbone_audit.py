# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from benchmarks.replayssm.audit_dspark_branch_backbone import audit, prefix_proposals


def test_prefix_pairing_uses_committed_tokens():
    rows = [
        {"accepted_draft_tokens": [], "recovery_token": 10},
        {"accepted_draft_tokens": [11, 12], "recovery_token": 13},
    ]
    assert list(prefix_proposals(rows)) == [(10,), (10, 11, 12, 13)]


def test_target_difference_is_failed_even_with_missing_async_cells(tmp_path):
    for mode, token in [("ar", 1), ("sync", 2)]:
        cell = tmp_path / "cells" / f"correctness_{mode}_eager_b1"
        cell.mkdir(parents=True)
        (cell / "cell_complete.json").write_text("{}")
        requests = [{"token_ids": [token], "top_logprobs": [{"a": 0.0}]}] * 4
        (cell / "requests.json").write_text(json.dumps(requests))
        if mode == "sync":
            rows = [
                {"request_id": "formal-000", "accepted_draft_count": 0},
                {"request_id": "formal-000", "accepted_draft_count": 2},
            ]
            (cell / "proposals.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
    result = audit(tmp_path)
    assert result["status"] == "failed"
    assert result["missing"] == ["async_jit", "async_cache"]
    assert result["acceptance"]["sync"]["verify_rounds"] == 1
    assert result["acceptance"]["sync"]["accepted_draft_tokens_per_round"] == 2


def test_identical_outputs_do_not_waive_proposal_difference(tmp_path):
    for mode in ["ar", "sync", "async_jit", "async_cache"]:
        cell = tmp_path / "cells" / f"correctness_{mode}_eager_b1"
        cell.mkdir(parents=True)
        (cell / "cell_complete.json").write_text("{}")
        requests = [{"token_ids": [10, 11, 12], "top_logprobs": [{}] * 3}] * 4
        (cell / "requests.json").write_text(json.dumps(requests))
        if mode == "ar":
            continue
        rows = []
        for i in range(4):
            for count, accepted, recovery in [(0, [], 10), (1, [11], 12)]:
                rows.append(
                    {
                        "request_id": f"formal-{i:03d}",
                        "accepted_draft_count": count,
                        "accepted_draft_tokens": accepted,
                        "recovery_token": recovery,
                        "draft_tokens": [11, 99 if mode == "async_jit" else 12, 13],
                    }
                )
        (cell / "proposals.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    result = audit(tmp_path)
    assert all(x["passed"] for x in result["target_parity"].values())
    assert result["status"] == "failed"
    assert result["forced_jit_vs_sync"]["paired_real_prefixes"] == 8
    assert result["forced_jit_vs_sync"]["equal_proposals"] == 0
