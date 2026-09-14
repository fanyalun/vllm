# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from benchmarks.replayssm.async_ssd_dspark_fanout import (
    parse_fan_outs,
    select_fan_out,
    timing_summary,
)


def test_parse_fan_outs_sorts_and_deduplicates() -> None:
    assert parse_fan_outs("12,3,6,3") == (3, 6, 12)


def test_timing_summary_pairs_window_with_previous_branch(tmp_path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    rows = [
        {
            "request_id": "ssd-performance-formal-000",
            "accepted_draft_count": 2,
            "cache_hit": False,
            "async_timing_pair_valid": False,
            "async_fan_out": 6,
            "async_batch_num_reqs": 1,
        },
        {
            "request_id": "ssd-performance-formal-000",
            "accepted_draft_count": 3,
            "cache_hit": True,
            "async_timing_pair_valid": True,
            "async_fan_out": 6,
            "async_batch_num_reqs": 1,
            "async_verify_window_seconds": 0.08,
            "async_previous_branch_build_seconds": 0.04,
            "async_next_proposal_wait_seconds": 0.003,
        },
        {
            "request_id": "warmup-performance-0",
            "accepted_draft_count": 1,
            "cache_hit": True,
            "async_timing_pair_valid": True,
            "async_fan_out": 6,
            "async_batch_num_reqs": 1,
            "async_verify_window_seconds": 9.0,
            "async_previous_branch_build_seconds": 9.0,
            "async_next_proposal_wait_seconds": 9.0,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "summary": {"completion_throughput_tok_s": 31.5},
                "metrics_delta": {
                    'vllm:async_draft_dspark_markov_branches_total{engine="0"}': 24,
                    'vllm:async_draft_dspark_backbone_refreshes_total{engine="0"}': 1,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = timing_summary(trace_path, result_path, verify_width=3, fan_out=6)

    assert summary["paired_timing_rounds"] == 1
    assert summary["verify_window_seconds"]["p05"] == 0.08
    assert summary["branch_build_seconds"]["p95"] == 0.04
    assert summary["fully_hidden_round_fraction"] == 1.0
    assert summary["sustainable_window_fit"] is True
    assert summary["cache"]["hit_rate"] == 1.0
    assert summary["runtime_branch_audit"]["passed"] is True


def test_select_fan_out_prefers_smallest_near_best_safe_cell() -> None:
    rows = [
        {
            "status": "complete",
            "fan_out": 6,
            "branches_per_round": 24,
            "completion_throughput_tok_s": 30.0,
            "conservative_window_fit": True,
            "sustainable_window_fit": True,
            "runtime_branch_audit": {"passed": True},
        },
        {
            "status": "complete",
            "fan_out": 12,
            "branches_per_round": 48,
            "completion_throughput_tok_s": 30.2,
            "conservative_window_fit": True,
            "sustainable_window_fit": True,
            "runtime_branch_audit": {"passed": True},
        },
        {
            "status": "complete",
            "fan_out": 24,
            "branches_per_round": 96,
            "completion_throughput_tok_s": 29.0,
            "conservative_window_fit": False,
            "sustainable_window_fit": False,
            "runtime_branch_audit": {"passed": True},
        },
    ]

    selection = select_fan_out(rows)

    assert selection["selected_fan_out"] == 6
    assert selection["largest_conservatively_hidden_fan_out"] == 12
