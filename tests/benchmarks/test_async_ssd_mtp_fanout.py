# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from benchmarks.replayssm.async_ssd_mtp_fanout import (
    parse_fan_outs,
    select_fan_out,
    timing_summary,
)


def test_parse_fan_outs_sorts_and_deduplicates() -> None:
    assert parse_fan_outs("12,3,6,3") == (3, 6, 12)


def test_select_fan_out_uses_largest_sustainable_cell() -> None:
    rows = [
        {
            "status": "complete",
            "fan_out": fan_out,
            "branches_per_round": 4 * fan_out,
            "sustainable_window_fit": safe,
            "completion_throughput_tok_s": 20.0,
            "overlap_coverage": {"p95_build_over_p05_window": coverage},
            "runtime_branch_audit": {"passed": True},
        }
        for fan_out, safe, coverage in (
            (3, True, 0.4),
            (6, True, 0.8),
            (12, False, 1.2),
        )
    ]

    selection = select_fan_out(rows)

    assert selection["selected_fan_out"] == 6
    assert selection["selected_branches_per_round"] == 24


def test_timing_summary_audits_dynamic_branch_count(tmp_path: Path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    records = [
        {
            "request_id": "qwen-formal-0",
            "accepted_draft_count": 1,
            "cache_hit": False,
        },
        {
            "request_id": "qwen-formal-0",
            "accepted_draft_count": 2,
            "cache_hit": True,
            "async_timing_pair_valid": True,
            "async_fan_out": 6,
            "async_batch_num_reqs": 1,
            "async_verify_window_seconds": 0.08,
            "async_previous_branch_build_seconds": 0.06,
            "async_next_proposal_wait_seconds": 0.002,
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "metrics_delta": {
                    'vllm:async_draft_fanout_branches_total{engine="0"}': 48,
                    'vllm:async_draft_fanout_build_rounds_total{engine="0"}': 2,
                },
                "summary": {"completion_throughput_tok_s": 30.0},
            }
        ),
        encoding="utf-8",
    )

    summary = timing_summary(trace_path, result_path, fan_out=6)

    assert summary["branches_per_round"] == 24
    assert summary["runtime_branch_audit"]["passed"]
    assert summary["overlap_coverage"]["fully_hidden_round_fraction"] == 1.0


def test_timing_summary_rejects_partial_branch_build(tmp_path: Path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "request_id": "qwen-formal-0",
                "accepted_draft_count": 1,
                "cache_hit": False,
                "async_timing_pair_valid": True,
                "async_fan_out": 6,
                "async_batch_num_reqs": 1,
                "async_verify_window_seconds": 0.08,
                "async_previous_branch_build_seconds": 0.06,
                "async_next_proposal_wait_seconds": 0.002,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "metrics_delta": {
                    'vllm:async_draft_fanout_branches_total{engine="0"}': 23,
                    'vllm:async_draft_fanout_build_rounds_total{engine="0"}': 1,
                },
                "summary": {"completion_throughput_tok_s": 30.0},
            }
        ),
        encoding="utf-8",
    )

    summary = timing_summary(trace_path, result_path, fan_out=6)

    assert not summary["runtime_branch_audit"]["passed"]


@pytest.mark.parametrize("value", ("", "0", "513"))
def test_parse_fan_outs_rejects_invalid_values(value) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_fan_outs(value)
