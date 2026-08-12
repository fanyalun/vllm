# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vllm.v1.core.sched.routed_experts_trace import (
    RoutedExpertsTraceWriter,
    build_decode_trace_metadata,
    is_target_model_moe_module,
)

ANALYSIS_PATH = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "replayssm"
    / "expert_routing_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("expert_routing_analysis", ANALYSIS_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def fake_vllm_config():
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        num_experts=256,
        num_experts_per_tok=2,
    )
    model_config = SimpleNamespace(hf_text_config=hf_config, model="fake-model")
    parallel_config = SimpleNamespace(
        tensor_parallel_size=2,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
    )
    scheduler_config = SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=4)
    cache_config = SimpleNamespace(
        use_replayssm=True,
        use_replayssm_spec=False,
        replayssm_buffer_len=16,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        num_speculative_tokens=0,
    )


def test_build_decode_trace_metadata_spec_acceptance():
    indices, kinds, accepted = build_decode_trace_metadata(
        num_scheduled=4,
        start_position=100,
        num_prompt_tokens=20,
        num_spec_tokens=3,
        num_accepted_drafts=1,
    )
    assert indices == [0, 1, 2, 3]
    assert kinds == [
        "spec_target",
        "spec_draft_1",
        "spec_draft_2",
        "spec_draft_3",
    ]
    assert accepted == [True, True, False, False]


def test_build_decode_trace_metadata_excludes_prefill():
    indices, kinds, accepted = build_decode_trace_metadata(
        num_scheduled=5,
        start_position=0,
        num_prompt_tokens=3,
        num_spec_tokens=0,
        num_accepted_drafts=0,
    )
    assert indices == [3, 4]
    assert kinds == ["ar_decode", "ar_decode"]
    assert accepted == [True, True]


def test_target_model_module_filter_excludes_mtp_drafter():
    assert is_target_model_moe_module("model.layers.0.mlp.experts")
    assert not is_target_model_moe_module(
        "draft_model.model.layers.0.mlp.experts"
    )
    assert not is_target_model_moe_module("mtp.layers.0.mlp.experts")


def test_trace_writer_uint8_boundary_and_offsets(tmp_path):
    trace_dir = tmp_path / "trace"
    writer = RoutedExpertsTraceWriter(
        {
            "output_dir": str(trace_dir),
            "run_name": "test",
            "decode_only": True,
        },
        fake_vllm_config(),
    )
    first = np.array([[[0, 255], [1, 254]]], dtype=np.int32)
    second = np.array([[[2, 253], [3, 252]]], dtype=np.int32)
    writer.write_request(
        first,
        scheduler_step=1,
        request_id="0",
        absolute_positions=[4],
        token_ids=[10],
        row_in_request=[0],
        route_kinds=["ar_decode"],
        accepted=[True],
    )
    writer.write_request(
        second,
        scheduler_step=2,
        request_id="0",
        absolute_positions=[5],
        token_ids=[11],
        row_in_request=[0],
        route_kinds=["ar_decode"],
        accepted=[True],
    )
    (trace_dir / "worker_complete").touch()
    writer.close()

    manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
    assert manifest["state"] == "complete"
    assert manifest["route_dtype"] == "uint8"
    assert manifest["route_shape"] == [2, 2, 2]
    stored = np.fromfile(trace_dir / "routes.bin", dtype=np.uint8).reshape(2, 2, 2)
    np.testing.assert_array_equal(
        stored, np.concatenate((first, second)).astype(np.uint8)
    )
    events = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [event["binary_row_offset"] for event in events] == [0, 1]


def test_trace_writer_failed_without_completion_marker(tmp_path):
    trace_dir = tmp_path / "trace"
    writer = RoutedExpertsTraceWriter(
        {"output_dir": str(trace_dir), "decode_only": True},
        fake_vllm_config(),
    )
    writer.close()
    manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert (trace_dir / "failure.json").is_file()


def test_distribution_metrics_known_cases():
    uniform = np.ones(10)
    concentrated = np.array([10, 0, 0, 0])
    assert analysis.gini(uniform) == 0
    assert analysis.gini(concentrated) == 0.75
    metrics = analysis.distribution_metrics(concentrated)
    assert metrics["active_experts"] == 1
    assert metrics["max_over_mean"] == 4
    assert metrics["hot_10pct_share"] == 1


def test_hot_selection_tie_break_and_pairwise_metrics():
    counts = np.ones((2, 256), dtype=np.int64)
    hot = analysis.select_hot_experts(counts)
    np.testing.assert_array_equal(hot[0], np.arange(26))
    hits = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
        ]
    )
    metrics = analysis.pairwise_hot_metrics(hits)
    assert metrics["intersection"][0, 1] == 1
    assert metrics["jaccard"][0, 1] == 1 / 3
    assert metrics["lift"][0, 1] == 1
    assert metrics["phi"][0, 1] == 0


def test_poisson_binomial_distribution():
    distribution = analysis.poisson_binomial(np.array([0.5, 0.5]))
    np.testing.assert_allclose(distribution, [0.25, 0.5, 0.25])
    assert distribution.sum() == 1


def test_spec_cumulative_masks_are_nested():
    kinds = np.array(
        ["spec_target", "spec_draft_1", "spec_draft_2", "spec_draft_3"]
    )
    previous = np.zeros(kinds.size, dtype=np.bool_)
    for _, stage_kinds in analysis.SPEC_STAGES:
        current = np.isin(kinds, stage_kinds)
        assert np.all(previous <= current)
        previous = current


def test_compare_outputs_reports_matching_prefix():
    result = analysis.compare_outputs(
        {0: [1, 2, 3], 1: [4, 5, 6]},
        {0: [1, 2, 3], 1: [4, 9, 6]},
    )
    assert result["matching_requests"] == 1
    assert result["mismatching_requests"] == 1
    assert result["matching_prefix_lengths"] == {0: 3, 1: 1}
    assert result["first_differences"][0]["first_difference"] == 1


def test_external_request_id_strips_vllm_random_suffix():
    output_map = {"request-with-hyphen": 0}
    assert (
        analysis.external_request_id("request-with-hyphen-deadbeef", output_map)
        == "request-with-hyphen"
    )
    assert analysis.external_request_id("request-with-hyphen", output_map) == (
        "request-with-hyphen"
    )


def test_token_id_hot_coverage_groups_occurrences():
    trace = SimpleNamespace(
        name="trace",
        token_ids=np.array([7, 8, 7]),
    )
    hits = np.array([[True, False], [False, False], [True, True]])
    rows = analysis.token_id_hot_coverage_rows(
        trace,
        hits,
        np.array([True, True, True]),
        "all_executed",
    )
    by_token = {row["token_id"]: row for row in rows}
    assert by_token[7]["occurrences"] == 2
    assert by_token[7]["mean_hot_layers"] == 1.5
    assert by_token[8]["strict_all_40_occurrences"] == 0
