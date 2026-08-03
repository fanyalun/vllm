# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module(module_name: str, file_name: str):
    module_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "features"
        / "speculative_decoding"
    )
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    module_path = module_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_module(
    "mtp_ep_load_balance_utils",
    "mtp_ep_load_balance_utils.py",
)
runtime = _load_module(
    "mtp_ep_experiment_runtime",
    "mtp_ep_experiment_runtime.py",
)
analysis = _load_module(
    "mtp_ep_experiment_analysis",
    "mtp_ep_experiment_analysis.py",
)
experiment = _load_module(
    "qwen3_6_mtp_ep_load_balance_experiment",
    "qwen3_6_mtp_ep_load_balance_experiment.py",
)
accuracy_limit = _load_module(
    "qwen3_6_predict_last_accuracy_limit",
    "qwen3_6_predict_last_accuracy_limit.py",
)


def _scheduler_output(num_scheduled_tokens, scheduled_spec_decode_tokens):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    )


def _model_runner_output(req_ids, routing_data):
    return SimpleNamespace(
        req_ids=req_ids,
        routed_experts=SimpleNamespace(routing_data=routing_data),
    )


def test_step_selection_skips_baseline_prefill_and_keeps_decode_step():
    prefill_output = _scheduler_output({"0": 4, "1": 5}, {})
    decode_output = _scheduler_output({"0": 1, "1": 1}, {})
    prefill_routing_data = np.zeros((9, 40, 2), dtype=np.int64)
    prefill_model_output = _model_runner_output(["0", "1"], prefill_routing_data)
    decode_routing_data = np.zeros((2, 40, 2), dtype=np.int64)
    decode_model_output = _model_runner_output(["0", "1"], decode_routing_data)

    assert not helper.should_capture_baseline_decode_step(prefill_output)
    assert helper.select_step_routing_data(
        prefill_output,
        prefill_model_output,
        use_spec_decode=False,
    ) is None
    prefill_decision = helper.classify_step_capture(
        prefill_output,
        prefill_model_output,
        worker_step_metadata={"req_ids": ["0", "1"], "has_prefill": True},
        use_spec_decode=False,
    )
    assert prefill_decision.captured_step is None
    assert prefill_decision.drop_reason == "prefill"

    decision = helper.classify_step_capture(
        decode_output,
        decode_model_output,
        worker_step_metadata={"req_ids": ["0", "1"], "has_prefill": False},
        use_spec_decode=False,
    )
    captured = decision.captured_step
    assert captured is not None
    assert captured.step_kind == "decode_only"
    assert captured.total_scheduled_tokens == 2
    np.testing.assert_array_equal(captured.routing_data, decode_routing_data)


def test_step_selection_only_keeps_mtp_verification_steps():
    no_spec_routing_data = np.arange(2 * 40 * 2, dtype=np.int64).reshape(2, 40, 2)
    no_spec_model_output = _model_runner_output(
        ["req_b", "req_a"],
        no_spec_routing_data,
    )

    no_spec_output = _scheduler_output({"req_b": 1, "req_a": 1}, {})
    assert not helper.should_capture_mtp_verification_step(no_spec_output)
    assert helper.select_step_routing_data(
        no_spec_output,
        no_spec_model_output,
        use_spec_decode=True,
    ) is None

    routing_data = np.arange(5 * 40 * 2, dtype=np.int64).reshape(5, 40, 2)
    model_output = _model_runner_output(["req_b", "req_a"], routing_data)
    mtp_output = _scheduler_output(
        {"req_b": 2, "req_a": 3},
        {"req_a": [7, 8], "req_b": [9]},
    )
    decision = helper.classify_step_capture(
        mtp_output,
        model_output,
        worker_step_metadata={
            "req_ids": ["req_b", "req_a"],
            "has_prefill": False,
        },
        use_spec_decode=True,
    )
    captured = decision.captured_step
    assert captured is not None
    assert captured.step_kind == "verification_only"
    assert captured.request_ids == ("req_b", "req_a")
    assert captured.total_scheduled_tokens == 5
    np.testing.assert_array_equal(captured.routing_data, routing_data)


def test_mixed_mtp_step_is_dropped_without_request_reslicing():
    routing_data = np.arange(3 * 40 * 2, dtype=np.int64).reshape(3, 40, 2)
    model_output = _model_runner_output(["req_a", "req_b"], routing_data)
    mtp_output = _scheduler_output({"req_a": 2, "req_b": 1}, {"req_a": [7]})
    decision = helper.classify_step_capture(
        mtp_output,
        model_output,
        worker_step_metadata={"req_ids": ["req_a", "req_b"], "has_prefill": False},
        use_spec_decode=True,
    )
    assert decision.captured_step is None
    assert decision.drop_reason == "mixed"


def test_counting_and_descending_reorder_are_correct():
    routing_data = np.zeros((2, 40, 2), dtype=np.int64)

    routing_data[0, 0, :] = [5, 1]
    routing_data[1, 0, :] = [5, 2]
    routing_data[0, 9, :] = [7, 7]
    routing_data[1, 9, :] = [3, 7]

    histograms = helper.count_layer_expert_histograms(
        routing_data,
        layers=(0, 9),
        num_experts=8,
    )

    expected_layer0 = np.array([0, 1, 1, 0, 0, 2, 0, 0])
    expected_layer9 = np.array([0, 0, 0, 1, 0, 0, 0, 3])
    np.testing.assert_array_equal(histograms[0], expected_layer0)
    np.testing.assert_array_equal(histograms[1], expected_layer9)

    sorted_counts, sorted_ids = helper.sort_experts_desc(histograms.astype(np.float64))
    np.testing.assert_array_equal(sorted_counts[0, :3], np.array([2.0, 1.0, 1.0]))
    np.testing.assert_array_equal(sorted_ids[0, :3], np.array([5, 1, 2]))
    np.testing.assert_array_equal(sorted_counts[1, :2], np.array([3.0, 1.0]))
    np.testing.assert_array_equal(sorted_ids[1, :2], np.array([7, 3]))


def test_token_destination_assignment_counts_topk_assignments():
    routing_data = np.array([[[0, 1, 1]], [[2, 3, 2]]], dtype=np.int64)
    expert_to_ep_rank = np.array([0, 1, 0, 1], dtype=np.int64)

    request_ids, position_ids, assignments = (
        helper.build_token_layer_destination_assignments(
            routing_data,
            ("req_a",),
            {"req_a": 2},
            expert_to_ep_rank=expert_to_ep_rank,
            ep_size=2,
            layers=(0,),
        )
    )

    np.testing.assert_array_equal(request_ids, np.array(["req_a", "req_a"]))
    np.testing.assert_array_equal(position_ids, np.array([0, 1], dtype=np.int16))
    np.testing.assert_array_equal(
        assignments[:, 0, :],
        np.array([[1, 2], [2, 1]], dtype=np.int16),
    )


def test_metric_logic_matches_balancedness_gini_and_relative_change():
    avg_histograms = np.array([[10.0, 3.0, 2.0, 1.0]])
    baseline_histograms = np.array([[4.0, 4.0, 4.0, 4.0]])

    rows = helper.build_condition_metrics(
        batch_size=32,
        draft_length=2,
        num_steps=5,
        layers=(0,),
        avg_histograms=avg_histograms,
        baseline_histograms=baseline_histograms,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["batch_size"] == 32
    assert row["draft_length"] == 2
    assert row["num_steps"] == 5
    assert row["avg_total_routed_assignments_per_step"] == 16.0
    assert row["balancedness"] == 0.4
    assert row["baseline_balancedness"] == 1.0
    assert row["balancedness_delta"] == -0.6
    assert row["balancedness_relative_change"] == -0.6
    assert row["gini"] > row["baseline_gini"]
    assert row["imbalance_change"] == "worsened"


def test_aggregate_worker_step_timings_uses_max_per_component():
    timing = helper.aggregate_worker_step_timings(
        [
            {
                "total_ms": 10.0,
                "attention_ms": 1.5,
                "routing_ms": 0.5,
                "prepare_ms": 2.0,
                "finalize_ms": 1.5,
                "ffn_ms": 4.0,
            },
            {
                "total_ms": 11.0,
                "attention_ms": 1.0,
                "routing_ms": 0.75,
                "prepare_ms": 1.0,
                "finalize_ms": 2.5,
                "ffn_ms": 3.0,
            },
        ]
    )
    assert timing.total_ms == 11.0
    assert timing.attention_ms == 1.5
    assert timing.routing_ms == 0.75
    assert timing.prepare_ms == 2.0
    assert timing.finalize_ms == 2.5
    assert timing.ffn_ms == 4.0
    assert timing.all2all_ms == 4.5
    assert timing.unattributed_ms == 0.25


def test_check_pending_timings_tolerates_missing_worker_counts(capsys):
    runtime._check_pending_timings(
        [{"pending_timings": 0}, None],
        batch_size=64,
        draft_length=0,
        round_idx=1,
    )
    captured = capsys.readouterr()
    assert "warning cleanup received partial pending timing counts" in captured.out


def test_run_recorded_round_preserves_original_exception_on_cleanup_issue(
    monkeypatch,
):
    class DummyRecorder:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class DummyExecutor:
        def collective_rpc(self, fn, timeout=None):
            if fn is runtime.start_condition_collection_worker:
                return [True]
            if fn is runtime.stop_condition_collection_worker:
                return [None]
            raise AssertionError(f"unexpected rpc function: {fn}")

    class DummyLLM:
        def generate(self, *args, **kwargs):
            raise TimeoutError("execute_model timed out")

    monkeypatch.setattr(runtime, "SchedulerStepRecorder", DummyRecorder)

    with pytest.raises(TimeoutError, match="execute_model timed out"):
        runtime._run_recorded_round(
            DummyLLM(),
            scheduler=SimpleNamespace(),
            model_executor=DummyExecutor(),
            sampling_params=SimpleNamespace(),
            prompt_batch=[],
            batch_size=64,
            draft_length=0,
            round_idx=0,
            use_spec_decode=False,
            layers=(0,),
            num_experts=256,
            expert_to_ep_rank=np.asarray([], dtype=np.int64),
            local_ep_rank=0,
            trace_steps_limit=0,
        )


def test_collect_hybrid_reload_timing_stats_worker_maps_replay_breakdown():
    class DummyModelRunner:
        def snapshot_hybrid_spec_reload_timing_stats(self):
            return SimpleNamespace(
                repair_copy_ms=3.5,
                repair_compute_ms=7.25,
                repair_row_count=11,
                repair_from_start_count=2,
                repair_from_resident_count=9,
                verify_attention_ms=13.0,
                layer_total_ms=31.5,
                verify_call_count=5,
                checkpoint_save_ms=1.75,
                post_replay_state_gather_ms=0.25,
                capture_materialize_ms=0.5,
                segment_start_save_ms=0.75,
                segment_start_wait_ms=0.125,
                tape_save_ms=2.25,
            )

    stats = runtime.collect_hybrid_reload_timing_stats_worker(
        SimpleNamespace(model_runner=DummyModelRunner())
    )

    assert stats["prepare_copy_ms"] == 3.5
    assert stats["repair_compute_ms"] == 7.25
    assert stats["verify_attention_ms"] == 13.0
    assert stats["spill_copy_ms"] == 2.25
    assert stats["layer_total_ms"] == 31.5
    assert stats["verify_call_count"] == 5
    assert stats["preload_total_ms"] == 4.0
    assert stats["preloaded_total_ms"] == 10.75
    assert stats["post_replay_state_gather_ms"] == 0.25
    assert stats["capture_materialize_ms"] == 0.5
    assert stats["segment_start_save_ms"] == 0.75
    assert stats["segment_start_wait_ms"] == 0.125


def test_accumulate_hybrid_reload_timing_stats_keeps_replay_breakdown():
    total = runtime._empty_hybrid_reload_timing_stats()
    worker_stats = {
        "preload_total_ms": 4.0,
        "preload_call_count": 1,
        "preload_req_count": 2,
        "preloaded_total_ms": 10.75,
        "preloaded_row_count": 11,
        "fallback_total_ms": 0.5,
        "fallback_row_count": 1,
        "prepare_copy_ms": 3.5,
        "repair_compute_ms": 7.25,
        "verify_attention_ms": 13.0,
        "spill_copy_ms": 2.25,
        "layer_total_ms": 31.5,
        "verify_call_count": 5,
        "repair_row_count": 11,
        "repair_from_start_count": 2,
        "repair_from_resident_count": 9,
        "checkpoint_save_ms": 1.75,
        "post_replay_state_gather_ms": 0.0,
        "capture_materialize_ms": 0.0,
        "segment_start_save_ms": 0.0,
        "segment_start_wait_ms": 0.0,
        "tape_save_ms": 2.25,
    }

    runtime._accumulate_hybrid_reload_timing_stats(total, worker_stats)

    assert total == worker_stats


def _rank_candidate_data(
    seq_ids,
    kinds,
    total_ms,
    ffn_ms,
    hist_values=None,
    *,
    last_seq_ids=None,
    num_ep_collectives=None,
    draft_ms=None,
    layer_ffn_ms=None,
    layer_local_routed_tokens=None,
    layer_local_active_experts=None,
    num_layers=1,
):
    size = len(seq_ids)
    if last_seq_ids is None:
        last_seq_ids = seq_ids
    if num_ep_collectives is None:
        num_ep_collectives = [
            int(last_seq_ids[idx]) - int(seq_ids[idx]) + 1 for idx in range(size)
        ]
    if draft_ms is None:
        draft_ms = [0.0] * size
    histograms = np.zeros((size, 1, 4), dtype=np.int64)
    if hist_values is not None:
        histograms[:, 0, :] = np.asarray(hist_values, dtype=np.int64)
    if num_layers != 1:
        histograms = np.zeros((size, num_layers, 4), dtype=np.int64)
    if layer_ffn_ms is None:
        layer_ffn_ms = np.asarray(ffn_ms, dtype=np.float64).reshape(size, 1)
    if layer_local_routed_tokens is None:
        layer_local_routed_tokens = np.full((size, num_layers), 2, dtype=np.int64)
    if layer_local_active_experts is None:
        layer_local_active_experts = np.ones((size, num_layers), dtype=np.int64)
    return {
        "candidate_first_ep_collective_seq_ids": np.asarray(seq_ids, dtype=np.int64),
        "candidate_last_ep_collective_seq_ids": np.asarray(
            last_seq_ids, dtype=np.int64
        ),
        "candidate_num_ep_collectives": np.asarray(
            num_ep_collectives, dtype=np.int64
        ),
        "candidate_step_kinds": np.asarray(kinds, dtype=np.str_),
        "candidate_step_total_ms": np.asarray(total_ms, dtype=np.float64),
        "candidate_step_draft_ms": np.asarray(draft_ms, dtype=np.float64),
        "candidate_step_ffn_ms": np.asarray(ffn_ms, dtype=np.float64),
        "candidate_step_total_tokens": np.full((size,), 2, dtype=np.int64),
        "candidate_step_histograms": histograms,
        "candidate_layer_ffn_ms": np.asarray(layer_ffn_ms, dtype=np.float64),
        "candidate_layer_local_routed_tokens": np.asarray(
            layer_local_routed_tokens, dtype=np.int64
        ),
        "candidate_layer_local_active_experts": np.asarray(
            layer_local_active_experts, dtype=np.int64
        ),
    }


def _add_token_routes(rank_data, per_step_assignments):
    counts = np.asarray(
        [assignments.shape[0] for assignments in per_step_assignments],
        dtype=np.int64,
    )
    num_steps = len(per_step_assignments)
    num_layers = per_step_assignments[0].shape[1]
    max_positions = max(assignments.shape[0] for assignments in per_step_assignments)
    rank_data["candidate_position_layer_ffn_ms"] = np.zeros(
        (num_steps, max_positions, num_layers),
        dtype=np.float64,
    )
    rank_data["candidate_position_layer_local_routed_tokens"] = np.zeros(
        (num_steps, max_positions, num_layers),
        dtype=np.int64,
    )
    rank_data["candidate_token_offsets"] = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(counts))
    )
    rank_data["candidate_token_request_ids"] = np.concatenate(
        [
            np.asarray([f"req_{idx}"] * assignments.shape[0], dtype=np.str_)
            for idx, assignments in enumerate(per_step_assignments)
        ],
        axis=0,
    )
    rank_data["candidate_token_position_ids"] = np.concatenate(
        [
            np.arange(assignments.shape[0], dtype=np.int16)
            for assignments in per_step_assignments
        ],
        axis=0,
    )
    rank_data["candidate_token_layer_destination_assignment_counts"] = (
        np.concatenate(per_step_assignments, axis=0).astype(np.int16)
    )
    return rank_data


def test_global_step_time_aggregation_aligns_rank_barriers_by_ordinal():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7, 8],
                ["verification_only", "verification_only"],
                [10.0, 12.0],
                [4.0, 6.0],
                [[1, 2, 0, 0], [0, 2, 1, 0]],
            ),
            _rank_candidate_data(
                [107, 110],
                ["verification_only", "verification_only"],
                [11.0, 9.0],
                [5.0, 3.0],
                [[0, 1, 3, 0], [2, 0, 0, 1]],
            ),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_array_equal(result.global_barrier_ids, np.array([0, 1]))
    np.testing.assert_array_equal(result.global_step_indices, np.array([7, 8]))
    np.testing.assert_array_equal(
        result.rank_barrier_first_ep_collective_seq_ids,
        np.array([[7, 107], [8, 110]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.global_step_kinds,
        np.array(["verification_only", "verification_only"]),
    )
    np.testing.assert_allclose(result.global_step_total_ms, np.array([11.0, 12.0]))
    np.testing.assert_allclose(result.global_step_ffn_ms, np.array([5.0, 6.0]))
    np.testing.assert_allclose(
        result.global_step_sorted_rank_ffn_ms,
        np.array([[5.0, 4.0], [6.0, 3.0]]),
    )
    np.testing.assert_allclose(
        result.global_step_ffn_max_mean_ratio,
        np.array([5.0 / 4.5, 6.0 / 4.5]),
    )
    np.testing.assert_allclose(result.global_step_other_ms, np.array([6.0, 6.0]))
    np.testing.assert_array_equal(
        result.global_step_histograms[0, 0],
        np.array([1, 3, 3, 0]),
    )
    assert result.num_global_candidate_steps == 2
    assert result.num_global_captured_steps == 2


def test_global_step_time_aggregation_keeps_identical_ep_spans():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7],
                ["verification_only"],
                [10.0],
                [4.0],
                [[1, 2, 0, 0]],
            ),
            _rank_candidate_data(
                [7],
                ["verification_only"],
                [11.0],
                [5.0],
                [[0, 1, 3, 0]],
            ),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_array_equal(result.global_barrier_ids, np.array([0]))
    np.testing.assert_array_equal(result.global_step_indices, np.array([7]))
    np.testing.assert_array_equal(
        result.global_step_kinds,
        np.array(["verification_only"]),
    )
    np.testing.assert_allclose(result.global_step_total_ms, np.array([11.0]))
    np.testing.assert_allclose(result.global_step_ffn_ms, np.array([5.0]))
    np.testing.assert_allclose(
        result.global_step_sorted_rank_ffn_ms,
        np.array([[5.0, 4.0]]),
    )
    np.testing.assert_allclose(
        result.global_step_ffn_max_mean_ratio,
        np.array([5.0 / 4.5]),
    )
    np.testing.assert_allclose(result.global_step_other_ms, np.array([6.0]))
    np.testing.assert_array_equal(
        result.global_step_histograms[0, 0],
        np.array([1, 3, 3, 0]),
    )
    assert result.num_global_candidate_steps == 1
    assert result.num_global_captured_steps == 1


def test_global_step_time_aggregation_sorts_rank_ffn_times_per_barrier():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7, 8],
                ["decode_only", "decode_only"],
                [12.0, 9.0],
                [7.0, 2.0],
            ),
            _rank_candidate_data(
                [7, 8],
                ["decode_only", "decode_only"],
                [8.0, 11.0],
                [3.0, 6.0],
            ),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )

    np.testing.assert_allclose(
        result.global_step_sorted_rank_ffn_ms,
        np.array(
            [
                [7.0, 3.0],
                [6.0, 2.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        result.global_step_ffn_max_mean_ratio,
        np.array([7.0 / 5.0, 6.0 / 4.0]),
    )


def test_critical_rank_time_components_use_one_physical_rank():
    critical_ranks, total_ms, ffn_ms, other_ms = (
        helper.compute_critical_rank_step_time_components(
            np.array([[20.0, 18.0], [15.0, 21.0]]),
            np.array(
                [
                    [[10.0, 1.0], [2.0, 8.0]],
                    [[9.0, 1.0], [3.0, 4.0]],
                ]
            ),
        )
    )

    np.testing.assert_array_equal(critical_ranks, np.array([0, 1]))
    np.testing.assert_allclose(total_ms, np.array([20.0, 21.0]))
    np.testing.assert_allclose(ffn_ms, np.array([11.0, 7.0]))
    np.testing.assert_allclose(other_ms, np.array([9.0, 14.0]))
    np.testing.assert_allclose(ffn_ms + other_ms, total_ms)


def test_global_step_time_aggregation_rejects_duplicate_span():
    try:
        helper.aggregate_global_step_time_components(
            [
                _rank_candidate_data(
                    [7, 7],
                    ["decode_only", "decode_only"],
                    [1, 2],
                    [1, 2],
                ),
                _rank_candidate_data([7], ["decode_only"], [1], [1]),
            ],
            data_parallel_size=2,
            layers=(0,),
            num_experts=4,
        )
    except ValueError as exc:
        assert "duplicate EP collective span" in str(exc)
    else:
        raise AssertionError("Expected duplicate span to fail.")


def test_global_step_time_aggregation_rejects_span_count_mismatch():
    try:
        helper.aggregate_global_step_time_components(
            [
                _rank_candidate_data(
                    [7],
                    ["decode_only"],
                    [1],
                    [1],
                    last_seq_ids=[8],
                    num_ep_collectives=[1],
                ),
                _rank_candidate_data([7], ["decode_only"], [1], [1]),
            ],
            data_parallel_size=2,
            layers=(0,),
            num_experts=4,
        )
    except ValueError as exc:
        assert "inconsistent EP span" in str(exc)
    else:
        raise AssertionError("Expected span count mismatch to fail.")


def test_global_step_time_aggregation_drops_unmatched_rank_tail():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7, 8],
                ["decode_only", "decode_only"],
                [1, 1],
                [1, 1],
            ),
            _rank_candidate_data([108], ["decode_only"], [1], [1]),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )

    np.testing.assert_array_equal(result.global_barrier_ids, np.array([0]))
    assert result.num_global_candidate_steps == 2
    assert result.num_global_captured_steps == 1
    assert result.num_global_non_target_dropped_steps == 1


def test_per_layer_sorted_rank_reorders_tokens_and_active_with_ffn():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7],
                ["decode_only"],
                [20.0],
                [0.0],
                num_layers=2,
                layer_ffn_ms=[[10.0, 1.0]],
                layer_local_routed_tokens=[[100, 10]],
                layer_local_active_experts=[[4, 1]],
            ),
            _rank_candidate_data(
                [7],
                ["decode_only"],
                [18.0],
                [0.0],
                num_layers=2,
                layer_ffn_ms=[[2.0, 8.0]],
                layer_local_routed_tokens=[[20, 80]],
                layer_local_active_experts=[[2, 3]],
            ),
        ],
        data_parallel_size=2,
        layers=(0, 1),
        num_experts=4,
    )
    np.testing.assert_allclose(
        result.global_step_sorted_rank_ffn_ms,
        np.array([[18.0, 3.0]]),
    )
    np.testing.assert_allclose(result.global_step_ffn_ms, np.array([11.0]))
    np.testing.assert_allclose(result.global_step_other_ms, np.array([9.0]))
    np.testing.assert_array_equal(
        result.global_step_sorted_rank_local_routed_tokens,
        np.array([[180, 30]]),
    )
    np.testing.assert_array_equal(
        result.global_step_sorted_rank_local_active_experts,
        np.array([[7, 3]]),
    )


def test_position_destination_routes_sum_all_source_ranks():
    result = helper.aggregate_global_step_time_components(
        [
            _add_token_routes(
                _rank_candidate_data(
                    [7],
                    ["verification_only"],
                    [10.0],
                    [1.0],
                    layer_ffn_ms=[[1.0]],
                ),
                [np.array([[[0, 2]]], dtype=np.int16)],
            ),
            _add_token_routes(
                _rank_candidate_data(
                    [107],
                    ["verification_only"],
                    [11.0],
                    [10.0],
                    layer_ffn_ms=[[10.0]],
                ),
                [np.array([[[0, 3]]], dtype=np.int16)],
            ),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )

    np.testing.assert_array_equal(
        result.rank_position_layer_local_routed_tokens[0, :, 0, 0],
        np.array([0, 5]),
    )
    np.testing.assert_array_equal(
        result.global_step_sorted_rank_local_routed_tokens,
        np.array([[5, 0]]),
    )
    np.testing.assert_array_equal(
        result.global_step_position_sorted_rank_local_routed_tokens[0, 0],
        np.array([5, 0]),
    )


def test_draft_timing_uses_rank_max_and_baseline_zero():
    baseline = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data([7], ["decode_only"], [10.0], [4.0]),
            _rank_candidate_data([7], ["decode_only"], [9.0], [3.0]),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_allclose(baseline.global_draft_ms, np.array([0.0]))

    drafted = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7], ["verification_only"], [10.0], [4.0], draft_ms=[1.5]
            ),
            _rank_candidate_data(
                [7], ["verification_only"], [9.0], [3.0], draft_ms=[2.5]
            ),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_allclose(drafted.global_draft_ms, np.array([1.5]))
    np.testing.assert_array_equal(
        drafted.global_critical_rank_indices,
        np.array([0]),
    )


def test_pending_draft_timing_is_recorded_after_step_is_queued(monkeypatch):
    class FakeEvent:
        def __init__(self, timestamp_ms):
            self.timestamp_ms = timestamp_ms
            self.synchronize_calls = 0

        def elapsed_time(self, other):
            return other.timestamp_ms - self.timestamp_ms

        def synchronize(self):
            self.synchronize_calls += 1

    state = runtime._WORKER_STATE
    execute_start = FakeEvent(0.0)
    execute_end = FakeEvent(10.0)
    draft_events = iter((FakeEvent(12.0), FakeEvent(14.0)))
    accumulator = runtime.StepAccumulator(
        step_index=3,
        execute_wall_start_s=0.0,
        execute_start_event=execute_start,
        execute_wall_end_s=0.010,
        execute_end_event=execute_end,
        completion_event=execute_end,
        owned_events=[execute_start, execute_end],
    )

    def record_fake_event(step):
        event = next(draft_events)
        step.owned_events.append(event)
        return event

    wall_times = iter((0.012, 0.014))
    monkeypatch.setattr(runtime, "_record_cuda_event", record_fake_event)
    monkeypatch.setattr(
        runtime.time,
        "perf_counter",
        lambda: next(wall_times, 1.0),
    )
    monkeypatch.setattr(runtime, "_push_nvtx_range", lambda *args: False)
    state.enabled = True
    state.pending_step_records.clear()
    state.current_step = None
    state.draft_measure_depth = 0
    pending_record = {
        "_accumulator": accumulator,
        "metadata": {"req_ids": [], "has_prefill": False},
    }
    state.pending_step_records.append(pending_record)

    try:
        result = runtime._measure_worker_section("draft", lambda: "drafted")
        resolved = runtime.pop_step_timing_worker(None, timeout_s=0.0)
    finally:
        state.enabled = False
        state.pending_step_records.clear()
        state.current_step = None
        state.draft_measure_depth = 0

    assert result == "drafted"
    assert resolved is not None
    assert resolved["timing"]["draft_wall_ms"] == 2.0
    assert resolved["timing"]["draft_gpu_ms"] == 2.0
    assert resolved["timing"]["verification_wall_ms"] == 10.0
    assert resolved["timing"]["verification_gpu_ms"] == 10.0
    assert resolved["trace"]["events"][-1]["label"] == "draft"
    assert accumulator.completion_event.synchronize_calls == 1


def test_cuda_event_pool_reuses_released_events():
    created = []

    def create_event():
        event = object()
        created.append(event)
        return event

    pool = runtime.CudaEventPool(create_event)
    first = pool.acquire()
    second = pool.acquire()
    assert pool.created == 2
    pool.release(first)
    assert pool.acquire() is first
    assert pool.created == 2
    pool.release(second)
    assert pool.available == 1


def test_interval_accounting_handles_draft_inside_and_outside_execute():
    execute = (0.0, 10.0)
    inside_and_outside_draft = [(2.0, 4.0), (12.0, 15.0)]

    assert (
        helper.subtract_interval_overlap_ms(
            execute,
            inside_and_outside_draft,
        )
        == 8.0
    )
    assert helper.interval_union_duration_ms(inside_and_outside_draft) == 5.0
    assert (
        helper.interval_union_duration_ms(
            [execute, *inside_and_outside_draft]
        )
        == 13.0
    )


def test_stage_wrapper_does_not_synchronize_cuda():
    source = inspect.getsource(runtime._measure_worker_section)
    assert ".synchronize(" not in source
    assert "_synchronize_device" not in source


def test_worker_hooks_cover_qwen3_next_top_level_moe():
    source = inspect.getsource(runtime._install_worker_hooks)
    assert "Qwen3NextSparseMoeBlock.forward" in source
    assert "patched_qwen3_next_sparse_moe_forward" in source


def test_analysis_rejects_schema_v9_raw(tmp_path):
    raw_path = tmp_path / "schema_v9.npz"
    np.savez(raw_path, schema_version=np.array([9], dtype=np.int64))

    try:
        analysis.load_condition_data(raw_path)
    except ValueError as exc:
        assert "Schema v9 and older" in str(exc)
    else:
        raise AssertionError("Expected schema v9 raw data to be rejected.")


def test_global_step_time_aggregation_keeps_prefill_global_step():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data([7], ["decode_only"], [10.0], [4.0]),
            _rank_candidate_data([7], ["prefill"], [11.0], [5.0]),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_array_equal(result.global_step_indices, np.array([7]))
    np.testing.assert_array_equal(result.global_step_kinds, np.array(["mixed_rank"]))
    assert result.num_global_prefill_dropped_steps == 1


def test_global_step_time_aggregation_keeps_mixed_global_step():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data([7], ["verification_only"], [10.0], [4.0]),
            _rank_candidate_data([7], ["mixed"], [11.0], [5.0]),
        ],
        data_parallel_size=2,
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_array_equal(result.global_step_indices, np.array([7]))
    np.testing.assert_array_equal(result.global_step_kinds, np.array(["mixed_rank"]))
    assert result.num_global_mixed_dropped_steps == 1


def test_global_step_time_aggregation_rejects_missing_join_key():
    try:
        helper.aggregate_global_step_time_components(
            [
                _rank_candidate_data([7], ["decode_only"], [10.0], [4.0]),
                _rank_candidate_data([-1], ["decode_only"], [11.0], [5.0]),
            ],
            data_parallel_size=2,
            layers=(0,),
            num_experts=4,
        )
    except ValueError as exc:
        assert "invalid EP collective span" in str(exc)
    else:
        raise AssertionError("Expected missing EP span to fail.")


def test_global_step_time_aggregation_rejects_negative_other_time():
    try:
        helper.aggregate_global_step_time_components(
            [
                _rank_candidate_data([7], ["decode_only"], [5.0], [6.0]),
                _rank_candidate_data([7], ["decode_only"], [4.0], [3.0]),
            ],
            data_parallel_size=2,
            layers=(0,),
            num_experts=4,
        )
    except ValueError as exc:
        assert "decomposition is incomplete" in str(exc)
    else:
        raise AssertionError("Expected negative Other time to fail.")


def test_global_step_time_summary_and_normalization_are_correct():
    summary = helper.summarize_global_step_time_components(
        step_total_ms=np.array([10.0, 14.0]),
        step_ffn_ms=np.array([4.0, 6.0]),
        step_other_ms=np.array([6.0, 8.0]),
    )
    assert summary["avg_step_total_ms"] == 12.0
    assert summary["avg_ffn_ms"] == 5.0
    assert summary["avg_other_ms"] == 7.0

    normalized = helper.normalize_global_time_components(
        summary, baseline_total_ms=8.0
    )
    assert normalized["normalized_ffn_ms"] == 0.625
    assert normalized["normalized_other_ms"] == 0.875
    assert normalized["ffn_share"] == 5.0 / 12.0
    assert normalized["other_share"] == 7.0 / 12.0


def test_step_time_rows_use_routed_expert_gpu_for_ffn_aliases():
    def condition(draft_length, *, kind, gpu_total, moe_gpu, routed_expert_gpu):
        return SimpleNamespace(
            data_parallel_size=2,
            timing_scope="global_cuda_event",
            timing_backend="cuda_event",
            global_barrier_ids=np.array([0], dtype=np.int64),
            global_step_kinds=np.array([kind], dtype=np.str_),
            rank_step_kinds=np.array([[kind, kind]], dtype=np.str_),
            global_verification_wall_ms=np.array([100.0], dtype=np.float64),
            global_iteration_wall_ms=np.array([110.0], dtype=np.float64),
            global_draft_wall_ms=np.array([0.0], dtype=np.float64),
            global_verification_gpu_total_ms=np.array(
                [gpu_total], dtype=np.float64
            ),
            global_attention_gpu_ms=np.array([20.0], dtype=np.float64),
            global_moe_gpu_ms=np.array([moe_gpu], dtype=np.float64),
            global_gpu_other_ms=np.array(
                [gpu_total - moe_gpu - 20.0], dtype=np.float64
            ),
            global_step_ffn_ms=np.array(
                [routed_expert_gpu], dtype=np.float64
            ),
            num_forward_steps_total=1,
            num_captured_steps=1,
            num_global_candidate_steps=1,
            num_global_captured_steps=1,
            num_dropped_steps=0,
            num_prefill_dropped_steps=0,
            num_mixed_dropped_steps=0,
            num_global_prefill_dropped_steps=0,
            num_global_mixed_dropped_steps=0,
            num_global_non_target_dropped_steps=0,
        )

    rows = analysis.build_step_time_rows(
        {"batch_sizes": (8,), "draft_lengths": (0, 2)},
        {
            (8, 0): condition(
                0,
                kind="decode_only",
                gpu_total=100.0,
                moe_gpu=70.0,
                routed_expert_gpu=10.0,
            ),
            (8, 2): condition(
                2,
                kind="verification_only",
                gpu_total=200.0,
                moe_gpu=150.0,
                routed_expert_gpu=40.0,
            ),
        },
    )

    draft_row = next(row for row in rows if row["draft_length"] == 2)
    assert draft_row["avg_moe_gpu_ms"] == 150.0
    assert draft_row["avg_routed_expert_gpu_ms"] == 40.0
    assert draft_row["avg_ffn_ms"] == 40.0
    assert draft_row["ffn_share"] == 40.0 / 200.0
    assert draft_row["normalized_moe_gpu"] == 150.0 / 100.0
    assert draft_row["normalized_ffn_ms"] == 40.0 / 100.0


def test_sorted_rank_ffn_time_rows_average_by_sorted_position():
    condition = SimpleNamespace(
        data_parallel_size=2,
        global_barrier_ids=np.array([0, 1], dtype=np.int64),
        global_step_kinds=np.array(["decode_only", "decode_only"]),
        rank_step_kinds=np.array(
            [["decode_only", "decode_only"], ["decode_only", "decode_only"]]
        ),
        global_step_sorted_rank_ffn_ms=np.array(
            [
                [7.0, 3.0],
                [6.0, 2.0],
            ],
            dtype=np.float64,
        ),
        global_step_ffn_max_mean_ratio=np.array(
            [7.0 / 5.0, 6.0 / 4.0],
            dtype=np.float64,
        ),
        num_global_captured_steps=2,
    )
    sorted_rows, imbalance_rows = analysis.build_sorted_rank_ffn_time_rows(
        {
            "batch_sizes": (32,),
            "draft_lengths": (0,),
        },
        {
            (32, 0): condition,
        },
    )

    assert [row["sorted_rank_position"] for row in sorted_rows] == [0, 1]
    np.testing.assert_allclose(
        [row["avg_local_ffn_ms"] for row in sorted_rows],
        np.array([6.5, 2.5]),
    )
    assert imbalance_rows[0]["avg_heaviest_minus_lightest_local_ffn_ms"] == 4.0
    assert imbalance_rows[0]["avg_step_ffn_max_mean_ratio"] == (
        (7.0 / 5.0 + 6.0 / 4.0) / 2.0
    )


def test_sorted_rank_summary_rows_average_ffn_tokens_and_active():
    condition = SimpleNamespace(
        data_parallel_size=2,
        layers=np.array([0, 1], dtype=np.int64),
        global_barrier_ids=np.array([0, 1], dtype=np.int64),
        global_step_kinds=np.array(["decode_only", "decode_only"]),
        rank_step_kinds=np.array(
            [["decode_only", "decode_only"], ["decode_only", "decode_only"]]
        ),
        rank_barrier_first_ep_collective_seq_ids=np.array(
            [[0, 0], [1, 1]], dtype=np.int64
        ),
        rank_barrier_last_ep_collective_seq_ids=np.array(
            [[0, 0], [1, 1]], dtype=np.int64
        ),
        rank_barrier_num_ep_collectives=np.ones((2, 2), dtype=np.int64),
        rank_step_total_ms=np.ones((2, 2), dtype=np.float64),
        rank_step_draft_ms=np.zeros((2, 2), dtype=np.float64),
        rank_layer_ffn_ms=np.ones((2, 2, 2), dtype=np.float64),
        rank_layer_local_routed_tokens=np.ones((2, 2, 2), dtype=np.int64),
        rank_layer_local_active_experts=np.ones((2, 2, 2), dtype=np.int64),
        global_step_sorted_rank_ffn_ms=np.array(
            [[18.0, 3.0], [10.0, 6.0]], dtype=np.float64
        ),
        global_step_sorted_rank_local_routed_tokens=np.array(
            [[180, 30], [100, 60]], dtype=np.int64
        ),
        global_step_sorted_rank_local_active_experts=np.array(
            [[7, 3], [5, 4]], dtype=np.int64
        ),
    )
    rows = analysis.build_sorted_rank_summary_rows(
        {"batch_sizes": (32,), "draft_lengths": (0,)},
        {(32, 0): condition},
    )
    assert rows[0]["sorted_rank_position"] == 0
    assert rows[0]["avg_ffn_ms"] == 14.0
    assert rows[0]["avg_local_routed_tokens"] == 140.0
    assert rows[0]["avg_local_active_experts"] == 6.0
    assert rows[1]["avg_ffn_ms"] == 4.5


def test_sorted_rank_summaries_exclude_mixed_rank_barriers():
    condition = SimpleNamespace(
        data_parallel_size=2,
        layers=np.array([0], dtype=np.int64),
        global_barrier_ids=np.array([0, 1], dtype=np.int64),
        global_step_kinds=np.array(["verification_only", "mixed_rank"]),
        rank_step_kinds=np.array(
            [
                ["verification_only", "verification_only"],
                ["verification_only", "prefill"],
            ]
        ),
        rank_step_total_ms=np.ones((2, 2), dtype=np.float64),
        rank_step_draft_ms=np.zeros((2, 2), dtype=np.float64),
        rank_layer_ffn_ms=np.ones((2, 2, 1), dtype=np.float64),
        rank_layer_local_routed_tokens=np.ones((2, 2, 1), dtype=np.int64),
        rank_layer_local_active_experts=np.ones((2, 2, 1), dtype=np.int64),
        global_step_sorted_rank_ffn_ms=np.array(
            [[10.0, 5.0], [100.0, 50.0]], dtype=np.float64
        ),
        global_step_sorted_rank_local_routed_tokens=np.array(
            [[20, 10], [2000, 1000]], dtype=np.int64
        ),
        global_step_sorted_rank_local_active_experts=np.array(
            [[4, 2], [40, 20]], dtype=np.int64
        ),
        global_step_ffn_max_mean_ratio=np.array([4.0 / 3.0, 4.0 / 3.0]),
        num_global_captured_steps=2,
    )
    manifest = {"batch_sizes": (16,), "draft_lengths": (6,)}
    results = {(16, 6): condition}

    summary_rows = analysis.build_sorted_rank_summary_rows(manifest, results)
    ffn_rows, _ = analysis.build_sorted_rank_ffn_time_rows(manifest, results)

    assert summary_rows[0]["avg_local_routed_tokens"] == 20.0
    assert summary_rows[0]["num_global_barriers"] == 1
    assert ffn_rows[0]["avg_local_ffn_ms"] == 10.0
    assert ffn_rows[0]["num_global_captured_steps"] == 1


def test_position_metric_matrix_includes_routed_token_distribution():
    rows = [
        {
            "verification_position": position,
            "sorted_rank_position": rank,
            "avg_attributed_ffn_ms": float(10 * position + rank),
            "avg_destination_routed_assignments": float(100 * position + rank),
        }
        for position in range(3)
        for rank in range(2)
    ]

    token_values = analysis._build_position_metric_matrix(
        rows,
        verification_positions=[0, 1, 2],
        rank_positions=[0, 1],
        metric_key="avg_destination_routed_assignments",
    )

    np.testing.assert_allclose(
        token_values,
        np.array([[0.0, 1.0], [100.0, 101.0], [200.0, 201.0]]),
    )


def test_position_breakdown_strictly_filters_global_verification_steps():
    condition = _drop_condition(
        np.zeros((3, 1, 2), dtype=np.int16),
        np.array([0, 1, 2], dtype=np.int16),
        global_step_kinds=np.array(
            ["verification_only", "mixed_rank", "verification_only"],
            dtype=np.str_,
        ),
        rank_step_kinds=np.array(
            [
                ["verification_only", "verification_only"],
                ["verification_only", "prefill"],
                ["verification_only", "prefill"],
            ],
            dtype=np.str_,
        ),
    )
    mask = analysis.strict_target_barrier_mask(condition, draft_length=2)

    np.testing.assert_array_equal(mask, np.array([True, False, False]))
    condition.global_step_position_sorted_rank_ffn_ms = np.array(
        [
            [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            [[100.0, 50.0], [200.0, 100.0], [300.0, 150.0]],
            [[1000.0, 500.0], [2000.0, 1000.0], [3000.0, 1500.0]],
        ]
    )
    condition.global_step_position_sorted_rank_local_routed_tokens = np.array(
        [
            [[8, 4], [7, 5], [6, 6]],
            [[800, 400], [700, 500], [600, 600]],
            [[8000, 4000], [7000, 5000], [6000, 6000]],
        ]
    )
    rows = analysis.build_position_breakdown_rows(
        {"batch_sizes": (8,), "draft_lengths": (2,)},
        {(8, 2): condition},
    )

    assert all(row["num_global_steps"] == 1 for row in rows)
    assert rows[0]["avg_attributed_ffn_ms"] == 1.0
    assert rows[0]["avg_destination_routed_assignments"] == 8.0


def _drop_condition(
    assignments,
    positions,
    *,
    global_step_kinds=None,
    rank_step_kinds=None,
    layers=(0,),
):
    assignments = np.asarray(assignments, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.int16)
    if global_step_kinds is None:
        global_step_kinds = ["verification_only"]
    if rank_step_kinds is None:
        rank_step_kinds = [["verification_only", "verification_only"]]
    offsets = [0]
    per_barrier = len(global_step_kinds)
    tokens_per_barrier = positions.shape[0] // per_barrier
    for barrier in range(per_barrier):
        offsets.append(offsets[-1] + tokens_per_barrier)
    return SimpleNamespace(
        schema_version=10,
        batch_size=8,
        draft_length=2,
        data_parallel_size=2,
        layers=np.asarray(layers, dtype=np.int64),
        global_barrier_ids=np.arange(per_barrier, dtype=np.int64),
        global_step_kinds=np.asarray(global_step_kinds, dtype=np.str_),
        rank_step_kinds=np.asarray(rank_step_kinds, dtype=np.str_),
        global_token_barrier_offsets=np.asarray(offsets, dtype=np.int64),
        global_token_source_ranks=np.zeros((positions.shape[0],), dtype=np.int16),
        global_token_request_ids=np.asarray(
            ["req_a"] * positions.shape[0],
            dtype=np.str_,
        ),
        global_token_position_ids=positions,
        global_token_layer_destination_assignment_counts=assignments,
    )


def _audit_condition(
    assignments,
    source_ranks,
    *,
    offsets,
    global_step_kinds,
    rank_step_kinds,
    rank_layer_routed_expert_gpu_ms,
    rank_layer_local_routed_tokens,
    step_histograms,
    global_step_sorted_rank_local_routed_tokens,
    rank_layer_local_active_experts=None,
    layers=(0,),
):
    assignments = np.asarray(assignments, dtype=np.int64)
    num_tokens, num_layers, num_ranks = assignments.shape
    num_barriers = len(global_step_kinds)
    if rank_layer_local_active_experts is None:
        rank_layer_local_active_experts = np.ones(
            (num_barriers, num_ranks, num_layers),
            dtype=np.int64,
        )
    return SimpleNamespace(
        schema_version=10,
        batch_size=8,
        draft_length=2,
        data_parallel_size=num_ranks,
        layers=np.asarray(layers, dtype=np.int64),
        global_barrier_ids=np.arange(num_barriers, dtype=np.int64),
        global_step_kinds=np.asarray(global_step_kinds, dtype=np.str_),
        rank_step_kinds=np.asarray(rank_step_kinds, dtype=np.str_),
        rank_layer_routed_expert_gpu_ms=np.asarray(
            rank_layer_routed_expert_gpu_ms,
            dtype=np.float64,
        ),
        rank_layer_local_routed_tokens=np.asarray(
            rank_layer_local_routed_tokens,
            dtype=np.int64,
        ),
        rank_layer_local_active_experts=np.asarray(
            rank_layer_local_active_experts,
            dtype=np.int64,
        ),
        step_histograms=np.asarray(step_histograms, dtype=np.int64),
        global_step_sorted_rank_local_routed_tokens=np.asarray(
            global_step_sorted_rank_local_routed_tokens,
            dtype=np.int64,
        ),
        global_token_barrier_offsets=np.asarray(offsets, dtype=np.int64),
        global_token_source_ranks=np.asarray(source_ranks, dtype=np.int64),
        global_token_request_ids=np.asarray(["req"] * num_tokens, dtype=np.str_),
        global_token_position_ids=np.zeros((num_tokens,), dtype=np.int16),
        global_token_layer_destination_assignment_counts=assignments,
    )


def test_routing_time_audit_sums_v10_destination_assignments_from_all_sources():
    condition = _audit_condition(
        [
            [[0, 2]],
            [[0, 3]],
            [[99, 0]],
            [[0, 99]],
        ],
        [0, 1, 0, 1],
        offsets=[0, 2, 4],
        global_step_kinds=["verification_only", "mixed_rank"],
        rank_step_kinds=[
            ["verification_only", "verification_only"],
            ["verification_only", "prefill"],
        ],
        rank_layer_routed_expert_gpu_ms=[
            [[1.0], [10.0]],
            [[1000.0], [1000.0]],
        ],
        rank_layer_local_routed_tokens=[
            [[0], [5]],
            [[99], [99]],
        ],
        step_histograms=[
            [[1, 1, 1, 2]],
            [[99, 99, 0, 0]],
        ],
        global_step_sorted_rank_local_routed_tokens=[
            [5, 0],
            [99, 99],
        ],
    )

    condition_rows, layer_rank_rows, sorted_rows = (
        analysis.build_routing_time_audit_rows(
            {"batch_sizes": (8,), "draft_lengths": (2,)},
            {(8, 2): condition},
        )
    )

    rank_1 = next(row for row in layer_rank_rows if row["physical_rank"] == 1)
    assert rank_1["avg_v10_destination_assignments"] == 5.0
    assert rank_1["avg_v8_emulated_self_assignments"] == 3.0
    assert condition_rows[0]["num_global_barriers"] == 1
    assert sorted_rows[0]["avg_ffn_sorted_v10_destination_tokens"] == 5.0
    assert sorted_rows[0]["avg_ffn_sorted_v8_emulated_self_tokens"] == 3.0


def test_routing_time_audit_keeps_ffn_sort_and_token_sort_independent():
    condition = _audit_condition(
        [
            [[3, 0], [8, 0]],
            [[0, 7], [0, 4]],
        ],
        [0, 1],
        offsets=[0, 2],
        global_step_kinds=["decode_only"],
        rank_step_kinds=[["decode_only", "decode_only"]],
        rank_layer_routed_expert_gpu_ms=[
            [[10.0, 2.0], [1.0, 9.0]],
        ],
        rank_layer_local_routed_tokens=[
            [[3, 8], [7, 4]],
        ],
        step_histograms=[
            [[10, 0], [12, 0]],
        ],
        global_step_sorted_rank_local_routed_tokens=[
            [7, 15],
        ],
        layers=(0, 1),
    )

    _, _, sorted_rows = analysis.build_routing_time_audit_rows(
        {"batch_sizes": (8,), "draft_lengths": (0,)},
        {(8, 0): condition},
    )

    first = next(row for row in sorted_rows if row["sorted_rank_position"] == 0)
    second = next(row for row in sorted_rows if row["sorted_rank_position"] == 1)
    assert first["avg_ffn_sorted_routed_expert_gpu_ms"] == 19.0
    assert first["avg_ffn_sorted_v10_destination_tokens"] == 7.0
    assert second["avg_ffn_sorted_v10_destination_tokens"] == 15.0
    assert first["avg_token_sorted_v10_destination_tokens"] == 15.0
    assert first["avg_token_sorted_routed_expert_gpu_ms"] == 3.0


def test_draft_drop_keeps_cutoff_position_and_closes_suffix():
    condition = _drop_condition(
        [
            [[2, 0]],
            [[0, 2]],
            [[0, 2]],
        ],
        [0, 1, 2],
    )

    cutoff_rows, layer_rows, step_rows, condition_rows = (
        analysis.build_draft_drop_rows(
            {"batch_sizes": (8,), "draft_lengths": (2,)},
            {(8, 2): condition},
        )
    )

    dest1_cutoff = next(
        row for row in cutoff_rows if row["destination_rank"] == 1
    )
    assert dest1_cutoff["baseline_assignments"] == 2
    assert dest1_cutoff["cutoff_position"] == 1
    assert layer_rows[0]["unique_dropped_draft_tokens"] == 1
    assert step_rows[0]["global_suffix_dropped_draft_tokens"] == 1
    assert step_rows[0]["global_suffix_drop_ratio"] == 0.5
    assert condition_rows[0]["mean_step_drop_ratio"] == 0.5
    assert condition_rows[0]["weighted_drop_ratio"] == 0.5


def test_draft_drop_counts_topk_assignments_but_deduplicates_tokens():
    condition = _drop_condition(
        [
            [[1, 1]],
            [[0, 2]],
        ],
        [0, 1],
        global_step_kinds=["verification_only"],
        rank_step_kinds=[["verification_only", "verification_only"]],
    )

    cutoff_rows, layer_rows, step_rows, _ = analysis.build_draft_drop_rows(
        {"batch_sizes": (8,), "draft_lengths": (2,)},
        {(8, 2): condition},
    )

    assert next(
        row for row in cutoff_rows if row["destination_rank"] == 1
    )["total_assignments"] == 3
    assert layer_rows[0]["unique_dropped_draft_tokens"] == 1
    assert step_rows[0]["global_suffix_dropped_draft_tokens"] == 1


def test_draft_drop_filters_mixed_global_barriers():
    condition = _drop_condition(
        [
            [[2, 0]],
            [[0, 2]],
            [[0, 2]],
            [[100, 0]],
            [[0, 100]],
            [[0, 100]],
        ],
        [0, 1, 2, 0, 1, 2],
        global_step_kinds=["verification_only", "mixed_rank"],
        rank_step_kinds=[
            ["verification_only", "verification_only"],
            ["verification_only", "prefill"],
        ],
    )

    _, _, step_rows, condition_rows = analysis.build_draft_drop_rows(
        {"batch_sizes": (8,), "draft_lengths": (2,)},
        {(8, 2): condition},
    )

    assert [row["global_barrier_id"] for row in step_rows] == [0]
    assert condition_rows[0]["num_verification_steps"] == 1


def test_active_expert_ratio_uses_all_ranks_layers_over_model_experts():
    condition = SimpleNamespace(
        data_parallel_size=2,
        layers=np.array([0, 1], dtype=np.int64),
        global_barrier_ids=np.array([0, 1], dtype=np.int64),
        global_step_kinds=np.array(["decode_only", "decode_only"]),
        rank_step_kinds=np.array(
            [["decode_only", "decode_only"], ["decode_only", "decode_only"]]
        ),
        rank_layer_local_active_experts=np.array(
            [
                [[2, 1], [1, 0]],
                [[4, 2], [2, 0]],
            ],
            dtype=np.int64,
        ),
    )
    rows = analysis.build_active_expert_ratio_rows(
        {
            "batch_sizes": (32,),
            "draft_lengths": (0,),
            "num_experts": 8,
        },
        {(32, 0): condition},
    )
    # barrier ratios: 4 / (2 * 8), 8 / (2 * 8)
    assert rows[0]["active_expert_ratio"] == 0.375


def test_close_ffn_component_folds_small_residual_into_ffn():
    closed = runtime._close_ffn_component(
        helper.StepTiming(
            total_ms=10.0,
            attention_ms=2.0,
            routing_ms=1.0,
            prepare_ms=1.0,
            finalize_ms=1.0,
            ffn_ms=4.0,
        )
    )
    assert closed == 5.0


def test_speedup_and_dataset_slicing_helpers_are_correct():
    rows = helper.build_speedup_rows(
        {
            (32, 0): 20.0,
            (32, 2): 10.0,
            (32, 4): 8.0,
            (64, 0): 30.0,
            (64, 2): 15.0,
            (64, 4): 12.0,
        },
        {
            (32, 0): 2000.0,
            (32, 2): 1000.0,
            (32, 4): 800.0,
            (64, 0): 3000.0,
            (64, 2): 1500.0,
            (64, 4): 1200.0,
        },
        {
            (32, 0): 100,
            (32, 2): 100,
            (32, 4): 100,
            (64, 0): 150,
            (64, 2): 150,
            (64, 4): 150,
        },
        batch_sizes=(32, 64),
        draft_lengths=(0, 2, 4),
    )
    assert next(
        row["tpot_speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 2
    ) == 2.0
    assert next(
        row["tpot_speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 0
    ) == 1.0
    assert next(
        row["decode_throughput_speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 0
    ) == 1.0
    np.testing.assert_array_equal(
        helper.select_dataset_indices(4, 8),
        np.array([0, 1, 2, 3]),
    )


def test_prompt_cache_roundtrip(tmp_path):
    prompt_items = [
        {"prompt_token_ids": [1, 2, 3]},
        {"prompt_token_ids": [4, 5]},
    ]
    cache_path = tmp_path / "prompt_cache.json"
    runtime.save_prompt_items_cache(prompt_items, cache_path)

    args = SimpleNamespace(prompt_cache_path=cache_path, batch_size=2)
    loaded = runtime.load_prompt_items(args)
    assert loaded == prompt_items


def test_stop_condition_collection_worker_clears_pending_timings():
    runtime._WORKER_STATE.pending_step_records.clear()
    runtime._WORKER_STATE.pending_step_records.append({"timing": {"total_ms": 1.0}})
    runtime._WORKER_STATE.pending_step_records.append({"timing": {"total_ms": 2.0}})
    runtime._WORKER_STATE.enabled = True

    result = runtime.stop_condition_collection_worker(None)

    assert result == {"pending_timings": 2}
    assert runtime._WORKER_STATE.enabled is False
    assert len(runtime._WORKER_STATE.pending_step_records) == 0


def test_collect_one_command_includes_prompt_cache_and_warmup(tmp_path):
    args = SimpleNamespace(
        model="m",
        dataset="d",
        dataset_config="cfg",
        dataset_split="split",
        num_samples=1024,
        data_parallel_size=4,
        max_tokens=128,
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        num_experts=256,
        layers=(0, 9),
        enforce_eager=True,
        warmup_rounds=1,
        max_num_batched_tokens=49152,
        hybrid_prediction_trace_mode="replay",
        hybrid_prediction_oracle_trace_root=tmp_path / "oracle",
        hybrid_prediction_target_accuracy=0.8,
        hybrid_prediction_sim_mode="exact_upper_bound",
        hybrid_prediction_sim_seed=7,
    )
    command = runtime._build_collect_one_command(
        args,
        tmp_path / "out",
        tmp_path / "entry.py",
        batch_size=32,
        draft_length=4,
        prompt_cache=tmp_path / "prompt_cache.json",
    )
    assert "--prompt-cache-path" in command
    assert str(tmp_path / "prompt_cache.json") in command
    assert "--warmup-rounds" in command
    assert command[command.index("--warmup-rounds") + 1] == "1"
    assert command[command.index("--data-parallel-size") + 1] == "4"
    assert command[command.index("--num-samples") + 1] == "1024"
    assert command[command.index("--max-num-batched-tokens") + 1] == "49152"
    assert command[command.index("--hybrid-prediction-trace-mode") + 1] == "replay"
    assert (
        command[command.index("--hybrid-prediction-oracle-trace-root") + 1]
        == str(tmp_path / "oracle")
    )
    assert (
        command[command.index("--hybrid-prediction-target-accuracy") + 1] == "0.8"
    )
    assert (
        command[command.index("--hybrid-prediction-sim-mode") + 1]
        == "exact_upper_bound"
    )
    assert command[command.index("--hybrid-prediction-sim-seed") + 1] == "7"
    assert command[2:5] == ["collect", "--internal-stage", "condition"]


def test_unified_cli_defaults_to_four_gpu_full_output_matrix():
    args = experiment.parse_args(["collect"])

    assert args.batch_sizes == [8, 16]
    assert args.draft_lengths == [0, 2, 4, 6]
    assert args.data_parallel_size == 4
    assert args.max_model_len == 768
    assert args.max_num_batched_tokens == 8192
    assert args.hybrid_prediction_trace_mode == "off"


def test_resolve_default_model_prefers_local_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        helper,
        "DEFAULT_LOCAL_MODEL",
        tmp_path / "Qwen3.6-35B-A3B",
    )
    monkeypatch.setattr(helper, "DEFAULT_HF_MODEL", "hf/model")

    helper.DEFAULT_LOCAL_MODEL.mkdir()
    assert helper.resolve_default_model() == str(helper.DEFAULT_LOCAL_MODEL)

    helper.DEFAULT_LOCAL_MODEL.rmdir()
    assert helper.resolve_default_model() == "hf/model"


def test_configure_hybrid_prediction_trace_worker_sets_exact_match_budget():
    class DummyRunner:
        def __init__(self) -> None:
            self.enabled = False
            self.override = None

        def reset_hybrid_spec_prediction_trace(self) -> None:
            self.enabled = False

        def disable_hybrid_spec_prediction_override(self) -> None:
            self.override = None

        def enable_hybrid_spec_prediction_trace(self) -> None:
            self.enabled = True

        def configure_hybrid_spec_prediction_override(
            self,
            *,
            mode,
            oracle_trace,
            exact_match_event_indices,
        ) -> None:
            self.override = {
                "mode": mode,
                "oracle_trace": oracle_trace,
                "exact_match_event_indices": exact_match_event_indices,
            }

    worker = SimpleNamespace(model_runner=DummyRunner())
    trace = [
        {"event_index": idx, "accepted_len": 2, "req_max_accept_len": 3, "draft_len": 2}
        for idx in range(5)
    ]

    result = runtime.configure_hybrid_prediction_trace_worker(
        worker,
        trace_mode="replay",
        oracle_trace=trace,
        target_accuracy=0.6,
        sim_mode="exact_upper_bound",
        sim_seed=0,
    )

    assert worker.model_runner.enabled is True
    assert result["trace_events"] == 5
    assert result["exact_match_events"] == 3
    assert worker.model_runner.override["mode"] == "exact_upper_bound"
    assert len(worker.model_runner.override["exact_match_event_indices"]) == 3


def test_load_rank_hybrid_prediction_trace_derives_req_local_indices(tmp_path):
    trace_path = tmp_path / "rank_00.npz"
    np.savez_compressed(
        trace_path,
        schema_version=np.asarray([runtime.HYBRID_PREDICTION_TRACE_SCHEMA_VERSION]),
        event_index=np.asarray([0, 1, 2], dtype=np.int64),
        req_id=np.asarray(["req-a", "req-b", "req-a"]),
        accepted_len=np.asarray([3, 1, 2], dtype=np.int64),
        baseline_predicted_len=np.asarray([5, 3, 4], dtype=np.int64),
        effective_predicted_len=np.asarray([5, 3, 4], dtype=np.int64),
        req_max_accept_len=np.asarray([5, 3, 4], dtype=np.int64),
        draft_len=np.asarray([4, 2, 3], dtype=np.int64),
    )

    trace = runtime.load_rank_hybrid_prediction_trace(trace_path)

    assert trace == [
        {
            "event_index": 0,
            "req_event_index": 0,
            "req_id": "req-a",
            "accepted_len": 3,
            "baseline_predicted_len": 5,
            "effective_predicted_len": 5,
            "req_max_accept_len": 5,
            "draft_len": 4,
            "output_token_ids": (),
        },
        {
            "event_index": 1,
            "req_event_index": 0,
            "req_id": "req-b",
            "accepted_len": 1,
            "baseline_predicted_len": 3,
            "effective_predicted_len": 3,
            "req_max_accept_len": 3,
            "draft_len": 2,
            "output_token_ids": (),
        },
        {
            "event_index": 2,
            "req_event_index": 1,
            "req_id": "req-a",
            "accepted_len": 2,
            "baseline_predicted_len": 4,
            "effective_predicted_len": 4,
            "req_max_accept_len": 4,
            "draft_len": 3,
            "output_token_ids": (),
        },
    ]


def test_save_and_load_rank_hybrid_prediction_trace_roundtrip_output_token_ids(
    tmp_path,
):
    trace_path = tmp_path / "rank_00.npz"
    args = SimpleNamespace(
        batch_size=128,
        draft_length=4,
        data_parallel_size=2,
        dp_rank=1,
    )
    data = SimpleNamespace(
        hybrid_prediction_trace_events=[
            {
                "event_index": 0,
                "req_event_index": 0,
                "req_id": "4-abcd1234",
                "accepted_len": 3,
                "baseline_predicted_len": 5,
                "effective_predicted_len": 5,
                "req_max_accept_len": 5,
                "draft_len": 4,
                "output_token_ids": (11, 12, 13, -1, -1),
            },
            {
                "event_index": 1,
                "req_event_index": 0,
                "req_id": "5-efgh5678",
                "accepted_len": 1,
                "baseline_predicted_len": 3,
                "effective_predicted_len": 3,
                "req_max_accept_len": 5,
                "draft_len": 4,
                "output_token_ids": (21, -1, -1, -1, -1),
            },
        ],
    )

    runtime.save_rank_hybrid_prediction_trace(trace_path, args, data)
    trace = runtime.load_rank_hybrid_prediction_trace(trace_path)

    assert trace == [
        {
            "event_index": 0,
            "req_event_index": 0,
            "req_id": "4-abcd1234",
            "accepted_len": 3,
            "baseline_predicted_len": 5,
            "effective_predicted_len": 5,
            "req_max_accept_len": 5,
            "draft_len": 4,
            "output_token_ids": (11, 12, 13, -1, -1),
        },
        {
            "event_index": 1,
            "req_event_index": 0,
            "req_id": "5-efgh5678",
            "accepted_len": 1,
            "baseline_predicted_len": 3,
            "effective_predicted_len": 3,
            "req_max_accept_len": 5,
            "draft_len": 4,
            "output_token_ids": (21, -1, -1, -1, -1),
        },
    ]


def test_safe_collective_rpc_skips_shutdown_executor():
    executor = SimpleNamespace(
        collective_rpc=lambda *args, **kwargs: ["unexpected"],
        rpc_broadcast_mq=None,
        is_failed=False,
        shutting_down=False,
    )

    assert runtime._safe_collective_rpc(executor, "noop", timeout=1) is None


def test_accuracy_limit_helpers_compute_oracle_verdict():
    compare_rows = [
        {
            "batch_size": 128,
            "draft_length": 4,
            "target_accuracy": 1.0,
            "vs_disabled_tpot_delta_ms": 1.0,
            "vs_disabled_throughput_delta_tok_s": -10.0,
        },
        {
            "batch_size": 128,
            "draft_length": 6,
            "target_accuracy": 1.0,
            "vs_disabled_tpot_delta_ms": -0.5,
            "vs_disabled_throughput_delta_tok_s": -5.0,
        },
    ]

    assert accuracy_limit._normalize_draft_lengths([4, 6]) == [0, 4, 6]
    assert accuracy_limit._accuracy_tag(0.8) == "080"
    assert (
        accuracy_limit._resolve_data_parallel_size(
            SimpleNamespace(data_parallel_size=None, local_gpu_ids="0,1")
        )
        == 2
    )
    assert accuracy_limit._oracle_beats_disabled(compare_rows) is True


def test_scheduler_capacity_helpers_use_local_batch_budget():
    local_max_num_seqs = runtime.get_local_max_num_seqs(
        batch_size=512,
        data_parallel_size=2,
    )
    assert local_max_num_seqs == 256
    assert (
        runtime.get_configured_max_num_batched_tokens(
            local_max_num_seqs,
            draft_length=6,
            max_num_batched_tokens_override=None,
        )
        == helper.DEFAULT_MAX_NUM_BATCHED_TOKENS
    )
    assert (
        runtime.get_configured_max_num_batched_tokens(
            local_max_num_seqs,
            draft_length=6,
            max_num_batched_tokens_override=49152,
        )
        == 49152
    )

    local_max_num_seqs = runtime.get_local_max_num_seqs(
        batch_size=1024,
        data_parallel_size=2,
    )
    assert local_max_num_seqs == 512
    assert (
        runtime.get_configured_max_num_batched_tokens(
            local_max_num_seqs,
            draft_length=6,
        )
        == helper.DEFAULT_MAX_NUM_BATCHED_TOKENS
    )


def test_scheduler_capacity_snapshot_reads_actual_vllm_config():
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=256,
                    max_num_batched_tokens=4096,
                    max_num_scheduled_tokens=2560,
                ),
                speculative_config=SimpleNamespace(
                    max_num_new_slots_for_drafting=6,
                ),
            )
        )
    )

    config = runtime._snapshot_scheduler_capacity_config(
        llm,
        local_max_num_seqs=256,
        configured_max_num_batched_tokens=4096,
    )

    assert config.to_log_dict() == {
        "local_max_num_seqs": 256,
        "configured_max_num_batched_tokens": 4096,
        "scheduler_max_num_seqs": 256,
        "scheduler_max_num_batched_tokens": 4096,
        "scheduler_max_num_scheduled_tokens": 2560,
        "speculative_max_num_new_slots_for_drafting": 6,
    }


def test_dp_sharding_covers_all_samples_without_overlap():
    shards = [
        helper.shard_global_batch_indices(
            num_samples=10,
            global_batch_size=4,
            round_idx=round_idx,
            dp_size=3,
            dp_rank=dp_rank,
        )
        for round_idx in range(helper.num_condition_rounds(10, 4))
        for dp_rank in range(3)
    ]
    flat = np.concatenate(shards, axis=0)
    np.testing.assert_array_equal(np.sort(flat), np.arange(10))


def test_tpot_formula_matches_decode_only_definition():
    output_lengths = np.array([5, 1, 3], dtype=np.int64)
    assert helper.compute_num_output_tokens_excluding_first(output_lengths) == 6
    assert helper.compute_tpot_ms(30.0, output_lengths) == 5.0
    assert helper.compute_tpot_ms_from_finished_stats(30.0, 6) == 5.0
    assert helper.compute_decode_throughput_tok_s(12, 300.0) == 40.0


def test_expert_to_ep_rank_mapping_and_rank_load_are_correct():
    expert_to_ep_rank = helper.build_expert_to_ep_rank(
        num_experts=5,
        ep_size=2,
    )
    np.testing.assert_array_equal(expert_to_ep_rank, np.array([0, 0, 0, 1, 1]))
    avg_histograms = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    rank_load = helper.build_rank_load_from_histograms(
        avg_histograms,
        expert_to_ep_rank,
        ep_size=2,
    )
    np.testing.assert_allclose(rank_load, np.array([[6.0, 9.0]]))


def test_validate_parallel_config_requires_tp1():
    args = SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1)
    try:
        runtime.validate_parallel_config(args)
    except ValueError as exc:
        assert "tensor_parallel_size=1" in str(exc)
    else:
        raise AssertionError("Expected tensor_parallel_size validation to fail.")


def test_extract_worker_step_metadata_supports_legacy_gpu_model_runner():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["req0", "req1"],
                num_reqs=2,
                num_computed_tokens_cpu=np.array([0, 8], dtype=np.int32),
                num_prompt_tokens=np.array([16, 8], dtype=np.int32),
            ),
            execute_model_state=None,
        )
    )
    metadata = runtime._extract_worker_step_metadata(worker)
    assert metadata["req_ids"] == ["req0", "req1"]
    assert metadata["has_prefill"] is True


def test_baseline_order_reordering_is_stable():
    avg_histograms = np.array(
        [
            [4.0, 2.0, 1.0, 0.0],
            [1.0, 3.0, 0.0, 2.0],
        ]
    )
    _, baseline_order = helper.sort_experts_desc(avg_histograms)
    target_histograms = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [7.0, 5.0, 1.0, 9.0],
        ]
    )
    reordered = helper.reorder_histograms_by_expert_order(
        target_histograms,
        baseline_order,
    )
    np.testing.assert_array_equal(reordered[0], np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_equal(reordered[1], np.array([5.0, 9.0, 7.0, 1.0]))
