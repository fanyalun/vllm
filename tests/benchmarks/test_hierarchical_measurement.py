# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from benchmarks.hierarchical.analyze_confidence import auc, policy_trigger
from benchmarks.hierarchical.analyze_failure import round_survival
from benchmarks.hierarchical.analyze_forward_kernels import map_graph
from benchmarks.hierarchical.analyze_gdn_mean import steady_cycles
from benchmarks.hierarchical.analyze_joint_confidence import trigger
from benchmarks.hierarchical.compare_signal_tradeoffs import combined_trigger
from benchmarks.hierarchical.cycle_worker import CycleWorker, pair_cycles
from benchmarks.hierarchical.disagreement_worker import (
    distribution_pair,
    rejection_outcome,
)
from benchmarks.hierarchical.measurement_worker import MeasurementWorker
from benchmarks.hierarchical.summarize_forward_stages import partition
from benchmarks.hierarchical.summarize_policy_matrix import summarize


def test_policy_matrix_does_not_certify_partial_requested_coverage(tmp_path):
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "samples": 16,
                "output_length": 512,
                "cells": [[1, "ar"]],
            }
        )
    )
    with pytest.raises(AssertionError):
        summarize(tmp_path)
    assert not (tmp_path / "MATRIX_AUDIT_COMPLETE").exists()


def test_piecewise_stop_uses_earliest_branch_and_keeps_final_round_exclusion():
    rounds = [
        {"inner_round": i, "accepted": a, "proposed": 4, "correction_margin": m}
        for i, (a, m) in enumerate(((0, 1.5), (2, 0.125), (0, 0.0), (0, 0.0)))
    ]
    assert combined_trigger(rounds, 3, 0.25, 2) is rounds[0]
    assert combined_trigger(rounds, 3, 0.25, 1) is rounds[1]
    assert combined_trigger(rounds[-1:], 3, 0.25, 2) is None


def test_joint_stop_requires_both_features_in_the_same_nonfinal_round():
    rounds = [
        {"accepted": a, "proposed": 4, "correction_margin": m}
        for a, m in ((2, 0.125), (0, 0.5), (1, 0.25), (0, 0.0))
    ]
    assert trigger(rounds, 1, 0.5) is rounds[2]
    assert trigger(rounds, 0, 0.5) is None
    assert trigger(rounds, 1, 0.25) is None


def test_confidence_policy_uses_first_strict_threshold_and_excludes_final_round():
    rounds = [
        {"accepted": 1, "proposed": 4, "draft_preverify": {"right": {"margin": gap}}}
        for gap in (0.5, 0.25, 0.0, 0.0)
    ]
    assert policy_trigger(rounds, "correction_margin_lt0.5") is rounds[1]
    assert policy_trigger(rounds, "inner_accept_le1") is rounds[0]
    assert policy_trigger(rounds[-1:], "correction_margin_lt0.5") is None
    minimal = [
        {"accepted": 1, "proposed": 4, "correction_margin": gap}
        for gap in (0.5, 0.25, 0.0, 0.0)
    ]
    assert policy_trigger(minimal, "correction_margin_lt0.5") is minimal[1]


def test_confidence_auc_counts_ties_as_half_without_inventing_missing_classes():
    rows = [
        {"margin": 2.0, "accepted": True},
        {"margin": 1.0, "accepted": True},
        {"margin": 1.0, "accepted": False},
    ]
    assert auc(rows) == 0.75
    assert auc(rows[:2]) is None


def test_distribution_comparison_preserves_shift_invariance_and_mutual_ranks():
    left = torch.tensor([3.0, 2.5, -1.0])
    same = distribution_pair(left, left + 100)
    assert same["js_nats"] == pytest.approx(0, abs=1e-12)
    assert same["tv"] == pytest.approx(0, abs=1e-12)
    flipped = distribution_pair(left, torch.tensor([2.5, 3.0, -1.0]))
    assert flipped["left"]["other_top1_rank"] == 2
    assert flipped["right"]["other_top1_rank"] == 2
    assert 0 < flipped["js_nats"] < 0.7
    assert flipped["left"]["other_top1_logit_gap"] == 0.5


def test_inner_rejection_outcomes_exclude_unreached_and_count_suffix_after_correction():
    earlier = rejection_outcome(5, 1, 3, 10)
    assert not earlier["target_reached"]
    assert earlier["correction_accepted"] is None
    assert earlier["suffix_accepted"] is None
    rejected = rejection_outcome(5, 1, 6, 10)
    assert rejected["target_reached"]
    assert rejected["correction_accepted"] is False
    corrected = rejection_outcome(5, 1, 9, 10)
    assert corrected["correction_accepted"] is True
    assert corrected["suffix_scheduled"] == 3
    assert corrected["suffix_accepted"] == 2


def test_cycle_yield_counts_only_tokens_returned_before_output_limit():
    request = {
        "cycles": [
            {"proposal_step": 0, "emitted": 3, "accepted": 2, "scheduled": 4},
            {"proposal_step": 1, "emitted": 5, "accepted": 4, "scheduled": 4},
            {"proposal_step": 2, "emitted": 5, "accepted": 4, "scheduled": 4},
        ]
    }
    cycles = steady_cycles(request, 6)
    assert len(cycles) == 1
    assert cycles[0]["returned"] == 2
    assert cycles[0]["emitted"] == 5


def test_stage_partition_counts_shared_routed_overlap_once():
    row = {
        "total_ms": 10,
        "spans": [
            {"phase": "moe_envelope", "start_ms": 1, "end_ms": 9},
            {"phase": "shared", "start_ms": 2, "end_ms": 6},
            {"phase": "routed", "start_ms": 4, "end_ms": 8},
        ],
    }
    result = partition(row)
    assert result == {
        "other": 2,
        "moe_other": 2,
        "shared": 2,
        "shared_overlap": 2,
        "routed": 2,
    }


def test_attention_stage_includes_nested_normalization_without_double_counting():
    row = {
        "total_ms": 5,
        "spans": [
            {"phase": "attention", "start_ms": 1, "end_ms": 4},
            {"phase": "norm", "start_ms": 2, "end_ms": 3},
        ],
    }
    assert partition(row) == {"other": 2, "attention": 3}


def test_reference_kernel_mapping_rejects_changed_kernel_sequence():
    reference = [{"name": "routed_gemm", "phase": "routed"}]
    graph = [
        {
            "cat": "kernel",
            "name": "unknown_gemm",
            "ts": 1,
            "dur": 1,
            "args": {"stream": 1},
        }
    ]
    with pytest.raises(AssertionError):
        map_graph(reference, graph)


def test_outer_rejection_distinguishes_partial_and_fully_wasted_inner_rounds():
    trace = [
        {"offset": 0, "accepted": 2, "emitted": 3},
        {"offset": 3, "accepted": 1, "emitted": 2},
        {"offset": 5, "accepted": 3, "emitted": 4},
        {"offset": 9, "accepted": 0, "emitted": 1},
    ]
    assert round_survival(trace, 4) == [3, 1, 0, 0]
    assert round_survival(trace, 10) == [3, 2, 4, 1]
    assert round_survival(trace, 0) == [0, 0, 0, 0]


def test_cycle_pairs_proposal_with_next_target_and_excludes_tail_proposal():
    spans = [
        {"phase": "target_sample", "step": 0},
        {
            "phase": "proposal",
            "step": 0,
            "start_ms": 10,
            "cpu_start_ms": 100,
            "stream_ms": 20,
        },
        {
            "phase": "target_sample",
            "step": 1,
            "end_ms": 45,
            "cpu_end_ms": 136,
            "scheduled": 12,
            "emitted": 10,
        },
        {
            "phase": "proposal",
            "step": 1,
            "start_ms": 46,
            "cpu_start_ms": 137,
            "stream_ms": 21,
        },
    ]
    assert pair_cycles(spans) == [
        {
            "step": 1,
            "proposal_step": 0,
            "scheduled": 12,
            "emitted": 10,
            "accepted": 9,
            "cycle_stream_ms": 35,
            "cycle_wall_ms": 36,
            "proposal_ms": 20,
        }
    ]


def test_cycle_worker_handles_prefill_without_draft_counts(monkeypatch):
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    worker = CycleWorker()
    worker.model_runner = SimpleNamespace(
        execute_model=lambda: None,
        sample=lambda hidden, batch: (None, torch.tensor([1]), None),
        speculator=SimpleNamespace(propose=lambda: None),
    )
    worker.begin_cycle_measurement()
    worker.model_runner.execute_model()
    worker.model_runner.sample(
        None, SimpleNamespace(num_draft_tokens_per_req=None, has_prefill=True)
    )
    measured = worker.collect_cycle_measurement()
    assert measured["cycles"] == []
    assert measured["spans"][1]["scheduled"] == 0
    assert measured["spans"][1]["emitted"] == 1


def test_drafting_timer_encloses_all_inner_rounds_once(monkeypatch):
    calls = []
    output = torch.zeros((1, 20), dtype=torch.int64)

    def propose(batch):
        calls.extend(["inner_round"] * 4)
        return output

    speculator = SimpleNamespace(
        propose=propose,
        record_verification=Mock(),
        last_trace=[{"emitted": 3}] * 4,
    )
    worker = MeasurementWorker()
    worker.model_runner = SimpleNamespace(speculator=speculator)
    start = Mock(record=lambda: calls.append("start"))
    end = Mock(record=lambda: calls.append("end"))
    start.elapsed_time.return_value = 17.0
    worker._event_pool = [(start, end)]
    clock = iter([1_000_000, 9_000_000])
    monkeypatch.setattr("time.perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    worker.begin_measurement("timing")
    assert speculator.propose(SimpleNamespace(has_prefill=False, num_reqs=1)) is output
    measured = worker.collect_measurement()

    assert calls == ["start"] + ["inner_round"] * 4 + ["end"]
    assert len(measured["proposals"]) == 1
    row = measured["proposals"][0]
    assert row["stream_elapsed_ms"] == 17.0
    assert row["cpu_submit_ms"] == 8.0
    assert row["actual_candidates"] == 12
    assert row["inner_rounds"] == 4


def test_acceptance_preserves_target_counts_before_buffer_reuse(monkeypatch):
    speculator = SimpleNamespace(propose=Mock(), record_verification=Mock())
    worker = MeasurementWorker()
    worker.model_runner = SimpleNamespace(speculator=speculator)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    original_verify = speculator.record_verification
    counts = torch.tensor([4])
    batch = SimpleNamespace(num_reqs=1, num_draft_tokens_per_req=[12])

    worker.begin_measurement("acceptance")
    speculator.record_verification(None, batch, counts)
    counts.fill_(1)
    batch.num_draft_tokens_per_req[0] = 20
    measured = worker.collect_measurement()

    original_verify.assert_called_once()
    assert measured["num_sampled"] == [4]
    assert measured["scheduled"] == [12]
    assert measured["proposals"] == []
