# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from benchmarks.hierarchical.analyze_failure import round_survival
from benchmarks.hierarchical.cycle_worker import CycleWorker, pair_cycles
from benchmarks.hierarchical.measurement_worker import MeasurementWorker


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
