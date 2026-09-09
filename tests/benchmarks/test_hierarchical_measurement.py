# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from benchmarks.hierarchical.measurement_worker import MeasurementWorker


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
