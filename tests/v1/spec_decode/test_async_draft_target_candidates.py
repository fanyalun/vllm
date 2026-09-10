# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.async_draft.speculator import AsyncDraftSpeculator


class Publication:
    def __init__(self, header):
        self.header = header

    def poll(self):
        return self.header is not None

    def recv(self):
        header, self.header = self.header, None
        return header


def test_disabling_export_keeps_internal_counters():
    proxy = object.__new__(AsyncDraftSpeculator)
    proxy._export_metrics = False
    proxy._metrics = {"cache_hits": 7}
    proxy._step_metrics = {"cache_hits": 2}
    assert proxy.take_metrics() == {}
    assert proxy._metrics == {"cache_hits": 7}
    assert proxy._step_metrics == {"cache_hits": 0}


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        "generation",
        "epoch",
        "request",
        "engine",
        "accepted",
        "recovery",
        "prefill",
        "transient",
        "force_jit",
    ],
)
def test_local_candidates_require_exact_identity_and_outcome(monkeypatch, mismatch):
    proxy = object.__new__(AsyncDraftSpeculator)
    key = ["engine", "request", 3, 2, 17]
    fields = {"engine": 0, "request": 1, "epoch": 2, "accepted": 3, "recovery": 4}
    if mismatch in fields:
        key[fields[mismatch]] = "wrong"
    header = {
        "generation": 8 if mismatch == "generation" else 9,
        "slot": 1,
        "keys": [tuple(key)],
    }
    proxy._candidate_connection = Publication(header)
    proxy._candidate_header = None
    proxy._candidate_tokens = torch.arange(8).view(2, 1, 4)
    batch = SimpleNamespace(
        generation=10,
        engine_instance_id="engine",
        req_ids=["request"],
        request_epochs=[3],
        transient=mismatch == "transient",
        is_prefilling_np=[mismatch == "prefill"],
    )
    slot = SimpleNamespace(
        num_sampled=torch.tensor([3]), last_sampled=torch.tensor([17])
    )
    monkeypatch.setenv("ASYNC_DRAFT_FORCE_JIT", str(int(mismatch == "force_jit")))
    result = proxy._find_local_candidate(batch, slot)
    if mismatch is None:
        assert result.tolist() == [4, 5, 6, 7]
    else:
        assert result is None


@pytest.mark.parametrize("invalid", [None, "generation", "hit", "counter"])
def test_deferred_response_validates_hit_without_double_counting(invalid):
    proxy = object.__new__(AsyncDraftSpeculator)
    proxy._pending_local_response = (10, 0)
    proxy.request_timeout = 1.0
    response = {
        "status": "ok",
        "generation": 11 if invalid == "generation" else 10,
        "slot": 0,
        "cache_hit_indices": [] if invalid == "hit" else [0],
        "metrics": {
            "cache_hits": 0 if invalid == "counter" else 1,
            "cache_misses": 0,
            "branch_build_seconds": 0.01,
        },
    }
    proxy._recv = lambda *args: response
    recorded = []
    proxy._record_metrics = recorded.append
    if invalid:
        with pytest.raises(RuntimeError):
            proxy._finish_local_response()
    else:
        proxy._finish_local_response()
        assert recorded == [{"cache_misses": 0, "branch_build_seconds": 0.01}]
        proxy._finish_local_response()
        assert len(recorded) == 1
