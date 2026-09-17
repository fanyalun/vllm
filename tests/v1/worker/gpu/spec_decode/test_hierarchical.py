# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.hierarchical.speculator import (
    HierarchicalSpeculator,
    accepted_prefix,
    refresh_graph_metadata,
    should_stop_inner,
)
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


@pytest.mark.parametrize(
    "policy,accepted,margin,expected",
    [
        ("low_error", 0, 1.5, True),
        ("low_error", 0, 2.0, False),
        ("low_error", 1, 0.25, False),
        ("low_error", 1, 0.125, True),
        ("balanced", 2, 0.5, True),
        ("balanced", 0, 1.0, False),
        ("aggressive", 3, 1.5, True),
        ("aggressive", 3, 2.0, False),
        ("aggressive", 4, 0.0, False),
        ("none", 0, 0.0, False),
    ],
)
def test_stop_policy_requires_correction_and_uses_strict_margin(
    policy, accepted, margin, expected
):
    assert should_stop_inner(policy, accepted, 4, margin) is expected


def test_batched_stop_compacts_survivors_and_preserves_request_mapping(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.hierarchical.batched import propose_gemma

    spec = SimpleNamespace(
        depth=4,
        rounds=2,
        device=torch.device("cpu"),
        config=SimpleNamespace(hierarchical_stop_policy="low_error"),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
        state=SimpleNamespace(layers={}),
        draft_tokens=torch.full((3, 10), -1, dtype=torch.int64),
        draft_lengths=torch.zeros(3, dtype=torch.int32),
        last_sampled=torch.zeros((3, 1), dtype=torch.int64),
    )
    HierarchicalSpeculator.reset_policy_metrics(spec)
    batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["a", "b", "c"],
        idx_mapping=torch.tensor([2, 0, 1]),
        seq_lens=torch.tensor([10, 20, 30]),
        has_structured_output_reqs=False,
    )
    calls = []

    def make_batch(spec, original, rows, positions, tokens):
        calls.append((list(rows), list(positions), tokens.clone()))
        return SimpleNamespace(idx_mapping=original.idx_mapping[rows]), {}, {}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.hierarchical.batched.make_batch", make_batch
    )
    spec.small = SimpleNamespace(
        propose=Mock(
            side_effect=[torch.tensor([[1, 2, 3, 4]] * 3), torch.tensor([[1, 2, 3, 4]])]
        )
    )

    def verify(batch, metadata, slots):
        n = len(batch.idx_mapping)
        spec.last_margins = torch.tensor([1.5] * (n * 5))
        predictions = torch.tensor([[9, 2, 3, 4, 8], [1, 2, 9, 4, 8]])
        if n == 1:
            predictions = torch.tensor([[1, 2, 3, 4, 8]])
        return predictions.flatten(), torch.zeros(n * 5, 2), None

    spec._verify = verify
    output = propose_gemma(
        spec,
        batch,
        {},
        {},
        torch.zeros(3, 2),
        torch.tensor([1, 1, 0]),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([[10], [20], [30]]),
        None,
        torch.zeros(3),
        torch.zeros(3, dtype=torch.int64),
    )
    assert calls[0][0] == [0, 1]
    assert calls[0][2][:, 0].tolist() == [30, 10]
    assert calls[1][0] == calls[2][0] == [1]
    assert calls[2][1] == [23]
    assert spec.draft_lengths.tolist() == [1, 8, 0]
    assert output[0, :1].tolist() == [9]
    assert output[1, :8].tolist() == [1, 2, 9, 1, 2, 3, 4, 8]
    assert output[2].eq(-1).all()
    assert spec.policy_metrics["early_stops"] == 1
    assert spec.policy_metrics["skipped_rounds"] == 1


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("max_num_seqs", 2, "batching requires"),
        ("async_scheduling", True, "async_scheduling=False"),
        ("enable_prefix_caching", True, "prefix caching disabled"),
        ("enable_prompt_embeds", True, "prompt embeddings"),
        ("multimodal_config", Mock(get_limit_per_prompt=lambda _: 1), "image"),
    ],
)
def test_unsupported_runtime_configuration_fails_before_loading(field, value, message):
    scheduler = SimpleNamespace(max_num_seqs=1, async_scheduling=False)
    cache = SimpleNamespace(enable_prefix_caching=False)
    model = SimpleNamespace(enable_prompt_embeds=False, multimodal_config=None)
    for owner in (scheduler, cache, model):
        if hasattr(owner, field):
            setattr(owner, field, value)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            preverify_gdn_update_policy="exact",
            inner_num_speculative_tokens=4,
            inner_num_rounds=4,
            num_speculative_tokens=20,
        ),
        scheduler_config=scheduler,
        cache_config=cache,
        model_config=model,
        lora_config=None,
    )
    with pytest.raises(ValueError, match=message):
        HierarchicalSpeculator(config, torch.device("cpu"))


def test_chunked_prefill_without_sample_proposes_nothing():
    proposer = object.__new__(HierarchicalSpeculator)
    proposer.draft_tokens = torch.empty((1, 20), dtype=torch.int64)
    proposer.draft_lengths = torch.ones(1, dtype=torch.int32)
    batch = SimpleNamespace(req_ids=["request"], num_reqs=1)
    result = proposer.propose(
        batch,
        None,
        None,
        None,
        None,
        torch.zeros(1, dtype=torch.int32),
        None,
        None,
        None,
        None,
        None,
    )
    assert proposer.draft_lengths.tolist() == [0]
    assert result.eq(-1).all()


def test_graph_metadata_refresh_preserves_captured_addresses():
    @dataclass
    class Metadata:
        width: int
        lengths: torch.Tensor

    captured = {"layer": Metadata(5, torch.tensor([8]))}
    address = captured["layer"].lengths.data_ptr()
    refresh_graph_metadata(captured, {"layer": Metadata(5, torch.tensor([17]))})
    assert captured["layer"].lengths.tolist() == [17]
    assert captured["layer"].lengths.data_ptr() == address
    with pytest.raises(ValueError, match="static structure"):
        refresh_graph_metadata(captured, {"layer": Metadata(4, torch.tensor([17]))})


@pytest.mark.parametrize("fail", [False, True])
def test_inner_mtp_refreshes_graph_lengths_and_restores_target_on_exit(fail):
    proposer = object.__new__(HierarchicalSpeculator)
    buffers = SimpleNamespace(
        seq_lens=torch.tensor([131]), query_start_loc=torch.tensor([0, 131])
    )
    proposer.small = SimpleNamespace(target_input_buffers=buffers)
    proposer.refresh_small_lengths = True
    proposer.saved_small_seq_lens = torch.empty_like(buffers.seq_lens)
    proposer.saved_small_query_start = torch.empty_like(buffers.query_start_loc)
    inner = SimpleNamespace(
        seq_lens=torch.tensor([136]), query_start_loc=torch.tensor([0, 5])
    )
    addresses = (buffers.seq_lens.data_ptr(), buffers.query_start_loc.data_ptr())
    try:
        with proposer._small_metadata(inner, 1):
            assert buffers.seq_lens.tolist() == [136]
            assert buffers.query_start_loc.tolist() == [0, 5]
            if fail:
                raise RuntimeError("draft failed")
    except RuntimeError:
        assert fail
    assert buffers.seq_lens.tolist() == [131]
    assert buffers.query_start_loc.tolist() == [0, 131]
    assert addresses == (
        buffers.seq_lens.data_ptr(),
        buffers.query_start_loc.data_ptr(),
    )
    assert inner.seq_lens.tolist() == [136]
    with proposer._small_metadata(inner, 0):
        assert buffers.seq_lens.tolist() == [131]


def test_scheduler_receives_only_materialized_candidates():
    handler = object.__new__(DraftTokensHandler)
    handler.req_ids = ["short", "empty"]
    handler.draft_tokens_np = np.array([[11, 12, -1, -1], [-1, -1, -1, -1]])
    handler.draft_lengths_np = np.array([2, 0])
    handler.copy_event = Mock()
    result = handler.get_draft_tokens()
    assert result.req_ids == ["short", "empty"]
    assert result.draft_token_ids == [[11, 12], []]
    handler.copy_event.synchronize.assert_called_once()


@pytest.mark.parametrize("mode", ["none", "ssm_mean", "input_mean", "replay_tail"])
def test_private_candidates_allocate_required_state_slots(monkeypatch, mode):
    class Layer:
        prefix = "layer"
        conv_kernel_size = 4

        def get_state_shape(self):
            return ((8, 2), (2, 2))

        def get_state_dtype(self):
            return (torch.float32, torch.float32)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.hierarchical.state.QwenGatedDeltaNetAttention",
        Layer,
    )
    state = PreverifyState(
        SimpleNamespace(modules=lambda: [Layer()]), 5, torch.device("cpu"), mode
    )
    expected = [[1, 2, 3, 4, 5]] if mode == "none" else [[0] * 5]
    assert state.state_indices.tolist() == expected
    slots = 6 if mode == "none" else 1
    assert all(cache.shape[0] == slots for cache in state.caches["layer"])


def test_attention_only_preverify_does_not_access_recurrent_state():
    state = PreverifyState(SimpleNamespace(modules=lambda: []), 5, torch.device("cpu"))
    state.begin(None, None, None, None)
    state.advance(3)
    with state.activate():
        assert state.snapshot() == {}


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("accepted", range(1, 6))
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("policy", ["three_level_p50", "windowed_three_level"])
@torch.inference_mode()
def test_three_level_batched_begin_copies_accepted_canonical_state(
    monkeypatch, align, accepted, dim_first, policy
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.worker.gpu.spec_decode.hierarchical import state as impl

    class Layer:
        prefix = "layer"
        conv_kernel_size = 4

        def get_state_shape(self):
            return ((7, 10) if dim_first else (10, 7), (2, 8, 8))

        kv_cache: tuple[torch.Tensor, torch.Tensor]

        def get_state_dtype(self):
            return (torch.bfloat16, torch.float32)

    monkeypatch.setattr(impl, "QwenGatedDeltaNetAttention", Layer)
    monkeypatch.setattr(impl, "is_conv_state_dim_first", lambda: dim_first)
    layer = Layer()
    layer.kv_cache = (
        torch.randn(
            (10, 7, 10) if dim_first else (10, 10, 7),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.randn(10, 2, 8, 8, device="cuda"),
    )
    canonical = tuple(x.clone() for x in layer.kv_cache)
    state = PreverifyState(
        SimpleNamespace(modules=lambda: [layer]),
        5,
        torch.device("cuda"),
        "replay_tail",
        policy,
    )
    table = torch.tensor(
        [[5, 2, 7, 1, 4, 0, 6, 3, 9, 8]], dtype=torch.int32, device="cuda"
    )
    model_state = SimpleNamespace(
        num_accepted_tokens_gpu=torch.tensor(
            [accepted], device="cuda", dtype=torch.int32
        ),
        _mamba_state_idx_gpu=torch.tensor([2], device="cuda", dtype=torch.int32),
        _align_mode=align,
    )
    batch = SimpleNamespace(
        idx_mapping=torch.tensor([0], device="cuda", dtype=torch.int32),
        req_ids=["request"],
    )
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"])])
    start = 2 if align else 0
    expected_conv = canonical[0][table[0, start]].index_select(
        1 if dim_first else 0,
        (torch.arange(8, device="cuda") + accepted - 1).clamp(max=9),
    )
    expected_state = canonical[1][table[0, start + accepted - 1]]
    for epoch in range(1, 3):
        state.begin(model_state, batch, (table,), config)
        if state.windowed:
            assert state.outer_epoch == epoch and state.request_id == "request"
            assert state.initialized and state.valid.item() == 1
            assert not state.tails and not state.repair_inputs
        torch.testing.assert_close(
            state.caches["layer"][0][0], expected_conv, rtol=0, atol=0
        )
        torch.testing.assert_close(
            state.caches["layer"][1][0], expected_state, rtol=0, atol=0
        )
        state.advance(4)
    for actual, expected in zip(layer.kv_cache, canonical, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("mode", ["none", "replay_tail"])
@pytest.mark.parametrize("size", [4, 8, 16, 32])
@torch.inference_mode()
def test_batched_private_state_tracks_compaction_rejection_and_new_epoch(
    monkeypatch, dim_first, mode, size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm.v1.worker.gpu.spec_decode.hierarchical import state as impl
    from vllm.v1.worker.gpu.spec_decode.hierarchical.batched_state import (
        BatchedPreverifyState,
    )

    class Layer:
        prefix = "layer"
        conv_kernel_size = 4
        kv_cache: tuple[torch.Tensor, torch.Tensor]

        def get_state_shape(self):
            return ((7, 10) if dim_first else (10, 7), (2, 8, 8))

        def get_state_dtype(self):
            return (torch.bfloat16, torch.float32)

    monkeypatch.setattr(impl, "QwenGatedDeltaNetAttention", Layer)
    monkeypatch.setattr(impl, "is_conv_state_dim_first", lambda: dim_first)
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: dim_first)
    layer = Layer()
    layer.kv_cache = (
        torch.randn(
            (10 * size, 7, 10) if dim_first else (10 * size, 10, 7),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.randn(10 * size, 2, 8, 8, device="cuda"),
    )
    canonical = tuple(x.clone() for x in layer.kv_cache)
    state = BatchedPreverifyState(
        SimpleNamespace(modules=lambda: [layer]),
        5,
        torch.device("cuda"),
        mode,
        "exact" if mode == "none" else "windowed_three_level",
        max_num_reqs=size,
    )
    table = torch.arange(10 * size, device="cuda", dtype=torch.int32).view(size, 10)
    model_state = SimpleNamespace(
        num_accepted_tokens_gpu=torch.arange(size, device="cuda") % 4 + 1,
        _align_mode=False,
    )
    batch = SimpleNamespace(
        num_reqs=size,
        req_ids=[f"request_{i}" for i in range(size)],
        idx_mapping=torch.arange(size, device="cuda", dtype=torch.int32).flip(0),
    )
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"])])
    state.begin(model_state, batch, [table], config)
    initial = state.snapshot()
    width = state.slots_per_req
    for i in range(size):
        bias = (size - 1 - i) % 4
        slot = i * width + (mode == "none")
        torch.testing.assert_close(
            initial["layer"][1][slot], canonical[1][i * 10 + bias], atol=0, rtol=0
        )
    state.select([batch.req_ids[-1], batch.req_ids[1]])
    consumed = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    state.advance(consumed)
    after = state.snapshot()
    for i in set(range(size)) - {1}:
        for x, y in zip(initial["layer"], after["layer"], strict=True):
            torch.testing.assert_close(
                x[i * width : (i + 1) * width],
                y[i * width : (i + 1) * width],
                atol=0,
                rtol=0,
            )
    if mode == "replay_tail":
        torch.testing.assert_close(
            initial["layer"][1], after["layer"][1], atol=0, rtol=0
        )
    state.invalidate()
    with pytest.raises(RuntimeError, match="invalid"):
        state.select(["request_1"])
    batch.req_ids = [f"new_{i}" for i in range(size)]
    state.begin(model_state, batch, [table], config)
    assert state.outer_epoch == 2
    with pytest.raises(RuntimeError, match="invalid"):
        state.select(["request_1"])
    for x, y in zip(layer.kv_cache, canonical, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.mark.parametrize("accepted", range(5))
@torch.inference_mode()
def test_three_level_repair_alternates_private_buffers_and_restores_target(
    monkeypatch, accepted
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
        replay_tail_update,
    )
    from vllm.v1.worker.gpu.spec_decode.hierarchical import state as impl

    class Layer:
        prefix = "layer"
        conv_kernel_size = 4
        num_k_heads, num_v_heads = 1, 2
        head_k_dim = head_v_dim = 128
        kv_cache: tuple[torch.Tensor, torch.Tensor]
        A_log: torch.Tensor
        dt_bias: torch.Tensor

        def get_state_shape(self):
            return ((4, 8), (2, 128, 128))

        def get_state_dtype(self):
            return (torch.bfloat16, torch.float32)

    monkeypatch.setattr(impl, "QwenGatedDeltaNetAttention", Layer)
    monkeypatch.setattr(impl, "is_conv_state_dim_first", lambda: True)
    layer = Layer()
    layer.A_log = torch.zeros(2, device="cuda")
    layer.dt_bias = torch.zeros(2, device="cuda")
    layer.kv_cache = (
        torch.randn(1, 4, 8, device="cuda"),
        torch.randn(1, 2, 128, 128, device="cuda"),
    )
    canonical = tuple(x.clone() for x in layer.kv_cache)
    original = layer.kv_cache
    state = PreverifyState(
        SimpleNamespace(modules=lambda: [layer]),
        5,
        torch.device("cuda"),
        "replay_tail",
        "three_level_p50",
        "repair_on_reject",
    )
    pointers = {state.caches["layer"][1].data_ptr(), state.tails["layer"].data_ptr()}
    state.caches["layer"][1].normal_(std=0.05)
    q, k = [
        torch.randn(5, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    v = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16)
    a, b = [torch.randn(5, 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    for window in range(3):
        initial = state.caches["layer"][1].clone()
        expected = torch.empty_like(initial)
        length = accepted + 1
        replay_tail_update(
            q[:length],
            k[:length],
            v[:length],
            a[:length],
            b[:length],
            layer.A_log,
            layer.dt_bias,
            initial,
            tail=expected,
            thresholds=state.thresholds,
        )
        state.update(layer, q, k, v, a, b)
        state.advance(accepted)
        torch.testing.assert_close(
            state.caches["layer"][1], expected, rtol=1e-3, atol=1e-3
        )
        assert state.direction == (window + 1) % 2
        assert {
            state.caches["layer"][1].data_ptr(),
            state.tails["layer"].data_ptr(),
        } == pointers
    with pytest.raises(RuntimeError, match="test recovery"), state.activate():
        assert layer.kv_cache is state.caches["layer"]
        raise RuntimeError("test recovery")
    assert layer.kv_cache is original
    for actual, expected in zip(layer.kv_cache, canonical, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["ssm_mean", "input_mean", "replay_tail"])
@pytest.mark.parametrize("accepted", range(5))
def test_approximate_state_survives_inner_rejection(monkeypatch, mode, accepted):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.hierarchical.state.is_conv_state_dim_first",
        lambda: False,
    )
    state = object.__new__(PreverifyState)
    state.mode = mode
    conv = torch.arange(16).view(1, 8, 2).float()
    temporal = torch.tensor([[[42.0]]])
    state.caches = {"layer": (conv, temporal)}
    state.advance(accepted)
    assert temporal.item() == 42
    offset = accepted if mode in ("ssm_mean", "replay_tail") else 0
    torch.testing.assert_close(conv[0, :3, 0], torch.arange(offset, offset + 3) * 2.0)


@pytest.mark.parametrize("accepted", range(5))
def test_inner_verification_stops_at_first_mismatch(accepted):
    draft = torch.tensor([1, 2, 3, 4])
    predictions = torch.tensor([1, 2, 3, 4, 5])
    if accepted < 4:
        predictions[accepted] = 9
    assert accepted_prefix(draft, predictions) == accepted


@pytest.mark.parametrize("accepted", range(5))
@pytest.mark.parametrize("dim_first", [False, True])
def test_preverify_restores_conv_and_temporal_at_same_accepted_position(
    monkeypatch,
    accepted,
    dim_first,
):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.hierarchical.state.is_conv_state_dim_first",
        lambda: dim_first,
    )
    state = object.__new__(PreverifyState)
    conv = torch.arange(16, dtype=torch.float32).view(1, 8, 2).repeat(6, 1, 1)
    if dim_first:
        conv = conv.transpose(1, 2).contiguous()
    temporal = torch.arange(6, dtype=torch.float32).view(6, 1, 1)
    state.caches = {"layer": (conv, temporal)}
    state.advance(accepted)
    history = conv[1].transpose(0, 1) if dim_first else conv[1]
    expected = torch.arange(accepted, accepted + 3).float() * 2
    torch.testing.assert_close(history[:3, 0], expected)
    assert temporal[1].item() == accepted + 1


@pytest.mark.parametrize("windowed", [False, True])
def test_preverify_restores_target_cache_bindings_after_forward_failure(windowed):
    state = object.__new__(PreverifyState)
    state.windowed = windowed
    state.valid = torch.ones(1, dtype=torch.int32)
    state.initialized, state.request_id = True, "request"
    canonical = (torch.ones(2), torch.ones(2))
    private = (torch.zeros(2), torch.zeros(2))
    layer = SimpleNamespace(kv_cache=canonical)
    state.layers = {"layer": layer}
    state.caches = {"layer": private}
    with pytest.raises(RuntimeError, match="forward failed"), state.activate():
        assert layer.kv_cache is private
        raise RuntimeError("forward failed")
    assert layer.kv_cache is canonical
    if windowed:
        assert state.valid.item() == 0
        assert not state.initialized and state.request_id is None


@pytest.mark.parametrize("accepted", [0, 2, 4])
def test_windowed_rejection_keeps_single_tail_and_advances_only_conv(
    monkeypatch, accepted
):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.hierarchical.state.is_conv_state_dim_first",
        lambda: False,
    )
    state = object.__new__(PreverifyState)
    state.windowed, state.mode, state.tail_policy = True, "replay_tail", "carry"
    state.thresholds = torch.tensor([0.95, 0.36328125])
    state.execution_optimized, state.conv_pointers = False, None
    state.direction = 0
    state.tails, state.repair_inputs, state.repair_graphs = {}, {}, {}
    conv = torch.arange(8).view(1, 8, 1).float()
    temporal = torch.tensor([[[[7.0]]]])
    state.caches = {"layer": (conv, temporal)}
    original = temporal.clone()
    for _ in range(3):
        state.advance(accepted)
        assert state.caches["layer"][1] is temporal
        assert torch.equal(temporal, original)
    assert state.direction == 0 and state.tails == {}
    expected = (torch.arange(8) + 3 * accepted).clamp(max=7).float()
    assert torch.equal(conv.flatten(), expected)


@pytest.mark.parametrize("accepted", [1, 3, 5])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("mode", ["none", "ssm_mean", "input_mean", "replay_tail"])
def test_outer_reset_copies_only_target_accepted_state(
    monkeypatch, accepted, dim_first, mode
):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.hierarchical.state.is_conv_state_dim_first",
        lambda: dim_first,
    )
    state = object.__new__(PreverifyState)
    state.mode = mode
    slot = 1 if mode == "none" else 0
    conv = torch.arange(160).view(10, 8, 2).float()
    if dim_first:
        conv = conv.transpose(1, 2).contiguous()
    temporal = torch.arange(10).view(10, 1, 1).float()
    canonical = (conv.clone(), temporal.clone())
    state.layers = {"layer": SimpleNamespace(kv_cache=(conv, temporal))}
    state.caches = {
        "layer": (torch.zeros_like(conv[:6]), torch.zeros_like(temporal[:6]))
    }
    model_state = SimpleNamespace(
        num_accepted_tokens_gpu=torch.tensor([accepted]),
        _mamba_state_idx_gpu=torch.tensor([1]),
        _align_mode=True,
    )
    batch = SimpleNamespace(idx_mapping=torch.tensor([0]))
    table = torch.tensor([[0, 2, 4, 6, 8, 9]])
    cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer"])]
    )
    state.begin(model_state, batch, (table,), cache_config)
    private_conv, private_temporal = state.caches["layer"]
    history = private_conv[slot].transpose(0, 1) if dim_first else private_conv[slot]
    expected = canonical[0][2].transpose(0, 1) if dim_first else canonical[0][2]
    torch.testing.assert_close(history[:3], expected[accepted - 1 : accepted + 2])
    assert private_temporal[slot].item() == table[0, accepted].item()
    with state.activate():
        state.layers["layer"].kv_cache[1].fill_(-1)
    state.begin(model_state, batch, (table,), cache_config)
    assert private_temporal[slot].item() == table[0, accepted].item()
    torch.testing.assert_close(conv, canonical[0])
    torch.testing.assert_close(temporal, canonical[1])


@pytest.mark.parametrize("mode", ["projection", "full"])
def test_grouped_gdn_preserves_sequential_residual_and_moe(mode):
    from vllm.v1.worker.gpu.spec_decode.hierarchical.grouped_gdn import (
        GroupedGDNPreverify,
    )

    class Norm:
        def __call__(self, hidden, residual):
            combined = hidden + residual
            return combined, combined

    class Decoder:
        def __init__(self, index):
            self.linear_attn = index
            self.post_attention_layernorm = Norm()
            self.mlp = lambda x: x * (index + 2)

    layers = [Decoder(i) for i in range(3)]
    model = SimpleNamespace(
        layers=layers,
        embed_input_ids=lambda ids: ids[:, None].to(torch.bfloat16),
        _maybe_add_hidden_state=lambda aux, *args: aux,
        norm=Norm(),
    )
    plan = object.__new__(GroupedGDNPreverify)
    plan.model, plan.by_start = model, {0: (0, 1, 2)}
    plan.normalize = lambda module, hidden, residual: module(hidden, residual)
    anchors = []

    def projections(group, anchor, serial):
        anchors.append(anchor.clone())
        return torch.stack([anchor * i for i in (1, 2, 3)]).to(torch.bfloat16), None

    plan.projections = projections
    plan.branches = lambda group, qkvz, ba, state: qkvz
    plan.branch = lambda layer, qkvz, ba, state: qkvz
    # Projection mode indexes the BA tensor even though this fake branch ignores it.
    if mode == "projection":
        plan.projections = lambda group, anchor, serial: (
            projections(group, anchor, serial)[0],
            torch.zeros(3),
        )
    ids = torch.tensor([1, 2])
    result = plan(ids, ids, mode, "replay_tail")
    anchor = ids[:, None].to(torch.bfloat16)
    current = anchor
    for i in range(3):
        residual = current + anchor * (i + 1)
        current = residual + layers[i].mlp(residual)
    torch.testing.assert_close(result, current)
    assert len(anchors) == 1
    torch.testing.assert_close(anchors[0], anchor.float())


def test_grouped_gdn_selects_middle_groups_and_rejects_short_layout():
    from vllm.v1.worker.gpu.spec_decode.hierarchical.grouped_gdn import selected_groups

    layout = ["linear_attention"] * 3 + ["full_attention"]
    assert selected_groups(layout * 10) == tuple(
        (4 * group, 4 * group + 1, 4 * group + 2) for group in range(2, 9)
    )
    with pytest.raises(ValueError, match="three-layer"):
        selected_groups(layout * 8)
    with pytest.raises(ValueError, match="three-layer"):
        selected_groups((layout * 10)[:10] + ["full_attention"] + (layout * 10)[11:])


def test_grouped_gdn_rejects_float32_conv_before_kernel_setup(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.hierarchical import speculator as impl

    proposer = object.__new__(HierarchicalSpeculator)
    proposer.preverify = SimpleNamespace(
        load_model=lambda target: None, model=object(), model_family="qwen3_6"
    )
    proposer.config = SimpleNamespace(
        preverify_gdn_mode="none",
        preverify_gdn_group_mode="full",
        preverify_gdn_update_policy="exact",
        preverify_gdn_tail_policy="carry",
        preverify_gdn_mode_window_size=5,
        preverify_gdn_tau_alpha=0.95,
        preverify_gdn_tau_beta=0.36328125,
        preverify_gdn_optimization="none",
    )
    proposer.depth, proposer.device = 4, torch.device("cuda")
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1), quant_config=None
    )
    state = SimpleNamespace(caches={"gdn": (torch.zeros(1), torch.zeros(1))})
    monkeypatch.setattr(impl, "PreverifyState", lambda *args: state)
    with pytest.raises(ValueError, match="BF16 Conv"):
        proposer.load_model(object())


@pytest.mark.parametrize("policy", ["exact", "three_level_p50", "windowed_three_level"])
@pytest.mark.parametrize("short_window", [False, True])
def test_four_rounds_compact_recoveries_and_start_after_computed_prefix(
    policy, short_window
):
    proposer = object.__new__(HierarchicalSpeculator)
    proposer.device = torch.device("cpu")
    proposer.config = SimpleNamespace(
        inner_method="mtp",
        preverify_gdn_update_policy=policy,
    )
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=9 if short_window else 128)
    )
    proposer.depth, proposer.rounds, proposer.capacity = 4, 4, 20
    proposer.draft_tokens = torch.empty((1, 20), dtype=torch.int64)
    proposer.draft_lengths = torch.zeros(1, dtype=torch.int32)
    proposer.last_sampled = torch.zeros((1, 1), dtype=torch.int64)
    proposer.state = Mock()
    proposer.state.windowed = policy == "windowed_three_level"
    proposer.model_state = Mock()
    proposer.kv_cache_config = Mock()
    proposer.block_tables = Mock()
    proposer.check_preverify = False
    proposer.refresh_small_lengths = False
    small = torch.tensor([[1, 2, 3, 4]])
    proposer.small = SimpleNamespace(propose=Mock(return_value=small))
    positions = []

    def make_batch(template, position, tokens):
        positions.append(position)
        return SimpleNamespace(input_ids=tokens, num_tokens=tokens.numel()), {}, {}

    proposer._batch = make_batch
    hidden = torch.zeros((5, 2))
    proposer._verify = Mock(
        side_effect=[
            (torch.tensor([9, 2, 3, 4, 9]), hidden, None),
            (torch.tensor([1, 9, 3, 4, 9]), hidden, None),
            (torch.tensor([1, 2, 3, 9, 9]), hidden, None),
            (torch.tensor([1, 2, 3, 4, 9]), hidden, None),
        ]
    )
    batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        has_structured_output_reqs=False,
        idx_mapping=torch.tensor([0]),
        seq_lens=torch.tensor([10]),
    )
    result = proposer.propose(
        batch,
        {},
        {},
        hidden,
        None,
        torch.tensor([3]),
        torch.tensor([2]),
        torch.tensor([[7]]),
        torch.zeros((1, 1)),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.int64),
    )
    expected_tokens = [9] if short_window else [9, 1, 9, 1, 2, 3, 9, 1, 2, 3, 4, 9]
    assert positions == ([8] if short_window else [8, 9, 11, 15])
    assert proposer.draft_lengths.tolist() == [len(expected_tokens)]
    assert result[0, : len(expected_tokens)].tolist() == expected_tokens
    assert result[0, len(expected_tokens) :].eq(-1).all()
    advances = [call.args[0] for call in proposer.state.advance.call_args_list]
    expected_advances = [0] if short_window else [0, 1, 3, 4]
    assert advances == (
        expected_advances[:-1] if policy != "exact" else expected_advances
    )
    if policy == "windowed_three_level":
        proposer.state.invalidate.assert_called_once()
