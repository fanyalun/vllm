# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.router import fused_topk_router
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)


@pytest.mark.parametrize(
    "policy, protected_top_k, expected",
    [
        ("batch_top_half", 2, [1, 3, 0, 2, -1, -1]),
        ("batch_max_gap", 2, [1, 3, 0, -1, -1, -1]),
        ("batch_top_half", 1, [1, 3, 0, 2, -1, -1]),
        ("batch_max_gap", 1, [1, -1, 0, -1, -1, -1]),
    ],
)
def test_batch_reference_protects_union_and_sorts_unsorted_slots(
    policy, protected_top_k, expected
):
    from benchmarks.kernels.moe_batch_policy_reference import batch_policy_reference

    ids = torch.tensor([[1, 3, 0, 2, 4, 5]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.125, 0.5, 0.0625, 0.03125, 0.03125]])
    logits = torch.zeros(1, 8).scatter_(1, ids.long(), weights.log())
    w, actual, stats = batch_policy_reference(
        weights, ids, logits, policy, protected_top_k=protected_top_k
    )
    assert actual.tolist() == [expected]
    assert stats["protected_unique"] == protected_top_k
    assert torch.equal(w, weights.masked_fill(actual < 0, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("policy", ["batch_top_half", "batch_max_gap"])
@pytest.mark.parametrize("case", ["union", "singleton", "empty", "gap_tie"])
def test_batch_policy_boundary_sets_and_ties(policy, case):
    from vllm.model_executor.layers.fused_moe.router.batch_expert_selection import (
        select_batch_experts,
    )

    if case == "union":
        ids = [[0, 1, 2, 3, 4], [2, 3, 0, 1, 5]]
        weights = [[0.5, 0.25, 0.125, 0.0625, 0.0625]] * 2
        expected = [
            [0, 1, 2, 3, 4],
            [2, 3, 0, 1, -1 if policy == "batch_top_half" else 5],
        ]
    elif case == "singleton":
        ids, weights = [[0, 1, 2]], [[0.5, 0.375, 0.125]]
        expected = ids
    elif case == "empty":
        ids, weights = [[0, 1]], [[0.75, 0.25]]
        expected = ids
    else:
        ids = [[0, 1, 2, 3, 4]]
        weights = [[0.40625, 0.3125, 0.125, 0.09375, 0.0625]]
        expected = [[0, 1, 2, 3 if policy == "batch_top_half" else -1, -1]]
    ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    weights = torch.tensor(weights, device="cuda")
    logits = torch.full((ids.shape[0], 8), -100.0, device="cuda")
    logits.scatter_(1, ids.long(), weights.log())
    actual_w, actual_i = select_batch_experts(weights, ids, logits, policy)
    assert actual_i.tolist() == expected
    assert torch.equal(actual_w, weights.masked_fill(actual_i < 0, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("policy", ["batch_top_half", "batch_max_gap"])
@pytest.mark.parametrize("tokens", [0, 1, 5, 33, 320, 640])
@pytest.mark.parametrize("uniform", [False, True])
def test_batch_routing_matches_reference_and_replays_padding(policy, tokens, uniform):
    from benchmarks.kernels.moe_batch_policy_reference import batch_policy_reference
    from vllm.model_executor.layers.fused_moe.router.batch_expert_selection import (
        select_batch_experts,
    )

    torch.manual_seed(34)
    logits = torch.randn(tokens, 256, device="cuda")
    if uniform:
        logits.zero_()
    scores, ids = logits.topk(8, dim=-1, sorted=False)
    ids = ids.int()
    weights = scores.softmax(-1)
    padding = torch.arange(tokens, device="cuda") % 3 == 2
    expected_w, expected_i, _ = batch_policy_reference(
        weights, ids, logits, policy, padding
    )
    actual_w, actual_i = select_batch_experts(weights, ids, logits, policy, padding)
    assert torch.equal(actual_i, expected_i)
    assert torch.equal(actual_w, expected_w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_w, captured_i = select_batch_experts(
            weights, ids, logits, policy, padding
        )
    for all_padding in (True, False, True):
        padding.fill_(all_padding)
        logits.neg_()
        scores, fresh_ids = logits.topk(8, dim=-1, sorted=False)
        ids.copy_(fresh_ids)
        weights.copy_(scores.softmax(-1))
        graph.replay()
        expected_w, expected_i, _ = batch_policy_reference(
            weights, ids, logits, policy, padding
        )
        assert torch.equal(captured_i, expected_i)
        assert torch.equal(captured_w, expected_w)


def test_call_level_routing_top_k_does_not_leak(monkeypatch):
    selected_top_ks = []

    def fake_fused_topk(*, hidden_states, topk, **kwargs):
        selected_top_ks.append(topk)
        shape = (hidden_states.shape[0], topk)
        return (
            torch.zeros(shape),
            torch.zeros(shape, dtype=torch.int32),
            torch.zeros(shape, dtype=torch.int32),
        )

    monkeypatch.setattr(fused_topk_router, "fused_topk", fake_fused_topk)
    router = FusedTopKRouter(top_k=8, global_num_experts=256, renormalize=False)
    hidden_states = torch.zeros(2, 4)
    router_logits = torch.zeros(2, 256)

    router._compute_routing(hidden_states, router_logits, None)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"routing_top_k": 4},
    )
    with override_forward_context(context):
        weights, ids = router._compute_routing(hidden_states, router_logits, None)
    router._compute_routing(hidden_states, router_logits, None)

    assert weights.shape == ids.shape == (2, 4)
    assert selected_top_ks == [8, 4, 8]
    assert router.top_k == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("policy", ["batch_top_half", "batch_max_gap"])
@pytest.mark.parametrize("tokens", [1, 5, 33])
def test_batch_router_dispatch_and_target_isolation(policy, tokens):
    from benchmarks.kernels.moe_batch_policy_reference import batch_policy_reference
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    torch.manual_seed(47)
    logits = torch.randn(tokens, 128, device="cuda")
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(128, 128, 128, device="cuda", dtype=torch.bfloat16) / 16
    w2 = torch.randn(128, 128, 64, device="cuda", dtype=torch.bfloat16) / 16
    router = FusedTopKRouter(top_k=8, global_num_experts=128)
    native_w, native_i = router.select_experts(x, logits)
    expected_w, _, _ = batch_policy_reference(native_w, native_i, logits, policy)
    reference = fused_experts(x, w1, w2, expected_w, native_i)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        is_padding=torch.zeros(tokens, device="cuda", dtype=torch.bool),
        additional_kwargs={
            "routing_batch_policy": policy,
            "routing_preserve_weights": True,
        },
    )

    def forward():
        w, ids = router.select_experts(x, logits)
        return fused_experts(x, w1, w2, w, ids)

    with override_forward_context(context):
        actual = forward()
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        graph.replay()
        torch.testing.assert_close(captured, reference, rtol=0.02, atol=0.01)
        context.is_padding.fill_(True)
        graph.replay()
        assert torch.count_nonzero(captured) == 0
    after_w, after_i = router.select_experts(x, logits)
    assert torch.equal(after_i, native_i)
    assert torch.equal(after_w, native_w)


def test_call_level_routing_top_k_validates_range(monkeypatch):
    monkeypatch.setattr(
        fused_topk_router,
        "fused_topk",
        None,
    )
    router = FusedTopKRouter(top_k=8, global_num_experts=256)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"routing_top_k": 9},
    )
    with (
        override_forward_context(context),
        torch.no_grad(),
        pytest.raises(ValueError, match=r"\[1, 8\]"),
    ):
        router._compute_routing(torch.zeros(1, 4), torch.zeros(1, 256), None)


def test_custom_router_honors_call_level_routing_top_k_without_leaking():
    selected_top_ks = []
    renormalize_values = []

    def custom_routing_function(*, hidden_states, topk, renormalize, **kwargs):
        selected_top_ks.append(topk)
        renormalize_values.append(renormalize)
        shape = (hidden_states.shape[0], topk)
        return torch.zeros(shape), torch.zeros(shape, dtype=torch.int32)

    router = CustomRoutingRouter(
        top_k=8,
        global_num_experts=128,
        custom_routing_function=custom_routing_function,
    )
    hidden_states = torch.zeros(2, 4)
    router_logits = torch.zeros(2, 128)

    router._compute_routing(hidden_states, router_logits, None)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"routing_top_k": 4},
    )
    with override_forward_context(context):
        weights, ids = router._compute_routing(hidden_states, router_logits, None)
    router._compute_routing(hidden_states, router_logits, None)

    assert weights.shape == ids.shape == (2, 4)
    assert selected_top_ks == [8, 4, 8]
    assert renormalize_values == [True, True, True]
    assert router.top_k == 8


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("top_h", [1, 4, 8])
@pytest.mark.parametrize("native_renormalize", [False, True])
def test_draft_weight_mode_retains_native_weights_and_leaves_target_unchanged(
    monkeypatch, custom, preserve, top_h, native_renormalize
):
    logits = torch.tensor([[0.2, 1.7, -0.4, 3.0, 1.1, 0.8, 2.5, 1.3, -1.0]])
    hidden = torch.zeros(1, 4)
    scales = torch.tensor([100.0, 0.1, 0.2, 0.01, 10.0, 4.0, 0.02, 2.0, 1.0])

    def route(*, gating_output, topk, renormalize, **kwargs):
        weights, ids = gating_output.softmax(-1).topk(topk, dim=-1)
        if renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        if custom:
            weights = weights * scales[ids]
        return weights.flip(-1).contiguous(), ids.flip(-1).to(torch.int32).contiguous()

    def fused(**kwargs):
        weights, ids = route(**kwargs)
        return weights, ids, torch.empty_like(ids)

    monkeypatch.setattr(fused_topk_router, "fused_topk", fused)
    kwargs = dict(top_k=8, global_num_experts=9, renormalize=native_renormalize)
    router = (
        CustomRoutingRouter(custom_routing_function=route, **kwargs)
        if custom
        else FusedTopKRouter(**kwargs)
    )
    target_weights, target_ids = router._compute_routing(hidden, logits, torch.int64)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={
            "routing_top_k": top_h,
            "routing_preserve_weights": preserve,
        },
    )
    with override_forward_context(context):
        weights, ids = router._compute_routing(hidden, logits, torch.int64)
    expected_weights, expected_ids = route(
        gating_output=logits,
        topk=8 if preserve else top_h,
        renormalize=native_renormalize,
    )
    expected_by_id = torch.zeros_like(logits).scatter_(
        1, expected_ids.long(), expected_weights
    )
    assert weights.shape == ids.shape == (1, top_h)
    assert torch.equal(
        ids.long().sort(-1).values, logits.topk(top_h).indices.sort(-1).values
    )
    torch.testing.assert_close(
        weights, expected_by_id.gather(1, ids.long()), rtol=0, atol=0
    )
    after_weights, after_ids = router._compute_routing(hidden, logits, torch.int64)
    assert torch.equal(after_weights, target_weights)
    assert torch.equal(after_ids, target_ids)
    assert router.top_k == 8 and router.renormalize == native_renormalize


def test_preserved_weights_keep_native_expert_order_for_tied_gates():
    def tied_routing(*, topk, **kwargs):
        ids = torch.arange(topk, dtype=torch.int32).unsqueeze(0)
        return (ids.float() + 1) / topk, ids

    router = CustomRoutingRouter(
        top_k=8, global_num_experts=8, custom_routing_function=tied_routing
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"routing_top_k": 4, "routing_preserve_weights": True},
    )
    with override_forward_context(context):
        weights, ids = router._compute_routing(
            torch.zeros(1, 4), torch.zeros(1, 8), None
        )
    assert ids.tolist() == [[0, 1, 2, 3]]
    assert weights.tolist() == [[0.125, 0.25, 0.375, 0.5]]


@pytest.mark.parametrize("invalid_id", [-1, 8])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_preserved_weights_handle_invalid_experts_during_graph_capture(
    invalid_id, all_invalid
):
    ids = torch.full((1, 8), invalid_id, dtype=torch.int32)
    if not all_invalid:
        ids[0, 2:] = torch.arange(6)
    weights = torch.arange(8).float().unsqueeze(0)
    router = CustomRoutingRouter(
        top_k=8,
        global_num_experts=8,
        custom_routing_function=lambda **kwargs: (weights, ids),
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"routing_top_k": 4, "routing_preserve_weights": True},
    )
    with override_forward_context(context):
        selected_weights, selected_ids = router._compute_routing(
            torch.zeros(1, 4), torch.zeros(1, 8), None
        )
    start = 0 if all_invalid else 2
    assert torch.equal(selected_weights, weights[:, start : start + 4])
    assert torch.equal(selected_ids, ids[:, start : start + 4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 4, 5, 16, 32, 33, 64])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("threshold", [0.125, 0.625])
def test_threshold_dispatch_matches_zero_weight_reference(tokens, uniform, threshold):
    """Skipped slots must be initialized on both packed and aligned paths."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.weight_threshold import (
        threshold_experts,
    )

    torch.manual_seed(0)
    logits = torch.randn(tokens, 128, device="cuda")
    if uniform:
        logits.zero_()
    scores, ids = logits.topk(8, dim=-1)
    ids = ids.to(torch.int32)
    q = scores.softmax(-1)
    weights = q * torch.linspace(0.1, 2, 128, device="cuda")[ids.long()]
    keep = q >= threshold
    actual_weights, actual_ids = threshold_experts(weights, ids, logits, threshold)
    assert torch.equal(actual_ids, ids.masked_fill(~keep, -1))
    assert torch.equal(actual_weights, weights.masked_fill(~keep, 0))
    renormalized, renormalized_ids = threshold_experts(
        weights, ids, logits, threshold, renormalize=True
    )
    mass = q.masked_fill(~keep, 0).sum(-1, keepdim=True)
    expected = weights.masked_fill(~keep, 0) / torch.where(mass > 0, mass, 1)
    torch.testing.assert_close(renormalized, expected)
    assert torch.equal(renormalized_ids, actual_ids)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(128, 128, 128, device="cuda", dtype=torch.bfloat16) / 16
    w2 = torch.randn(128, 128, 64, device="cuda", dtype=torch.bfloat16) / 16
    reference = fused_experts(x, w1, w2, actual_weights, ids)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"routing_min_weight": threshold},
    )
    with override_forward_context(context):
        actual = fused_experts(x, w1, w2, actual_weights, actual_ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_experts(x, w1, w2, actual_weights, actual_ids)
        graph.replay()
        torch.accelerator.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
    torch.testing.assert_close(captured, reference, rtol=0.02, atol=0.01)
    actual_ids.fill_(-1)
    actual_weights.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.count_nonzero(captured) == 0
