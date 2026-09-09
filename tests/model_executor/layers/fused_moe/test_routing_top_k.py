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
