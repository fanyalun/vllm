# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental top-p within native top-8, enabled only in benchmark workers."""

import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _truncate(W, IDS, G, C, OW, OI, E: tl.constexpr, P: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 8)
    ids = tl.load(IDS + row * 8 + col)
    weights = tl.load(W + row * 8 + col)
    if P == 1.0:
        keep = tl.full((8,), True, tl.int1)
        mass = 1.0
    else:
        scores = tl.load(G + row * E + ids).to(tl.float32)
        probs = tl.exp(scores - tl.max(scores, 0))
        probs = probs / tl.sum(probs, 0)
        precedes = (scores[None, :] > scores[:, None]) | (
            (scores[None, :] == scores[:, None]) & (col[None, :] < col[:, None])
        )
        before = tl.sum(tl.where(precedes, probs[None, :], 0.0), 1)
        keep = before < P
        mass = tl.sum(tl.where(keep, probs, 0.0), 0)
    tl.store(OW + row * 8 + col, tl.where(keep, weights / mass, 0.0))
    tl.store(OI + row * 8 + col, tl.where(keep, ids, -1))
    h = tl.sum(keep.to(tl.int32), 0)
    tl.atomic_add(C + h - 1, 1)


def truncate(weights, ids, logits, counts, p):
    if weights.shape[-1] != 8 or not 0 < p <= 1:
        raise ValueError("Top-p benchmark requires native top-8 and 0 < p <= 1")
    if weights.shape[0] * 8 * 4 > logits.shape[1]:
        raise ValueError("Top-p requires naive MoE assignment: 4 * tokens * 8 <= E")
    out_weights, out_ids = torch.empty_like(weights), torch.empty_like(ids)
    _truncate[(weights.shape[0],)](
        weights,
        ids,
        logits,
        counts,
        out_weights,
        out_ids,
        logits.shape[1],
        p,
        num_warps=4,
    )
    return out_weights, out_ids


def install(p):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.utils.torch_utils import direct_register_custom_op

    def operation(
        weights: torch.Tensor,
        ids: torch.Tensor,
        logits: torch.Tensor,
        counts: torch.Tensor,
        probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return truncate(weights, ids, logits, counts, probability)

    def fake(weights, ids, logits, counts, probability):
        return torch.empty_like(weights), torch.empty_like(ids)

    direct_register_custom_op(
        op_name="benchmark_moe_top_p",
        op_func=operation,
        mutates_args=["counts"],
        fake_impl=fake,
    )

    def patch(cls):
        init, route = cls.__init__, cls._compute_routing

        def initialize(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self.benchmark_budget_counts = torch.zeros(
                8, dtype=torch.int64, device="cuda"
            )

        def routing(
            self, hidden_states, router_logits, indices_type, *, input_ids=None
        ):
            weights, ids = route(
                self, hidden_states, router_logits, indices_type, input_ids=input_ids
            )
            if "routing_top_k" not in get_forward_context().additional_kwargs:
                return weights, ids
            return torch.ops.vllm.benchmark_moe_top_p(
                weights,
                ids,
                router_logits,
                self.benchmark_budget_counts,
                p,
            )

        cls.__init__, cls._compute_routing = initialize, routing

    for cls in (FusedTopKRouter, CustomRoutingRouter):
        patch(cls)


class TopPWorker:
    def reset_budget_counts(self):
        for module in self.model_runner.model.modules():
            router = getattr(module, "router", None)
            if hasattr(router, "benchmark_budget_counts"):
                router.benchmark_budget_counts.zero_()

    def collect_budget_counts(self):
        return {
            name: module.router.benchmark_budget_counts.cpu().tolist()
            for name, module in self.model_runner.model.named_modules()
            if hasattr(getattr(module, "router", None), "benchmark_budget_counts")
        }


if "MOE_SKIP_BENCH_TOP_P" in os.environ:
    install(float(os.environ["MOE_SKIP_BENCH_TOP_P"]))
