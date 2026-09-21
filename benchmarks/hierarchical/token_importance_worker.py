# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma Pre-Verify expert-pool experiment; enable only in benchmark workers."""

import os

import torch

from vllm.triton_utils import tl, triton

MODE = os.environ.get("PREVERIFY_EXPERT_POOL", "none")
IMPORTANCE = None
POOL = None


@triton.jit
def _scores(
    Q,
    K,
    TABLE,
    LENGTH,
    OUT,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    LIMIT: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    CAP: tl.constexpr,
):
    head = tl.program_id(0)
    pos = tl.program_id(1) * 32 + tl.arange(0, 32)
    dim = tl.arange(0, BD)
    length = tl.load(LENGTH)
    valid = (pos < length) & (pos < LIMIT)
    if WINDOW >= 0:
        valid &= pos >= length - 1 - WINDOW
    block = tl.load(TABLE + pos // BS, mask=valid, other=0)
    key = tl.load(
        K
        + block[:, None] * KS0
        + (head // (H // HK)) * KS1
        + (pos[:, None] % BS) * KS2
        + dim[None, :],
        mask=valid[:, None] & (dim[None, :] < D),
        other=0,
    ).to(tl.float32)
    query = tl.load(Q + (M - 1) * QS0 + head * QS1 + dim, mask=dim < D, other=0).to(
        tl.float32
    )
    score = tl.sum(key * query[None, :], 1) * SCALE
    if CAP > 0:
        score = CAP * (2.0 / (1.0 + tl.exp(2.0 * -score / CAP)) - 1.0)
    tl.store(
        OUT + head * LIMIT + pos,
        tl.where(valid, score, -float("inf")),
        mask=pos < LIMIT,
    )


@triton.jit
def _probabilities(
    SCORES,
    LENGTH,
    OUT,
    M: tl.constexpr,
    BM: tl.constexpr,
    LIMIT: tl.constexpr,
    BL: tl.constexpr,
):
    head = tl.program_id(0)
    pos = tl.arange(0, BL)
    scores = tl.load(SCORES + head * LIMIT + pos, pos < LIMIT, -float("inf"))
    maximum = tl.max(scores, 0)
    denominator = tl.sum(tl.exp(scores - maximum), 0)
    row = tl.arange(0, BM)
    source = tl.load(LENGTH) - M + row
    selected = tl.load(SCORES + head * LIMIT + source, row < M, -float("inf"))
    tl.store(OUT + head * M + row, tl.exp(selected - maximum) / denominator, row < M)


def last_attention(
    query, cache, table, lengths, window=-1, scale=1.0, cap=0.0, limit=1024
):
    m, heads, dim = query.shape
    if cache.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Only unquantized KV caches are supported")
    if query.stride(-1) != 1 or cache.stride(-1) != 1:
        raise ValueError("Contiguous head dimensions are required")
    scores = torch.empty((heads, limit), device=query.device, dtype=torch.float32)
    probabilities = torch.empty((heads, m), device=query.device, dtype=torch.float32)
    _scores[(heads, triton.cdiv(limit, 32))](
        query,
        cache,
        table,
        lengths,
        scores,
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        m,
        heads,
        cache.shape[1],
        dim,
        triton.next_power_of_2(dim),
        cache.shape[2],
        limit,
        window,
        scale,
        cap,
        num_warps=4,
    )
    _probabilities[(heads,)](
        scores,
        lengths,
        probabilities,
        m,
        triton.next_power_of_2(m),
        limit,
        triton.next_power_of_2(limit),
    )
    return probabilities.mean(0)


def select_pool(weights, ids, logits, importance):
    m, k = ids.shape
    experts = logits.shape[1]
    if k != 8 or m < 3:
        raise ValueError("Expert-pool probe requires native top8 and anchor + 2 drafts")
    native = logits.float().gather(1, ids.long()).softmax(-1)
    rows = torch.arange(m, device=ids.device)
    relevance = importance * ((rows > 0) & (rows < m - 1))
    active_counts = torch.zeros(experts, device=ids.device, dtype=torch.int64)
    active_counts.scatter_add_(
        0,
        ids[1:].long().flatten(),
        torch.ones_like(ids[1:], dtype=torch.int64).flatten(),
    )
    active = active_counts > 0
    score = torch.zeros(experts, device=ids.device, dtype=torch.float32)
    score.scatter_add_(0, ids.long().flatten(), (native * relevance[:, None]).flatten())
    order = score.masked_fill(~active, -float("inf")).argsort(
        descending=True, stable=True
    )
    count = (active.sum() * 3 + 4) // 5
    pool = torch.zeros_like(active).scatter_(
        0, order, torch.arange(experts, device=ids.device) < count
    )
    keep = pool[ids.long()]
    retained_mass = (native * keep).sum(-1, keepdim=True)
    result = weights * keep / retained_mass.clamp_min(1e-30)
    counters = torch.stack(
        (
            active.sum(),
            pool.sum(),
            keep.sum(),
            (keep.sum(-1) == 0).sum(),
            torch.full((), m, dtype=torch.int64, device=ids.device),
            torch.ones((), dtype=torch.int64, device=ids.device),
        )
    )
    return result, pool, counters


def is_preverify():
    from vllm.forward_context import get_forward_context

    return "preverify_gdn_mode" in get_forward_context().additional_kwargs


def mask_assignment(expert_ids, pool):
    valid = (expert_ids >= 0) & (expert_ids < pool.numel())
    return torch.where(
        valid & pool[expert_ids.clamp(0, pool.numel() - 1)], expert_ids, -1
    )


def install():
    from vllm.model_executor.layers.fused_moe import fused_moe
    from vllm.model_executor.layers.fused_moe.experts import triton_moe
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    original_attention = TritonAttentionImpl.forward

    def attention(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        global IMPORTANCE
        result = original_attention(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        if MODE == "attention" and is_preverify():
            if self.sinks is not None or not attn_metadata.causal:
                raise ValueError("Sink or noncausal attention is unsupported")
            IMPORTANCE = last_attention(
                query,
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                self.sliding_window[0],
                self.scale,
                self.logits_soft_cap or 0.0,
            )
        return result

    original_init = CustomRoutingRouter.__init__
    original_route = CustomRoutingRouter._compute_routing

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.pool_counters = torch.zeros(6, dtype=torch.int64, device="cuda")

    def route(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        global POOL
        weights, ids = original_route(
            self, hidden_states, router_logits, indices_type, input_ids=input_ids
        )
        if is_preverify():
            relevance = (
                IMPORTANCE
                if MODE == "attention"
                else torch.ones(ids.shape[0], device=ids.device)
            )
            if MODE in ("attention", "routing"):
                weights, POOL, counts = select_pool(
                    weights, ids, router_logits, relevance
                )
            else:
                retained = (ids >= 0) & (weights != 0)
                counts = torch.stack(
                    (
                        torch.zeros((), dtype=torch.int64, device=ids.device),
                        torch.zeros((), dtype=torch.int64, device=ids.device),
                        retained.sum(),
                        (retained.sum(-1) == 0).sum(),
                        torch.full((), ids.shape[0], device=ids.device),
                        torch.ones((), dtype=torch.int64, device=ids.device),
                    )
                )
            self.pool_counters.add_(counts)
        return weights, ids

    original_assignment = fused_moe._prepare_expert_assignment

    def assignment(*args, **kwargs):
        sorted_ids, expert_ids, padded = original_assignment(*args, **kwargs)
        if MODE in ("attention", "routing") and is_preverify():
            expert_ids = mask_assignment(expert_ids, POOL)
        return sorted_ids, expert_ids, padded

    TritonAttentionImpl.forward = attention
    CustomRoutingRouter.__init__ = initialize
    CustomRoutingRouter._compute_routing = route
    fused_moe._prepare_expert_assignment = assignment
    triton_moe._prepare_expert_assignment = assignment


class TokenImportanceWorker:
    def begin_pool_measurement(self):
        if not hasattr(self, "pool_measurement_installed"):
            speculator = self.model_runner.speculator
            if speculator is not None:
                original = speculator.propose

                def propose(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if self.pool_measurement_active:
                        for row in speculator.last_trace:
                            self.inner_counts[0] += row["proposed"]
                            self.inner_counts[1] += row["accepted"]
                            self.inner_counts[2] += 1
                    return result

                speculator.propose = propose
            self.pool_measurement_installed = True
        self.inner_counts = [0, 0, 0]
        self.pool_measurement_active = True
        self.reset_pool_counters()

    def collect_pool_measurement(self):
        self.pool_measurement_active = False
        return {
            "inner_counts": self.inner_counts,
            "layers": self.collect_pool_counters(),
        }

    def reset_pool_counters(self):
        for module in self.model_runner.model.modules():
            router = getattr(module, "router", None)
            if hasattr(router, "pool_counters"):
                router.pool_counters.zero_()

    def collect_pool_counters(self):
        return {
            name: module.router.pool_counters.cpu().tolist()
            for name, module in self.model_runner.model.named_modules()
            if hasattr(getattr(module, "router", None), "pool_counters")
        }


if MODE in ("attention", "routing") or os.environ.get(
    "PREVERIFY_POOL_ACCEPTANCE_AUDIT"
):
    install()
