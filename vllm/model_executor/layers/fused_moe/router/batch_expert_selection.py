# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _top_two(
    IDS,
    LOGITS,
    PAD,
    TOP,
    K: tl.constexpr,
    E: tl.constexpr,
    HAS_PAD: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BK)
    ids = tl.load(IDS + row * K + col, col < K, other=-1)
    valid = (col < K) & (ids >= 0) & (ids < E)
    if HAS_PAD:
        valid = valid & ~tl.load(PAD + row)
    scores = tl.load(LOGITS + row * E + ids, valid, other=-float("inf"))
    ahead = valid[None, :] & (
        (scores[None, :] > scores[:, None])
        | ((scores[None, :] == scores[:, None]) & (ids[None, :] < ids[:, None]))
    )
    rank = tl.sum(ahead.to(tl.int32), 1)
    tl.store(TOP + row * K + col, valid & (rank < 2), col < K)


@triton.jit
def _aggregate(
    W,
    IDS,
    TOP,
    PAD,
    A,
    P,
    U,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    HAS_PAD: tl.constexpr,
    B: tl.constexpr,
):
    expert = tl.program_id(0)
    chunk = tl.program_id(1)
    slot = chunk * B + tl.arange(0, B)
    valid = slot < M * K
    if HAS_PAD:
        valid = valid & ~tl.load(PAD + slot // K, valid, other=True)
    ids = tl.load(IDS + slot, valid, other=-1)
    match = valid & (ids == expert)
    w = tl.load(W + slot, match, other=0).to(tl.float32)
    protected = tl.load(TOP + slot, match, other=False)
    offset = chunk * E + expert
    tl.store(A + offset, tl.sum(w, 0))
    tl.store(P + offset, tl.sum(protected.to(tl.int32), 0) > 0)
    tl.store(U + offset, tl.sum(match.to(tl.int32), 0) > 0)


@triton.jit
def _select(
    A,
    P,
    U,
    KEEP,
    E: tl.constexpr,
    CHUNKS: tl.constexpr,
    HALF: tl.constexpr,
    BE: tl.constexpr,
):
    expert = tl.arange(0, BE)
    score = tl.full((BE,), 0, tl.float32)
    protected = tl.full((BE,), False, tl.int1)
    active = tl.full((BE,), False, tl.int1)
    for chunk in range(CHUNKS):
        offset = chunk * E + expert
        score += tl.load(A + offset, expert < E, other=0)
        protected |= tl.load(P + offset, expert < E, other=False)
        active |= tl.load(U + offset, expert < E, other=False)
    candidate = active & ~protected & (expert < E)
    count = tl.sum(candidate.to(tl.int32), 0)
    ranked = tl.sort(tl.where(candidate, score, -float("inf")), descending=True)
    if HALF:
        n = (count + 1) // 2
    else:
        following = tl.gather(ranked, tl.minimum(expert + 1, BE - 1), 0)
        gaps = tl.where(expert < count - 1, ranked - following, -float("inf"))
        largest = tl.max(gaps, 0)
        cut = tl.min(tl.where(gaps == largest, expert + 1, BE), 0)
        n = tl.where((count < 2) | (largest <= 0), count, cut)
    threshold = tl.sum(tl.where(expert == tl.maximum(n - 1, 0), ranked, 0), 0)
    greater = candidate & (score > threshold)
    equal = candidate & (score == threshold)
    tie_rank = tl.cumsum(equal.to(tl.int32), 0)
    chosen = greater | (equal & (tie_rank <= n - tl.sum(greater.to(tl.int32), 0)))
    tl.store(KEEP + expert, protected | (chosen & (n > 0)), expert < E)


@triton.jit
def _mask(
    W,
    IDS,
    PAD,
    KEEP,
    OW,
    OI,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    HAS_PAD: tl.constexpr,
    B: tl.constexpr,
):
    slot = tl.program_id(0) * B + tl.arange(0, B)
    ids = tl.load(IDS + slot, slot < M * K, other=-1)
    valid = (slot < M * K) & (ids >= 0) & (ids < E)
    if HAS_PAD:
        valid = valid & ~tl.load(PAD + slot // K, slot < M * K, other=True)
    keep = valid & tl.load(KEEP + ids, valid, other=False)
    w = tl.load(W + slot, slot < M * K, other=0)
    tl.store(OW + slot, tl.where(keep, w, 0), slot < M * K)
    tl.store(OI + slot, tl.where(keep, ids, -1), slot < M * K)


def select_batch_experts(weights, ids, logits, policy, is_padding=None):
    """Prune native normalized Qwen routes with a batch-wide Top-2 union."""
    if policy not in ("batch_top_half", "batch_max_gap"):
        raise ValueError(f"Unknown MoE-Skip batch policy: {policy}")
    if not weights.is_cuda:
        raise ValueError("MoE-Skip batch routing requires CUDA")
    if not all(t.is_contiguous() for t in (weights, ids, logits)):
        raise ValueError("MoE-Skip batch routing requires contiguous inputs")
    m, k = ids.shape
    e = logits.shape[1]
    if weights.shape != ids.shape or logits.shape[0] != m or k < 2:
        raise ValueError("Invalid native top-k batch routing shapes")
    if is_padding is not None and (
        is_padding.shape != (m,) or not is_padding.is_contiguous()
    ):
        raise ValueError("Invalid batch routing padding mask")
    out_weights, out_ids = torch.empty_like(weights), torch.empty_like(ids)
    if m == 0:
        return out_weights, out_ids
    chunks = triton.cdiv(m * k, 256)
    top = torch.empty_like(ids, dtype=torch.bool)
    aggregate = torch.empty((chunks, e), device=ids.device, dtype=torch.float32)
    protected = torch.empty_like(aggregate, dtype=torch.bool)
    active = torch.empty_like(protected)
    keep = torch.empty((e,), device=ids.device, dtype=torch.bool)
    has_pad = is_padding is not None
    _top_two[(m,)](
        ids, logits, is_padding, top, k, e, has_pad, triton.next_power_of_2(k)
    )
    _aggregate[(e, chunks)](
        weights,
        ids,
        top,
        is_padding,
        aggregate,
        protected,
        active,
        m,
        k,
        e,
        has_pad,
        256,
    )
    _select[(1,)](
        aggregate,
        protected,
        active,
        keep,
        e,
        chunks,
        policy == "batch_top_half",
        triton.next_power_of_2(e),
    )
    _mask[(chunks,)](
        weights, ids, is_padding, keep, out_weights, out_ids, m, k, e, has_pad, 256
    )
    return out_weights, out_ids
