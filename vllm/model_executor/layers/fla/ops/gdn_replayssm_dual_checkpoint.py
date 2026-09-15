# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _reset_dual_checkpoint(
    write_pos,
    cache_base,
    is_flush,
    head_slot,
    previous_len,
    indices,
    query_start,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    blk = tl.load(indices + row * STRIDE, row < N, 0)
    length = tl.load(query_start + row + 1, row < N, 0) - tl.load(
        query_start + row, row < N, 0
    )
    valid = (row < N) & (blk > 0) & (length > 0)
    tl.store(write_pos + blk, 0, valid)
    tl.store(cache_base + blk, 0, valid)
    tl.store(is_flush + blk, 0, valid)
    tl.store(head_slot + blk, 0, valid)
    tl.store(previous_len + blk, 0, valid)


def reset_gdn_dual_checkpoint(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    head_slot: torch.Tensor,
    previous_len: torch.Tensor,
    indices: torch.Tensor,
    query_start: torch.Tensor,
) -> None:
    """Prefill initializes state0, including on reused or preempted blocks."""
    n = indices.numel()
    _reset_dual_checkpoint[(1,)](
        write_pos,
        cache_base,
        is_flush,
        head_slot,
        previous_len,
        indices,
        query_start,
        n,
        indices.stride(0),
        triton.next_power_of_2(max(1, n)),
    )


@triton.jit
def _commit_dual_checkpoint(
    write_pos,
    cache_base,
    is_flush,
    head_slot,
    previous_len,
    accepted,
    indices,
    query_start,
    first_decode,
    N: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    ACCEPT_STRIDE: tl.constexpr,
    W: tl.constexpr,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    blk = tl.load(indices + row * INDEX_STRIDE, row < N, other=0)
    valid = (row < N) & (blk > 0)
    reset = tl.load(first_decode + row, valid, other=0) != 0
    prev = tl.load(previous_len + blk, valid, other=0)
    acc = tl.load(accepted + row * ACCEPT_STRIDE, valid, other=0)
    h = tl.load(write_pos + blk, valid, other=0)
    base = tl.load(cache_base + blk, valid, other=0)
    flushed = tl.load(is_flush + blk, valid, other=0) != 0
    head = tl.load(head_slot + blk, valid, other=0)
    t = tl.load(query_start + row + 1, row < N, other=0) - tl.load(
        query_start + row, row < N, other=0
    )
    promote = (prev > 0) & (acc == prev) & ~reset
    # A preceding flush moved the checkpoint to that window's start.
    base = tl.where(flushed, (base + h) & (L - 1), base)
    h = tl.where(flushed, 0, h) + tl.where(prev > 0, acc, 0)
    head = tl.where(promote, 1 - head, head)
    h = tl.where(promote | reset, 0, h)
    base = tl.where(promote | reset, 0, base)
    head = tl.where(reset, 0, head)
    tl.store(write_pos + blk, h, valid)
    tl.store(cache_base + blk, base, valid)
    tl.store(is_flush + blk, h + t > W, valid)
    tl.store(head_slot + blk, head, valid)
    tl.store(previous_len + blk, t, valid)


def commit_gdn_dual_checkpoint(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    head_slot: torch.Tensor,
    previous_len: torch.Tensor,
    accepted: torch.Tensor,
    indices: torch.Tensor,
    query_start: torch.Tensor,
    first_decode: torch.Tensor,
    hard_cap: int,
) -> None:
    """Commit the preceding verification before launching the next one."""
    n = indices.numel()
    _commit_dual_checkpoint[(1,)](
        write_pos,
        cache_base,
        is_flush,
        head_slot,
        previous_len,
        accepted,
        indices,
        query_start,
        first_decode,
        n,
        indices.stride(0),
        accepted.stride(0),
        hard_cap,
        triton.next_power_of_2(hard_cap),
        triton.next_power_of_2(max(1, n)),
    )


@triton.jit
def _flush_dual_checkpoint(
    state0,
    state1,
    d_cache,
    k_cache,
    g_cache,
    indices,
    write_pos,
    cache_base,
    is_flush,
    head_slot,
    STATE_STRIDE: tl.constexpr,
    D_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    G_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    iv, row, hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    blk = tl.load(indices + row * INDEX_STRIDE)
    if blk <= 0:
        return
    if tl.load(is_flush + blk) == 0:
        return
    h = tl.load(write_pos + blk)
    base = tl.load(cache_base + blk)
    head = tl.load(head_slot + blk)
    state = state0
    if head != 0:
        state = state1
    c = tl.arange(0, BC)
    v = iv * BV + tl.arange(0, BV)
    k = tl.arange(0, BK)
    phys = (base + c) & (L - 1)
    g = tl.load(g_cache + blk * G_STRIDE + hv * L + phys, c < h, 0)
    total = tl.sum(g, 0)
    decay = tl.where(c < h, tl.exp(total - tl.cumsum(g)), 0.0)
    d = (
        tl.load(
            d_cache + blk * D_STRIDE + (hv * L + phys[None, :]) * V + v[:, None],
            (v[:, None] < V) & (c[None, :] < h),
            0,
        ).to(tl.float32)
        * decay[None, :]
    )
    keys = tl.load(
        k_cache
        + blk * K_STRIDE
        + ((hv // (HV // H)) * L + phys[:, None]) * K
        + k[None, :],
        (c[:, None] < h) & (k[None, :] < K),
        0,
    ).to(tl.float32)
    p = state + blk * STATE_STRIDE + hv * V * K + v[:, None] * K + k[None, :]
    mask = (v[:, None] < V) & (k[None, :] < K)
    s = tl.load(p, mask, 0).to(tl.float32)
    s = tl.dot(d, keys, acc=tl.exp(total) * s, input_precision=PRECISION)
    tl.store(p, s, mask)


def flush_gdn_dual_checkpoint(
    state0: torch.Tensor,
    state1: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    head_slot: torch.Tensor,
    precision: str,
) -> None:
    """Fold history before verify can overwrite any ring entries."""
    _, hv, v, k = state0.shape
    h, length = k_cache.shape[1:3]
    _flush_dual_checkpoint[(triton.cdiv(v, 32), indices.numel(), hv)](
        state0,
        state1,
        d_cache,
        k_cache,
        g_cache,
        indices,
        write_pos,
        cache_base,
        is_flush,
        head_slot,
        state0.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        indices.stride(0),
        h,
        hv,
        v,
        k,
        length,
        max(16, length),
        32,
        triton.next_power_of_2(k),
        precision,
    )
