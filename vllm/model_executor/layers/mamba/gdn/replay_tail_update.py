# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal block update with shared gates and one in-place private checkpoint."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _replay_tail_update(
    Q,
    K,
    V,
    A,
    B,
    A_LOG,
    DT,
    STATE,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(0)
    value_tile = tl.program_id(1)
    key_head = head // (HV // H)
    ts = tl.arange(0, 16)
    vs = value_tile * BV + tl.arange(0, BV)
    qnorm = tl.full((16,), 1e-6, tl.float32)
    knorm = tl.full((16,), 1e-6, tl.float32)
    for tile in range(tl.cdiv(DK, BK)):
        ks = tile * BK + tl.arange(0, BK)
        mask = (ts[:, None] < T) & (ks[None, :] < DK)
        q = tl.load(Q + ts[:, None] * QS + key_head * DK + ks[None, :], mask, 0).to(
            tl.float32
        )
        k = tl.load(K + ts[:, None] * KS + key_head * DK + ks[None, :], mask, 0).to(
            tl.float32
        )
        qnorm += tl.sum(q * q, 1)
        knorm += tl.sum(k * k, 1)
    qscale = tl.rsqrt(qnorm) * DK**-0.5
    kscale = tl.rsqrt(knorm)
    initial_k = tl.zeros((16, BV), tl.float32)
    initial_q = tl.zeros((16, BV), tl.float32)
    kk = tl.zeros((16, 16), tl.float32)
    qk = tl.zeros((16, 16), tl.float32)
    for tile in range(tl.cdiv(DK, BK)):
        ks = tile * BK + tl.arange(0, BK)
        mask = (ts[:, None] < T) & (ks[None, :] < DK)
        q = (
            tl.load(Q + ts[:, None] * QS + key_head * DK + ks[None, :], mask, 0).to(
                tl.float32
            )
            * qscale[:, None]
        )
        k = (
            tl.load(K + ts[:, None] * KS + key_head * DK + ks[None, :], mask, 0).to(
                tl.float32
            )
            * kscale[:, None]
        )
        offsets = head * DV * DK + vs[:, None] * DK + ks[None, :]
        state_mask = (vs[:, None] < DV) & (ks[None, :] < DK)
        state = tl.load(STATE + offsets, state_mask, 0).to(tl.float32)
        initial_k += tl.dot(k, tl.trans(state), input_precision="tf32x3")
        initial_q += tl.dot(q, tl.trans(state), input_precision="tf32x3")
        kk += tl.dot(k, tl.trans(k), input_precision="tf32x3")
        qk += tl.dot(q, tl.trans(k), input_precision="tf32x3")
    v = tl.load(
        V + ts[:, None] * VS + head * DV + vs[None, :],
        (ts[:, None] < T) & (vs[None, :] < DV),
        0,
    ).to(tl.float32)
    x = tl.load(A + head).to(tl.float32) + tl.load(DT + head).to(tl.float32)
    softplus = tl.maximum(x, 0) + tl.log(1 + tl.exp(-tl.abs(x)))
    g = -tl.exp(tl.load(A_LOG + head).to(tl.float32)) * softplus
    beta = tl.sigmoid(tl.load(B + head).to(tl.float32))
    exp_g = tl.exp((ts + 1) * g)
    lower = (ts[:, None] > ts[None, :]) & (ts[:, None] < T)
    # Mask the exponent before exp to avoid overflow in the unused upper triangle.
    decay = tl.exp(tl.maximum(ts[:, None] - ts[None, :], 0) * g)
    neg_a = tl.where(lower, -beta * decay * kk, 0.0)
    inverse = neg_a
    for i in tl.static_range(2, T):
        row = tl.sum(tl.where((ts == i)[:, None], neg_a, 0.0), 0)
        row += tl.sum(row[:, None] * inverse, 0)
        inverse = tl.where((ts == i)[:, None], row, inverse)
    inverse += (ts[:, None] == ts[None, :]).to(tl.float32)
    rhs = beta * (v - exp_g[:, None] * initial_k)
    delta = tl.dot(inverse, rhs, input_precision="tf32x3")
    causal = (ts[:, None] >= ts[None, :]) & (ts[:, None] < T)
    weights = tl.where(causal, decay * qk, 0.0)
    output = exp_g[:, None] * initial_q + tl.dot(
        weights, delta, input_precision="tf32x3"
    )
    tl.store(
        OUT + (ts[:, None] * HV + head) * DV + vs[None, :],
        output,
        (ts[:, None] < T) & (vs[None, :] < DV),
    )
    tail_delta = (
        delta * tl.where(ts < T, tl.exp(tl.maximum(T - 1 - ts, 0) * g), 0.0)[:, None]
    )
    # All initial-state projections are complete before any in-place writes.
    # Each program owns full key rows for its value tile.
    for tile in range(tl.cdiv(DK, BK)):
        ks = tile * BK + tl.arange(0, BK)
        k = tl.load(
            K + ts[:, None] * KS + key_head * DK + ks[None, :],
            (ts[:, None] < T) & (ks[None, :] < DK),
            0,
        ).to(tl.float32)
        k *= kscale[:, None]
        offsets = head * DV * DK + vs[:, None] * DK + ks[None, :]
        state_mask = (vs[:, None] < DV) & (ks[None, :] < DK)
        state = tl.load(STATE + offsets, state_mask, 0).to(tl.float32)
        tail = tl.exp(T * g) * state + tl.dot(
            tl.trans(tail_delta), k, input_precision="tf32x3"
        )
        tl.store(STATE + offsets, tail, state_mask)


def replay_tail_update(q, k, v, a, b, a_log, dt_bias, state):
    """Return all causal outputs and overwrite state with the full window tail.

    The caller supplies one gate projection per head, from the first input
    token. No per-position recurrent states or replay history are retained.
    """
    tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1:]
    if not 1 <= tokens <= 5 or value_heads % heads:
        raise ValueError("Replay-tail GDN requires 1..5 tokens and integral heads")
    if a.shape != (1, value_heads) or b.shape != (1, value_heads):
        raise ValueError("Replay-tail GDN requires first-token gates only")
    if state.shape != (1, value_heads, value_dim, key_dim):
        raise ValueError("Replay-tail GDN requires exactly one private state")
    if not state.is_contiguous() or state.dtype != torch.float32:
        raise ValueError("Replay-tail GDN requires contiguous FP32 state")
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    _replay_tail_update[(value_heads, triton.cdiv(value_dim, 32))](
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        out,
        tokens,
        heads,
        value_heads,
        key_dim,
        value_dim,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        32,
        32,
        num_warps=4,
    )
    return out
