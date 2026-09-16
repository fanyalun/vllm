# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal recurrence with per-token gates and one private final checkpoint."""

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
    AS: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(0)
    vs = tl.program_id(1) * BV + tl.arange(0, BV)
    ks = tl.arange(0, BK)
    key_head = head // (HV // H)
    offsets = head * DV * DK + vs[:, None] * DK + ks[None, :]
    mask = (vs[:, None] < DV) & (ks[None, :] < DK)
    state = tl.load(STATE + offsets, mask, 0).to(tl.float32)
    a_log = tl.load(A_LOG + head).to(tl.float32)
    dt = tl.load(DT + head).to(tl.float32)
    for t in range(T):
        q = tl.load(Q + t * QS + key_head * DK + ks, ks < DK, 0).to(tl.float32)
        k = tl.load(K + t * KS + key_head * DK + ks, ks < DK, 0).to(tl.float32)
        v = tl.load(V + t * VS + head * DV + vs, vs < DV, 0).to(tl.float32)
        x = tl.load(A + t * AS + head).to(tl.float32) + dt
        softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        g = -tl.exp(a_log) * softplus
        beta = tl.sigmoid(tl.load(B + t * BS + head).to(tl.float32))
        q *= tl.rsqrt(tl.sum(q * q) + 1e-6)
        k *= tl.rsqrt(tl.sum(k * k) + 1e-6)
        q *= DK**-0.5
        state *= tl.exp(g)
        delta = (v - tl.sum(state * k[None, :], 1)) * beta
        state += delta[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(OUT + (t * HV + head) * DV + vs, output, vs < DV)
    tl.store(STATE + offsets, state, mask)


def replay_tail_update(q, k, v, a, b, a_log, dt_bias, state):
    """Return all causal outputs and overwrite state with the full window tail."""
    tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1:]
    if not 1 <= tokens <= 5 or value_heads % heads:
        raise ValueError("Replay-tail GDN requires 1..5 tokens and integral heads")
    if a.shape != (tokens, value_heads) or b.shape != (tokens, value_heads):
        raise ValueError("Replay-tail GDN requires per-token gates")
    if state.shape != (1, value_heads, value_dim, key_dim):
        raise ValueError("Replay-tail GDN requires exactly one private state")
    if not state.is_contiguous() or state.dtype != torch.float32:
        raise ValueError("Replay-tail GDN requires contiguous FP32 state")
    if any(x.stride(-1) != 1 for x in (q, k, v, a, b)):
        raise ValueError("Replay-tail GDN requires contiguous head dimensions")
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    _replay_tail_update[(value_heads, triton.cdiv(value_dim, 16))](
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
        a.stride(0),
        b.stride(0),
        triton.next_power_of_2(key_dim),
        16,
        num_warps=4,
    )
    return out


@triton.jit(do_not_specialize=["accepted"])
def _advance_conv(
    CONV,
    accepted,
    CHANNELS: tl.constexpr,
    LENGTH: tl.constexpr,
    CS: tl.constexpr,
    TS: tl.constexpr,
    BC: tl.constexpr,
    BT: tl.constexpr,
):
    channels = tl.program_id(0) * BC + tl.arange(0, BC)
    positions = tl.arange(0, BT)
    source = tl.minimum(positions + accepted, LENGTH - 1)
    mask = (channels[:, None] < CHANNELS) & (positions[None, :] < LENGTH)
    values = tl.load(CONV + channels[:, None] * CS + source[None, :] * TS, mask, 0)
    # Each program owns complete channels; finish overlapping reads before writes.
    tl.debug_barrier()
    tl.store(CONV + channels[:, None] * CS + positions[None, :] * TS, values, mask)


def advance_replay_tail_conv(conv, accepted, dim_first):
    """Shift the private Conv history without index tensors or temporary copies."""
    channels, length = (
        (conv.shape[1], conv.shape[2]) if dim_first else (conv.shape[2], conv.shape[1])
    )
    cs, ts = (
        (conv.stride(1), conv.stride(2))
        if dim_first
        else (conv.stride(2), conv.stride(1))
    )
    _advance_conv[(triton.cdiv(channels, 32),)](
        conv,
        accepted,
        channels,
        length,
        cs,
        ts,
        32,
        triton.next_power_of_2(length),
        num_warps=4,
    )
