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
    TAIL,
    THRESHOLDS,
    ACTION_COUNTS,
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
    CONDITIONAL: tl.constexpr = False,
    READOUT: tl.constexpr = True,
    EFFECTIVE_GATES: tl.constexpr = False,
    RECORD_ACTIONS: tl.constexpr = False,
):
    head = tl.program_id(0)
    vs = tl.program_id(1) * BV + tl.arange(0, BV)
    ks = tl.arange(0, BK)
    key_head = head // (HV // H)
    offsets = head * DV * DK + vs[:, None] * DK + ks[None, :]
    mask = (vs[:, None] < DV) & (ks[None, :] < DK)
    state = tl.load(STATE + offsets, mask, 0).to(tl.float32)
    if not EFFECTIVE_GATES:
        a_log = tl.load(A_LOG + head).to(tl.float32)
        dt = tl.load(DT + head).to(tl.float32)
    if CONDITIONAL:
        threshold_alpha = tl.load(THRESHOLDS)
        threshold_beta = tl.load(THRESHOLDS + 1)
    for t in range(T):
        if EFFECTIVE_GATES:
            g = tl.load(A + t * AS + head).to(tl.float32)
            beta = tl.load(B + t * BS + head).to(tl.float32)
        else:
            x = tl.load(A + t * AS + head).to(tl.float32) + dt
            softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
            g = -tl.exp(a_log) * softplus
            beta = tl.sigmoid(tl.load(B + t * BS + head).to(tl.float32))
        full = True
        action = 0
        if CONDITIONAL:
            full = beta.to(tl.bfloat16).to(tl.float32) >= threshold_beta
        if full:
            k = tl.load(K + t * KS + key_head * DK + ks, ks < DK, 0).to(tl.float32)
            v = tl.load(V + t * VS + head * DV + vs, vs < DV, 0).to(tl.float32)
            k *= tl.rsqrt(tl.sum(k * k) + 1e-6)
            state *= tl.exp(g)
            delta = (v - tl.sum(state * k[None, :], 1)) * beta
            state += delta[:, None] * k[None, :]
        elif CONDITIONAL:
            action = 2
            if tl.exp(g) <= threshold_alpha:
                state *= tl.exp(g)
                action = 1
        if RECORD_ACTIONS and tl.program_id(1) == 0:
            tl.atomic_add(ACTION_COUNTS + action, 1)
        if READOUT:
            q = tl.load(Q + t * QS + key_head * DK + ks, ks < DK, 0).to(tl.float32)
            q *= tl.rsqrt(tl.sum(q * q) + 1e-6)
            q *= DK**-0.5
            output = tl.sum(state * q[None, :], 1)
            tl.store(OUT + (t * HV + head) * DV + vs, output, vs < DV)
    tl.store(TAIL + offsets, state, mask)


def replay_tail_update(
    q,
    k,
    v,
    a,
    b,
    a_log,
    dt_bias,
    state,
    *,
    tail=None,
    thresholds=None,
    out=None,
    readout=True,
    value_tile=16,
    num_warps=4,
    effective_gates=False,
    action_counts=None,
):
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
    if thresholds is not None:
        if tail is None or tail.data_ptr() == state.data_ptr():
            raise ValueError("Three-level GDN requires an independent tail buffer")
        if thresholds.shape != (2,) or thresholds.dtype != torch.float32:
            raise ValueError("Three-level GDN requires two FP32 device thresholds")
        if thresholds.device != state.device:
            raise ValueError("Thresholds and state must share a device")
    if tail is None:
        tail = state
    if (
        tail.shape != state.shape
        or tail.dtype != state.dtype
        or not tail.is_contiguous()
    ):
        raise ValueError("Tail must match the contiguous FP32 input state")
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    elif readout and (out.shape != v.shape or not out.is_contiguous()):
        raise ValueError("Output must be contiguous and match the value shape")
    _replay_tail_update[(value_heads, triton.cdiv(value_dim, value_tile))](
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        out,
        tail,
        thresholds,
        action_counts,
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
        value_tile,
        thresholds is not None,
        readout,
        effective_gates,
        action_counts is not None,
        num_warps=num_warps,
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


@triton.jit(do_not_specialize=["accepted"])
def _advance_conv_many(
    POINTERS,
    accepted,
    CHANNELS: tl.constexpr,
    LENGTH: tl.constexpr,
    CS: tl.constexpr,
    TS: tl.constexpr,
    BT: tl.constexpr,
):
    conv = tl.load(POINTERS + tl.program_id(1)).to(tl.pointer_type(tl.bfloat16))
    _advance_conv(conv, accepted, CHANNELS, LENGTH, CS, TS, 32, BT)


def advance_replay_tail_convs(pointers, template, accepted, dim_first):
    """Advance identically laid out private BF16 histories in one launch."""
    channel_axis, time_axis = (1, 2) if dim_first else (2, 1)
    channels, length = template.shape[channel_axis], template.shape[time_axis]
    _advance_conv_many[(triton.cdiv(channels, 32), pointers.numel())](
        pointers,
        accepted,
        channels,
        length,
        template.stride(channel_axis),
        template.stride(time_axis),
        triton.next_power_of_2(length),
        num_warps=4,
    )


@triton.jit
def _begin_private_states(
    DESCRIPTORS, REQUEST, ACCEPTED, SOURCE, ALIGN: tl.constexpr, BLOCK: tl.constexpr
):
    row = DESCRIPTORS + tl.program_id(1) * 15
    conv = tl.load(row).to(tl.pointer_type(tl.bfloat16))
    state = tl.load(row + 1).to(tl.pointer_type(tl.float32))
    out_conv = tl.load(row + 2).to(tl.pointer_type(tl.bfloat16))
    out_state = tl.load(row + 3).to(tl.pointer_type(tl.float32))
    table = tl.load(row + 4).to(tl.pointer_type(tl.int32))
    request = tl.load(REQUEST)
    bias = tl.load(ACCEPTED + request).to(tl.int64) - 1
    source = tl.load(SOURCE + request).to(tl.int64) if ALIGN else 0
    conv_index = tl.load(table + source).to(tl.int64)
    state_index = tl.load(table + source + bias).to(tl.int64)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    state_size = tl.load(row + 13)
    state_stride = tl.load(row + 14)
    values = tl.load(
        state + state_index * state_stride + offsets, offsets < state_size, 0
    )
    tl.store(out_state + offsets, values, offsets < state_size)
    conv_stride = tl.load(row + 5)
    channel_stride, time_stride = tl.load(row + 6), tl.load(row + 7)
    out_channel_stride, out_time_stride = tl.load(row + 8), tl.load(row + 9)
    source_length, length, channels = (
        tl.load(row + 10).to(tl.int64),
        tl.load(row + 11).to(tl.int64),
        tl.load(row + 12).to(tl.int64),
    )
    channel, time = offsets // length, offsets % length
    mask = channel < channels
    values_conv = tl.load(
        conv
        + conv_index * conv_stride
        + channel * channel_stride
        + tl.minimum(time + bias, source_length - 1) * time_stride,
        mask,
        0,
    )
    tl.store(
        out_conv + channel * out_channel_stride + time * out_time_stride,
        values_conv,
        mask,
    )


def begin_replay_tail_states(descriptors, elements, request, accepted, source, align):
    """Copy canonical accepted states to disjoint private buffers."""
    _begin_private_states[(triton.cdiv(elements, 1024), descriptors.shape[0])](
        descriptors,
        request,
        accepted,
        source,
        align,
        1024,
        num_warps=4,
    )
