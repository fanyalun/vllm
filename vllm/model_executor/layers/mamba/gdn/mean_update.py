# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-state mean update for the experimental Qwen pre-verifier."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mean_update(
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
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(0)
    value_tile = tl.program_id(1)
    key_head = head // (HV // H)
    ts = tl.arange(0, BT)
    ks = tl.arange(0, BK)
    vs = value_tile * BV + tl.arange(0, BV)
    keys = tl.load(
        K + ts[:, None] * KS + key_head * DK + ks[None, :],
        (ts[:, None] < T) & (ks[None, :] < DK),
        0,
    ).to(tl.float32)
    key = tl.sum(keys, 0) / T
    key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
    values = tl.load(
        V + ts[:, None] * VS + head * DV + vs[None, :],
        (ts[:, None] < T) & (vs[None, :] < DV),
        0,
    ).to(tl.float32)
    value = tl.sum(values, 0) / T
    a = tl.sum(tl.load(A + ts * AS + head, ts < T, 0).to(tl.float32), 0) / T
    b = tl.sum(tl.load(B + ts * BS + head, ts < T, 0).to(tl.float32), 0) / T
    x = a + tl.load(DT + head).to(tl.float32)
    softplus = tl.maximum(x, 0) + tl.log(1 + tl.exp(-tl.abs(x)))
    decay = tl.exp(-tl.exp(tl.load(A_LOG + head)) * softplus)
    beta = tl.sigmoid(b)
    offsets = head * DV * DK + vs[:, None] * DK + ks[None, :]
    mask = (vs[:, None] < DV) & (ks[None, :] < DK)
    state = tl.load(STATE + offsets, mask, 0).to(tl.float32) * decay
    delta = (value - tl.sum(state * key[None, :], 1)) * beta
    state += delta[:, None] * key[None, :]
    for t in range(T):
        query = tl.load(Q + t * QS + key_head * DK + ks, ks < DK, 0).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6) * DK**-0.5
        output = tl.sum(state * query[None, :], 1)
        tl.store(OUT + (t * HV + head) * DV + vs, output, vs < DV)
    tl.store(STATE + offsets, state, mask)


def mean_state_update(q, k, v, a, b, a_log, dt_bias, state):
    """Update one private state once, then read it with each token's query.

    Keys/values and pre-activation gate inputs are averaged over actual tokens.
    The mean key is normalized after pooling. Query normalization remains per token.
    """
    tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1:]
    if not 1 <= tokens <= 5 or value_heads % heads:
        raise ValueError("Mean GDN requires 1..5 tokens and integral value/key heads")
    if state.shape != (1, value_heads, value_dim, key_dim):
        raise ValueError("Mean GDN requires exactly one private recurrent state")
    if not state.is_contiguous() or state.dtype != torch.float32:
        raise ValueError("Mean GDN requires contiguous FP32 recurrent state")
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    _mean_update[(value_heads, triton.cdiv(value_dim, 32))](
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
        triton.next_power_of_2(tokens),
        triton.next_power_of_2(key_dim),
        32,
        num_warps=4,
    )
    return out
