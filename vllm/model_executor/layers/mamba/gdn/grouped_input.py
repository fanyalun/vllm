# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three independent GDN branches sharing one pre-normalization input."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _select(P0, P1, P2, group):
    return tl.where(group == 0, P0, tl.where(group == 1, P1, P2))


@triton.jit
def _norm(
    X,
    W0,
    W1,
    W2,
    Y,
    H: tl.constexpr,
    T: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, B)
    x = tl.load(X + token * H + c, c < H, 0).to(tl.float32)
    w = tl.load(_select(W0, W1, W2, group) + c, c < H, 0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x) / H + EPS) * (1.0 + w)
    tl.store(Y + (group * T + token) * H + c, y, c < H)


def grouped_norm(x, weights, eps, output_dtype=None):
    out = torch.empty((3, *x.shape), device=x.device, dtype=output_dtype or x.dtype)
    _norm[(x.shape[0], 3)](
        x,
        *weights,
        out,
        x.shape[1],
        x.shape[0],
        eps,
        triton.next_power_of_2(x.shape[1]),
    )
    return out


@triton.jit
def _add_norm(
    X,
    R,
    W,
    Y,
    RES,
    H: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * H + c, c < H, 0).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(R + row * H + c, c < H, 0).to(tl.float32)
    w = tl.load(W + c, c < H, 0).to(tl.float32) + 1.0
    y = x * tl.rsqrt(tl.sum(x * x) / H + EPS) * w
    tl.store(Y + row * H + c, y, c < H)
    tl.store(RES + row * H + c, x, c < H)


def fused_add_norm(x, residual, weight, eps):
    """Qwen/Gemma normalization, retaining the FP32 sum until normalization."""
    output, combined = torch.empty_like(x), torch.empty_like(x)
    _add_norm[(x.shape[0],)](
        x,
        residual if residual is not None else x,
        weight,
        output,
        combined,
        x.shape[1],
        eps,
        residual is not None,
        triton.next_power_of_2(x.shape[1]),
    )
    return output, combined


@triton.jit
def _linear(
    X,
    W0,
    W1,
    W2,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(1)
    m = tl.arange(0, 16)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    w = _select(W0, W1, W2, group)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + k
        x = tl.load(
            X + group * M * K + m[:, None] * K + ks[None, :],
            (m[:, None] < M) & (ks[None, :] < K),
            0,
        )
        weight = tl.load(
            w + n[None, :] * K + ks[:, None], (n[None, :] < N) & (ks[:, None] < K), 0
        )
        acc = tl.dot(x, weight, acc)
    tl.store(
        Y + group * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


def grouped_linear(x, weights):
    """Multiply three small inputs by distinct, existing row-major weights."""
    groups, tokens, width = x.shape
    if groups != 3 or len(weights) != 3 or not x.is_contiguous():
        raise ValueError("Grouped linear requires three contiguous inputs")
    n = weights[0].shape[0]
    if any(w.shape != (n, width) or not w.is_contiguous() for w in weights):
        raise ValueError("Grouped linear requires equal contiguous weight shapes")
    out = torch.empty((3, tokens, n), dtype=x.dtype, device=x.device)
    block_n = 32 if n <= 64 else 64
    _linear[(triton.cdiv(n, block_n), 3)](
        x, *weights, out, tokens, n, width, block_n, 128, num_warps=4
    )
    return out


@triton.jit
def _conv(
    X,
    C0,
    C1,
    C2,
    W0,
    W1,
    W2,
    Y,
    T: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    CS: tl.constexpr,
    TS: tl.constexpr,
    SLOT: tl.constexpr,
    SS: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.program_id(1)
    c = tl.program_id(0) * B + tl.arange(0, B)
    state = _select(C0, C1, C2, group) + SLOT * SS + c * CS
    weight = _select(W0, W1, W2, group) + c * 4
    h0 = tl.load(state, c < D, 0)
    h1 = tl.load(state + TS, c < D, 0)
    h2 = tl.load(state + 2 * TS, c < D, 0)
    w0 = tl.load(weight, c < D, 0)
    w1 = tl.load(weight + 1, c < D, 0)
    w2 = tl.load(weight + 2, c < D, 0)
    w3 = tl.load(weight + 3, c < D, 0)
    # Every program owns complete channels; preserve history before overwriting.
    tl.debug_barrier()
    tl.store(state, h1, c < D)
    tl.store(state + TS, h2, c < D)
    for t in range(T):
        x = tl.load(X + (group * T + t) * P + c, c < D, 0)
        # Match causal_conv1d_update's BF16 products and FP32 accumulation.
        value = (h0 * w0).to(tl.float32)
        value += (h1 * w1).to(tl.float32)
        value += (h2 * w2).to(tl.float32)
        value += (x * w3).to(tl.float32)
        value = value / (1 + tl.exp(-value))
        tl.store(Y + (group * T + t) * D + c, value, c < D)
        tl.store(state + (t + 2) * TS, x, c < D)
        h0, h1, h2 = h1, h2, x


def grouped_conv(qkvz, convs, weights, qkv_width, slot, dim_first):
    """Advance independent private histories from an already aligned boundary."""
    c = convs[0]
    cs, ts = (c.stride(1), c.stride(2)) if dim_first else (c.stride(2), c.stride(1))
    out = torch.empty(
        (*qkvz.shape[:2], qkv_width), device=qkvz.device, dtype=qkvz.dtype
    )
    _conv[(triton.cdiv(qkv_width, 128), 3)](
        qkvz,
        *convs,
        *weights,
        out,
        qkvz.shape[1],
        qkv_width,
        qkvz.shape[2],
        cs,
        ts,
        slot,
        c.stride(0),
        128,
    )
    return out


@triton.jit
def _recurrent(
    QKV,
    BA,
    AL0,
    AL1,
    AL2,
    DT0,
    DT1,
    DT2,
    S0,
    S1,
    S2,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    TAIL: tl.constexpr,
    BV: tl.constexpr,
):
    group, head = tl.program_id(2), tl.program_id(0)
    vs = tl.program_id(1) * BV + tl.arange(0, BV)
    ks = tl.arange(0, DK)
    offsets = head * DV * DK + vs[:, None] * DK + ks[None, :]
    state_ptr = _select(S0, S1, S2, group)
    base = 0 if TAIL else HV * DV * DK
    state = tl.load(state_ptr + base + offsets, vs[:, None] < DV, 0)
    a_log = tl.load(_select(AL0, AL1, AL2, group) + head).to(tl.float32)
    dt = tl.load(_select(DT0, DT1, DT2, group) + head).to(tl.float32)
    qkv_width: tl.constexpr = 2 * H * DK + HV * DV
    key_head = head // (HV // H)
    for t in range(T):
        p = QKV + (group * T + t) * qkv_width
        q = tl.load(p + key_head * DK + ks).to(tl.float32)
        k = tl.load(p + H * DK + key_head * DK + ks).to(tl.float32)
        v = tl.load(p + 2 * H * DK + head * DV + vs, vs < DV, 0).to(tl.float32)
        a = tl.load(BA + (group * T + t) * HV * 2 + HV + head).to(tl.float32)
        b = tl.load(BA + (group * T + t) * HV * 2 + head).to(tl.float32)
        x = a + dt
        softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        g = -tl.exp(a_log) * softplus
        q *= tl.rsqrt(tl.sum(q * q) + 1e-6)
        k *= tl.rsqrt(tl.sum(k * k) + 1e-6)
        q *= DK**-0.5
        state *= tl.exp(g)
        delta = (v - tl.sum(state * k[None, :], 1)) * tl.sigmoid(b)
        state += delta[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(OUT + ((group * T + t) * HV + head) * DV + vs, output, vs < DV)
        if not TAIL:
            tl.store(
                state_ptr + (t + 1) * HV * DV * DK + offsets, state, vs[:, None] < DV
            )
    if TAIL:
        tl.store(state_ptr + offsets, state, vs[:, None] < DV)


def grouped_recurrent(
    qkv, ba, a_logs, dt_biases, states, heads, value_heads, key_dim, value_dim, tail
):
    out = torch.empty(
        (*qkv.shape[:2], value_heads * value_dim), device=qkv.device, dtype=qkv.dtype
    )
    _recurrent[(value_heads, triton.cdiv(value_dim, 16), 3)](
        qkv,
        ba,
        *a_logs,
        *dt_biases,
        *states,
        out,
        qkv.shape[1],
        heads,
        value_heads,
        key_dim,
        value_dim,
        tail,
        16,
        num_warps=4,
    )
    return out


@triton.jit
def _gated_norm(
    X,
    Z,
    W0,
    W1,
    W2,
    Y,
    T: tl.constexpr,
    HV: tl.constexpr,
    DV: tl.constexpr,
    P: tl.constexpr,
    EPS: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    token, head = row // HV, row % HV
    c = tl.arange(0, DV)
    offset = (group * T * HV + row) * DV + c
    x = tl.load(X + offset).to(tl.float32)
    z = tl.load(Z + (group * T + token) * P + P - HV * DV + head * DV + c).to(
        tl.float32
    )
    w = tl.load(_select(W0, W1, W2, group) + c).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x) / DV + EPS) * w * z * tl.sigmoid(z)
    tl.store(Y + offset, y)


def grouped_gated_norm(core, qkvz, weights, heads, dim, eps):
    out = torch.empty_like(core)
    _gated_norm[(core.shape[1] * heads, 3)](
        core, qkvz, *weights, out, core.shape[1], heads, dim, qkvz.shape[2], eps
    )
    return out
