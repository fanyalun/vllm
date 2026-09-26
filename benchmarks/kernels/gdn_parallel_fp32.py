# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-block GDN reference using FP32 triangular solve and prefix factors."""

import torch

from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.triton_utils import tl, triton


@triton.jit
def _triangular_coefficients(
    K,
    G,
    BETA,
    A,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    nh = tl.program_id(0)
    n, h = nh // H, nh % H
    ts = tl.arange(0, BT)
    ds = tl.arange(0, BD)
    keys = tl.load(
        K + ((n * T + ts[:, None]) * H + h) * D + ds[None, :],
        mask=(ts[:, None] < T) & (ds[None, :] < D),
        other=0,
    )
    g = tl.load(G + (n * T + ts) * H + h, mask=ts < T, other=0)
    beta = tl.load(BETA + (n * T + ts) * H + h, mask=ts < T, other=0)
    causal = (ts[:, None] > ts[None, :]) & (ts[:, None] < T) & (ts[None, :] < T)
    decay = tl.exp(tl.where(causal, g[:, None] - g[None, :], 0))
    product = tl.dot(keys, tl.trans(keys), input_precision="tf32x3")
    output = tl.where(causal, product * decay * beta[:, None], 0)
    tl.store(
        A + ((n * T + ts[:, None]) * H + h) * BT + ts[None, :],
        output,
        mask=ts[:, None] < T,
    )


@triton.jit
def _parallel_updates(
    Q,
    K,
    V,
    G,
    BETA,
    INITIAL,
    INVERSE,
    UPDATES,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    nh = tl.program_id(0)
    n, h = nh // H, nh % H
    ts = tl.arange(0, BT)
    ds = tl.arange(0, BD)
    vs = tl.program_id(1) * 16 + tl.arange(0, 16)
    offsets = ((n * T + ts[:, None]) * H + h) * D + ds[None, :]
    mask = (ts[:, None] < T) & (ds[None, :] < D)
    k = tl.load(K + offsets, mask=mask, other=0)
    state = tl.load(
        INITIAL + nh.to(tl.int64) * D * D + vs[:, None] * D + ds[None, :],
        mask=(vs[:, None] < D) & (ds[None, :] < D),
        other=0,
    )
    state_k = tl.dot(k, tl.trans(state), input_precision="tf32x3")
    g = tl.load(G + (n * T + ts) * H + h, mask=ts < T, other=0)
    beta = tl.load(BETA + (n * T + ts) * H + h, mask=ts < T, other=0)
    values = tl.load(
        V + ((n * T + ts[:, None]) * H + h) * D + vs[None, :],
        mask=(ts[:, None] < T) & (vs[None, :] < D),
        other=0,
    ).to(tl.float32)
    rhs = beta[:, None] * (values - tl.exp(g)[:, None] * state_k)
    inverse = tl.load(
        INVERSE + ((n * T + ts[:, None]) * H + h) * BT + ts[None, :],
        mask=ts[:, None] < T,
        other=0,
    )
    updates = tl.dot(inverse, rhs, input_precision="tf32x3")
    q = tl.load(Q + offsets, mask=mask, other=0)
    state_q = tl.dot(q, tl.trans(state), input_precision="tf32x3")
    destination = ((n * T + ts[:, None]) * H + h) * D + vs[None, :]
    mask_out = (ts[:, None] < T) & (vs[None, :] < D)
    tl.store(UPDATES + destination, updates, mask=mask_out)
    tl.store(OUT + destination, tl.exp(g)[:, None] * state_q, mask=mask_out)


@triton.jit
def _parallel_output(
    Q,
    K,
    G,
    UPDATES,
    BASE,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    nh = tl.program_id(0)
    n, h = nh // H, nh % H
    ts = tl.arange(0, BT)
    ds = tl.arange(0, BD)
    vs = tl.program_id(1) * 32 + tl.arange(0, 32)
    offsets = ((n * T + ts[:, None]) * H + h) * D + ds[None, :]
    mask = (ts[:, None] < T) & (ds[None, :] < D)
    q = tl.load(Q + offsets, mask=mask, other=0)
    k = tl.load(K + offsets, mask=mask, other=0)
    scores = tl.dot(q, tl.trans(k), input_precision="tf32x3")
    g = tl.load(G + (n * T + ts) * H + h, mask=ts < T, other=0)
    causal = (ts[:, None] >= ts[None, :]) & (ts[:, None] < T) & (ts[None, :] < T)
    decay = tl.exp(tl.where(causal, g[:, None] - g[None, :], 0))
    scores = tl.where(causal, scores * decay, 0)
    destination = ((n * T + ts[:, None]) * H + h) * D + vs[None, :]
    mask_out = (ts[:, None] < T) & (vs[None, :] < D)
    updates = tl.load(UPDATES + destination, mask=mask_out, other=0)
    base = tl.load(BASE + destination, mask=mask_out, other=0)
    output = (base + tl.dot(scores, updates, input_precision="tf32x3")) * D**-0.5
    tl.store(OUT + destination, output, mask=mask_out)


def parallel_gdn_fp32_fused(q, k, v, g, beta, initial):
    q, k = q.float(), k.float()
    q = (q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).contiguous()
    k = (k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).contiguous()
    cumulative = g.float().cumsum(1)
    beta = beta.float().contiguous()
    batch, length, heads, dim = v.shape
    tile = max(16, triton.next_power_of_2(length))
    matrix = torch.empty(
        batch, length, heads, tile, device=v.device, dtype=torch.float32
    )
    _triangular_coefficients[(batch * heads,)](
        k,
        cumulative,
        beta,
        matrix,
        length,
        heads,
        dim,
        tile,
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    inverse = solve_tril(A=matrix, output_dtype=torch.float32)
    updates = torch.empty_like(v, dtype=torch.float32)
    base = torch.empty_like(updates)
    output = torch.empty_like(v)
    _parallel_updates[(batch * heads, triton.cdiv(dim, 16))](
        q,
        k,
        v.contiguous(),
        cumulative,
        beta,
        initial,
        inverse,
        updates,
        base,
        length,
        heads,
        dim,
        tile,
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    _parallel_output[(batch * heads, triton.cdiv(dim, 32))](
        q,
        k,
        cumulative,
        updates,
        base,
        output,
        length,
        heads,
        dim,
        tile,
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output, k, updates, cumulative


def parallel_gdn_fp32(q, k, v, g, beta, initial):
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q = q.float()
        k = k.float()
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        qh, kh, vh = [x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v.float())]
        cumulative = g.float().cumsum(1)
        gh = cumulative.transpose(1, 2)
        bh = beta.float().transpose(1, 2)
        length = q.shape[1]
        positions = torch.arange(length, device=q.device)
        causal = positions[:, None] >= positions[None, :]
        difference = torch.where(causal, gh[..., :, None] - gh[..., None, :], 0)
        decay = difference.exp() * causal
        products = kh @ kh.transpose(-1, -2)
        lower = products * decay * bh[..., :, None]
        lower = torch.tril(lower, diagonal=-1)
        rhs = bh[..., None] * (
            vh
            - gh.exp()[..., None] * (initial @ kh.transpose(-1, -2)).transpose(-1, -2)
        )
        updates = torch.linalg.solve_triangular(
            lower, rhs, upper=False, unitriangular=True
        )
        readout = gh.exp()[..., None] * (initial @ qh.transpose(-1, -2)).transpose(
            -1, -2
        )
        readout += ((qh @ kh.transpose(-1, -2)) * decay) @ updates
        output = (readout * q.shape[-1] ** -0.5).permute(0, 2, 1, 3).to(v.dtype)
        return (
            output,
            k.contiguous(),
            updates.permute(0, 2, 1, 3).contiguous(),
            cumulative,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
