# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only parallel GDN verify with one final-state write.

Derived from gdn_replayssm_spec_circular_kernel. Retains its window solve,
removes committed-history processing and ring writes, and adds the final state.
"""

from vllm.model_executor.layers.mamba.ops.replayssm_config import get_replayssm_config
from vllm.triton_utils import tl, triton


@triton.jit
def parallel_last_kernel(
    mixed_qkv,  # [total_tokens, qkv_dim]  packed, channel-last (q|k|v)
    a,  # [total_tokens, HV]
    b,  # [total_tokens, HV]
    A_log,  # [HV] fp32
    dt_bias,  # [HV] fp32
    o,  # [total_tokens, HV, V]  preallocated output
    h0,  # [num_slots, HV, V, K]  read-only state before the window
    ht,  # [num_slots, HV, V, K]
    query_start_loc,  # [B+1] int  packed cu_seqlens
    ssm_state_indices,  # [B] int  physical block per request
    scale,
    stride_mqkv_t: tl.constexpr,  # per-token stride of mixed_qkv
    stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_o_t: tl.constexpr,  # per-token stride of o (= HV*V)
    stride_state_slot: tl.constexpr,
    stride_qsl: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    mask_v = o_v < V

    # --- per-request packed window ---
    bos = tl.load(query_start_loc + i_n * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc + (i_n + 1) * stride_qsl).to(tl.int64)
    spec_len = eos - bos  # full window length

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)

    # output pointer (packed): token (bos + o_s), value-head i_hv, dim o_v
    p_o = o + (bos + o_s[:, None]) * stride_o_t + i_hv * V + o_v[None, :]

    if state_idx <= NULL_BLOCK_ID:
        return
    mask_s = o_s < spec_len
    out_mask = mask_s[:, None] & mask_v[None, :]

    # ------------------------------------------------------------------
    # Gates, beta and window-local cumulative decay.
    # ------------------------------------------------------------------
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    a_s = tl.load(a + (bos + o_s) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    b_s = tl.load(b + (bos + o_s) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    x = a_s + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, tl.sigmoid(b_s), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    if USE_QK_L2NORM_IN_KERNEL:
        qnorm_acc = tl.zeros([BS], dtype=tl.float32)
        knorm_acc = tl.zeros([BS], dtype=tl.float32)
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            ld = mask_s[:, None] & mask_kt[None, :]
            qn = tl.load(
                mixed_qkv
                + (bos + o_s[:, None]) * stride_mqkv_t
                + i_h * K
                + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            knn = tl.load(
                mixed_qkv
                + (bos + o_s[:, None]) * stride_mqkv_t
                + H * K
                + i_h * K
                + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            qnorm_acc += tl.sum(qn * qn, axis=1)
            knorm_acc += tl.sum(knn * knn, axis=1)
        q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-6), 0.0)
        k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-6), 0.0)
    else:
        q_rnorm = tl.where(mask_s, 1.0, 0.0)
        k_rnorm = tl.where(mask_s, 1.0, 0.0)

    # ------------------------------------------------------------------
    # State projections and within-window matrices.
    # ------------------------------------------------------------------
    hw_q = tl.zeros([BV, BS], dtype=tl.float32)
    hw_k = tl.zeros([BV, BS], dtype=tl.float32)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)

    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        ld_s = mask_s[:, None] & mask_kt[None, :]
        q_tile = tl.load(
            mixed_qkv + (bos + o_s[:, None]) * stride_mqkv_t + i_h * K + o_kt[None, :],
            mask=ld_s,
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            mixed_qkv
            + (bos + o_s[:, None]) * stride_mqkv_t
            + H * K
            + i_h * K
            + o_kt[None, :],
            mask=ld_s,
            other=0.0,
        ).to(tl.float32)
        q_tile = q_tile * (q_rnorm * scale)[:, None]
        k_tile = k_tile * k_rnorm[:, None]

        p_h0 = (
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        sc_tile = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0).to(
            tl.float32
        )
        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT, input_precision=DOT_PRECISION)
        kq_mat += tl.dot(k_tile, qT, input_precision=DOT_PRECISION)

        hw_q += tl.dot(sc_tile, qT, input_precision=DOT_PRECISION)
        hw_k += tl.dot(sc_tile, kT, input_precision=DOT_PRECISION)

    # ------------------------------------------------------------------
    # strictly-lower A and T = (I + A)^{-1}.
    # ------------------------------------------------------------------
    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)

    b_Ai = -A_mat
    for ii in range(2, BS):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)

    # ------------------------------------------------------------------
    # R and D_spec = R @ T^T.
    # ------------------------------------------------------------------
    p_v = (
        mixed_qkv
        + (bos + o_s[None, :]) * stride_mqkv_t
        + 2 * H * K
        + i_hv * V
        + o_v[:, None]
    )
    v_tile = tl.load(p_v, mask=mask_v[:, None] & mask_s[None, :], other=0.0).to(
        tl.float32
    )
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    D_spec = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]

    # ------------------------------------------------------------------
    # outputs.
    # ------------------------------------------------------------------
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF

    tl.store(p_o, tl.trans(O_tile).to(p_o.dtype.element_ty), mask=out_mask)

    # Materialize only the final state after the full verification window.
    g_end = tl.sum(tl.where(o_s == spec_len - 1, G_s, 0.0), axis=0)
    weights = tl.where(mask_s, tl.exp(g_end - G_s), 0.0)
    final_delta = D_spec * weights[None, :]
    o_final = tl.arange(0, max(16, BS))
    gather_s = tl.minimum(o_final, BS - 1)
    final_delta = tl.gather(
        final_delta, tl.broadcast_to(gather_s[None, :], [BV, max(16, BS)]), 1
    )
    final_delta = tl.where((o_final < spec_len)[None, :], final_delta, 0.0)
    final_rnorm = tl.gather(k_rnorm, gather_s, 0)
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        k_tile = tl.load(
            mixed_qkv
            + (bos + o_final[:, None]) * stride_mqkv_t
            + H * K
            + i_h * K
            + o_kt[None, :],
            mask=(o_final < spec_len)[:, None] & mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tile *= final_rnorm[:, None]
        offset = (
            state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        start_state = tl.load(
            h0 + offset, mask=mask_v[:, None] & mask_kt[None, :], other=0.0
        )
        final_state = tl.dot(
            final_delta,
            k_tile,
            acc=tl.exp(g_end) * start_state,
            input_precision=DOT_PRECISION,
        )
        tl.store(ht + offset, final_state, mask=mask_v[:, None] & mask_kt[None, :])


def parallel_last(inputs, initial_state, final_state, out):
    x = inputs
    bv, warps, nk, stages = get_replayssm_config(
        "gdn_spec_verify",
        max_spec_len=x.width,
        head_k_dim=128,
    )
    bv = bv or 64
    parallel_last_kernel[(triton.cdiv(128, bv), x.batch, 32)](
        x.qkv,
        x.a,
        x.b,
        x.a_log,
        x.bias,
        out,
        initial_state,
        final_state,
        x.qsl,
        x.slots,
        128**-0.5,
        x.qkv.stride(0),
        x.a.stride(0),
        x.b.stride(0),
        out.stride(0),
        initial_state.stride(0),
        x.qsl.stride(0),
        x.slots.stride(0),
        H=16,
        HV=32,
        K=128,
        V=128,
        BK=128,
        BV=bv,
        BS=max(4, triton.next_power_of_2(x.width)),
        NK=nk,
        BKT=128 // nk,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        NULL_BLOCK_ID=0,
        DOT_PRECISION="tf32",
        num_warps=warps,
        num_stages=stages,
    )
    return out
