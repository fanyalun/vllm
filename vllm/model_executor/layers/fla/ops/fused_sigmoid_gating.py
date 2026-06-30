# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch

from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "SAVE_G_BETA": lambda args: args["saved_g_out"] is not None,
        "USE_PRECOMPUTED_G_BETA": lambda args: args["precomputed_g"] is not None,
        "WRITE_OUTPUT": lambda args: args["o"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    saved_g_out,
    saved_beta_out,
    precomputed_g,
    precomputed_beta,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    SAVE_G_BETA: tl.constexpr,
    USE_PRECOMPUTED_G_BETA: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    if WRITE_OUTPUT:
        p_q = q + (bos * H + i_h) * K + o_k
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if not USE_PRECOMPUTED_G_BETA:
        p_A_log = A_log + i_hv
        if not IS_KDA:
            p_a = a + bos * HV + i_hv
            p_dt_bias = dt_bias + i_hv
        else:
            p_a = a + (bos * HV + i_hv) * K + o_k
            p_dt_bias = dt_bias + i_hv * K + o_k
        p_b = b + bos * HV + i_hv
        if SAVE_G_BETA:
            p_saved_g = saved_g_out + (bos * HV + i_hv)
            p_saved_beta = saved_beta_out + (bos * HV + i_hv)
    else:
        p_precomputed_g = precomputed_g + (bos * HV + i_hv)
        p_precomputed_beta = precomputed_beta + (bos * HV + i_hv)

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if WRITE_OUTPUT:
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)

        if USE_PRECOMPUTED_G_BETA:
            if not IS_KDA:
                b_g = tl.load(p_precomputed_g).to(tl.float32)
            else:
                b_g = tl.load(p_precomputed_g + o_k, mask=mask_k, other=0).to(
                    tl.float32
                )
            b_beta = tl.load(p_precomputed_beta).to(tl.float32)
        else:
            b_b = tl.load(p_b).to(tl.float32)
            x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
            softplus_x = tl.where(
                beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
            )
            b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
            b_beta = tl.sigmoid(b_b)
            if SAVE_G_BETA:
                if not IS_KDA:
                    tl.store(p_saved_g, b_g.to(saved_g_out.dtype.element_ty))
                else:
                    tl.store(
                        p_saved_g + o_k,
                        b_g.to(saved_g_out.dtype.element_ty),
                        mask=mask_k,
                    )
                tl.store(
                    p_saved_beta,
                    b_beta.to(saved_beta_out.dtype.element_ty),
                )

        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
            if WRITE_OUTPUT:
                b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        if WRITE_OUTPUT:
            b_q = b_q * scale

        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        if WRITE_OUTPUT:
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        if WRITE_OUTPUT:
            p_q += H * K
            p_o += HV * V
        p_k += H * K
        p_v += HV * V
        if USE_PRECOMPUTED_G_BETA:
            if not IS_KDA:
                p_precomputed_g += HV
            else:
                p_precomputed_g += HV * K
            p_precomputed_beta += HV
        else:
            if not IS_KDA:
                p_a += HV
            else:
                p_a += HV * K
            p_b += HV
            if SAVE_G_BETA:
                if not IS_KDA:
                    p_saved_g += HV
                else:
                    p_saved_g += HV * K
                p_saved_beta += HV


def _validate_token_major_tensor(
    tensor: torch.Tensor | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    if tensor is None:
        return
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
        )


def _launch_fused_sigmoid_gating_delta_rule(
    *,
    A_log: torch.Tensor | None,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float | None,
    initial_state: torch.Tensor,
    inplace_final_state: bool,
    final_state_out: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    is_kda: bool,
    saved_g_out: torch.Tensor | None = None,
    saved_beta_out: torch.Tensor | None = None,
    precomputed_g: torch.Tensor | None = None,
    precomputed_beta: torch.Tensor | None = None,
    write_output: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if saved_g_out is not None and precomputed_g is not None:
        raise ValueError(
            "saved_g_out and precomputed_g are mutually exclusive modes."
        )
    if (saved_g_out is None) != (saved_beta_out is None):
        raise ValueError("saved_g_out and saved_beta_out must be provided together.")
    if (precomputed_g is None) != (precomputed_beta is None):
        raise ValueError(
            "precomputed_g and precomputed_beta must be provided together."
        )
    if (A_log is None or a is None or b is None or dt_bias is None) and (
        precomputed_g is None
    ):
        raise ValueError("baseline mode requires A_log, a, b, and dt_bias.")
    if (saved_g_out is not None or precomputed_g is not None) and is_kda:
        raise ValueError("tape capture/replay does not support KDA mode.")
    if write_output:
        if q is None:
            raise ValueError("q is required when write_output=True.")
    elif q is None:
        q = k

    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    batch_dim = q.shape[0] if q is not None else k.shape[0]
    if cu_seqlens is not None and batch_dim != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {batch_dim} "
            "when using `cu_seqlens`. Please flatten variable-length inputs "
            "before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    token_major_shape = (T, HV)
    _validate_token_major_tensor(
        saved_g_out,
        name="saved_g_out",
        expected_shape=token_major_shape,
    )
    _validate_token_major_tensor(
        saved_beta_out,
        name="saved_beta_out",
        expected_shape=token_major_shape,
    )
    _validate_token_major_tensor(
        precomputed_g,
        name="precomputed_g",
        expected_shape=token_major_shape,
    )
    _validate_token_major_tensor(
        precomputed_beta,
        name="precomputed_beta",
        expected_shape=token_major_shape,
    )

    o = q.new_empty(NK, *v.shape) if write_output and q is not None else None
    if inplace_final_state:
        if final_state_out is not None:
            raise ValueError(
                "final_state_out is incompatible with inplace_final_state=True."
            )
        final_state = initial_state
    else:
        expected_shape = (T, HV, V, K)
        if final_state_out is not None:
            if final_state_out.shape != expected_shape:
                raise ValueError(
                    "final_state_out must have shape "
                    f"{expected_shape}, got {tuple(final_state_out.shape)}."
                )
            final_state = final_state_out
        else:
            allocator = q if q is not None else k
            final_state = allocator.new_empty(
                expected_shape, dtype=initial_state.dtype
            )

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=None if a is None else a.contiguous(),
        b=None if b is None else b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=None if q is None else q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        saved_g_out=saved_g_out,
        saved_beta_out=saved_beta_out,
        precomputed_g=precomputed_g,
        precomputed_beta=precomputed_beta,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (None if o is None else o.squeeze(0)), final_state


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    final_state_out: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """Baseline fused sigmoid-gating delta-rule update."""
    o, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        final_state_out=final_state_out,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
    )
    assert o is not None
    return o, final_state


def fused_sigmoid_gating_delta_rule_update_capture_tape(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = False,
    final_state_out: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if inplace_final_state:
        raise ValueError(
            "capture_tape requires inplace_final_state=False so the per-token "
            "final states remain materialized for predicted checkpoint selection."
        )
    total_tokens = k.shape[1]
    hv = v.shape[2]
    saved_g = torch.empty((total_tokens, hv), dtype=torch.float32, device=k.device)
    saved_beta = torch.empty(
        (total_tokens, hv),
        dtype=torch.float32,
        device=k.device,
    )
    o, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=False,
        final_state_out=final_state_out,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        saved_g_out=saved_g,
        saved_beta_out=saved_beta,
    )
    assert o is not None
    return o, final_state, saved_g, saved_beta


def fused_sigmoid_gating_delta_rule_replay_from_tape(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    final_state_out: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> torch.Tensor:
    _, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=None,
        a=None,
        b=None,
        dt_bias=None,
        q=None,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        scale=None,
        initial_state=initial_state,
        inplace_final_state=False,
        final_state_out=final_state_out,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        precomputed_g=g,
        precomputed_beta=beta,
        write_output=False,
    )
    return final_state
