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
        "WRITE_FINAL_STATE": lambda args: args["ht"] is not None,
        "WRITE_RESIDENT_FINAL_STATE": (
            lambda args: args["resident_final_state_out"] is not None
        ),
        "INITIAL_STATE_ROW_IDS": (
            lambda args: args["ssm_state_indices"] is not None
            and args["ssm_state_indices"].ndim == 1
        ),
        "RESIDENT_STATE_ROW_IDS": (
            lambda args: args["resident_state_indices"] is not None
            and args["resident_state_indices"].ndim == 1
        ),
        "WRITE_RESIDENT_BY_TOKEN": (
            lambda args: args["resident_token_indices"] is not None
        ),
        "WRITE_SHADOW_TAPE": lambda args: args["shadow_key_out"] is not None,
        "READ_SHADOW_TAPE": lambda args: args["shadow_src_begin"] is not None,
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
    shadow_key_out,
    shadow_value_out,
    shadow_g_out,
    shadow_beta_out,
    precomputed_g,
    precomputed_beta,
    h0,
    ht,
    resident_final_state_out,
    cu_seqlens,
    ssm_state_indices,
    resident_state_indices,
    resident_token_indices,
    shadow_req_slots,
    shadow_src_begin,
    num_accepted_tokens,
    scale,
    shadow_max_seq_len: tl.int64,
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
    stride_resident_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_resident_indices_seq: tl.constexpr,
    stride_resident_indices_tok: tl.constexpr,
    stride_resident_token_indices: tl.constexpr,
    stride_shadow_req_slots: tl.constexpr,
    stride_shadow_src_begin: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    SAVE_G_BETA: tl.constexpr,
    USE_PRECOMPUTED_G_BETA: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
    WRITE_FINAL_STATE: tl.constexpr,
    WRITE_RESIDENT_FINAL_STATE: tl.constexpr,
    INITIAL_STATE_ROW_IDS: tl.constexpr,
    RESIDENT_STATE_ROW_IDS: tl.constexpr,
    WRITE_RESIDENT_BY_TOKEN: tl.constexpr,
    WRITE_SHADOW_TAPE: tl.constexpr,
    READ_SHADOW_TAPE: tl.constexpr,
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

    req_slot = 0
    src_begin = 0
    if WRITE_SHADOW_TAPE or READ_SHADOW_TAPE:
        req_slot = tl.load(
            shadow_req_slots + i_n * stride_shadow_req_slots
        ).to(tl.int64)
    if READ_SHADOW_TAPE:
        src_begin = tl.load(
            shadow_src_begin + i_n * stride_shadow_src_begin
        ).to(tl.int64)

    if WRITE_OUTPUT:
        p_q = q + (bos * H + i_h) * K + o_k
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    if READ_SHADOW_TAPE:
        shadow_pos = req_slot * shadow_max_seq_len + src_begin
        p_k = k + ((shadow_pos * H + i_h) * K + o_k)
        p_v = v + ((shadow_pos * HV + i_hv) * V + o_v)
    else:
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
        if READ_SHADOW_TAPE:
            shadow_pos = req_slot * shadow_max_seq_len + src_begin
            p_precomputed_g = precomputed_g + (shadow_pos * HV + i_hv)
            p_precomputed_beta = precomputed_beta + (shadow_pos * HV + i_hv)
        else:
            p_precomputed_g = precomputed_g + (bos * HV + i_hv)
            p_precomputed_beta = precomputed_beta + (bos * HV + i_hv)
    if WRITE_SHADOW_TAPE:
        shadow_pos = req_slot * shadow_max_seq_len
        p_shadow_k = shadow_key_out + ((shadow_pos * H + i_h) * K + o_k)
        p_shadow_v = shadow_value_out + ((shadow_pos * HV + i_hv) * V + o_v)
        p_shadow_g = shadow_g_out + (shadow_pos * HV + i_hv)
        p_shadow_beta = shadow_beta_out + (shadow_pos * HV + i_hv)

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if INITIAL_STATE_ROW_IDS:
                state_idx = tl.load(
                    ssm_state_indices + i_n * stride_indices_seq
                ).to(tl.int64)
                if state_idx < 0:
                    return
            else:
                if IS_SPEC_DECODING:
                    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                else:
                    i_t = 0
                state_idx = tl.load(
                    ssm_state_indices + i_n * stride_indices_seq + i_t
                ).to(tl.int64)
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
        if WRITE_SHADOW_TAPE:
            tl.store(p_shadow_k, b_k.to(shadow_key_out.dtype.element_ty), mask=mask_k)
            tl.store(p_shadow_v, b_v.to(shadow_value_out.dtype.element_ty), mask=mask_v)

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
        if WRITE_SHADOW_TAPE:
            if not IS_KDA:
                tl.store(p_shadow_g, b_g.to(shadow_g_out.dtype.element_ty))
            else:
                tl.store(
                    p_shadow_g + o_k,
                    b_g.to(shadow_g_out.dtype.element_ty),
                    mask=mask_k,
                )
            tl.store(
                p_shadow_beta,
                b_beta.to(shadow_beta_out.dtype.element_ty),
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

        if WRITE_FINAL_STATE:
            if INPLACE_FINAL_STATE:
                if INITIAL_STATE_ROW_IDS:
                    final_state_idx = tl.load(
                        ssm_state_indices + i_n * stride_indices_seq
                    ).to(tl.int64)
                    should_write_final = final_state_idx >= 0
                else:
                    final_state_idx = tl.load(
                        ssm_state_indices + i_n * stride_indices_seq + i_t
                    ).to(tl.int64)
                    should_write_final = final_state_idx > 0
                if should_write_final:
                    p_ht = ht + final_state_idx * stride_final_state_token
                    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        if WRITE_RESIDENT_FINAL_STATE:
            if WRITE_RESIDENT_BY_TOKEN:
                resident_state_idx = tl.load(
                    resident_state_indices + i_n * stride_resident_indices_seq
                ).to(tl.int64)
                resident_token_idx = tl.load(
                    resident_token_indices
                    + i_n * stride_resident_token_indices
                ).to(tl.int64)
                should_write_resident = resident_token_idx == i_t
                if not RESIDENT_STATE_ROW_IDS:
                    should_write_resident = (
                        should_write_resident and resident_state_idx > 0
                    )
            else:
                resident_state_idx = tl.load(
                    resident_state_indices
                    + i_n * stride_resident_indices_seq
                    + i_t * stride_resident_indices_tok
                ).to(tl.int64)
                should_write_resident = (
                    resident_state_idx >= 0
                    if RESIDENT_STATE_ROW_IDS
                    else resident_state_idx > 0
                )
            if should_write_resident:
                p_resident = (
                    resident_final_state_out
                    + resident_state_idx * stride_resident_state_token
                )
                p_resident = (
                    p_resident
                    + i_hv * V * K
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(
                    p_resident,
                    b_h.to(p_resident.dtype.element_ty),
                    mask=mask_h,
                )

        if WRITE_OUTPUT:
            p_q += H * K
            p_o += HV * V
        p_k += H * K
        p_v += HV * V
        if WRITE_SHADOW_TAPE:
            p_shadow_k += H * K
            p_shadow_v += HV * V
            p_shadow_g += HV
            p_shadow_beta += HV
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


def _validate_shadow_tape_tensor(
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
    resident_final_state_out: torch.Tensor | None,
    resident_state_indices: torch.Tensor | None,
    resident_token_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    is_kda: bool,
    saved_g_out: torch.Tensor | None = None,
    saved_beta_out: torch.Tensor | None = None,
    precomputed_g: torch.Tensor | None = None,
    precomputed_beta: torch.Tensor | None = None,
    shadow_key_out: torch.Tensor | None = None,
    shadow_value_out: torch.Tensor | None = None,
    shadow_g_out: torch.Tensor | None = None,
    shadow_beta_out: torch.Tensor | None = None,
    shadow_req_slots: torch.Tensor | None = None,
    shadow_src_begin: torch.Tensor | None = None,
    shadow_max_seq_len: int | None = None,
    write_output: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    write_shadow_tape = shadow_key_out is not None
    read_shadow_tape = shadow_src_begin is not None

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
    if (resident_final_state_out is None) != (resident_state_indices is None):
        raise ValueError(
            "resident_final_state_out and resident_state_indices must be "
            "provided together."
        )
    if resident_token_indices is not None and resident_final_state_out is None:
        raise ValueError(
            "resident_token_indices requires resident_final_state_out and "
            "resident_state_indices."
        )
    if (
        write_shadow_tape
        != (shadow_value_out is not None)
        or write_shadow_tape != (shadow_g_out is not None)
        or write_shadow_tape != (shadow_beta_out is not None)
    ):
        raise ValueError(
            "shadow_key_out, shadow_value_out, shadow_g_out, and "
            "shadow_beta_out must be provided together."
        )
    if (write_shadow_tape or read_shadow_tape) and shadow_req_slots is None:
        raise ValueError("shadow_req_slots is required for shadow tape mode.")
    if read_shadow_tape and shadow_max_seq_len is None:
        raise ValueError("shadow_max_seq_len is required for shadow replay.")
    if write_shadow_tape and shadow_max_seq_len is None:
        raise ValueError("shadow_max_seq_len is required for shadow capture.")
    if (A_log is None or a is None or b is None or dt_bias is None) and (
        precomputed_g is None
    ):
        raise ValueError("baseline mode requires A_log, a, b, and dt_bias.")
    if (saved_g_out is not None or precomputed_g is not None) and is_kda:
        raise ValueError("tape capture/replay does not support KDA mode.")
    if write_output:
        if q is None:
            raise ValueError("q is required when write_output=True.")
    elif q is None and not read_shadow_tape:
        q = k

    if read_shadow_tape:
        if cu_seqlens is None:
            raise ValueError("shadow replay requires cu_seqlens.")
        if precomputed_g is None or precomputed_beta is None:
            raise ValueError("shadow replay requires precomputed g/beta.")
        if shadow_req_slots is None or shadow_src_begin is None:
            raise ValueError(
                "shadow replay requires shadow_req_slots and shadow_src_begin."
            )
        _, shadow_seq_len, H, K = k.shape
        _, shadow_seq_len_v, HV, V = v.shape
        if shadow_seq_len_v != shadow_seq_len:
            raise ValueError("shadow key/value tapes must share max seq len.")
        if shadow_max_seq_len != shadow_seq_len:
            raise ValueError(
                "shadow_max_seq_len must match shadow key/value tensors."
            )
        T = int(cu_seqlens[-1].item())
        B = 1
        N = len(cu_seqlens) - 1
    else:
        B, T, H, K = k.shape
        HV = v.shape[2]
        V = v.shape[-1]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    batch_dim = 1 if read_shadow_tape else (q.shape[0] if q is not None else k.shape[0])
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
    if not read_shadow_tape:
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
    if write_shadow_tape:
        assert shadow_key_out is not None
        assert shadow_value_out is not None
        assert shadow_g_out is not None
        assert shadow_beta_out is not None
        assert shadow_max_seq_len is not None
        shadow_rows = shadow_key_out.shape[0]
        _validate_shadow_tape_tensor(
            shadow_key_out,
            name="shadow_key_out",
            expected_shape=(shadow_rows, shadow_max_seq_len, H, K),
        )
        _validate_shadow_tape_tensor(
            shadow_value_out,
            name="shadow_value_out",
            expected_shape=(shadow_rows, shadow_max_seq_len, HV, V),
        )
        _validate_shadow_tape_tensor(
            shadow_g_out,
            name="shadow_g_out",
            expected_shape=(shadow_rows, shadow_max_seq_len, HV),
        )
        _validate_shadow_tape_tensor(
            shadow_beta_out,
            name="shadow_beta_out",
            expected_shape=(shadow_rows, shadow_max_seq_len, HV),
        )
    if resident_final_state_out is not None:
        expected_state_shape = (HV, V, K)
        if resident_final_state_out.ndim != 4 or (
            resident_final_state_out.shape[1:] != expected_state_shape
        ):
            raise ValueError(
                "resident_final_state_out must have shape "
                f"(num_states, {HV}, {V}, {K}), got "
                f"{tuple(resident_final_state_out.shape)}."
            )

    o = q.new_empty(NK, *v.shape) if write_output and q is not None else None
    write_final_state = inplace_final_state or final_state_out is not None
    if inplace_final_state:
        if final_state_out is not None:
            raise ValueError(
                "final_state_out is incompatible with inplace_final_state=True."
            )
        final_state = initial_state
    elif final_state_out is not None:
        expected_shape = (T, HV, V, K)
        if final_state_out.shape != expected_shape:
            raise ValueError(
                "final_state_out must have shape "
                f"{expected_shape}, got {tuple(final_state_out.shape)}."
            )
        final_state = final_state_out
    elif resident_final_state_out is None and not write_output:
        allocator = q if q is not None else k
        final_state = allocator.new_empty(
            (T, HV, V, K), dtype=initial_state.dtype
        )
        write_final_state = True
    else:
        final_state = None

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = 1 if final_state is None else final_state.stride(0)
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()
    if resident_final_state_out is None:
        stride_resident_state_token = 1
        stride_resident_indices_seq, stride_resident_indices_tok = 1, 1
    else:
        stride_resident_state_token = resident_final_state_out.stride(0)
        assert resident_state_indices is not None
        if resident_state_indices.ndim == 1:
            stride_resident_indices_seq = resident_state_indices.stride(0)
            stride_resident_indices_tok = 1
        else:
            (
                stride_resident_indices_seq,
                stride_resident_indices_tok,
            ) = resident_state_indices.stride()
    stride_resident_token_indices = (
        1 if resident_token_indices is None else resident_token_indices.stride(0)
    )
    stride_shadow_req_slots = (
        1 if shadow_req_slots is None else shadow_req_slots.stride(0)
    )
    stride_shadow_src_begin = (
        1 if shadow_src_begin is None else shadow_src_begin.stride(0)
    )

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
        shadow_key_out=shadow_key_out,
        shadow_value_out=shadow_value_out,
        shadow_g_out=shadow_g_out,
        shadow_beta_out=shadow_beta_out,
        precomputed_g=precomputed_g,
        precomputed_beta=precomputed_beta,
        h0=initial_state,
        ht=final_state,
        resident_final_state_out=resident_final_state_out,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        shadow_req_slots=shadow_req_slots,
        shadow_src_begin=shadow_src_begin,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        shadow_max_seq_len=1 if shadow_max_seq_len is None else shadow_max_seq_len,
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
        stride_resident_state_token=stride_resident_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_resident_indices_seq=stride_resident_indices_seq,
        stride_resident_indices_tok=stride_resident_indices_tok,
        stride_resident_token_indices=stride_resident_token_indices,
        stride_shadow_req_slots=stride_shadow_req_slots,
        stride_shadow_src_begin=stride_shadow_src_begin,
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
    resident_final_state_out: torch.Tensor | None = None,
    resident_state_indices: torch.Tensor | None = None,
    resident_token_indices: torch.Tensor | None = None,
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
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
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
    resident_final_state_out: torch.Tensor | None = None,
    resident_state_indices: torch.Tensor | None = None,
    resident_token_indices: torch.Tensor | None = None,
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
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        saved_g_out=saved_g,
        saved_beta_out=saved_beta,
    )
    assert o is not None
    return o, final_state, saved_g, saved_beta


def fused_sigmoid_gating_delta_rule_update_capture_shadow(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    shadow_key_out: torch.Tensor,
    shadow_value_out: torch.Tensor,
    shadow_g_out: torch.Tensor,
    shadow_beta_out: torch.Tensor,
    shadow_req_slots: torch.Tensor,
    shadow_max_seq_len: int,
    final_state_out: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    resident_final_state_out: torch.Tensor | None = None,
    resident_state_indices: torch.Tensor | None = None,
    resident_token_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    o, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
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
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        shadow_key_out=shadow_key_out,
        shadow_value_out=shadow_value_out,
        shadow_g_out=shadow_g_out,
        shadow_beta_out=shadow_beta_out,
        shadow_req_slots=shadow_req_slots,
        shadow_max_seq_len=shadow_max_seq_len,
    )
    assert o is not None
    return o, final_state


def fused_sigmoid_gating_delta_rule_update_capture_shadow_resident(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    shadow_key_out: torch.Tensor,
    shadow_value_out: torch.Tensor,
    shadow_g_out: torch.Tensor,
    shadow_beta_out: torch.Tensor,
    shadow_req_slots: torch.Tensor,
    shadow_max_seq_len: int,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    resident_final_state_out: torch.Tensor,
    resident_state_indices: torch.Tensor,
    resident_token_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> torch.Tensor:
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            "capture_shadow_resident requires 1D initial-state row ids."
        )
    if resident_state_indices.ndim != 1:
        raise ValueError(
            "capture_shadow_resident requires 1D resident-state row ids."
        )
    o, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        scale=None,
        initial_state=initial_state,
        inplace_final_state=False,
        final_state_out=None,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        shadow_key_out=shadow_key_out,
        shadow_value_out=shadow_value_out,
        shadow_g_out=shadow_g_out,
        shadow_beta_out=shadow_beta_out,
        shadow_req_slots=shadow_req_slots,
        shadow_max_seq_len=shadow_max_seq_len,
    )
    assert o is not None
    assert final_state is None
    return o


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
    resident_final_state_out: torch.Tensor | None = None,
    resident_state_indices: torch.Tensor | None = None,
    resident_token_indices: torch.Tensor | None = None,
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
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        precomputed_g=g,
        precomputed_beta=beta,
        write_output=False,
    )
    assert final_state is not None
    return final_state


def fused_sigmoid_gating_delta_rule_replay_from_shadow(
    shadow_key: torch.Tensor,
    shadow_value: torch.Tensor,
    shadow_g: torch.Tensor,
    shadow_beta: torch.Tensor,
    *,
    shadow_req_slots: torch.Tensor,
    shadow_src_begin: torch.Tensor,
    shadow_max_seq_len: int,
    initial_state: torch.Tensor,
    final_state_out: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    resident_final_state_out: torch.Tensor | None = None,
    resident_state_indices: torch.Tensor | None = None,
    resident_token_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> torch.Tensor | None:
    _, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=None,
        a=None,
        b=None,
        dt_bias=None,
        q=None,
        k=shadow_key,
        v=shadow_value,
        beta=1.0,
        threshold=20.0,
        scale=None,
        initial_state=initial_state,
        inplace_final_state=False,
        final_state_out=final_state_out,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        precomputed_g=shadow_g,
        precomputed_beta=shadow_beta,
        shadow_req_slots=shadow_req_slots,
        shadow_src_begin=shadow_src_begin,
        shadow_max_seq_len=shadow_max_seq_len,
        write_output=False,
    )
    return final_state


def fused_sigmoid_gating_delta_rule_replay_from_shadow_resident(
    shadow_key: torch.Tensor,
    shadow_value: torch.Tensor,
    shadow_g: torch.Tensor,
    shadow_beta: torch.Tensor,
    *,
    shadow_req_slots: torch.Tensor,
    shadow_src_begin: torch.Tensor,
    shadow_max_seq_len: int,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    resident_final_state_out: torch.Tensor,
    resident_state_indices: torch.Tensor,
    resident_token_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> None:
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            "replay_from_shadow_resident requires 1D initial-state row ids."
        )
    if resident_state_indices.ndim != 1:
        raise ValueError(
            "replay_from_shadow_resident requires 1D resident-state row ids."
        )
    _, final_state = _launch_fused_sigmoid_gating_delta_rule(
        A_log=None,
        a=None,
        b=None,
        dt_bias=None,
        q=None,
        k=shadow_key,
        v=shadow_value,
        beta=1.0,
        threshold=20.0,
        scale=None,
        initial_state=initial_state,
        inplace_final_state=False,
        final_state_out=None,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        resident_final_state_out=resident_final_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        precomputed_g=shadow_g,
        precomputed_beta=shadow_beta,
        shadow_req_slots=shadow_req_slots,
        shadow_src_begin=shadow_src_begin,
        shadow_max_seq_len=shadow_max_seq_len,
        write_output=False,
    )
    assert final_state is None
