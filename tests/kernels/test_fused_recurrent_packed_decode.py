# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_float32_gdn_prefill_uses_recurrent_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seq_len, num_q_heads = 1, 8, 2
    num_v_heads, head_dim = 4, 128

    q = torch.randn(
        batch,
        seq_len,
        num_q_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch,
        seq_len,
        num_v_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    g = -torch.rand(
        batch,
        seq_len,
        num_v_heads,
        device=device,
        dtype=torch.float32,
    )
    beta = torch.rand_like(g)
    initial_state = torch.randn(
        batch,
        num_v_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    output_buffer = torch.empty_like(v.squeeze(0))

    actual_output, actual_state = ChunkGatedDeltaRule.forward_native(
        None,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=output_buffer,
    )
    expected_output, per_token_states = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        inplace_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, per_token_states[-1:])
    torch.testing.assert_close(output_buffer, expected_output.squeeze(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_float32_gdn_prefill_rejects_multiple_sequences():
    q = torch.zeros(1, 4, 1, 128, device="cuda", dtype=torch.float32)
    v = torch.zeros(1, 4, 2, 128, device="cuda", dtype=torch.float32)
    g = torch.zeros(1, 4, 2, device="cuda", dtype=torch.float32)
    initial_state = torch.zeros(2, 2, 128, 128, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="exactly one sequence"):
        ChunkGatedDeltaRule.forward_native(
            None,
            q=q,
            k=q,
            v=v,
            g=g,
            beta=g,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, 2, 4], device="cuda"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device("cuda")

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (slot 0 is the null block; include PAD_SLOT_ID=-1).
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).to(dtype).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    torch.testing.assert_close(out_packed, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)
