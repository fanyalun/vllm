# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_replay_from_tape,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_capture_tape,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type
HAS_ACCELERATOR = bool(DEVICE)


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    seq_len = 1  # seq_len is 1 for decode
    num_tokens = num_reqs * seq_len
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(
        total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    # Index 0 is reserved as NULL_BLOCK_ID for continuous batching state slots.
    state_indices = (
        torch.randperm(total_entries - 1, dtype=torch.int32)[:num_tokens] + 1
    )
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * (num_speculative_tokens + 1)
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(
        total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    # Index 0 is reserved as NULL_BLOCK_ID for continuous batching state slots.
    state_indices = (
        torch.randperm(total_entries - 1, dtype=torch.int32)[:num_tokens] + 1
    ).view(num_reqs, num_speculative_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_speculative_tokens + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(
        0, num_tokens + 1, num_speculative_tokens + 1, dtype=torch.int32
    )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="Need accelerator device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_capture_tape_matches_update(
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    num_reqs = 2
    num_tokens_per_req = 3
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    total_tokens = num_reqs * num_tokens_per_req

    query = torch.rand(
        1, total_tokens, num_k_heads, head_k_dim, dtype=dtype
    )
    key = torch.rand(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype)
    value = torch.rand(1, total_tokens, num_v_heads, head_v_dim, dtype=dtype)
    A_log = torch.rand(num_v_heads, dtype=dtype)
    dt_bias = torch.rand(num_v_heads, dtype=dtype)
    a = torch.rand(total_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(total_tokens, num_v_heads, dtype=dtype)
    initial_state = torch.rand(
        num_reqs + 1,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=dtype,
    )
    initial_state[0].zero_()
    final_state_out = torch.empty(
        total_tokens,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=dtype,
    )
    state_indices = (
        torch.arange(1, num_reqs + 1, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, num_tokens_per_req)
        .contiguous()
    )
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        num_tokens_per_req,
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32)

    ref_out, ref_final = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=initial_state.clone(),
        inplace_final_state=False,
        final_state_out=final_state_out.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )
    out, final_states, saved_g, saved_beta = (
        fused_sigmoid_gating_delta_rule_update_capture_tape(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=initial_state.clone(),
            final_state_out=final_state_out.clone(),
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )
    )

    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_states, ref_final, atol=1e-2, rtol=1e-2)
    assert tuple(saved_g.shape) == (total_tokens, num_v_heads)
    assert tuple(saved_beta.shape) == (total_tokens, num_v_heads)


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="Need accelerator device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_replay_from_tape_recovers_final_state(
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(1)
    num_reqs = 2
    num_tokens_per_req = 4
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    total_tokens = num_reqs * num_tokens_per_req

    query = torch.rand(
        1, total_tokens, num_k_heads, head_k_dim, dtype=dtype
    )
    key = torch.rand(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype)
    value = torch.rand(1, total_tokens, num_v_heads, head_v_dim, dtype=dtype)
    A_log = torch.rand(num_v_heads, dtype=dtype)
    dt_bias = torch.rand(num_v_heads, dtype=dtype)
    a = torch.rand(total_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(total_tokens, num_v_heads, dtype=dtype)
    initial_state = torch.rand(
        num_reqs + 1,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=dtype,
    )
    initial_state[0].zero_()
    state_indices = (
        torch.arange(1, num_reqs + 1, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, num_tokens_per_req)
        .contiguous()
    )
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        num_tokens_per_req,
        dtype=torch.int32,
    )

    _, captured_final, saved_g, saved_beta = (
        fused_sigmoid_gating_delta_rule_update_capture_tape(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=initial_state.clone(),
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
    )
    replay_final = fused_sigmoid_gating_delta_rule_replay_from_tape(
        k=key,
        v=value,
        g=saved_g,
        beta=saved_beta,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(replay_final, captured_final, atol=1e-2, rtol=1e-2)
