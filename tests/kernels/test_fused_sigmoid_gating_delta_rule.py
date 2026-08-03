# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating as fused_sigmoid_gating_module,
)
from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_replay_from_shadow,
    fused_sigmoid_gating_delta_rule_replay_from_shadow_resident,
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_replay_from_tape,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_capture_shadow,
    fused_sigmoid_gating_delta_rule_update_capture_shadow_resident,
    fused_sigmoid_gating_delta_rule_update_capture_tape,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type
HAS_ACCELERATOR = bool(DEVICE)


def test_fused_sigmoid_gating_delta_rule_capture_tape_forwards_resident_write_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return torch.empty(1, 2, 1, 1), torch.empty(2, 1, 1, 1)

    monkeypatch.setattr(
        fused_sigmoid_gating_module,
        "_launch_fused_sigmoid_gating_delta_rule",
        fake_launch,
    )

    resident_state_out = torch.empty(3, 1, 1, 1)
    resident_state_indices = torch.tensor([[1, 0]], dtype=torch.int32)
    out, final_states, saved_g, saved_beta = (
        fused_sigmoid_gating_delta_rule_update_capture_tape(
            A_log=torch.zeros(1),
            a=torch.zeros(2, 1),
            b=torch.zeros(2, 1),
            dt_bias=torch.zeros(1),
            q=torch.zeros(1, 2, 1, 1),
            k=torch.zeros(1, 2, 1, 1),
            v=torch.zeros(1, 2, 1, 1),
            initial_state=torch.zeros(2, 1, 1, 1),
            resident_final_state_out=resident_state_out,
            resident_state_indices=resident_state_indices,
            ssm_state_indices=torch.tensor([[1, 1]], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        )
    )

    assert isinstance(out, torch.Tensor)
    assert isinstance(final_states, torch.Tensor)
    assert tuple(saved_g.shape) == (2, 1)
    assert tuple(saved_beta.shape) == (2, 1)
    assert captured["resident_final_state_out"] is resident_state_out
    assert captured["resident_state_indices"] is resident_state_indices


def test_fused_sigmoid_gating_delta_rule_capture_shadow_forwards_shadow_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return torch.empty(1, 2, 1, 1), torch.empty(2, 1, 1, 1)

    monkeypatch.setattr(
        fused_sigmoid_gating_module,
        "_launch_fused_sigmoid_gating_delta_rule",
        fake_launch,
    )

    resident_state_out = torch.empty(3, 1, 1, 1)
    resident_state_indices = torch.tensor([1], dtype=torch.int32)
    resident_token_indices = torch.tensor([0], dtype=torch.int32)
    shadow_key = torch.empty(3, 2, 1, 1)
    shadow_value = torch.empty(3, 2, 1, 1)
    shadow_g = torch.empty(3, 2, 1)
    shadow_beta = torch.empty(3, 2, 1)
    shadow_req_slots = torch.tensor([2], dtype=torch.long)
    out, final_states = fused_sigmoid_gating_delta_rule_update_capture_shadow(
        A_log=torch.zeros(1),
        a=torch.zeros(2, 1),
        b=torch.zeros(2, 1),
        dt_bias=torch.zeros(1),
        q=torch.zeros(1, 2, 1, 1),
        k=torch.zeros(1, 2, 1, 1),
        v=torch.zeros(1, 2, 1, 1),
        initial_state=torch.zeros(2, 1, 1, 1),
        shadow_key_out=shadow_key,
        shadow_value_out=shadow_value,
        shadow_g_out=shadow_g,
        shadow_beta_out=shadow_beta,
        shadow_req_slots=shadow_req_slots,
        shadow_max_seq_len=2,
        resident_final_state_out=resident_state_out,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        ssm_state_indices=torch.tensor([[1, 1]], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
    )

    assert isinstance(out, torch.Tensor)
    assert isinstance(final_states, torch.Tensor)
    assert captured["shadow_key_out"] is shadow_key
    assert captured["shadow_value_out"] is shadow_value
    assert captured["shadow_g_out"] is shadow_g
    assert captured["shadow_beta_out"] is shadow_beta
    assert captured["shadow_req_slots"] is shadow_req_slots
    assert captured["resident_state_indices"] is resident_state_indices
    assert captured["resident_token_indices"] is resident_token_indices
    assert captured["final_state_out"] is None


def test_fused_sigmoid_gating_delta_rule_replay_from_shadow_forwards_shadow_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return None, torch.empty(2, 1, 1, 1)

    monkeypatch.setattr(
        fused_sigmoid_gating_module,
        "_launch_fused_sigmoid_gating_delta_rule",
        fake_launch,
    )

    shadow_key = torch.zeros(2, 3, 1, 1)
    shadow_value = torch.zeros(2, 3, 1, 1)
    shadow_g = torch.zeros(2, 3, 1)
    shadow_beta = torch.zeros(2, 3, 1)
    shadow_req_slots = torch.tensor([1], dtype=torch.long)
    shadow_src_begin = torch.tensor([1], dtype=torch.int32)
    final_states = fused_sigmoid_gating_delta_rule_replay_from_shadow(
        shadow_key=shadow_key,
        shadow_value=shadow_value,
        shadow_g=shadow_g,
        shadow_beta=shadow_beta,
        shadow_req_slots=shadow_req_slots,
        shadow_src_begin=shadow_src_begin,
        shadow_max_seq_len=3,
        initial_state=torch.zeros(2, 1, 1, 1),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor([[1]], dtype=torch.int32),
    )

    assert isinstance(final_states, torch.Tensor)
    assert captured["k"] is shadow_key
    assert captured["v"] is shadow_value
    assert captured["precomputed_g"] is shadow_g
    assert captured["precomputed_beta"] is shadow_beta
    assert captured["shadow_req_slots"] is shadow_req_slots
    assert captured["shadow_src_begin"] is shadow_src_begin


def test_fused_sigmoid_gating_delta_rule_capture_shadow_resident_forwards_shadow_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return torch.empty(1, 2, 1, 1), None

    monkeypatch.setattr(
        fused_sigmoid_gating_module,
        "_launch_fused_sigmoid_gating_delta_rule",
        fake_launch,
    )

    out = fused_sigmoid_gating_delta_rule_update_capture_shadow_resident(
        A_log=torch.zeros(1),
        a=torch.zeros(2, 1),
        b=torch.zeros(2, 1),
        dt_bias=torch.zeros(1),
        q=torch.zeros(1, 2, 1, 1),
        k=torch.zeros(1, 2, 1, 1),
        v=torch.zeros(1, 2, 1, 1),
        initial_state=torch.zeros(1, 1, 1, 1),
        shadow_key_out=torch.zeros(1, 2, 1, 1),
        shadow_value_out=torch.zeros(1, 2, 1, 1),
        shadow_g_out=torch.zeros(1, 2, 1),
        shadow_beta_out=torch.zeros(1, 2, 1),
        shadow_req_slots=torch.tensor([0], dtype=torch.long),
        shadow_max_seq_len=2,
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32),
        resident_final_state_out=torch.zeros(2, 1, 1, 1),
        resident_state_indices=torch.tensor([1], dtype=torch.int32),
        resident_token_indices=torch.tensor([0], dtype=torch.int32),
    )

    assert isinstance(out, torch.Tensor)
    assert captured["final_state_out"] is None
    assert captured["ssm_state_indices"].tolist() == [0]
    assert captured["resident_state_indices"].tolist() == [1]


def test_fused_sigmoid_gating_delta_rule_replay_from_shadow_resident_forwards_shadow_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(
        fused_sigmoid_gating_module,
        "_launch_fused_sigmoid_gating_delta_rule",
        fake_launch,
    )

    fused_sigmoid_gating_delta_rule_replay_from_shadow_resident(
        shadow_key=torch.zeros(1, 2, 1, 1),
        shadow_value=torch.zeros(1, 2, 1, 1),
        shadow_g=torch.zeros(1, 2, 1),
        shadow_beta=torch.zeros(1, 2, 1),
        shadow_req_slots=torch.tensor([0], dtype=torch.long),
        shadow_src_begin=torch.tensor([0], dtype=torch.int32),
        shadow_max_seq_len=2,
        initial_state=torch.zeros(2, 1, 1, 1),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor([1], dtype=torch.int32),
        resident_final_state_out=torch.zeros(2, 1, 1, 1),
        resident_state_indices=torch.tensor([1], dtype=torch.int32),
        resident_token_indices=torch.tensor([1], dtype=torch.int32),
    )

    assert captured["final_state_out"] is None
    assert captured["write_output"] is False
    assert captured["ssm_state_indices"].tolist() == [1]
    assert captured["resident_state_indices"].tolist() == [1]


def test_replay_from_shadow_supports_workspace_target_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_launch(**kwargs):
        initial_state = kwargs["initial_state"]
        cu_seqlens = kwargs["cu_seqlens"]
        ssm_state_indices = kwargs["ssm_state_indices"]
        resident_final_state_out = kwargs["resident_final_state_out"]
        resident_state_indices = kwargs["resident_state_indices"]
        final_state = torch.empty(
            int(cu_seqlens[-1].item()),
            *initial_state.shape[1:],
            dtype=initial_state.dtype,
        )
        for row, (start, end) in enumerate(
            zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
        ):
            state = initial_state[int(ssm_state_indices[row, 0].item())].clone()
            for token_idx in range(start, end):
                state = state + 1
                final_state[token_idx].copy_(state)
            resident_final_state_out[
                int(resident_state_indices[row].item())
            ].copy_(state)
        return None, final_state

    monkeypatch.setattr(
        fused_sigmoid_gating_module,
        "_launch_fused_sigmoid_gating_delta_rule",
        fake_launch,
    )

    common_kwargs = dict(
        shadow_key=torch.zeros(1, 2, 1, 1),
        shadow_value=torch.zeros(1, 2, 1, 1),
        shadow_g=torch.zeros(1, 2, 1),
        shadow_beta=torch.zeros(1, 2, 1),
        shadow_req_slots=torch.tensor([0], dtype=torch.long),
        shadow_src_begin=torch.tensor([0], dtype=torch.int32),
        shadow_max_seq_len=2,
        initial_state=torch.tensor([[[[0.0]]], [[[3.0]]]]),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor([[1]], dtype=torch.int32),
        resident_token_indices=torch.tensor([1], dtype=torch.int32),
    )
    resident_ssm_state = torch.zeros(4, 1, 1, 1)
    resident_workspace = torch.zeros(3, 1, 1, 1)

    fused_sigmoid_gating_delta_rule_replay_from_shadow(
        resident_final_state_out=resident_ssm_state,
        resident_state_indices=torch.tensor([2], dtype=torch.int32),
        **common_kwargs,
    )
    fused_sigmoid_gating_delta_rule_replay_from_shadow(
        resident_final_state_out=resident_workspace,
        resident_state_indices=torch.tensor([1], dtype=torch.int32),
        **common_kwargs,
    )

    torch.testing.assert_close(resident_ssm_state[2], resident_workspace[1])


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
    if not HAS_ACCELERATOR:
        pytest.skip("accelerator is required for Triton kernel coverage")
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
    if not HAS_ACCELERATOR:
        pytest.skip("accelerator is required for Triton kernel coverage")
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
@pytest.mark.parametrize(
    ("query_lens", "predicted_slots"),
    [
        ([3], [1]),
        ([2, 3], [0, 2]),
    ],
)
def test_fused_sigmoid_gating_delta_rule_capture_tape_direct_resident_write_matches_post_copy(
    dtype: torch.dtype,
    query_lens: list[int],
    predicted_slots: list[int],
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(2)
    num_reqs = len(query_lens)
    max_query_len = max(query_lens)
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    total_tokens = sum(query_lens)

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
        .expand(-1, max_query_len)
        .contiguous()
    )
    resident_state_indices = torch.zeros(
        (num_reqs, max_query_len),
        dtype=torch.int32,
    )
    cu_seqlens_list = [0]
    for row, query_len in enumerate(query_lens):
        resident_state_indices[row, predicted_slots[row]] = row + 1
        cu_seqlens_list.append(cu_seqlens_list[-1] + query_len)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32)

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
        use_qk_l2norm_in_kernel=True,
    )
    resident_state_out = torch.full(
        (num_reqs + 1, num_v_heads, head_v_dim, head_k_dim),
        -1,
        dtype=initial_state.dtype,
        device=initial_state.device,
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
            resident_final_state_out=resident_state_out,
            resident_state_indices=resident_state_indices,
            use_qk_l2norm_in_kernel=True,
        )
    )

    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_states, ref_final, atol=1e-2, rtol=1e-2)
    assert tuple(saved_g.shape) == (total_tokens, num_v_heads)
    assert tuple(saved_beta.shape) == (total_tokens, num_v_heads)
    for row, predicted_slot in enumerate(predicted_slots):
        global_slot = cu_seqlens_list[row] + predicted_slot
        torch.testing.assert_close(
            resident_state_out[row + 1],
            ref_final[global_slot],
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="Need accelerator device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("query_lens", "predicted_slots"),
    [
        ([3], [1]),
        ([2, 3], [0, 2]),
    ],
)
def test_fused_sigmoid_gating_delta_rule_capture_shadow_without_final_state_matches_observables(
    dtype: torch.dtype,
    query_lens: list[int],
    predicted_slots: list[int],
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(3)
    num_reqs = len(query_lens)
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    total_tokens = sum(query_lens)
    max_query_len = max(query_lens)

    query = torch.rand(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype)
    key = torch.rand(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype)
    value = torch.rand(1, total_tokens, num_v_heads, head_v_dim, dtype=dtype)
    A_log = torch.rand(num_v_heads, dtype=dtype)
    dt_bias = torch.rand(num_v_heads, dtype=dtype)
    a = torch.rand(total_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(total_tokens, num_v_heads, dtype=dtype)
    initial_state = torch.rand(
        num_reqs,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=dtype,
    )
    initial_state_row_ids = torch.arange(num_reqs, dtype=torch.int32)
    resident_state_indices = torch.tensor(
        [0, 2][:num_reqs],
        dtype=torch.int32,
    )
    resident_token_indices = torch.tensor(predicted_slots, dtype=torch.int32)
    cu_seqlens_list = [0]
    for query_len in query_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + query_len)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32)
    shadow_req_slots = torch.tensor(
        [2, 0][:num_reqs],
        dtype=torch.long,
    )
    shadow_rows = int(shadow_req_slots.max().item()) + 1
    resident_state_out_ref = torch.full(
        (3, num_v_heads, head_v_dim, head_k_dim),
        -1,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    resident_state_out_no_final = resident_state_out_ref.clone()
    final_state_out = torch.empty(
        total_tokens,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=initial_state.dtype,
    )

    shadow_ref = {
        "key": torch.zeros(
            shadow_rows,
            max_query_len,
            num_k_heads,
            head_k_dim,
            dtype=dtype,
        ),
        "value": torch.zeros(
            shadow_rows,
            max_query_len,
            num_v_heads,
            head_v_dim,
            dtype=dtype,
        ),
        "g": torch.zeros(
            shadow_rows,
            max_query_len,
            num_v_heads,
            dtype=torch.float32,
        ),
        "beta": torch.zeros(
            shadow_rows,
            max_query_len,
            num_v_heads,
            dtype=torch.float32,
        ),
    }
    shadow_no_final = {
        name: tensor.clone() for name, tensor in shadow_ref.items()
    }

    ref_out, ref_final = fused_sigmoid_gating_delta_rule_update_capture_shadow(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=initial_state.clone(),
        shadow_key_out=shadow_ref["key"],
        shadow_value_out=shadow_ref["value"],
        shadow_g_out=shadow_ref["g"],
        shadow_beta_out=shadow_ref["beta"],
        shadow_req_slots=shadow_req_slots,
        shadow_max_seq_len=max_query_len,
        final_state_out=final_state_out.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=initial_state_row_ids,
        resident_final_state_out=resident_state_out_ref,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        use_qk_l2norm_in_kernel=True,
    )
    out_no_final, final_state_none = fused_sigmoid_gating_delta_rule_update_capture_shadow(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=initial_state.clone(),
        shadow_key_out=shadow_no_final["key"],
        shadow_value_out=shadow_no_final["value"],
        shadow_g_out=shadow_no_final["g"],
        shadow_beta_out=shadow_no_final["beta"],
        shadow_req_slots=shadow_req_slots,
        shadow_max_seq_len=max_query_len,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=initial_state_row_ids,
        resident_final_state_out=resident_state_out_no_final,
        resident_state_indices=resident_state_indices,
        resident_token_indices=resident_token_indices,
        use_qk_l2norm_in_kernel=True,
    )

    assert final_state_none is None
    torch.testing.assert_close(out_no_final, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        resident_state_out_no_final,
        resident_state_out_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(shadow_no_final["key"], shadow_ref["key"])
    torch.testing.assert_close(shadow_no_final["value"], shadow_ref["value"])
    torch.testing.assert_close(shadow_no_final["g"], shadow_ref["g"])
    torch.testing.assert_close(shadow_no_final["beta"], shadow_ref["beta"])
    assert isinstance(ref_final, torch.Tensor)


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
