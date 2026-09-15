# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-path tests for fused Qwen3.5 GDN MTP decode."""

from __future__ import annotations

import types
from typing import cast
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(80)):
    pytest.skip(
        reason="Fused GDN MTP decode requires CUDA compute capability 8.0+.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)
from vllm.utils.torch_utils import _encode_layer_name  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
SPEC_TOKENS = NUM_SPEC + 1
H = 1
HV = 8
K = 128
V = 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
EPS = 1e-6


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("extreme", [False, True])
@torch.inference_mode()
def test_mean_state_update_matches_one_pooled_write_and_token_queries(tokens, extreme):
    from vllm.model_executor.layers.mamba.gdn.mean_update import mean_state_update

    torch.manual_seed(42)
    h, hv, dim = 2, 4, 128
    q, k = [
        torch.randn(tokens, h, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(tokens, hv, dim, device="cuda", dtype=torch.bfloat16)
    a, b = [
        torch.randn(tokens, hv, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    if extreme:
        a[:, ::2] = 80
        a[:, 1::2] = -80
        k.zero_()
    a_log = torch.randn(hv, device="cuda")
    dt = torch.randn(hv, device="cuda")
    pool = torch.randn(3, hv, dim, dim, device="cuda") * 0.05
    state = pool[1:2]
    canaries = pool[[0, 2]].clone()
    reference = state.clone()
    for _ in range(2):
        if tokens == 1:
            from vllm.third_party.flash_linear_attention.ops import (
                fused_recurrent_gated_delta_rule,
            )

            single_output, single_state = fused_recurrent_gated_delta_rule(
                q=q.unsqueeze(0),
                k=k.unsqueeze(0),
                v=v.unsqueeze(0),
                g=(
                    -a_log.exp() * torch.nn.functional.softplus(a.float() + dt)
                ).unsqueeze(0),
                beta=b.float().sigmoid().unsqueeze(0),
                initial_state=state.clone(),
                inplace_final_state=False,
                cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                use_qk_l2norm_in_kernel=True,
            )
        key = k.float().mean(0)
        key *= (key.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        key = key.repeat_interleave(hv // h, 0)
        decay = (
            -a_log.exp() * torch.nn.functional.softplus(a.float().mean(0) + dt)
        ).exp()
        reference *= decay[None, :, None, None]
        delta = v.float().mean(0) - (reference[0] * key[:, None, :]).sum(-1)
        delta *= b.float().mean(0).sigmoid()[:, None]
        reference += delta[None, :, :, None] * key[None, :, None, :]
        query = q.float()
        query *= (query.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * dim**-0.5
        query = query.repeat_interleave(hv // h, 1)
        expected = torch.einsum("hvk,thk->thv", reference[0], query)
        actual = mean_state_update(q, k, v, a, b, a_log, dt, state)
        torch.testing.assert_close(state, reference, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(pool[[0, 2]], canaries, rtol=0, atol=0)
        if tokens == 1:
            torch.testing.assert_close(state, single_state, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(actual, single_output[0], rtol=0.01, atol=0.002)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("extreme", [False, True])
@torch.inference_mode()
def test_replay_tail_preserves_causal_outputs_and_overwrites_only_final_state(
    tokens, extreme
):
    from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
        replay_tail_update,
    )

    torch.manual_seed(42)
    h, hv, dim = 16, 32, 128
    # Packed views exercise the strides used by the real post-conv Q/K/V path.
    packed = torch.randn(tokens, 2 * h * dim + hv * dim, device="cuda")
    q, k, v = packed.split([h * dim, h * dim, hv * dim], -1)
    q, k, v = q.view(tokens, h, dim), k.view(tokens, h, dim), v.view(tokens, hv, dim)
    a, b = [torch.randn(1, hv, device="cuda") for _ in range(2)]
    if extreme:
        a[:, ::2], a[:, 1::2] = 80, -80
        b[:, ::2], b[:, 1::2] = -80, 80
    a_log, dt = [torch.randn(hv, device="cuda") for _ in range(2)]
    pool = torch.randn(3, hv, dim, dim, device="cuda") * 0.05
    state = pool[1:2]
    canaries = pool[[0, 2]].clone()
    reference = state.clone()
    query = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * dim**-0.5
    key = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    query, key = [x.repeat_interleave(hv // h, 1) for x in (query, key)]
    decay = (-a_log.exp() * torch.nn.functional.softplus(a + dt)).exp()
    beta = b.sigmoid()
    for _ in range(4):
        outputs = []
        for t in range(tokens):
            reference *= decay[:, :, None, None]
            delta = beta[0, :, None] * (
                v[t] - (reference[0] * key[t, :, None, :]).sum(-1)
            )
            reference += delta[None, :, :, None] * key[t, None, :, None, :]
            outputs.append((reference[0] * query[t, :, None, :]).sum(-1))
        actual = replay_tail_update(q, k, v, a, b, a_log, dt, state)
        torch.testing.assert_close(actual, torch.stack(outputs), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(state, reference, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(pool[[0, 2]], canaries, rtol=0, atol=0)
    before = state.clone()
    expected = replay_tail_update(q, k, v, a, b, a_log, dt, state).clone()
    expected_state = state.clone()
    state.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = replay_tail_update(q, k, v, a, b, a_log, dt, state)
    state.copy_(before)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


class _TestGatedNorm:
    def __init__(self, weight: torch.Tensor, activation: str) -> None:
        self.weight = weight
        self.bias = None
        self.eps = EPS
        self.group_size = None
        self.norm_before_gate = True
        self.activation = activation

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(
            x,
            self.weight,
            None,
            z=z,
            eps=self.eps,
            norm_before_gate=True,
            activation=self.activation,
        )


def _make_vllm_config():
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.additional_config = {"gdn_prefill_backend": "cutedsl"}
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return config


def _build_layer(
    vllm_config,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    output_gate_activation: str,
):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=K,
        value_dim=HV * V,
        activation="silu",
        A_log=a_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=(conv_state, ssm_state),
        norm=_TestGatedNorm(norm_weight, output_gate_activation),
        layer_norm_epsilon=EPS,
        gdn_decode_kernel="cuda",
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "split_ba",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@pytest.mark.parametrize(
    "num_v_heads,expected",
    [
        pytest.param(2, True, id="ratio1"),
        pytest.param(4, True, id="ratio2"),
        pytest.param(6, True, id="ratio3"),
        pytest.param(8, True, id="ratio4"),
        pytest.param(16, True, id="ratio8"),
        pytest.param(5, False, id="non-integral-ratio"),
        pytest.param(10, False, id="unsupported-ratio5"),
    ],
)
def test_fused_mtp_head_ratio_guard(num_v_heads: int, expected: bool) -> None:
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")

    layer = types.SimpleNamespace(
        num_k_heads=2,
        num_v_heads=num_v_heads,
        kv_cache=(None, torch.empty(1, dtype=torch.float32, device="cuda")),
        gdn_decode_kernel="cuda",
    )
    attn_metadata = types.SimpleNamespace(
        spec_state_indices_tensor=torch.ones(
            1, SPEC_TOKENS, dtype=torch.int32, device="cuda"
        ),
        spec_sequence_masks=object(),
        num_decodes=0,
        num_spec_decodes=1,
    )

    assert (
        QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode(
            cast(QwenGatedDeltaNetAttention, layer),
            cast(GDNAttentionMetadata, attn_metadata),
        )
        is expected
    )


@torch.inference_mode()
def test_fused_forward_uses_packed_entrypoint() -> None:
    """Fused mode keeps projected QKVZ and BA packed through the model op."""
    device = torch.device("cuda")
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=device)
    mixed_qkvz = torch.randn(
        num_tokens, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device
    )
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_fused_gdn_decode=True,
        norm=types.SimpleNamespace(
            weight=torch.empty(V, dtype=torch.bfloat16, device=device)
        ),
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        in_proj_qkvz=lambda _: (mixed_qkvz, None),
        in_proj_ba=lambda _: (ba, None),
        out_proj=lambda x: (x, None),
    )
    layer.forward_cuda = types.MethodType(
        QwenGatedDeltaNetAttention.forward_cuda, layer
    )
    layer._forward_cuda_exact = types.MethodType(
        QwenGatedDeltaNetAttention._forward_cuda_exact, layer
    )

    def packed_op(
        actual_qkvz: torch.Tensor,
        actual_ba: torch.Tensor,
        output: torch.Tensor,
        *,
        layer_name: str,
    ) -> None:
        assert actual_qkvz is mixed_qkvz
        assert actual_ba is ba
        assert layer_name == _encode_layer_name(PREFIX)
        output.fill_(1)

    with (
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core_fused_norm_packed",
            side_effect=packed_op,
        ) as packed_mock,
        patch.object(torch.ops.vllm, "qwen_gdn_attention_core") as triton_mock,
    ):
        output = layer.forward_cuda(hidden_states)

    packed_mock.assert_called_once()
    triton_mock.assert_not_called()
    torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize(
    "seq_lens,query_lens,draft_tokens,expected_fused_calls",
    [
        pytest.param([128], [SPEC_TOKENS], [NUM_SPEC], 1, id="pure-mtp"),
        pytest.param(
            [128, 96],
            [SPEC_TOKENS, 64],
            [NUM_SPEC, -1],
            0,
            id="mixed-mtp-falls-back",
        ),
        pytest.param([96], [64], [-1], 0, id="pure-prefill"),
        pytest.param([128], [1], [-1], 0, id="pure-decode"),
    ],
)
@pytest.mark.parametrize("output_gate_activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_fused_model_path_matches_reference(
    seq_lens: list[int],
    query_lens: list[int],
    draft_tokens: list[int],
    expected_fused_calls: int,
    output_gate_activation: str,
) -> None:
    """Fused MTP and its mixed/prefill/decode fallbacks match the reference."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                batch.batch_size, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
        )

    state_indices = [
        indices
        for indices in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if indices is not None and indices.numel() > 0
    ]
    pool_size = max(int(indices.max().item()) for indices in state_indices) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state_seed = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = 0.01 * torch.randn(
        pool_size, *temporal_state_shape, dtype=torch.float32, device=device
    )
    a_log = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    conv_weight = 0.1 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    num_tokens = batch.compute_num_tokens()
    mixed_qkv = 0.1 * torch.randn(
        num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device
    )
    b = 0.1 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    a = 0.1 * torch.randn_like(b)
    output_gate = 0.1 * torch.randn(
        num_tokens, HV, V, dtype=torch.bfloat16, device=device
    )
    mixed_qkvz = torch.cat((mixed_qkv, output_gate.flatten(1)), dim=-1)
    ba = torch.cat((b, a), dim=-1)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    reference_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    reference_out = torch.zeros_like(output_gate)
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        reference_layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b,
            a=a,
            core_attn_out=reference_out,
        )
    reference_out = reference_layer.norm(reference_out, output_gate)

    fused_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    context.no_compile_layers = {PREFIX: fused_layer}
    fused_out = torch.zeros_like(output_gate)
    fused_op = qwen_gdn_linear_attn.ops.fused_gdn_decode_post_conv_mtp
    with (
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=context),
        patch.object(
            qwen_gdn_linear_attn.ops,
            "fused_gdn_decode_post_conv_mtp",
            wraps=fused_op,
        ) as fused_mock,
    ):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz.clone(),
            ba,
            fused_out,
            layer_name=_encode_layer_name(PREFIX),
        )

    assert fused_mock.call_count == expected_fused_calls
    torch.testing.assert_close(
        fused_layer.kv_cache[0], reference_layer.kv_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(fused_out, reference_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fused_layer.kv_cache[1],
        reference_layer.kv_cache[1],
        atol=3e-2,
        rtol=3e-2,
    )
