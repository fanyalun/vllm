# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preverify-only grouped GDN execution over the shared Target parameters."""

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.grouped_input import (
    fused_add_norm,
    grouped_conv,
    grouped_gated_norm,
    grouped_linear,
    grouped_norm,
    grouped_recurrent,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    qwen_gdn_mean_projected,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


def selected_groups(layer_types):
    groups, current = [], []
    for i, kind in enumerate(layer_types):
        if kind == "linear_attention":
            current.append(i)
        elif current:
            groups.append(tuple(current))
            current = []
    if current:
        groups.append(tuple(current))
    if len(groups) < 9 or any(len(group) != 3 for group in groups[2:9]):
        raise ValueError("Grouped GDN requires three-layer GDN groups 3 through 9")
    return tuple(groups[2:9])


class GroupedGDNPreverify:
    """Stateless execution plan; all intermediate tensors belong to one forward."""

    def __init__(self, model, vllm_config):
        parallel = vllm_config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.use_sequence_parallel_moe
            or vllm_config.quant_config is not None
            or vllm_config.lora_config is not None
        ):
            raise ValueError("Grouped GDN requires unquantized TP1/PP1 without LoRA/SP")
        self.model = model.model
        if (
            self.model.start_layer != 0
            or self.model.end_layer != len(self.model.layers)
            or self.model.use_sequence_parallel
            or any(decoder.layer_scale for decoder in self.model.layers)
        ):
            raise ValueError("Grouped GDN requires a complete unscaled Qwen decoder")
        self.groups = selected_groups(self.model.config.layer_types)
        self.by_start = {group[0]: group for group in self.groups}
        for group in self.groups:
            for index in group:
                decoder = self.model.layers[index]
                layer = decoder.linear_attn
                if (
                    decoder.layer_scale
                    or layer.gqa_interleaved_layout
                    or layer.conv_kernel_size != 4
                    or layer.head_k_dim != 128
                    or layer.head_v_dim != 128
                    or layer.activation not in ("silu", "swish")
                    or layer.norm.activation not in ("silu", "swish")
                    or not layer.norm.norm_before_gate
                    or layer.conv1d.bias is not None
                ):
                    raise ValueError("Unsupported grouped Qwen GDN layout")
                for weight in (
                    layer.in_proj_qkvz.weight,
                    layer.in_proj_ba.weight,
                    layer.out_proj.weight,
                ):
                    if weight.dtype != torch.bfloat16 or not weight.is_cuda:
                        raise ValueError("Grouped GDN requires CUDA BF16 weights")

    def projections(self, group, anchor, serial=False):
        decoders = [self.model.layers[i] for i in group]
        layers = [d.linear_attn for d in decoders]
        if serial:
            normalized = [
                d.input_layernorm(anchor).to(torch.bfloat16) for d in decoders
            ]
            qkvz = torch.stack(
                [
                    layer.in_proj_qkvz(x)[0]
                    for layer, x in zip(layers, normalized, strict=True)
                ]
            )
            ba = torch.stack(
                [
                    layer.in_proj_ba(x)[0]
                    for layer, x in zip(layers, normalized, strict=True)
                ]
            )
            return qkvz, ba
        normalized = grouped_norm(
            anchor,
            [d.input_layernorm.weight for d in decoders],
            decoders[0].input_layernorm.variance_epsilon,
            output_dtype=torch.bfloat16,
        )
        qkvz = grouped_linear(
            normalized, [layer.in_proj_qkvz.weight for layer in layers]
        )
        ba = grouped_linear(normalized, [layer.in_proj_ba.weight for layer in layers])
        return qkvz, ba

    @staticmethod
    def normalize(module, hidden, residual):
        return fused_add_norm(hidden, residual, module.weight, module.variance_epsilon)

    def branch(self, layer, qkvz, ba, state_mode):
        if state_mode == "none":
            return layer.forward_cuda_projected(qkvz, ba)
        metadata = get_forward_context().attn_metadata[layer.prefix]
        return qwen_gdn_mean_projected(layer, qkvz, ba, metadata, state_mode)

    def branches(self, group, qkvz, ba, state_mode):
        layers = [self.model.layers[i].linear_attn for i in group]
        first = layers[0]
        convs, states = zip(*(layer.kv_cache for layer in layers), strict=True)
        tail = state_mode == "replay_tail"
        slot = 0 if tail else 1
        qkv = grouped_conv(
            qkvz,
            convs,
            [layer.conv1d.weight for layer in layers],
            first.conv_dim,
            slot,
            is_conv_state_dim_first(),
        )
        core = grouped_recurrent(
            qkv,
            ba,
            [layer.A_log for layer in layers],
            [layer.dt_bias for layer in layers],
            states,
            first.num_k_heads,
            first.num_v_heads,
            first.head_k_dim,
            first.head_v_dim,
            tail,
        )
        normalized = grouped_gated_norm(
            core,
            qkvz,
            [layer.norm.weight for layer in layers],
            first.num_v_heads,
            first.head_v_dim,
            first.norm.eps,
        )
        return grouped_linear(normalized, [layer.out_proj.weight for layer in layers])

    def __call__(self, input_ids, positions, mode, state_mode):
        if mode not in ("projection", "full", "serial"):
            raise ValueError(f"Unsupported grouped mode: {mode}")
        if state_mode not in ("none", "replay_tail") or not 1 <= input_ids.numel() <= 5:
            raise ValueError("Grouped GDN requires none/replay_tail and 1..5 tokens")
        model = self.model
        hidden = model.embed_input_ids(input_ids)
        residual = None
        aux = model._maybe_add_hidden_state([], 0, hidden, residual)
        index = 0
        while index < len(model.layers):
            group = self.by_start.get(index)
            if group is None:
                decoder = model.layers[index]
                hidden, residual = self.normalize(
                    decoder.input_layernorm, hidden, residual
                )
                if decoder.layer_type == "linear_attention":
                    hidden = decoder.linear_attn(hidden)
                else:
                    hidden = decoder.self_attn(
                        positions=positions, hidden_states=hidden
                    )
                hidden, residual = self.normalize(
                    decoder.post_attention_layernorm, hidden, residual
                )
                hidden = decoder.mlp(hidden)
                model._maybe_add_hidden_state(aux, index + 1, hidden, residual)
                index += 1
                continue
            # Match fused_add_rms_norm: normalize the FP32 sum, round the residual.
            anchor = (
                hidden.float()
                if residual is None
                else hidden.float() + residual.float()
            )
            qkvz, ba = self.projections(group, anchor, serial=mode == "serial")
            outputs = (
                self.branches(group, qkvz, ba, state_mode) if mode == "full" else None
            )
            for offset, layer_index in enumerate(group):
                decoder = model.layers[layer_index]
                residual = anchor.to(hidden.dtype) if offset == 0 else hidden + residual
                output = (
                    outputs[offset]
                    if outputs is not None
                    else self.branch(
                        decoder.linear_attn, qkvz[offset], ba[offset], state_mode
                    )
                )
                hidden, residual = self.normalize(
                    decoder.post_attention_layernorm, output, residual
                )
                hidden = decoder.mlp(hidden)
                model._maybe_add_hidden_state(aux, layer_index + 1, hidden, residual)
            index += 3
        hidden, _ = self.normalize(model.norm, hidden, residual)
        return (hidden, aux) if aux else hidden
