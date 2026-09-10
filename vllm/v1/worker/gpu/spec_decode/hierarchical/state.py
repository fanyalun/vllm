# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


class PreverifyState:
    """Private GDN candidates; attention-only models need no recurrent copies."""

    def __init__(self, model, width: int, device: torch.device):
        self.width = width
        self.layers = {
            layer.prefix: layer
            for layer in model.modules()
            if isinstance(layer, QwenGatedDeltaNetAttention)
        }
        self.caches = {}
        for name, layer in self.layers.items():
            self.caches[name] = tuple(
                torch.zeros((width + 1, *shape), dtype=dtype, device=device)
                for shape, dtype in zip(
                    layer.get_state_shape(), layer.get_state_dtype(), strict=True
                )
            )
        self.state_indices = torch.arange(
            1, width + 1, dtype=torch.int32, device=device
        ).view(1, width)
        self.num_accepted = torch.ones(1, dtype=torch.int32, device=device)

    def _copy_conv(self, destination, source, bias):
        axis = 2 if is_conv_state_dim_first() else 1
        positions = torch.arange(source.shape[axis], device=source.device)
        positions = (positions + bias).clamp(max=source.shape[axis] - 1)
        destination.copy_(source.index_select(axis, positions))

    def begin(self, model_state, input_batch, block_tables, kv_cache_config):
        if not self.layers:
            return
        req_idx = input_batch.idx_mapping
        bias = (model_state.num_accepted_tokens_gpu[req_idx] - 1).to(torch.int64)
        source = (
            model_state._mamba_state_idx_gpu[req_idx].to(torch.int64)
            if model_state._align_mode
            else torch.zeros_like(req_idx)
        )
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if name not in self.layers:
                    continue
                conv, temporal = self.layers[name].kv_cache
                private_conv, private_temporal = self.caches[name]
                table = block_tables[gid][0]
                conv_idx = table[source].to(torch.int64)
                temporal_idx = table[source + bias].to(torch.int64)
                self._copy_conv(private_conv[1:2], conv.index_select(0, conv_idx), bias)
                private_temporal[1:2].copy_(temporal.index_select(0, temporal_idx))

    def advance(self, accepted_drafts: int):
        for conv, temporal in self.caches.values():
            self._copy_conv(conv[1:2], conv[1:2], accepted_drafts)
            temporal[1:2].copy_(
                temporal[accepted_drafts + 1 : accepted_drafts + 2].clone()
            )

    def snapshot(self):
        return {
            name: tuple(x.clone() for x in cache) for name, cache in self.caches.items()
        }

    def restore(self, snapshot):
        for name, cache in self.caches.items():
            for destination, source in zip(cache, snapshot[name], strict=True):
                destination.copy_(source)

    @contextmanager
    def activate(self):
        original = {}
        try:
            for name, layer in self.layers.items():
                original[name] = layer.kv_cache
                layer.kv_cache = self.caches[name]
            yield
        finally:
            for name, cache in original.items():
                self.layers[name].kv_cache = cache
