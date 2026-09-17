# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned GDN slots across compacted hierarchical inner batches."""

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    advance_replay_tail_convs,
    windowed_replay_tail_update,
)
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class BatchedPreverifyState(PreverifyState):
    def __init__(self, *args, max_num_reqs, **kwargs):
        super().__init__(*args, **kwargs)
        if self.layers and not (self.windowed or self.mode == "none"):
            raise ValueError("Batched Qwen requires native or windowed GDN")
        self.max_num_reqs = max_num_reqs
        self.slots_per_req = self.width + 1 if self.mode == "none" else 1
        self.caches = {
            name: tuple(x.repeat((max_num_reqs,) + (1,) * (x.ndim - 1)) for x in cache)
            for name, cache in self.caches.items()
        }
        device = self.valid.device
        self.valid = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.state_indices = torch.empty(
            (max_num_reqs, self.width), dtype=torch.int32, device=device
        )
        self.num_accepted = torch.ones(max_num_reqs, dtype=torch.int32, device=device)
        self.sequence_mask = torch.ones(max_num_reqs, dtype=torch.bool, device=device)
        self.request_slots = {}
        self.active_slots = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self.active_count = 0
        self.conv_pointers = torch.tensor(
            [x[0].data_ptr() for x in self.caches.values()],
            dtype=torch.uint64,
            device=device,
        )

    def invalidate(self):
        super().invalidate()
        self.request_slots = {}
        self.active_count = 0

    def begin(self, model_state, input_batch, block_tables, kv_cache_config):
        self.invalidate()
        n = input_batch.num_reqs
        if n > self.max_num_reqs or len(set(input_batch.req_ids)) != n:
            raise ValueError("Private GDN requests must have unique bounded slots")
        if not self.layers:
            return
        if self.windowed:
            self._begin_batched(model_state, input_batch, block_tables, kv_cache_config)
            self.request_slots = dict(zip(input_batch.req_ids, range(n), strict=True))
            self.outer_epoch += 1
            self.valid[:n].fill_(1)
            self.initialized = True
            return
        req_idx = input_batch.idx_mapping
        bias = (model_state.num_accepted_tokens_gpu[req_idx] - 1).long()
        source = (
            model_state._mamba_state_idx_gpu[req_idx].long()
            if model_state._align_mode
            else torch.zeros_like(req_idx)
        )
        rows = torch.arange(n, device=self.valid.device)
        destination = rows * self.slots_per_req + (self.mode == "none")
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            table = block_tables[gid]
            for name in group.layer_names:
                if name not in self.layers:
                    continue
                conv, ssm = self.layers[name].kv_cache
                private_conv, private_ssm = self.caches[name]
                source_conv = conv[table[rows, source].long()]
                shifted = torch.empty_like(private_conv[:n])
                self._copy_conv(shifted, source_conv, bias[:, None])
                private_conv[destination] = shifted
                private_ssm[destination] = ssm[table[rows, source + bias].long()]
        self.request_slots = dict(zip(input_batch.req_ids, range(n), strict=True))
        self.outer_epoch += 1
        self.valid[:n].fill_(1)
        self.initialized = True

    def _copy_conv(self, destination, source, bias):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        axis = 2 if is_conv_state_dim_first() else 1
        positions = torch.arange(destination.shape[axis], device=source.device)
        positions = (positions + bias).clamp(max=source.shape[axis] - 1).long()
        if axis == 2:
            positions = positions[:, None, :].expand(-1, source.shape[1], -1)
        else:
            positions = positions[:, :, None].expand(-1, -1, source.shape[2])
        destination.copy_(source.gather(axis, positions))

    def select(self, req_ids):
        if len(set(req_ids)) != len(req_ids):
            raise ValueError("Active GDN requests must have unique private slots")
        if not self.initialized or any(x not in self.request_slots for x in req_ids):
            raise RuntimeError("Batched GDN request slot is invalid")
        n = len(req_ids)
        slots = torch.tensor(
            [self.request_slots[x] for x in req_ids],
            device=self.valid.device,
            dtype=torch.int64,
        )
        self.active_slots[:n].copy_(slots)
        if self.mode == "none":
            indices = slots[:, None] * self.slots_per_req + torch.arange(
                1, self.width + 1, device=slots.device
            )
        else:
            indices = slots[:, None].expand(n, self.width)
        self.state_indices[:n].copy_(indices)
        self.active_count = n

    def advance(self, accepted_drafts):
        slots = self.active_slots[: self.active_count]
        if self.windowed:
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            advance_replay_tail_convs(
                self.conv_pointers,
                next(iter(self.caches.values()))[0],
                accepted_drafts,
                is_conv_state_dim_first(),
                slots=slots,
            )
            return
        indices = slots * self.slots_per_req + (self.mode == "none")
        for conv, ssm in self.caches.values():
            source = conv[indices]
            shifted = torch.empty_like(source)
            self._copy_conv(shifted, source, accepted_drafts[:, None])
            conv[indices] = shifted
            if self.mode == "none":
                ssm[indices] = ssm[indices + accepted_drafts]

    def update(self, layer, q, k, v, a, b, query_start_loc=None):
        if not self.initialized or not self.windowed:
            raise RuntimeError("Batched approximate GDN requires initialized windows")
        return windowed_replay_tail_update(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            self.caches[layer.prefix][1],
            query_start_loc=query_start_loc,
            valid=self.valid,
            thresholds=self.thresholds,
            window_size=self.window_size,
            optimization=self.optimization,
            action_counts=self.action_counts,
            state_slots=self.active_slots[: self.active_count],
        )
