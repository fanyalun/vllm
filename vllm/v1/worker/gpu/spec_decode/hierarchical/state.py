# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    advance_replay_tail_conv,
    advance_replay_tail_convs,
    begin_replay_tail_states,
    replay_tail_update,
    windowed_replay_tail_update,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


class PreverifyState:
    """Private GDN candidates; attention-only models need no recurrent copies."""

    def __init__(
        self,
        model,
        width: int,
        device: torch.device,
        mode="none",
        update_policy="exact",
        tail_policy="carry",
        window_size=5,
        tau_alpha=0.95,
        tau_beta=0.36328125,
        optimization="none",
    ):
        self.width = width
        self.mode = mode
        self.update_policy = update_policy
        self.tail_policy = tail_policy
        self.windowed = update_policy == "windowed_three_level"
        self.window_size = window_size
        self.optimization = optimization
        self.request_id = None
        self.outer_epoch = 0
        self.valid = torch.zeros(1, dtype=torch.int32, device=device)
        self.initialized = False
        self.direction = 0
        self.tails = {}
        self.repair_inputs = {}
        self.repair_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.begin_descriptors: dict[
            int, tuple[tuple[tuple[int, ...], ...], torch.Tensor]
        ] = {}
        self.thresholds = None
        self.action_counts = None
        self.execution_optimized = update_policy != "exact"
        self.kernel_tuned = update_policy != "exact"
        self.batch_constants = {}
        if update_policy != "exact":
            if (
                update_policy not in ("three_level_p50", "windowed_three_level")
                or mode != "replay_tail"
            ):
                raise ValueError("Three-level GDN requires replay_tail")
            self.thresholds = torch.tensor(
                [tau_alpha, tau_beta] if self.windowed else [0.98, 0.36328125],
                device=device,
            )
            self.counts = torch.arange(width + 1, dtype=torch.int32, device=device)
            self.sequence_mask = torch.ones(1, dtype=torch.bool, device=device)
            for length in range(1, width + 1):
                self.batch_constants[length] = (
                    torch.arange(length, dtype=torch.int64, device=device),
                    torch.arange(length, dtype=torch.int32, device=device),
                    torch.tensor([0, length], dtype=torch.int32, device=device),
                )
        if tail_policy not in ("carry", "repair_on_reject") or (
            tail_policy == "repair_on_reject" and update_policy != "three_level_p50"
        ):
            raise ValueError("Invalid three-level tail policy")
        if mode not in ("none", "ssm_mean", "input_mean", "replay_tail"):
            raise ValueError(f"Unknown preverify GDN mode: {mode}")
        self.layers = {
            layer.prefix: layer
            for layer in model.modules()
            if isinstance(layer, QwenGatedDeltaNetAttention)
        }
        self.caches = {}
        for name, layer in self.layers.items():
            shapes = list(layer.get_state_shape())
            if mode != "none":
                conv_shape = list(shapes[0])
                axis = 1 if is_conv_state_dim_first() else 0
                conv_shape[axis] = layer.conv_kernel_size - 1
                if mode in ("ssm_mean", "replay_tail"):
                    conv_shape[axis] += width
                shapes[0] = tuple(conv_shape)
            self.caches[name] = tuple(
                torch.zeros(
                    (width + 1 if mode == "none" else 1, *shape),
                    dtype=dtype,
                    device=device,
                )
                for shape, dtype in zip(shapes, layer.get_state_dtype(), strict=True)
            )
            if self.thresholds is not None and not self.windowed:
                self.tails[name] = torch.empty_like(self.caches[name][1])
                if tail_policy == "repair_on_reject":
                    self.repair_inputs[name] = tuple(
                        torch.empty(
                            (width, *shape),
                            dtype=layer.get_state_dtype()[0],
                            device=device,
                        )
                        for shape in (
                            (layer.num_k_heads, layer.head_k_dim),
                            (layer.num_v_heads, layer.head_v_dim),
                            (layer.num_v_heads,),
                            (layer.num_v_heads,),
                        )
                    )
        self.state_indices = torch.arange(
            1, width + 1, dtype=torch.int32, device=device
        ).view(1, width)
        if mode != "none":
            self.state_indices.zero_()
        self.num_accepted = torch.ones(1, dtype=torch.int32, device=device)
        self.conv_pointers = None
        if self.thresholds is not None and self.caches:
            convs = [cache[0] for cache in self.caches.values()]
            if all(
                x.dtype == torch.bfloat16
                and x.shape == convs[0].shape
                and x.stride() == convs[0].stride()
                for x in convs
            ):
                self.conv_pointers = torch.tensor(
                    [x.data_ptr() for x in convs], dtype=torch.uint64, device=device
                )

    def _copy_conv(self, destination, source, bias):
        axis = 2 if is_conv_state_dim_first() else 1
        positions = torch.arange(destination.shape[axis], device=source.device)
        positions = (positions + bias).clamp(max=source.shape[axis] - 1)
        destination.copy_(source.index_select(axis, positions))

    def begin(self, model_state, input_batch, block_tables, kv_cache_config):
        if getattr(self, "windowed", False):
            self.invalidate()
            if len(input_batch.req_ids) != 1:
                raise ValueError("Windowed GDN requires B1")
        if not self.layers:
            return
        if getattr(self, "execution_optimized", False) is True:
            self._begin_batched(model_state, input_batch, block_tables, kv_cache_config)
            if self.windowed:
                self.request_id = input_batch.req_ids[0]
                self.outer_epoch += 1
                self.valid.fill_(1)
                self.initialized = True
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
                start = 1 if getattr(self, "mode", "none") == "none" else 0
                self._copy_conv(
                    private_conv[start : start + 1],
                    conv.index_select(0, conv_idx),
                    bias,
                )
                private_temporal[start : start + 1].copy_(
                    temporal.index_select(0, temporal_idx)
                )

    def _begin_batched(self, model_state, input_batch, block_tables, kv_cache_config):
        batched = getattr(self, "max_num_reqs", 1) > 1
        channel_axis, time_axis = (1, 2) if is_conv_state_dim_first() else (2, 1)
        rows: list[tuple] = []
        elements = 0
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if name not in self.layers:
                    continue
                conv, temporal = self.layers[name].kv_cache
                out_conv, out_temporal = self.caches[name]
                if (
                    conv.dtype != torch.bfloat16
                    or temporal.dtype != torch.float32
                    or temporal.shape[1:] != out_temporal.shape[1:]
                    or temporal.stride()[1:] != out_temporal.stride()[1:]
                    or not out_temporal.is_contiguous()
                    or block_tables[gid].dtype != torch.int32
                ):
                    raise ValueError(
                        "Batched private initialization requires BF16 Conv "
                        "and contiguous FP32 state"
                    )
                rows.append(
                    (
                        conv.data_ptr(),
                        temporal.data_ptr(),
                        out_conv.data_ptr(),
                        out_temporal.data_ptr(),
                        block_tables[gid].data_ptr(),
                        conv.stride(0),
                        conv.stride(channel_axis),
                        conv.stride(time_axis),
                        out_conv.stride(channel_axis),
                        out_conv.stride(time_axis),
                        conv.shape[time_axis],
                        out_conv.shape[time_axis],
                        out_conv.shape[channel_axis],
                        out_temporal[0].numel(),
                        temporal.stride(0),
                    )
                )
                if batched:
                    rows[-1] += (
                        block_tables[gid].stride(0),
                        out_conv.stride(0),
                        out_temporal.stride(0),
                    )
                elements = max(elements, out_conv[0].numel(), out_temporal[0].numel())
        signature = tuple(rows)
        if len(rows) != len(self.layers) or (
            not batched and input_batch.idx_mapping.numel() != 1
        ):
            raise ValueError("Private initialization requires all GDN layers and B1")
        cached = self.begin_descriptors.get(self.direction)
        if cached is None or cached[0] != signature:
            descriptors = torch.tensor(
                rows, dtype=torch.uint64, device=input_batch.idx_mapping.device
            )
            self.begin_descriptors[self.direction] = signature, descriptors
        else:
            descriptors = cached[1]
        begin_replay_tail_states(
            descriptors,
            elements,
            input_batch.idx_mapping,
            model_state.num_accepted_tokens_gpu,
            model_state._mamba_state_idx_gpu
            if model_state._align_mode
            else input_batch.idx_mapping,
            model_state._align_mode,
            batched=batched,
        )

    def advance(self, accepted_drafts: int):
        mode = getattr(self, "mode", "none")
        if getattr(self, "thresholds", None) is not None:
            if (
                self.tail_policy == "repair_on_reject"
                and accepted_drafts + 1 < self.window_width
            ):
                self._repair(accepted_drafts + 1)
            if not getattr(self, "windowed", False):
                for name, (conv, initial) in self.caches.items():
                    tail = self.tails[name]
                    self.caches[name] = (conv, tail)
                    self.tails[name] = initial
                self.direction = 1 - self.direction
            if self.execution_optimized and self.conv_pointers is not None:
                advance_replay_tail_convs(
                    self.conv_pointers,
                    next(iter(self.caches.values()))[0],
                    accepted_drafts,
                    is_conv_state_dim_first(),
                )
                return
        if mode == "input_mean":
            return
        for conv, temporal in self.caches.values():
            if mode == "replay_tail" and conv.is_cuda:
                advance_replay_tail_conv(
                    conv, accepted_drafts, is_conv_state_dim_first()
                )
                continue
            if mode in ("ssm_mean", "replay_tail"):
                self._copy_conv(conv, conv, accepted_drafts)
                continue
            self._copy_conv(conv[1:2], conv[1:2], accepted_drafts)
            temporal[1:2].copy_(
                temporal[accepted_drafts + 1 : accepted_drafts + 2].clone()
            )

    def _repair(self, consumed):
        if not self.execution_optimized:
            self._repair_prefix(consumed)
            return
        key = self.direction, consumed
        if key not in self.repair_graphs:
            for length in range(1, self.width):
                self._repair_prefix(length)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._repair_prefix(length)
                self.repair_graphs[self.direction, length] = graph
        self.repair_graphs[key].replay()

    def _repair_prefix(self, length):
        for name, (_, initial) in self.caches.items():
            k, v, a, b = [x[:length] for x in self.repair_inputs[name]]
            layer = self.layers[name]
            replay_tail_update(
                k,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                initial,
                tail=self.tails[name],
                thresholds=self.thresholds,
                out=v,
                readout=False,
                value_tile=8 if self.kernel_tuned and length < 3 else 16,
                num_warps=8 if self.kernel_tuned and length >= 3 else 4,
            )

    def invalidate(self):
        self.valid.zero_()
        self.initialized = False
        self.request_id = None

    def update(self, layer, q, k, v, a, b, query_start_loc=None):
        name = layer.prefix
        self.window_width = q.shape[0]
        if self.windowed:
            if not self.initialized or query_start_loc is None:
                raise RuntimeError("Windowed GDN requires initialized request state")
            return windowed_replay_tail_update(
                q,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                self.caches[name][1],
                query_start_loc=query_start_loc,
                valid=self.valid,
                thresholds=self.thresholds,
                window_size=self.window_size,
                optimization=self.optimization,
                action_counts=self.action_counts,
            )
        if self.tail_policy == "repair_on_reject":
            for target, source in zip(
                self.repair_inputs[name], (k, v, a, b), strict=True
            ):
                target[: q.shape[0]].copy_(source)
        return replay_tail_update(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            self.caches[name][1],
            tail=self.tails[name],
            thresholds=self.thresholds,
            value_tile=8 if self.kernel_tuned and q.shape[0] < 3 else 16,
            num_warps=8 if self.kernel_tuned and q.shape[0] >= 3 else 4,
            action_counts=self.action_counts,
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
        except BaseException:
            if getattr(self, "windowed", False):
                self.invalidate()
            raise
        finally:
            for name, cache in original.items():
                self.layers[name].kv_cache = cache
