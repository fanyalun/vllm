# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import gc
import os
import signal
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import destroy_model_parallel
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.platforms import current_platform
from vllm.platforms.interface import set_assigned_physical_gpu_ids
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.spec_decode.async_draft.cache import (
    BranchCache,
    CachedBranch,
    select_branches_within_budget,
    select_recovery_candidates,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.ipc import (
    ASYNC_DRAFT_FAN_OUT,
    AsyncDraftBatch,
    AsyncDraftResponse,
    make_ring_slots,
    response_error,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.weights import (
    materialize_standalone_eagle_weights,
)
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class DraftCapacityError(RuntimeError):
    pass


class DraftBlockPool:
    def __init__(self, runner: GPUModelRunner):
        self.runner = runner
        self.block_tables = runner.block_tables
        self.num_groups = len(runner.kv_cache_config.kv_cache_groups)
        num_blocks = runner.kv_cache_config.num_blocks
        if num_blocks <= 1:
            raise ValueError("Async draft KV cache must contain at least two blocks")
        self.free_blocks = [
            list(range(num_blocks - 1, 0, -1)) for _ in range(self.num_groups)
        ]
        self.block_refcounts = [[0] * num_blocks for _ in range(self.num_groups)]
        self.allocations: dict[str, list[list[int]]] = {}
        self.request_slots: dict[str, int] = {}
        self.request_epochs: dict[str, int] = {}
        self.free_request_slots = list(range(runner.max_num_reqs - 1, -1, -1))

    def _release(self, req_id: str) -> None:
        allocations = self.allocations.pop(req_id, None)
        if allocations is not None:
            for group, blocks in enumerate(allocations):
                for block in blocks:
                    refcounts = self.block_refcounts[group]
                    if refcounts[block] <= 0:
                        raise RuntimeError(
                            "Async draft KV block refcount underflow: "
                            f"group={group}, block={block}"
                        )
                    refcounts[block] -= 1
                    if refcounts[block] == 0:
                        self.free_blocks[group].append(block)
        req_slot = self.request_slots.pop(req_id, None)
        if req_slot is not None:
            self.free_request_slots.append(req_slot)
            for group in range(self.num_groups):
                self.block_tables.num_blocks.np[group, req_slot] = 0
        self.request_epochs.pop(req_id, None)

    def release(self, request_ids: list[str]) -> None:
        for req_id in request_ids:
            self._release(req_id)
        self.block_tables.num_blocks.copy_to_uva()

    def reset(self, request_ids: list[str]) -> None:
        self.release(request_ids)

    def clone(
        self,
        source_id: str,
        branch_ids: list[str],
        required_seq_lens: np.ndarray,
        mutation_start_positions: np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.clone_many(
            [source_id] * len(branch_ids),
            branch_ids,
            required_seq_lens,
            mutation_start_positions,
        )

    def clone_many(
        self,
        source_ids: list[str],
        branch_ids: list[str],
        required_seq_lens: np.ndarray,
        mutation_start_positions: np.ndarray | None = None,
    ) -> torch.Tensor:
        if self.num_groups != 1:
            raise NotImplementedError(
                "Async EAGLE3 branch cloning currently requires one KV cache group"
            )
        if len(source_ids) != len(branch_ids):
            raise ValueError("Async draft clone source and branch counts differ")
        if len(required_seq_lens) != len(branch_ids):
            raise ValueError("Async draft clone sequence lengths must match branches")
        if mutation_start_positions is not None and len(
            mutation_start_positions
        ) != len(branch_ids):
            raise ValueError("Async draft clone mutation positions must match branches")
        for source_id in source_ids:
            if source_id not in self.allocations:
                raise KeyError(f"Unknown async draft source request {source_id!r}")
        if any(branch_id in self.allocations for branch_id in branch_ids):
            raise ValueError("Async draft branch IDs must be unique and new")
        if len(branch_ids) > len(self.free_request_slots):
            raise DraftCapacityError(
                "Async draft request slot pool exhausted during clone: "
                f"need={len(branch_ids)}, "
                f"available={len(self.free_request_slots)}"
            )

        block_costs = [
            self.clone_new_block_count(
                source_id,
                int(seq_len),
                None
                if mutation_start_positions is None
                else int(mutation_start_positions[index]),
            )
            for index, (source_id, seq_len) in enumerate(
                zip(source_ids, required_seq_lens)
            )
        ]
        if sum(block_costs) > len(self.free_blocks[0]):
            raise DraftCapacityError(
                "Async draft KV block pool exhausted during clone: "
                f"need={sum(block_costs)}, "
                f"available={len(self.free_blocks[0])}"
            )

        branch_slots: list[int] = []
        copy_sources: list[int] = []
        copy_destinations: list[int] = []
        block_size = self.block_tables.block_sizes[0]
        for index, (source_id, branch_id, seq_len) in enumerate(
            zip(source_ids, branch_ids, required_seq_lens)
        ):
            source_blocks = self.allocations[source_id][0]
            needed = (int(seq_len) + block_size - 1) // block_size
            shared_count = self.clone_shared_prefix_blocks(
                source_id,
                int(seq_len),
                None
                if mutation_start_positions is None
                else int(mutation_start_positions[index]),
            )
            destination_blocks = list(source_blocks[:shared_count])
            for block in destination_blocks:
                self.block_refcounts[0][block] += 1

            copied_source_blocks = source_blocks[
                shared_count : min(len(source_blocks), needed)
            ]
            new_count = needed - shared_count
            new_blocks = [self.free_blocks[0].pop() for _ in range(new_count)]
            for block in new_blocks:
                if self.block_refcounts[0][block] != 0:
                    raise RuntimeError(
                        f"Async draft free KV block {block} has live references"
                    )
                self.block_refcounts[0][block] = 1
            destination_blocks.extend(new_blocks)
            copy_sources.extend(copied_source_blocks)
            copy_destinations.extend(new_blocks[: len(copied_source_blocks)])

            req_slot = self.free_request_slots.pop()
            branch_slots.append(req_slot)
            self.request_slots[branch_id] = req_slot
            self.request_epochs[branch_id] = -1
            self.allocations[branch_id] = [destination_blocks]
            self.block_tables.append_block_ids(
                req_slot, (destination_blocks,), overwrite=True
            )

        self.block_tables.apply_staged_writes()
        self._copy_kv_blocks(copy_sources, copy_destinations)
        return torch.tensor(
            branch_slots,
            dtype=torch.int32,
            device=runner_device(self.runner),
        )

    def clone_shared_prefix_blocks(
        self,
        source_id: str,
        seq_len: int,
        mutation_start_position: int | None = None,
    ) -> int:
        source_blocks = self.allocations[source_id][0]
        block_size = self.block_tables.block_sizes[0]
        needed = (seq_len + block_size - 1) // block_size
        if mutation_start_position is not None:
            if mutation_start_position < 0:
                raise ValueError("Async draft mutation position must be non-negative")
            return min(
                mutation_start_position // block_size,
                len(source_blocks),
                needed,
            )
        mutation_window = 2 * self.runner.num_speculative_steps + 1
        cow_tail_blocks = (mutation_window + block_size - 1) // block_size + 1
        return min(max(len(source_blocks) - cow_tail_blocks, 0), needed)

    def clone_new_block_count(
        self,
        source_id: str,
        seq_len: int,
        mutation_start_position: int | None = None,
    ) -> int:
        block_size = self.block_tables.block_sizes[0]
        needed = (seq_len + block_size - 1) // block_size
        return needed - self.clone_shared_prefix_blocks(
            source_id, seq_len, mutation_start_position
        )

    def _copy_kv_blocks(
        self,
        source_blocks: list[int],
        destination_blocks: list[int],
    ) -> None:
        if not source_blocks:
            return
        source = torch.tensor(
            source_blocks, dtype=torch.long, device=runner_device(self.runner)
        )
        destination = torch.tensor(
            destination_blocks[: len(source_blocks)],
            dtype=torch.long,
            device=runner_device(self.runner),
        )
        for kv_cache in self.runner.kv_caches:
            if not isinstance(kv_cache, torch.Tensor):
                raise TypeError("Async EAGLE3 requires tensor attention KV caches")
            block_dim = 1 if kv_cache.shape[0] == 2 else 0
            source_values = kv_cache.index_select(block_dim, source)
            kv_cache.index_copy_(block_dim, destination, source_values)

    def ensure(
        self,
        req_ids: list[str],
        request_epochs: list[int],
        required_seq_lens: np.ndarray,
    ) -> torch.Tensor:
        if not (len(req_ids) == len(request_epochs) == len(required_seq_lens)):
            raise ValueError(
                "Async draft request IDs, epochs, and sequence lengths must "
                "have equal length"
            )
        req_slots: list[int] = []
        block_sizes = self.block_tables.block_sizes
        changed = False

        for req_id, epoch, seq_len in zip(req_ids, request_epochs, required_seq_lens):
            if self.request_epochs.get(req_id) != epoch:
                self._release(req_id)

        new_request_count = sum(req_id not in self.request_slots for req_id in req_ids)
        if new_request_count > len(self.free_request_slots):
            raise DraftCapacityError(
                "Async draft request slot pool exhausted: "
                f"need={new_request_count}, "
                f"available={len(self.free_request_slots)}"
            )
        missing_blocks = [0] * self.num_groups
        for req_id, seq_len in zip(req_ids, required_seq_lens):
            allocations = self.allocations.get(req_id)
            for group, block_size in enumerate(block_sizes):
                allocated = len(allocations[group]) if allocations else 0
                needed = (int(seq_len) + block_size - 1) // block_size
                missing_blocks[group] += max(needed - allocated, 0)
        for group, missing in enumerate(missing_blocks):
            if missing > len(self.free_blocks[group]):
                raise DraftCapacityError(
                    "Async draft KV block pool exhausted: "
                    f"group={group}, need={missing}, "
                    f"available={len(self.free_blocks[group])}"
                )

        for req_id, epoch, seq_len in zip(req_ids, request_epochs, required_seq_lens):
            if req_id not in self.request_slots:
                self.request_slots[req_id] = self.free_request_slots.pop()
                self.request_epochs[req_id] = epoch
                self.allocations[req_id] = [[] for _ in range(self.num_groups)]

            req_slot = self.request_slots[req_id]
            req_slots.append(req_slot)
            allocations = self.allocations[req_id]
            new_block_ids: list[list[int]] = []
            for group, block_size in enumerate(block_sizes):
                needed = (int(seq_len) + block_size - 1) // block_size
                missing = needed - len(allocations[group])
                blocks = [self.free_blocks[group].pop() for _ in range(missing)]
                for block in blocks:
                    if self.block_refcounts[group][block] != 0:
                        raise RuntimeError(
                            "Async draft free KV block has live references: "
                            f"group={group}, block={block}"
                        )
                    self.block_refcounts[group][block] = 1
                allocations[group].extend(blocks)
                new_block_ids.append(blocks)
            if any(new_block_ids):
                self.block_tables.append_block_ids(
                    req_slot,
                    tuple(new_block_ids),
                    overwrite=False,
                )
                changed = True

        if changed:
            self.block_tables.apply_staged_writes()
        return torch.tensor(
            req_slots,
            dtype=torch.int32,
            device=runner_device(self.runner),
        )


def runner_device(runner: GPUModelRunner) -> torch.device:
    return runner.device


def _branch_cudagraph_capture_sizes(
    num_speculative_tokens: int,
    max_num_reqs: int,
) -> set[int]:
    branches_per_request = (num_speculative_tokens + 1) * ASYNC_DRAFT_FAN_OUT
    return {
        branches_per_request * (1 << power)
        for power in range(max_num_reqs.bit_length())
        if 1 << power <= max_num_reqs
    }


def _standalone_load_draft(
    runner: GPUModelRunner,
    target_model_path: str,
) -> list[dict[str, object]]:
    speculator = runner.speculator
    if not isinstance(speculator, EagleSpeculator):
        raise TypeError(
            "Async draft child expected EagleSpeculator, got "
            f"{type(speculator).__name__}"
        )

    draft_config = runner.vllm_config.speculative_config
    assert draft_config is not None
    with set_current_vllm_config(runner.vllm_config):
        speculator.model = get_model(
            vllm_config=runner.vllm_config,
            model_config=draft_config.draft_model_config,
        )
    materialized = materialize_standalone_eagle_weights(
        speculator.model, target_model_path
    )
    speculator._validate_local_argmax_reduction()
    speculator.recorded_greedy_logits = torch.empty(
        runner.max_num_reqs,
        runner.num_speculative_steps,
        speculator.vocab_size,
        dtype=runner.vllm_config.model_config.dtype,
        device=runner.device,
    )
    speculator.recorded_feedback_hidden_states = torch.empty(
        runner.max_num_reqs,
        runner.num_speculative_steps,
        speculator.hidden_size,
        dtype=runner.vllm_config.model_config.dtype,
        device=runner.device,
    )

    from vllm.config import get_layers_from_vllm_config

    speculator.draft_attn_layer_names = set(
        get_layers_from_vllm_config(
            runner.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        ).keys()
    )
    runner.model_state = init_model_state(
        runner.vllm_config,
        speculator.model,
        None,
        runner.device,
    )
    runner.decode_query_len = runner.num_speculative_steps + 1
    return [weight.to_dict() for weight in materialized]


def _initialize_draft_kv_cache(runner: GPUModelRunner) -> None:
    free_memory, total_memory = torch.cuda.mem_get_info(runner.device)
    utilization = runner.cache_config.gpu_memory_utilization
    allocator_headroom = int(total_memory * (1.0 - utilization))
    available_memory = free_memory - allocator_headroom
    if available_memory <= 0:
        raise RuntimeError(
            "No memory available for standalone draft KV cache: "
            f"free={free_memory}, total={total_memory}, "
            f"gpu_memory_utilization={utilization}"
        )
    kv_cache_spec = runner.get_kv_cache_spec()
    kv_cache_config = get_kv_cache_configs(
        runner.vllm_config, [kv_cache_spec], [available_memory]
    )[0]
    runner.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    runner.initialize_kv_cache(kv_cache_config)


def _capture_standalone_draft(runner: GPUModelRunner) -> dict[str, object]:
    speculator = runner.speculator
    if not isinstance(speculator, DraftModelSpeculator):
        raise TypeError(
            "Async draft child expected DraftModelSpeculator, got "
            f"{type(speculator).__name__}"
        )
    prefill_manager = getattr(speculator, "prefill_cudagraph_manager", None)
    decode_manager = getattr(speculator, "decode_cudagraph_manager", None)
    if prefill_manager is None or not prefill_manager.needs_capture():
        return {
            "mode": CUDAGraphMode.NONE.name,
            "prefill_graphs": 0,
            "decode_graphs": 0,
        }

    attn_states = {}
    capture_descs = {
        desc for descs in prefill_manager._capture_descs.values() for desc in descs
    }
    for desc in capture_descs:
        num_reqs = desc.num_reqs or min(desc.num_tokens, prefill_manager.max_num_reqs)
        warmup = prepare_inputs_to_capture(
            num_reqs,
            desc.num_tokens,
            runner.model_state,
            speculator.input_buffers,
            runner.block_tables,
            runner.attn_groups,
            runner.kv_cache_config,
            skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
        )
        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
            captured = warmup
        else:
            captured = prepare_inputs_to_capture(
                num_reqs,
                desc.num_tokens,
                runner.model_state,
                speculator.input_buffers,
                runner.block_tables,
                runner.attn_groups,
                runner.kv_cache_config,
            )
        attn_states[desc] = AttentionStatePair(warmup, captured)

    speculator.capture(attn_states)
    torch.cuda.synchronize(runner.device)
    return {
        "mode": prefill_manager.cudagraph_mode.name,
        "prefill_graphs": len(prefill_manager.graphs),
        "decode_graphs": len(decode_manager.graphs) if decode_manager else 0,
    }


def _make_input_batch(
    batch: AsyncDraftBatch,
    ring_slot: Any,
    idx_mapping: torch.Tensor,
) -> InputBatch:
    num_reqs = batch.num_reqs
    num_tokens = batch.num_tokens
    num_tokens_after_padding = batch.num_tokens_after_padding
    query_start_loc = ring_slot.query_start_loc[: batch.num_reqs_after_padding + 1]
    seq_lens = ring_slot.seq_lens[: batch.num_reqs_after_padding]
    idx_mapping_np = idx_mapping.cpu().numpy().astype(np.int32, copy=False)
    logits_indices = query_start_loc[1 : num_reqs + 1] - 1
    cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
    cu_num_logits = torch.arange(
        num_reqs + 1, dtype=torch.int32, device=idx_mapping.device
    )
    return InputBatch(
        req_ids=batch.req_ids,
        num_reqs=num_reqs,
        num_reqs_after_padding=batch.num_reqs_after_padding,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=idx_mapping,
        expanded_local_pos=torch.zeros(
            num_reqs, dtype=torch.int32, device=idx_mapping.device
        ),
        num_scheduled_tokens=batch.num_scheduled_tokens,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_after_padding,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=query_start_loc,
        query_start_loc_np=batch.query_start_loc_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=torch.from_numpy(batch.seq_lens_cpu_upper_bound),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=batch.num_computed_tokens_np,
        prefill_len_np=batch.prefill_len_np,
        num_computed_prefill_tokens_np=batch.num_computed_prefill_tokens_np,
        is_prefilling_np=batch.is_prefilling_np,
        max_seq_len_np=None,
        input_ids=ring_slot.input_ids[:num_tokens_after_padding],
        positions=ring_slot.positions[:num_tokens_after_padding],
        is_padding=torch.zeros(
            num_tokens_after_padding,
            dtype=torch.bool,
            device=idx_mapping.device,
        ),
        logits_indices=logits_indices,
        cu_num_logits=cu_num_logits,
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def _slice_proposal_batch(
    batch: AsyncDraftBatch,
    ring_slot: Any,
    indices: list[int],
) -> tuple[AsyncDraftBatch, Any]:
    device = ring_slot.input_ids.device
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    token_slices = [
        slice(batch.query_start_loc_np[index], batch.query_start_loc_np[index + 1])
        for index in indices
    ]
    query_lengths = np.array(
        [token_slice.stop - token_slice.start for token_slice in token_slices],
        dtype=np.int32,
    )
    query_start_loc_np = np.zeros(len(indices) + 1, dtype=np.int32)
    np.cumsum(query_lengths, out=query_start_loc_np[1:])
    num_tokens = int(query_start_loc_np[-1])

    def pack_tokens(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor[token_slice] for token_slice in token_slices])

    sliced_slot = SimpleNamespace(
        input_ids=pack_tokens(ring_slot.input_ids),
        positions=pack_tokens(ring_slot.positions),
        query_start_loc=torch.tensor(
            query_start_loc_np, dtype=torch.int32, device=device
        ),
        seq_lens=ring_slot.seq_lens.index_select(0, index_tensor),
        aux_hidden_states=pack_tokens(ring_slot.aux_hidden_states),
        num_sampled=ring_slot.num_sampled.index_select(0, index_tensor),
        num_rejected=ring_slot.num_rejected.index_select(0, index_tensor),
        last_sampled=ring_slot.last_sampled.index_select(0, index_tensor),
        next_prefill_tokens=ring_slot.next_prefill_tokens.index_select(0, index_tensor),
        temperature=ring_slot.temperature.index_select(0, index_tensor),
        seeds=ring_slot.seeds.index_select(0, index_tensor),
    )
    sliced_batch = AsyncDraftBatch(
        generation=batch.generation,
        slot=batch.slot,
        engine_instance_id=batch.engine_instance_id,
        req_ids=[batch.req_ids[index] for index in indices],
        request_epochs=[batch.request_epochs[index] for index in indices],
        transient=batch.transient,
        overlap_budget_seconds=batch.overlap_budget_seconds,
        num_reqs=len(indices),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_reqs_after_padding=len(indices),
        num_scheduled_tokens=batch.num_scheduled_tokens[indices].copy(),
        query_start_loc_np=query_start_loc_np,
        seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound[indices].copy(),
        num_computed_tokens_np=batch.num_computed_tokens_np[indices].copy(),
        prefill_len_np=batch.prefill_len_np[indices].copy(),
        num_computed_prefill_tokens_np=(
            batch.num_computed_prefill_tokens_np[indices].copy()
        ),
        is_prefilling_np=batch.is_prefilling_np[indices].copy(),
    )
    return sliced_batch, sliced_slot


def _execute_jit_proposal(
    runner: GPUModelRunner,
    block_pool: DraftBlockPool,
    branch_cache: BranchCache,
    batch: AsyncDraftBatch,
    ring_slot: Any,
    aux_hidden_splits: tuple[int, ...],
    num_speculative_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    proposal_steps = num_speculative_steps or runner.num_speculative_steps
    required_seq_lens = (
        batch.seq_lens_cpu_upper_bound[: batch.num_reqs] + proposal_steps
    )
    evictions = 0
    try:
        idx_mapping = block_pool.ensure(
            batch.req_ids,
            batch.request_epochs,
            required_seq_lens,
        )
    except DraftCapacityError:
        discarded = branch_cache.discard_all()
        evictions = len(discarded)
        block_pool.release(discarded)
        idx_mapping = block_pool.ensure(
            batch.req_ids,
            batch.request_epochs,
            required_seq_lens,
        )
    input_batch = _make_input_batch(batch, ring_slot, idx_mapping)

    block_tables = runner.block_tables.gather_block_tables(
        idx_mapping, batch.num_reqs_after_padding
    )
    slot_mappings = runner.block_tables.compute_slot_mappings(
        idx_mapping,
        input_batch.query_start_loc,
        input_batch.positions,
        input_batch.num_tokens_after_padding,
    )
    attn_metadata = runner.model_state.prepare_attn(
        input_batch,
        runner.cudagraph_manager.cudagraph_mode
        if runner.cudagraph_manager is not None
        else runner.compilation_config.cudagraph_mode,
        block_tables,
        slot_mappings,
        runner.attn_groups,
        runner.kv_cache_config,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, runner.kv_cache_config
    )

    max_num_reqs = runner.max_num_reqs
    last_sampled = torch.zeros(max_num_reqs, 1, dtype=torch.int64, device=runner.device)
    next_prefill_tokens = torch.zeros(
        max_num_reqs, dtype=torch.int32, device=runner.device
    )
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=runner.device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=runner.device)
    last_sampled[idx_mapping, 0] = ring_slot.last_sampled[: batch.num_reqs]
    next_prefill_tokens[idx_mapping] = ring_slot.next_prefill_tokens[: batch.num_reqs]
    temperature[idx_mapping] = ring_slot.temperature[: batch.num_reqs]
    seeds[idx_mapping] = ring_slot.seeds[: batch.num_reqs]

    aux_hidden_states = list(
        torch.split(
            ring_slot.aux_hidden_states[: batch.num_tokens_after_padding],
            aux_hidden_splits,
            dim=-1,
        )
    )
    assert isinstance(runner.speculator, DraftModelSpeculator)
    speculator = runner.speculator
    previous_num_steps = speculator.num_speculative_steps
    speculator.num_speculative_steps = proposal_steps
    try:
        draft_tokens = speculator.propose(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings_by_layer,
            last_hidden_states=aux_hidden_states[-1],
            aux_hidden_states=aux_hidden_states,
            num_sampled=ring_slot.num_sampled[: batch.num_reqs],
            num_rejected=ring_slot.num_rejected[: batch.num_reqs],
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
        )
    finally:
        speculator.num_speculative_steps = previous_num_steps
    feedback_hidden_states = speculator.recorded_feedback_hidden_states[
        : batch.num_reqs
    ].clone()
    return draft_tokens.clone(), feedback_hidden_states, evictions


def _run_proposal(
    runner: GPUModelRunner,
    block_pool: DraftBlockPool,
    branch_cache: BranchCache,
    ring_slots: list[Any],
    batch: AsyncDraftBatch,
    aux_hidden_splits: tuple[int, ...],
) -> tuple[
    dict[str, float | int],
    torch.Tensor,
    list[int],
    list[dict[str, Any] | None] | None,
]:
    start = time.perf_counter()
    ring_slot = ring_slots[batch.slot]
    num_reqs = batch.num_reqs
    speculator = runner.speculator
    assert isinstance(speculator, DraftModelSpeculator)
    feedback_hidden_states = torch.empty(
        num_reqs,
        runner.num_speculative_steps,
        speculator.hidden_size,
        dtype=runner.vllm_config.model_config.dtype,
        device=runner.device,
    )
    hits = 0
    hit_indices: list[int] = []
    cache_evictions = 0
    miss_indices: list[int] = []
    trace_top2: list[dict[str, Any] | None] | None = None
    force_jit = os.environ.get("ASYNC_DRAFT_FORCE_JIT", "0") == "1"
    if batch.transient or force_jit:
        miss_indices = list(range(num_reqs))
    else:
        accepted_counts = (ring_slot.num_sampled[:num_reqs] - 1).cpu().tolist()
        recovery_tokens = ring_slot.last_sampled[:num_reqs].cpu().tolist()
        for index, (req_id, epoch, accepted, recovery) in enumerate(
            zip(
                batch.req_ids,
                batch.request_epochs,
                accepted_counts,
                recovery_tokens,
            )
        ):
            key = (
                batch.engine_instance_id,
                req_id,
                epoch,
                accepted,
                recovery,
            )
            branch = branch_cache.pop(key)
            if branch is None:
                miss_indices.append(index)
                discarded = branch_cache.discard_request(req_id)
                block_pool.release(discarded)
                continue
            ring_slot.draft_tokens[index].copy_(branch.tokens)
            feedback_hidden_states[index].copy_(branch.feedback_hidden_states)
            discarded = branch_cache.discard_request(req_id)
            block_pool.release([branch.branch_id, *discarded])
            hits += 1
            hit_indices.append(index)

    if miss_indices:
        if len(miss_indices) == num_reqs:
            miss_batch, miss_slot = batch, ring_slot
        else:
            miss_batch, miss_slot = _slice_proposal_batch(
                batch, ring_slot, miss_indices
            )
        miss_tokens, miss_hidden_states, jit_evictions = _execute_jit_proposal(
            runner,
            block_pool,
            branch_cache,
            miss_batch,
            miss_slot,
            aux_hidden_splits,
        )
        if os.environ.get("REPLAYSSM_SPEC_DECODE_TRACE_LOGITS") == "1":
            top_values, top_ids = (
                speculator.recorded_greedy_logits[: len(miss_indices)]
                .float()
                .topk(2, dim=-1)
            )
            values = top_values.cpu().tolist()
            ids = top_ids.cpu().tolist()
            trace_top2 = [None] * num_reqs
            for miss_row, output_row in enumerate(miss_indices):
                trace_top2[output_row] = {
                    "draft_top2": [
                        {
                            "token_ids": step_ids,
                            "logits": step_values,
                            "gap": step_values[0] - step_values[1],
                        }
                        for step_ids, step_values in zip(
                            ids[miss_row], values[miss_row]
                        )
                    ]
                }
        cache_evictions += jit_evictions
        for miss_row, output_row in enumerate(miss_indices):
            ring_slot.draft_tokens[output_row].copy_(miss_tokens[miss_row])
            feedback_hidden_states[output_row].copy_(miss_hidden_states[miss_row])

    if batch.transient:
        block_pool.release(batch.req_ids)
    elapsed = time.perf_counter() - start
    misses = num_reqs - hits
    metrics = {
        "cache_hits": hits,
        "cache_misses": misses,
        "jit_fallbacks": misses,
        "cache_evictions": cache_evictions,
        "wait_seconds": elapsed,
        "branch_build_seconds": 0.0,
        "fan_out": ASYNC_DRAFT_FAN_OUT,
    }
    return metrics, feedback_hidden_states, hit_indices, trace_top2


def _run_glue_decode(
    runner: GPUModelRunner,
    block_pool: DraftBlockPool,
    ring_slot: Any,
    batch: AsyncDraftBatch,
    feedback_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    speculator = runner.speculator
    assert isinstance(speculator, EagleSpeculator)
    batch_size = batch.num_reqs
    num_steps = runner.num_speculative_steps
    query_len = num_steps + 1
    device = runner.device

    target_query_end = ring_slot.query_start_loc[1 : batch_size + 1]
    recovery_indices = (target_query_end - ring_slot.num_rejected[:batch_size] - 1).to(
        torch.long
    )
    recovery_positions = ring_slot.positions.index_select(0, recovery_indices)
    combined_target_hidden_states = speculator.model.combine_hidden_states(
        ring_slot.aux_hidden_states[: batch.num_tokens_after_padding]
    )
    recovery_hidden_states = combined_target_hidden_states.index_select(
        0, recovery_indices
    )

    glue_input_ids = torch.cat(
        (
            ring_slot.last_sampled[:batch_size, None],
            ring_slot.draft_tokens[:batch_size],
        ),
        dim=1,
    )
    glue_hidden_states = torch.cat(
        (recovery_hidden_states[:, None], feedback_hidden_states), dim=1
    )
    position_offsets = torch.arange(query_len, device=device)
    glue_positions = recovery_positions[:, None] + position_offsets
    total_tokens = batch_size * query_len
    query_start_loc = torch.arange(
        0,
        total_tokens + 1,
        query_len,
        dtype=torch.int32,
        device=device,
    )
    seq_lens = recovery_positions + query_len
    glue_ids = [
        f"__async_glue_{batch.generation}_{request_index}"
        for request_index in range(batch_size)
    ]
    idx_mapping = block_pool.clone_many(
        batch.req_ids,
        glue_ids,
        seq_lens.cpu().numpy().astype(np.int32, copy=False),
        recovery_positions.cpu().numpy().astype(np.int32, copy=False),
    )

    speculator.input_buffers.input_ids[:total_tokens].copy_(glue_input_ids.reshape(-1))
    speculator.input_buffers.positions[:total_tokens].copy_(glue_positions.reshape(-1))
    speculator.input_buffers.query_start_loc[: batch_size + 1].copy_(query_start_loc)
    speculator.input_buffers.seq_lens[:batch_size].copy_(seq_lens)
    speculator.hidden_states[:total_tokens].copy_(
        glue_hidden_states.reshape(total_tokens, -1)
    )
    speculator.draft_max_seq_len = min(
        int(seq_lens.max().item()), speculator.max_model_len
    )

    runner.block_tables.gather_block_tables(idx_mapping, batch_size)
    slot_mappings = runner.block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        speculator.input_buffers.positions[:total_tokens],
        total_tokens,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, runner.kv_cache_config
    )
    attn_metadata = speculator._build_draft_attn_metadata(
        num_reqs=batch_size,
        num_reqs_padded=batch_size,
        num_tokens_padded=total_tokens,
        num_query_per_req=query_len,
        causal=True,
    )
    last_hidden_states, output_hidden_states = speculator._run_model(
        total_tokens,
        attn_metadata,
        slot_mappings_by_layer,
        num_tokens_across_dp=None,
    )
    logits = speculator.model.compute_logits(last_hidden_states).view(
        batch_size, query_len, -1
    )
    output_hidden_states = output_hidden_states.view(batch_size, query_len, -1)
    candidates = select_recovery_candidates(logits, ring_slot.draft_tokens[:batch_size])
    return candidates, output_hidden_states, recovery_positions, glue_ids


def _decode_fanout_branches(
    runner: GPUModelRunner,
    branch_slots: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    speculator = runner.speculator
    assert isinstance(speculator, EagleSpeculator)
    num_branches = input_ids.shape[0]
    num_steps = runner.num_speculative_steps
    query_start_loc = torch.arange(
        num_branches + 1, dtype=torch.int32, device=runner.device
    )
    seq_lens = positions + 1
    decode_manager = speculator.decode_cudagraph_manager
    decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
        decode_manager,
        num_branches,
        num_branches,
        uniform_token_count=1,
        dp_size=speculator.dp_size,
        dp_rank=speculator.dp_rank,
    )

    speculator.input_buffers.input_ids[:num_branches].copy_(input_ids)
    speculator.input_buffers.positions[:num_branches].copy_(positions)
    speculator.input_buffers.query_start_loc[: num_branches + 1].copy_(query_start_loc)
    speculator.input_buffers.seq_lens[:num_branches].copy_(seq_lens)
    speculator.hidden_states[:num_branches].copy_(hidden_states)
    speculator.idx_mapping[:num_branches].copy_(branch_slots)

    for step in range(num_steps):
        positions = speculator.input_buffers.positions[:num_branches]
        speculator.draft_max_seq_len = min(
            int(speculator.input_buffers.seq_lens[:num_branches].max().item()),
            speculator.max_model_len,
        )

        runner.block_tables.gather_block_tables(branch_slots, num_branches)
        slot_mappings = runner.block_tables.compute_slot_mappings(
            branch_slots,
            query_start_loc,
            positions,
            decode_batch_desc.num_tokens,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, runner.kv_cache_config
        )
        attn_metadata = speculator._build_draft_attn_metadata(
            num_reqs=num_branches,
            num_reqs_padded=decode_batch_desc.num_reqs or num_branches,
            num_tokens_padded=decode_batch_desc.num_tokens,
        )
        speculator.current_draft_step.fill_(step)
        if decode_batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert decode_manager is not None
            decode_manager.run_fullgraph(decode_batch_desc)
        else:
            speculator._generate_draft(
                num_branches,
                decode_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings_by_layer,
                num_tokens_across_dp,
                decode_batch_desc.cg_mode,
            )
    return (
        speculator.draft_tokens[:num_branches].clone(),
        speculator.recorded_feedback_hidden_states[:num_branches].clone(),
    )


def _build_fanout_branches(
    runner: GPUModelRunner,
    block_pool: DraftBlockPool,
    branch_cache: BranchCache,
    ring_slot: Any,
    batch: AsyncDraftBatch,
    feedback_hidden_states: torch.Tensor,
    hit_indices: list[int],
    aux_hidden_splits: tuple[int, ...],
) -> tuple[float, int]:
    if batch.transient or os.environ.get("ASYNC_DRAFT_FORCE_JIT", "0") == "1":
        return 0.0, 0
    start = time.perf_counter()
    cache_evictions = 0
    if hit_indices:
        hit_batch, hit_slot = _slice_proposal_batch(batch, ring_slot, hit_indices)
        with torch.cuda.nvtx.range("async_draft:canonical_commit"):
            _, _, commit_evictions = _execute_jit_proposal(
                runner,
                block_pool,
                branch_cache,
                hit_batch,
                hit_slot,
                aux_hidden_splits,
                num_speculative_steps=1,
            )
        cache_evictions += commit_evictions
    try:
        with torch.cuda.nvtx.range("async_draft:glue_decode"):
            (
                candidates,
                glue_hidden_states,
                recovery_positions,
                glue_ids,
            ) = _run_glue_decode(
                runner,
                block_pool,
                ring_slot,
                batch,
                feedback_hidden_states,
            )
    except DraftCapacityError:
        skipped = (
            batch.num_reqs * (runner.num_speculative_steps + 1) * ASYNC_DRAFT_FAN_OUT
        )
        return time.perf_counter() - start, skipped
    batch_size, num_positions, fan_out = candidates.shape
    if fan_out != ASYNC_DRAFT_FAN_OUT:
        raise RuntimeError(
            f"Unexpected async draft fan-out {fan_out}, expected {ASYNC_DRAFT_FAN_OUT}"
        )

    branch_ids: list[str] = []
    branch_keys: list[tuple[str, str, int, int, int]] = []
    branch_input_ids: list[torch.Tensor] = []
    branch_positions: list[torch.Tensor] = []
    branch_hidden_states: list[torch.Tensor] = []
    branch_seq_lens: list[int] = []
    branch_request_indices: list[int] = []
    branch_accepted_counts: list[int] = []
    branch_candidate_indices: list[int] = []
    branch_shared_prefix_blocks: list[int] = []
    block_size = block_pool.block_tables.block_sizes[0]
    for request_index in range(batch_size):
        for accepted_count in range(num_positions):
            for candidate_index in range(fan_out):
                branch_id = (
                    f"__async_branch_{batch.generation}_{request_index}_"
                    f"{accepted_count}_{candidate_index}"
                )
                candidate = candidates[request_index, accepted_count, candidate_index]
                position = recovery_positions[request_index] + accepted_count + 1
                branch_ids.append(branch_id)
                branch_seq_lens.append(
                    int(position.item()) + runner.num_speculative_steps
                )
                branch_keys.append(
                    (
                        batch.engine_instance_id,
                        batch.req_ids[request_index],
                        batch.request_epochs[request_index],
                        accepted_count,
                        int(candidate.item()),
                    )
                )
                branch_input_ids.append(candidate)
                branch_positions.append(position)
                branch_hidden_states.append(
                    glue_hidden_states[request_index, accepted_count]
                )
                branch_request_indices.append(request_index)
                branch_accepted_counts.append(accepted_count)
                branch_candidate_indices.append(candidate_index)
                branch_shared_prefix_blocks.append(
                    block_pool.clone_shared_prefix_blocks(
                        glue_ids[request_index],
                        branch_seq_lens[-1],
                        int(position.item()),
                    )
                )

    selected = select_branches_within_budget(
        branch_seq_lens,
        branch_request_indices,
        branch_accepted_counts,
        branch_candidate_indices,
        block_size=block_size,
        available_slots=len(block_pool.free_request_slots),
        available_blocks=len(block_pool.free_blocks[0]),
        shared_prefix_blocks=branch_shared_prefix_blocks,
    )
    cache_evictions += len(branch_ids) - len(selected)
    branch_slots: list[torch.Tensor] = []
    for request_index in range(batch_size):
        request_selected = [
            index
            for index in selected
            if branch_request_indices[index] == request_index
        ]
        if not request_selected:
            continue
        selected_branch_ids = [branch_ids[index] for index in request_selected]
        selected_seq_lens = [branch_seq_lens[index] for index in request_selected]
        slots = block_pool.clone(
            glue_ids[request_index],
            selected_branch_ids,
            np.asarray(selected_seq_lens, dtype=np.int32),
            np.asarray(
                [int(branch_positions[index].item()) for index in request_selected],
                dtype=np.int32,
            ),
        )
        branch_slots.append(slots)

    block_pool.release(glue_ids)

    if not branch_slots:
        torch.cuda.synchronize(runner.device)
        return time.perf_counter() - start, cache_evictions
    slots_tensor = torch.cat(branch_slots)
    with torch.cuda.nvtx.range("async_draft:tree_decode"):
        tokens, hidden_states = _decode_fanout_branches(
            runner,
            slots_tensor,
            torch.stack([branch_input_ids[index] for index in selected]),
            torch.stack([branch_positions[index] for index in selected]),
            torch.stack([branch_hidden_states[index] for index in selected]),
        )
    for output_index, branch_index in enumerate(selected):
        branch_cache.add(
            branch_keys[branch_index],
            CachedBranch(
                branch_id=branch_ids[branch_index],
                tokens=tokens[output_index].clone(),
                feedback_hidden_states=hidden_states[output_index].clone(),
            ),
        )
    torch.cuda.synchronize(runner.device)
    return time.perf_counter() - start, cache_evictions


def _wait_for_shutdown_after_error(connection: Connection) -> None:
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        if message.get("command") == "shutdown":
            connection.send({"status": "shutdown"})
            return


def run_async_draft_child(
    connection: Connection,
    vllm_config: VllmConfig,
    physical_device_id: int,
    init_port: int | None = None,
) -> None:
    runner: GPUModelRunner | None = None
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        set_assigned_physical_gpu_ids([physical_device_id])
        visible_device_id = current_platform.logical_device_id_to_visible_device_id(0)
        device = torch.device(f"cuda:{visible_device_id}")
        torch.cuda.set_device(device)

        child_config = copy.deepcopy(vllm_config)
        assert child_config.speculative_config is not None
        target_max_num_reqs = child_config.scheduler_config.max_num_seqs
        target_max_num_tokens = child_config.scheduler_config.max_num_batched_tokens
        num_speculative_tokens = child_config.speculative_config.num_speculative_tokens
        branch_capacity = (
            target_max_num_reqs * (num_speculative_tokens + 1) * ASYNC_DRAFT_FAN_OUT
        )
        child_config.scheduler_config.max_num_seqs = (
            2 * target_max_num_reqs + branch_capacity
        )
        child_config.scheduler_config.max_num_batched_tokens = max(
            target_max_num_tokens,
            branch_capacity,
        )
        if child_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            branch_capture_sizes = _branch_cudagraph_capture_sizes(
                num_speculative_tokens,
                target_max_num_reqs,
            )
            capture_sizes = set(
                child_config.compilation_config.cudagraph_capture_sizes or ()
            )
            capture_sizes.update(branch_capture_sizes)
            child_config.compilation_config.cudagraph_capture_sizes = sorted(
                capture_sizes
            )
            child_config.compilation_config.max_cudagraph_capture_size = max(
                capture_sizes
            )
        child_config.speculative_config.async_draft_device = None
        child_config.scheduler_config.async_scheduling = False
        # Target-only block-count overrides are useful for forcing scheduler
        # preemption. The standalone Draft owns a separate block pool and must
        # profile its actual free memory instead of inheriting that override.
        child_config.cache_config.num_gpu_blocks_override = None
        if child_config.model_config.enforce_eager:
            child_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            child_config.compilation_config.cudagraph_capture_sizes = []
        child_config.parallel_config.assigned_physical_gpu_ids = [physical_device_id]
        child_config.parallel_config.distributed_executor_backend = "uni"
        child_config.instance_id = f"{vllm_config.instance_id}_async_draft"

        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

        port = init_port or get_open_port()
        init_method = f"tcp://127.0.0.1:{port}"
        with set_current_vllm_config(child_config):
            init_worker_distributed_environment(
                child_config,
                rank=0,
                distributed_init_method=init_method,
                local_rank=0,
                backend="nccl",
            )
            runner = GPUModelRunner(child_config, device)
            materialized_weights = _standalone_load_draft(
                runner, child_config.model_config.model
            )
            _initialize_draft_kv_cache(runner)
            draft_cudagraph = _capture_standalone_draft(runner)

        draft_hf_config = child_config.speculative_config.draft_model_config.hf_config
        layer_ids = getattr(
            draft_hf_config,
            "eagle_aux_hidden_state_layer_ids",
            None,
        )
        if layer_ids is None:
            eagle_config = getattr(draft_hf_config, "eagle_config", {}) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        num_aux_hidden_states = len(layer_ids) if layer_ids else 3
        target_hidden_size = child_config.model_config.get_hidden_size()
        aux_hidden_splits = (target_hidden_size,) * num_aux_hidden_states
        ring_slots = make_ring_slots(
            max_num_reqs=target_max_num_reqs,
            max_num_tokens=target_max_num_tokens,
            num_speculative_tokens=runner.num_speculative_steps,
            aux_hidden_size=sum(aux_hidden_splits),
            dtype=child_config.model_config.dtype,
            device=device,
        )
        response_events = [
            torch.cuda.Event(interprocess=True) for _ in range(len(ring_slots))
        ]
        response_event_handles = [event.ipc_handle() for event in response_events]
        torch.cuda.synchronize(device)
        connection.send(
            {
                "status": "ready",
                "ring_slots": ring_slots,
                "response_event_handles": response_event_handles,
                "aux_hidden_splits": aux_hidden_splits,
                "kv_num_blocks": runner.kv_cache_config.num_blocks,
                "materialized_weights": materialized_weights,
                "cudagraph": draft_cudagraph,
                "pid": os.getpid(),
                "physical_device_id": physical_device_id,
            }
        )
        block_pool = DraftBlockPool(runner)
        branch_cache = BranchCache()
        pending_branch_build_seconds = 0.0
        pending_cache_evictions = 0
        last_generation = [-1] * len(ring_slots)
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            command = message.get("command")
            if command == "shutdown":
                connection.send({"status": "shutdown"})
                break
            if command == "release":
                request_ids = message["request_ids"]
                block_pool.release(branch_cache.discard_requests(request_ids))
                block_pool.release(request_ids)
                connection.send(
                    {
                        "status": "ok",
                        "metrics": {
                            "branch_build_seconds": (pending_branch_build_seconds),
                            "cache_evictions": pending_cache_evictions,
                        },
                    }
                )
                pending_branch_build_seconds = 0.0
                pending_cache_evictions = 0
                continue
            if command == "reset":
                request_ids = message["request_ids"]
                block_pool.release(branch_cache.discard_requests(request_ids))
                block_pool.reset(request_ids)
                connection.send(
                    {
                        "status": "ok",
                        "metrics": {
                            "branch_build_seconds": (pending_branch_build_seconds),
                            "cache_evictions": pending_cache_evictions,
                        },
                    }
                )
                pending_branch_build_seconds = 0.0
                pending_cache_evictions = 0
                continue
            if command != "propose":
                raise ValueError(f"Unknown async draft command: {command!r}")

            batch = AsyncDraftBatch(**message["batch"])
            if batch.slot < 0 or batch.slot >= len(ring_slots):
                raise ValueError(f"Invalid ring slot {batch.slot}")
            if batch.generation <= last_generation[batch.slot]:
                raise ValueError(
                    "Stale async draft generation: "
                    f"slot={batch.slot}, generation={batch.generation}, "
                    f"last={last_generation[batch.slot]}"
                )
            last_generation[batch.slot] = batch.generation
            try:
                (
                    metrics,
                    feedback_hidden_states,
                    hit_indices,
                    trace_top2,
                ) = _run_proposal(
                    runner,
                    block_pool,
                    branch_cache,
                    ring_slots,
                    batch,
                    aux_hidden_splits,
                )
                metrics["branch_build_seconds"] = pending_branch_build_seconds
                metrics["cache_evictions"] += pending_cache_evictions
                pending_branch_build_seconds = 0.0
                pending_cache_evictions = 0
                response_events[batch.slot].record(torch.cuda.current_stream(device))
                connection.send(
                    asdict(
                        AsyncDraftResponse(
                            generation=batch.generation,
                            slot=batch.slot,
                            status="ok",
                            num_reqs=batch.num_reqs,
                            metrics=metrics,
                            cache_hit_indices=hit_indices,
                            trace_top2=trace_top2,
                        )
                    )
                )
                (
                    pending_branch_build_seconds,
                    pending_cache_evictions,
                ) = _build_fanout_branches(
                    runner,
                    block_pool,
                    branch_cache,
                    ring_slots[batch.slot],
                    batch,
                    feedback_hidden_states,
                    hit_indices,
                    aux_hidden_splits,
                )
            except BaseException as error:
                response = response_error(batch.generation, batch.slot, error)
                response["traceback"] = traceback.format_exc()
                connection.send(response)
                _wait_for_shutdown_after_error(connection)
                return
    except BaseException as error:
        try:
            connection.send(
                {
                    "status": "fatal",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            _wait_for_shutdown_after_error(connection)
        except BaseException:
            pass
    finally:
        if runner is not None:
            try:
                runner.shutdown()
            except BaseException:
                logger.exception("Failed to shut down async draft runner")
        destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
