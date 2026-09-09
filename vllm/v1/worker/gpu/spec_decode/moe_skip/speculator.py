# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_decode_inputs,
    update_draft_inputs,
)
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _is_internal_request(req_id: str) -> bool:
    return req_id.startswith(("_warmup_", "_profile_"))


def _external_request_id(req_id: str) -> str:
    external_id, separator, suffix = req_id.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(char in "0123456789abcdef" for char in suffix)
    ):
        return external_id
    return req_id


class MoeSkipSpeculator(BaseSpeculator):
    """Same-model drafter that evaluates fewer routed experts per token."""

    supports_mm_inputs = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        assert speculative_config.method == "moe_skip"
        assert speculative_config.moe_skip_top_h is not None
        self.top_h = speculative_config.moe_skip_top_h
        self.num_speculative_steps = speculative_config.num_speculative_tokens

        scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.hidden_size = vllm_config.model_config.get_hidden_size()

        self.input_buffers = InputBuffers(
            self.max_num_reqs, self.max_num_tokens, device
        )
        self.position_dims = 3 if vllm_config.model_config.uses_mrope else 1
        self.mrope_positions = torch.zeros(
            self.position_dims,
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.initial_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.sample_src_positions = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.current_draft_step = torch.zeros((), dtype=torch.int64, device=device)
        self.draft_tokens = torch.full(
            (self.max_num_reqs, self.num_speculative_steps),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.draft_logits: torch.Tensor | None = None
        self.hidden_states = torch.zeros(
            self.max_num_reqs,
            self.hidden_size,
            dtype=vllm_config.model_config.dtype,
            device=device,
        )
        self.query_start_loc_np = np.arange(self.max_num_reqs + 1, dtype=np.int32)
        self.ones_np = np.ones(self.max_num_reqs, dtype=np.int32)
        self.zeros_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.false_np = np.zeros(self.max_num_reqs, dtype=np.bool_)
        self.logits_indices = torch.arange(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.cu_num_logits = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.cu_num_logits_np = np.arange(self.max_num_reqs + 1, dtype=np.int32)
        self.seq_lens_cpu_upper_bound = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device="cpu"
        )
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.scratch_state_indices: dict[int, torch.Tensor] = {}
        self.trace_dir = (
            Path(envs.VLLM_MOE_SKIP_TRACE_DIR) if envs.VLLM_MOE_SKIP_TRACE_DIR else None
        )
        self.trace_path: Path | None = None
        self.pending_draft_top8: dict[str, list[list[int]]] = {}
        self.pending_draft_top2_logits: dict[str, list[list[float]]] = {}
        self.pending_draft_argmax_tokens: dict[str, list[int]] = {}
        self.verify_steps: dict[str, int] = {}
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.trace_dir / "raw_trace.jsonl"
            self.draft_top8_tokens = torch.full(
                (self.max_num_reqs, self.num_speculative_steps, 8),
                -1,
                dtype=torch.int64,
                device=device,
            )
            self.draft_top2_logits = torch.empty(
                (self.max_num_reqs, self.num_speculative_steps, 2),
                dtype=torch.float32,
                device=device,
            )

    def load_model(self, target_model: nn.Module) -> None:
        architecture = type(target_model).__name__
        if architecture == "Qwen3_5MoeForConditionalGeneration":
            self.model = target_model.language_model
            self.model_family = "qwen3_6"
        elif architecture == "Qwen3_5MoeForCausalLM":
            self.model = target_model
            self.model_family = "qwen3_6"
        elif architecture == "Gemma4ForConditionalGeneration":
            self.model = target_model.language_model
            self.model_family = "gemma4"
        elif architecture == "Gemma4ForCausalLM":
            self.model = target_model
            self.model_family = "gemma4"
        else:
            raise ValueError(
                f"MoE-Skip only supports Qwen3.6 MoE and Gemma4 MoE; got {architecture}"
            )
        self.logits_model = target_model

        moe_layers = [
            module for module in target_model.modules() if isinstance(module, MoERunner)
        ]
        if not moe_layers:
            raise ValueError("MoE-Skip target has no modular FusedMoE layers")
        expected_router_type = (
            FusedTopKRouter if self.model_family == "qwen3_6" else CustomRoutingRouter
        )
        for layer in moe_layers:
            if layer.is_monolithic or not isinstance(
                layer.router, expected_router_type
            ):
                raise ValueError(
                    "MoE-Skip requires modular FusedMoE layers with "
                    f"{expected_router_type.__name__} for {self.model_family}"
                )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        if self.model_family == "qwen3_6" and not isinstance(
            model_state, MambaHybridModelState
        ):
            raise ValueError("Qwen3.6 MoE-Skip requires the hybrid model state")
        if self.model_family == "gemma4" and not isinstance(
            model_state, DefaultModelState
        ):
            raise ValueError("Gemma4 MoE-Skip requires the default attention state")
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = target_attn_groups
        if isinstance(model_state, MambaHybridModelState):
            self.scratch_state_indices = model_state.moe_skip_state_index_buffers(
                kv_cache_config
            )
        else:
            self.scratch_state_indices = {}

    def _apply_draft_state_indices(
        self,
        attn_metadata: dict[str, Any],
        num_reqs: int,
    ) -> None:
        if isinstance(self.model_state, MambaHybridModelState):
            self.model_state.apply_moe_skip_state_indices(
                attn_metadata,
                self.attn_groups,
                self.scratch_state_indices,
                num_reqs,
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(self) -> None:
        manager = self.decode_cudagraph_manager
        assert manager is not None
        if not manager.needs_capture():
            return
        logger.info("Capturing CUDA graphs for the MoE-Skip drafter...")
        self.idx_mapping.zero_()
        self.current_draft_step.zero_()
        for indices in self.scratch_state_indices.values():
            indices.zero_()
        if manager.use_breakable_cg:
            manager.init_breakable_cg_runner(self.model)
        manager.capture(
            self._generate_draft,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing MoE-Skip CUDA graphs",
        )

    def _make_draft_batch(
        self,
        input_batch: InputBatch,
        batch_desc: BatchExecutionDescriptor,
        step: int,
    ) -> InputBatch:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens
        self.seq_lens_cpu_upper_bound.zero_()
        torch.add(
            input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            step + 1,
            out=self.seq_lens_cpu_upper_bound[:num_reqs],
        )
        self.seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        self.query_start_loc_np[: num_reqs + 1] = np.arange(
            num_reqs + 1, dtype=np.int32
        )
        self.query_start_loc_np[num_reqs + 1 :] = num_reqs
        return replace(
            input_batch,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=self.idx_mapping[:num_reqs],
            idx_mapping_np=input_batch.idx_mapping_np,
            expanded_idx_mapping=self.idx_mapping[:num_reqs],
            expanded_local_pos=self.input_buffers.positions[:num_reqs].to(torch.int32),
            num_scheduled_tokens=self.ones_np[:num_reqs],
            num_tokens=num_reqs,
            num_tokens_after_padding=num_tokens_padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_np=self.query_start_loc_np[: num_reqs_padded + 1],
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            seq_lens_cpu_upper_bound=self.seq_lens_cpu_upper_bound[:num_reqs_padded],
            dcp_local_seq_lens=None,
            num_computed_tokens_np=self.zeros_np[:num_reqs],
            prefill_len_np=self.zeros_np[:num_reqs],
            num_computed_prefill_tokens_np=self.zeros_np[:num_reqs],
            is_prefilling_np=self.false_np[:num_reqs],
            has_prefill=False,
            input_ids=self.input_buffers.input_ids[:num_tokens_padded],
            positions=self.input_buffers.positions[:num_tokens_padded],
            is_padding=self.input_buffers.is_padding[:num_tokens_padded],
            logits_indices=self.logits_indices[:num_reqs],
            cu_num_logits=self.cu_num_logits[: num_reqs + 1],
            cu_num_logits_np=self.cu_num_logits_np[: num_reqs + 1],
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=1,
        )

    def _build_attn(
        self,
        input_batch: InputBatch,
        batch_desc: BatchExecutionDescriptor,
        block_tables: tuple[torch.Tensor, ...],
        step: int,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        draft_batch = self._make_draft_batch(input_batch, batch_desc, step)
        slot_mappings = self.block_tables.compute_slot_mappings(
            self.idx_mapping[: input_batch.num_reqs],
            draft_batch.query_start_loc,
            draft_batch.positions,
            batch_desc.num_tokens,
        )
        attn_metadata = self.model_state.prepare_attn(
            draft_batch,
            batch_desc.cg_mode,
            block_tables,
            slot_mappings,
            self.attn_groups,
            self.kv_cache_config,
        )
        self._apply_draft_state_indices(
            attn_metadata, batch_desc.num_reqs or input_batch.num_reqs
        )
        return (
            attn_metadata,
            build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config),
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(
            self.input_buffers.input_ids[:num_tokens]
        )
        positions = self.input_buffers.positions[:num_tokens]
        if self.position_dims > 1:
            self.mrope_positions[:, :num_tokens].copy_(positions.unsqueeze(0))
            positions = self.mrope_positions[:, :num_tokens]
        model_inputs = {
            "input_ids": None,
            "positions": positions,
            "inputs_embeds": inputs_embeds,
        }
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            is_padding=self.input_buffers.is_padding[:num_tokens],
            additional_forward_kwargs={"routing_top_k": self.top_h},
        ):
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                assert self.decode_cudagraph_manager is not None
                hidden_states = self.decode_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                hidden_states = self.model(**model_inputs)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._apply_draft_state_indices(attn_metadata, num_reqs)
        hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        logits = self.logits_model.compute_logits(hidden_states[:num_reqs])
        assert logits is not None
        if self.trace_dir is not None:
            top8_logits, top8_tokens = torch.topk(logits, k=8, dim=-1)
            self.draft_top8_tokens[:num_reqs].index_copy_(
                1,
                self.current_draft_step.view(1),
                top8_tokens.unsqueeze(1),
            )
            self.draft_top2_logits[:num_reqs].index_copy_(
                1,
                self.current_draft_step.view(1),
                top8_logits[:, :2].float().unsqueeze(1),
            )
        draft_tokens = torch.argmax(logits, dim=-1)
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states[:num_reqs],
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            self.sample_src_positions,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states
        del next_prefill_tokens, temperature, seeds, mm_inputs
        if skip_attn_for_dummy_run:
            return self.draft_tokens[: input_batch.num_reqs]

        num_reqs = input_batch.num_reqs
        self.idx_mapping[:num_reqs].copy_(input_batch.idx_mapping)
        self.idx_mapping[num_reqs:].zero_()
        self.initial_tokens[:num_reqs].copy_(last_sampled[input_batch.idx_mapping, 0])
        self.input_buffers.positions[:num_reqs].copy_(input_batch.seq_lens)
        self.input_buffers.positions[:num_reqs].sub_(num_rejected)
        self.input_buffers.positions[:num_reqs].sub_(1)
        self.sample_src_positions[:num_reqs].copy_(
            self.input_buffers.positions[:num_reqs]
        )
        prepare_decode_inputs(
            self.initial_tokens[:num_reqs],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.sample_src_positions,
            self.max_model_len,
            self.max_num_reqs,
        )
        self.input_buffers.is_padding[:num_reqs].fill_(False)
        self.input_buffers.is_padding[num_reqs:].fill_(True)
        self.draft_tokens[:num_reqs].fill_(-1)

        manager = self.decode_cudagraph_manager
        assert manager is not None
        batch_desc, _ = dispatch_cg_and_sync_dp(
            manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=1,
            dp_rank=0,
            need_eager=is_profile,
            dp_sync=dp_sync,
        )
        block_tables = self.block_tables.gather_block_tables(
            self.idx_mapping[:num_reqs],
            num_reqs_padded=batch_desc.num_reqs or num_reqs,
        )
        if isinstance(self.model_state, MambaHybridModelState):
            self.scratch_state_indices = self.model_state.prepare_moe_skip_scratch(
                input_batch, block_tables, self.kv_cache_config
            )

        for step in range(self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            step_attn_metadata, step_slot_mappings = self._build_attn(
                input_batch, batch_desc, block_tables, step
            )
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    step_attn_metadata,
                    step_slot_mappings,
                    None,
                    batch_desc.cg_mode,
                )
        if self.trace_dir is not None and not dummy_run and not is_profile:
            draft_top8 = self.draft_top8_tokens[:num_reqs].cpu().tolist()
            draft_top2_logits = self.draft_top2_logits[:num_reqs].cpu().tolist()
            draft_argmax_tokens = self.draft_tokens[:num_reqs].cpu().tolist()
            for req_id, top8, top2_logits, argmax_tokens in zip(
                input_batch.req_ids,
                draft_top8,
                draft_top2_logits,
                draft_argmax_tokens,
                strict=True,
            ):
                if _is_internal_request(req_id):
                    continue
                self.pending_draft_top8[req_id] = top8
                self.pending_draft_top2_logits[req_id] = top2_logits
                self.pending_draft_argmax_tokens[req_id] = argmax_tokens
        return self.draft_tokens[:num_reqs]

    def record_verification(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
    ) -> None:
        if self.trace_path is None or input_batch.num_draft_tokens_per_req is None:
            return

        target_top1 = torch.argmax(logits, dim=-1).cpu().tolist()
        target_top2_logits, target_top2_tokens = torch.topk(logits, k=2, dim=-1)
        target_top2_logits = target_top2_logits.float().cpu().tolist()
        target_top2_tokens = target_top2_tokens.cpu().tolist()
        sampled_counts = num_sampled[: input_batch.num_reqs].cpu().tolist()
        rows = []
        for req_idx, req_id in enumerate(input_batch.req_ids):
            if _is_internal_request(req_id):
                continue
            num_draft = int(input_batch.num_draft_tokens_per_req[req_idx])
            if num_draft == 0:
                continue
            draft_top8 = self.pending_draft_top8.pop(req_id, None)
            if draft_top8 is None:
                raise RuntimeError(
                    f"Missing MoE-Skip draft trace for request {req_id!r}"
                )
            draft_top2_logits = self.pending_draft_top2_logits.pop(req_id, None)
            if draft_top2_logits is None:
                raise RuntimeError(
                    f"Missing MoE-Skip draft logit trace for request {req_id!r}"
                )
            draft_argmax_tokens = self.pending_draft_argmax_tokens.pop(req_id, None)
            if draft_argmax_tokens is None:
                raise RuntimeError(
                    f"Missing MoE-Skip draft argmax trace for request {req_id!r}"
                )
            verify_step = self.verify_steps.get(req_id, 0)
            self.verify_steps[req_id] = verify_step + 1
            accepted = min(max(int(sampled_counts[req_idx]) - 1, 0), num_draft)
            logits_start = int(input_batch.cu_num_logits_np[req_idx])
            draft_argmax_tokens_gpu = self.draft_tokens[req_idx, :num_draft]
            draft_topk_tokens_gpu = self.draft_top8_tokens[req_idx, :num_draft, :2]
            draft_runner_up_tokens_gpu = torch.where(
                draft_topk_tokens_gpu[:, 0] == draft_argmax_tokens_gpu,
                draft_topk_tokens_gpu[:, 1],
                draft_topk_tokens_gpu[:, 0],
            )
            draft_pair_tokens_gpu = torch.stack(
                (draft_argmax_tokens_gpu, draft_runner_up_tokens_gpu), dim=1
            )
            target_draft_pair_logits = (
                torch.gather(
                    logits[logits_start : logits_start + num_draft],
                    1,
                    draft_pair_tokens_gpu,
                )
                .float()
                .cpu()
                .tolist()
            )
            for position in range(num_draft):
                top1_logit, top2_logit = draft_top2_logits[position]
                draft_argmax_token = draft_argmax_tokens[position]
                top8_tokens = draft_top8[position]
                if draft_argmax_token not in top8_tokens:
                    raise RuntimeError(
                        "MoE-Skip draft argmax is absent from traced Top-8"
                    )
                draft_runner_up_token = next(
                    token for token in top8_tokens if token != draft_argmax_token
                )
                argmax_ordered_top8 = [
                    draft_argmax_token,
                    *(token for token in top8_tokens if token != draft_argmax_token),
                ]
                target_offset = logits_start + position
                target_top1_logit, target_top2_logit = target_top2_logits[target_offset]
                target_draft_top1_logit, target_draft_top2_logit = (
                    target_draft_pair_logits[position]
                )
                rows.append(
                    {
                        "draft_length": self.num_speculative_steps,
                        "request_id": _external_request_id(req_id),
                        "engine_request_id": req_id,
                        "verify_step": verify_step,
                        "draft_position": position + 1,
                        "draft_top8_token_ids": top8_tokens,
                        "draft_argmax_ordered_top8_token_ids": argmax_ordered_top8,
                        "draft_argmax_token_id": draft_argmax_token,
                        "draft_runner_up_token_id": draft_runner_up_token,
                        "draft_top1_logit": top1_logit,
                        "draft_top2_logit": top2_logit,
                        "draft_top1_minus_top2": top1_logit - top2_logit,
                        "target_top1_token_id": target_top1[target_offset],
                        "target_top2_token_ids": target_top2_tokens[target_offset],
                        "target_top1_logit": target_top1_logit,
                        "target_top2_logit": target_top2_logit,
                        "target_top1_minus_top2": (
                            target_top1_logit - target_top2_logit
                        ),
                        "target_logit_for_draft_top1": target_draft_top1_logit,
                        "target_logit_for_draft_top2": target_draft_top2_logit,
                        "target_draft_top1_minus_draft_top2": (
                            target_draft_top1_logit - target_draft_top2_logit
                        ),
                        "accepted_draft_tokens": accepted,
                        "valid_mask": True,
                    }
                )
        if rows:
            with self.trace_path.open("a", encoding="utf-8") as trace_file:
                for row in rows:
                    trace_file.write(json.dumps(row, sort_keys=True) + "\n")
