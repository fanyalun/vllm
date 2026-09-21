# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import os
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState
from vllm.v1.worker.gpu.spec_decode.moe_skip.speculator import MoeSkipSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.workspace import use_workspace_lane

logger = init_logger(__name__)


def accepted_prefix(draft: torch.Tensor, predictions: torch.Tensor) -> int:
    """Number of consecutive greedy matches before the first rejection."""
    matches = draft.eq(predictions[: draft.numel()])
    return int(matches.to(torch.int32).cumprod(0).sum().item())


def should_stop_inner(policy, accepted, proposed, margin):
    if accepted >= proposed or policy == "none":
        return False
    threshold = {
        "low_error": 2.0 if accepted == 0 else 0.25,
        "balanced": 1.0,
        "aggressive": 2.0,
    }[policy]
    return margin < threshold


def refresh_graph_metadata(destination, source):
    """Refresh captured tensor addresses without replacing graph-owned buffers."""
    if isinstance(destination, torch.Tensor):
        destination.copy_(source)
    elif is_dataclass(destination):
        for field in fields(destination):
            refresh_graph_metadata(
                getattr(destination, field.name), getattr(source, field.name)
            )
    elif isinstance(destination, dict):
        for key in destination:
            refresh_graph_metadata(destination[key], source[key])
    elif isinstance(destination, (list, tuple)):
        for old, new in zip(destination, source, strict=True):
            refresh_graph_metadata(old, new)
    elif destination != source:
        raise ValueError("Pre-verify graph metadata changed its static structure")


class HierarchicalSpeculator(BaseSpeculator):
    """Accelerate shared-weight MoE-Skip with a short MTP or DSpark drafter."""

    supports_mm_inputs = False
    draft_logits = None
    last_logits: torch.Tensor
    last_margins: torch.Tensor

    def __init__(self, vllm_config, device):
        from vllm.v1.worker.gpu.spec_decode import init_speculator

        self.vllm_config = vllm_config
        self.device = device
        config = vllm_config.speculative_config
        self.config = config
        self.depth = config.inner_num_speculative_tokens
        self.rounds = config.inner_num_rounds
        self.capacity = config.num_speculative_tokens
        scheduler = vllm_config.scheduler_config
        self.max_num_reqs = scheduler.max_num_seqs
        architectures = set(
            getattr(vllm_config.model_config, "architectures", None) or ()
        )
        if self.max_num_reqs > 1 and not (
            architectures
            and architectures
            <= {
                "Gemma4ForCausalLM",
                "Gemma4ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
            }
            and config.inner_method == "mtp"
        ):
            raise ValueError("hierarchical batching requires Qwen or Gemma MTP")
        if self.max_num_reqs > 1 and (
            config.preverify_gdn_group_mode != "none"
            or (
                config.preverify_gdn_mode != "none"
                and config.preverify_gdn_update_policy != "windowed_three_level"
            )
        ):
            raise ValueError("Batched GDN requires native or ungrouped windowed state")
        if scheduler.async_scheduling:
            raise ValueError("hierarchical currently requires async_scheduling=False")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("hierarchical currently requires prefix caching disabled")
        if vllm_config.lora_config is not None:
            raise ValueError("hierarchical does not support LoRA")
        model_config = vllm_config.model_config
        if model_config.enable_prompt_embeds:
            raise ValueError("hierarchical does not support prompt embeddings")
        if config.preverify_gdn_update_policy == "windowed_three_level":
            text = model_config.hf_text_config
            if model_config.dtype != torch.bfloat16 or any(
                getattr(text, key, None) != value
                for key, value in {
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                }.items()
            ):
                raise ValueError("Windowed GDN requires BF16 Qwen3.6-35B-A3B")
        mm_config = model_config.multimodal_config
        if mm_config is not None and any(
            mm_config.get_limit_per_prompt(modality) > 0
            for modality in ("image", "video")
        ):
            raise ValueError(
                "hierarchical requires limit_mm_per_prompt={'image': 0, 'video': 0}"
            )
        self.inner_config = copy.copy(vllm_config)
        self.inner_config.speculative_config = config.make_inner_config()
        self.small = init_speculator(self.inner_config, device)
        self.preverify_config = copy.copy(vllm_config)
        self.preverify_config.speculative_config = replace(
            config,
            method="moe_skip",
            # Revalidate without the previously resolved native top-k.
            moe_skip_top_h=None
            if config.moe_skip_batch_policy is not None
            else config.moe_skip_top_h,
            model=None,
            inner_method=None,
            dspark_draft_topk=None,
            preverify_gdn_mode="none",
            preverify_gdn_group_mode="none",
            preverify_gdn_update_policy="exact",
            preverify_gdn_tail_policy="carry",
            preverify_gdn_mode_window_size=5,
            preverify_gdn_tau_alpha=0.95,
            preverify_gdn_tau_beta=0.36328125,
            preverify_gdn_optimization="none",
        )
        self.preverify = MoeSkipSpeculator(self.preverify_config, device)
        self.buffers = InputBuffers(
            self.max_num_reqs, scheduler.max_num_batched_tokens, device
        )
        self.draft_tokens = torch.full(
            (self.max_num_reqs, self.capacity), -1, dtype=torch.int64, device=device
        )
        self._draft_lengths = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.draft_lengths = self._draft_lengths
        self.last_sampled = torch.zeros(
            (self.max_num_reqs, 1), dtype=torch.int64, device=device
        )
        self.reset_policy_metrics()
        self.last_trace: list[dict[str, int]] = []
        trace_dir = os.environ.get("VLLM_HIERARCHICAL_TRACE_DIR")
        self.trace_path = Path(trace_dir) / "rounds.jsonl" if trace_dir else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.check_preverify = (
            os.environ.get("VLLM_HIERARCHICAL_CHECK_PREVERIFY") == "1"
        )
        if self.check_preverify and (
            config.preverify_gdn_mode != "none"
            or config.preverify_gdn_group_mode != "none"
        ):
            raise ValueError("Mean GDN cannot use exact sequential preverify checks")
        self.pending_req_id = None
        self.preverify_graphs = {}
        self.use_preverify_graphs = False
        self.preverify_graph_replays = 0

    def reset_policy_metrics(self):
        self.policy_metrics = dict(
            proposals=0,
            inner_rounds=0,
            inner_proposed=0,
            inner_accepted=0,
            early_stops=0,
            skipped_rounds=0,
            batch_round_calls=0,
        )

    def load_model(self, target_model):
        self.preverify.load_model(target_model)
        self.model = self.preverify.model
        self.logits_model = target_model
        state_cls: Any = PreverifyState
        state_kwargs = {}
        if getattr(self, "max_num_reqs", 1) > 1:
            from vllm.v1.worker.gpu.spec_decode.hierarchical.batched_state import (
                BatchedPreverifyState,
            )

            state_cls = BatchedPreverifyState
            state_kwargs["max_num_reqs"] = self.max_num_reqs
        self.state = state_cls(
            self.model,
            self.depth + 1,
            self.device,
            self.config.preverify_gdn_mode,
            self.config.preverify_gdn_update_policy,
            self.config.preverify_gdn_tail_policy,
            self.config.preverify_gdn_mode_window_size,
            self.config.preverify_gdn_tau_alpha,
            self.config.preverify_gdn_tau_beta,
            self.config.preverify_gdn_optimization,
            **state_kwargs,
        )
        self.grouped_gdn = None
        if (
            self.config.preverify_gdn_mode != "none"
            or self.config.preverify_gdn_group_mode != "none"
        ):
            if self.preverify.model_family != "qwen3_6" or self.device.type != "cuda":
                raise ValueError("Mean GDN preverify requires Qwen3.6 on CUDA")
            if self.vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Mean GDN preverify currently requires TP=1")
            if self.vllm_config.quant_config is not None:
                raise ValueError("Mean GDN preverify requires unquantized weights")
            if any(
                cache[1].dtype != torch.float32 for cache in self.state.caches.values()
            ):
                raise ValueError("Mean GDN preverify requires FP32 recurrent state")
        if self.config.preverify_gdn_group_mode != "none":
            from vllm.v1.worker.gpu.spec_decode.hierarchical.grouped_gdn import (
                GroupedGDNPreverify,
            )

            if any(
                cache[0].dtype != torch.bfloat16 for cache in self.state.caches.values()
            ):
                raise ValueError("Grouped GDN requires BF16 Conv state")
            self.grouped_gdn = GroupedGDNPreverify(self.model, self.vllm_config)
        if self.preverify.model_family == "qwen3_6" and not self.state.layers:
            raise ValueError("hierarchical requires Qwen GDN layers")
        self.small.load_model(target_model)
        self.layer_outputs: dict[int, tuple[torch.Tensor, ...]] = {}
        self.tracing_layers = False
        if self.check_preverify:
            for index, layer in enumerate(self.model.model.layers):

                def record_layer(module, inputs, output, index=index):
                    if self.tracing_layers:
                        self.layer_outputs[index] = tuple(x.clone() for x in output)

                layer.register_forward_hook(record_layer)

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        expected_state = (
            MambaHybridModelState
            if self.preverify.model_family == "qwen3_6"
            else DefaultModelState
        )
        if not isinstance(model_state, expected_state):
            raise ValueError(f"hierarchical requires {expected_state.__name__}")
        if (
            isinstance(model_state, MambaHybridModelState)
            and model_state.recoverssm is not None
        ):
            raise ValueError("hierarchical does not support RecoverSSM")
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups, _, _ = init_attn_backend(
            kv_cache_config, self.vllm_config, self.device
        )
        self.small.set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.refresh_small_lengths = (
            self.preverify.model_family == "gemma4"
            and self.config.inner_method == "mtp"
        )
        if self.refresh_small_lengths:
            self.saved_small_seq_lens = torch.empty_like(
                target_input_buffers.seq_lens[:1]
            )
            self.saved_small_query_start = torch.empty_like(
                target_input_buffers.query_start_loc[:2]
            )

    @contextmanager
    def _small_metadata(self, batch, inner_round):
        if not self.refresh_small_lengths or inner_round == 0:
            yield
            return
        # Gemma's Q-only MTP prefill graph captures the Target's Triton lengths.
        # Inner batches use private buffers; restore Target lengths after replay.
        buffers = self.small.target_input_buffers
        seq_lens = buffers.seq_lens[:1]
        query_start = buffers.query_start_loc[:2]
        self.saved_small_seq_lens.copy_(seq_lens)
        self.saved_small_query_start.copy_(query_start)
        try:
            seq_lens.copy_(batch.seq_lens)
            query_start.copy_(batch.query_start_loc)
            yield
        finally:
            seq_lens.copy_(self.saved_small_seq_lens)
            query_start.copy_(self.saved_small_query_start)

    def init_cudagraph_manager(self, cudagraph_mode):
        self.small.init_cudagraph_manager(cudagraph_mode)
        if self.config.inner_method == "mtp":
            from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
                SpeculatorCudaGraphManager,
            )

            self.small.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
                self.vllm_config, self.device, cudagraph_mode, self.capacity + 1
            )
        self.use_preverify_graphs = cudagraph_mode.has_full_cudagraphs()

    def capture(self):
        self.small.capture()

    def record_verification(self, logits, input_batch, num_sampled):
        if self.trace_path is None or self.pending_req_id not in input_batch.req_ids:
            return
        scheduled = int(input_batch.num_draft_tokens_per_req[0])
        record = {
            "request_id": self.pending_req_id,
            "inner_method": self.config.inner_method,
            "inner_rounds": self.last_trace,
            "outer_proposed": int(self.draft_lengths[0].item()),
            "outer_scheduled": scheduled,
            "outer_accepted": int(num_sampled[0].item()) - 1,
            "candidate_tokens": self.draft_tokens[0, :scheduled].tolist(),
            "preverify_graph_widths": sorted(self.preverify_graphs),
            "preverify_graph_replays": self.preverify_graph_replays,
        }
        with self.trace_path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        self.pending_req_id = None

    def _batch(self, template, position: int, tokens: torch.Tensor):
        width = tokens.numel()
        padded = width
        buffers = self.buffers
        optimized = getattr(self.state, "execution_optimized", False) is True
        if not optimized:
            buffers.input_ids[:padded].zero_()
            buffers.positions[:padded].zero_()
            buffers.is_padding[:padded].fill_(True)
        buffers.input_ids[:width].copy_(tokens)
        if optimized:
            indices, local_positions, query_start = self.state.batch_constants[width]
            torch.add(indices, position, out=buffers.positions[:width])
            buffers.query_start_loc[:2].copy_(query_start)
        else:
            buffers.positions[:width].copy_(
                torch.arange(position, position + width, device=self.device)
            )
            buffers.query_start_loc[:2].copy_(
                torch.tensor([0, width], dtype=torch.int32, device=self.device)
            )
        buffers.seq_lens[:1].fill_(position + width)
        buffers.is_padding[:width].zero_()
        zeros = np.zeros(1, dtype=np.int32)
        batch = replace(
            template,
            num_reqs_after_padding=1,
            expanded_idx_mapping=template.idx_mapping.expand(width),
            expanded_local_pos=local_positions
            if optimized
            else torch.arange(width, dtype=torch.int32, device=self.device),
            num_scheduled_tokens=np.array([width], dtype=np.int32),
            num_tokens=width,
            num_tokens_after_padding=padded,
            num_draft_tokens=width - 1,
            num_draft_tokens_per_req=np.array([width - 1], dtype=np.int32),
            query_start_loc=buffers.query_start_loc[:2],
            query_start_loc_np=np.array([0, width], dtype=np.int32),
            seq_lens=buffers.seq_lens[:1],
            seq_lens_cpu_upper_bound=torch.tensor(
                [self.vllm_config.model_config.max_model_len], dtype=torch.int32
            ),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.array([position], dtype=np.int32),
            prefill_len_np=zeros,
            num_computed_prefill_tokens_np=zeros,
            is_prefilling_np=np.zeros(1, dtype=np.bool_),
            has_prefill=False,
            input_ids=buffers.input_ids[:padded],
            positions=buffers.positions[:padded],
            is_padding=buffers.is_padding[:padded],
            logits_indices=indices
            if optimized
            else torch.arange(width, dtype=torch.int64, device=self.device),
            cu_num_logits=buffers.query_start_loc[:2],
            cu_num_logits_np=np.array([0, width], dtype=np.int32),
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=width,
        )
        tables = self.block_tables.gather_block_tables(batch.idx_mapping, 1)
        slots = self.block_tables.compute_slot_mappings(
            batch.idx_mapping, batch.query_start_loc, batch.positions, padded
        )
        metadata = self.model_state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            tables,
            slots,
            self.attn_groups,
            self.kv_cache_config,
        )
        if self.state.layers:
            self._set_gdn_metadata(batch, width, metadata)
        return (
            batch,
            metadata,
            build_slot_mappings_by_layer(slots, self.kv_cache_config),
        )

    def _set_gdn_metadata(self, batch, width, metadata):
        n = batch.num_reqs
        if getattr(self, "max_num_reqs", 1) > 1:
            self.state.select(batch.req_ids)
        gdn = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=n,
            num_spec_decode_tokens=batch.num_tokens,
            num_actual_tokens=batch.num_tokens,
            spec_query_start_loc=batch.query_start_loc,
            spec_state_indices_tensor=self.state.state_indices[:n, :width],
            spec_sequence_masks=self.state.sequence_mask[:n]
            if hasattr(self.state, "sequence_mask")
            else torch.ones(n, dtype=torch.bool, device=self.device),
            num_accepted_tokens=self.state.num_accepted[:n],
        )
        for name in self.state.layers:
            metadata[name] = gdn

    def _verify(self, batch, metadata, slots):
        width = batch.num_tokens
        key = width if batch.num_reqs == 1 else (batch.num_reqs, width)
        if getattr(self.state, "thresholds", None) is not None:
            key = (width, self.state.direction)
            if self.state.action_counts is not None:
                key = (width, self.state.direction, "actions")
            self.state.window_width = width
            if self.state.windowed:
                if not self.state.initialized or (
                    getattr(self, "max_num_reqs", 1) == 1
                    and self.state.request_id != batch.req_ids[0]
                ):
                    raise RuntimeError("Windowed GDN request slot is invalid")
                key = (
                    batch.num_reqs,
                    width,
                    self.state.update_policy,
                    self.state.window_size,
                    self.state.optimization,
                    self.state.action_counts is not None,
                )
        if not self.use_preverify_graphs or self.check_preverify:
            return self._verify_eager(batch, metadata, slots)
        if key not in self.preverify_graphs:
            before = self.state.snapshot()
            action_counts = getattr(self.state, "action_counts", None)
            counts_before = action_counts.clone() if action_counts is not None else None
            for _ in range(3):
                self._verify_eager(batch, metadata, slots)
                self.state.restore(before)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._verify_eager(batch, metadata, slots)
            self.state.restore(before)
            if counts_before is not None and action_counts is not None:
                action_counts.copy_(counts_before)
            self.preverify_graphs[key] = (
                graph,
                output,
                metadata,
                slots,
                self.last_logits,
                self.last_margins,
            )
            logger.info(
                "Captured hierarchical pre-verifier CUDA graph: width=%d", width
            )
        graph, output, captured_metadata, captured_slots, logits, margins = (
            self.preverify_graphs[key]
        )
        refresh_graph_metadata(captured_metadata, metadata)
        refresh_graph_metadata(captured_slots, slots)
        graph.replay()
        self.last_logits = logits
        self.last_margins = margins
        self.preverify_graph_replays += 1
        return output

    def _verify_eager(self, batch, metadata, slots):
        positions = batch.positions
        if self.vllm_config.model_config.uses_mrope:
            positions = positions.unsqueeze(0).expand(3, -1)
        with (
            self.state.activate(),
            use_workspace_lane(0),
            set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=batch.num_tokens_after_padding,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slots,
                batch_descriptor=BatchDescriptor(
                    num_tokens=batch.num_tokens_after_padding
                ),
                is_padding=batch.is_padding,
                additional_forward_kwargs={
                    "routing_top_k": self.config.moe_skip_top_h,
                    "routing_min_weight": self.config.moe_skip_min_weight,
                    "routing_batch_policy": self.config.moe_skip_batch_policy,
                    "routing_preserve_weights": self.config.moe_skip_weight_mode
                    == "preserve",
                    "preverify_gdn_mode": self.config.preverify_gdn_mode,
                    "preverify_gdn_state": self.state,
                },
            ),
        ):
            self.tracing_layers = self.check_preverify
            try:
                if self.config.preverify_gdn_group_mode == "none":
                    output = self.model(input_ids=batch.input_ids, positions=positions)
                else:
                    assert self.grouped_gdn is not None
                    output = self.grouped_gdn(
                        batch.input_ids,
                        positions,
                        self.config.preverify_gdn_group_mode,
                        self.config.preverify_gdn_mode,
                    )
            finally:
                self.tracing_layers = False
        if isinstance(output, tuple):
            hidden, aux = output
        else:
            hidden, aux = output, None
        logits = self.logits_model.compute_logits(hidden)
        self.last_logits = logits
        top2 = logits.topk(2, dim=-1).values.float()
        self.last_margins = top2[:, 0] - top2[:, 1]
        return logits.argmax(-1), hidden, aux

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        if hasattr(self, "_draft_lengths"):
            self.draft_lengths = self._draft_lengths[: input_batch.num_reqs]
        self.draft_tokens.fill_(-1)
        self.draft_lengths.zero_()
        self.last_trace = []
        if input_batch.req_ids and all(
            req_id.startswith("_warmup_") for req_id in input_batch.req_ids
        ):
            # Runner warmup schedules the full capacity without asking for lengths.
            self.draft_tokens.zero_()
            self.draft_lengths.fill_(self.capacity)
            return self.draft_tokens[: input_batch.num_reqs]
        if dummy_run or is_profile or skip_attn_for_dummy_run:
            self.small.propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync=dp_sync,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )
            return self.draft_tokens[: input_batch.num_reqs]
        if input_batch.num_reqs == 0:
            if getattr(self.state, "windowed", False):
                self.state.invalidate()
            return self.draft_tokens[:0]
        if mm_inputs is not None:
            raise ValueError("hierarchical supports unstructured text requests only")
        if getattr(self, "max_num_reqs", 1) > 1:
            from vllm.v1.worker.gpu.spec_decode.hierarchical.batched import (
                propose_gemma,
            )

            return propose_gemma(
                self,
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
            )
        if int(num_sampled[0].item()) == 0:
            return self.draft_tokens
        if input_batch.has_structured_output_reqs or mm_inputs is not None:
            raise ValueError("hierarchical supports unstructured text requests only")
        tables = self.block_tables.gather_block_tables(input_batch.idx_mapping, 1)
        self.state.begin(self.model_state, input_batch, tables, self.kv_cache_config)
        position = int((input_batch.seq_lens - num_rejected)[0].item())
        anchor = last_sampled[input_batch.idx_mapping, 0].clone()
        batch, metadata, slots = input_batch, attn_metadata, slot_mappings
        hidden, aux = last_hidden_states, aux_hidden_states
        count = 0
        if hasattr(self, "policy_metrics"):
            self.policy_metrics["proposals"] += 1
        self.last_trace = []
        for round_idx in range(self.rounds):
            width = min(
                self.depth + 1,
                self.vllm_config.model_config.max_model_len - position,
            )
            if width <= 0:
                break
            self.last_sampled[0, 0] = anchor[0]
            with self._small_metadata(batch, round_idx):
                small_tokens = self.small.propose(
                    batch,
                    metadata,
                    slots,
                    hidden,
                    aux if self.config.inner_method == "dspark" else None,
                    num_sampled,
                    num_rejected,
                    self.last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                )[0, : width - 1].clone()
            batch, metadata, slots = self._batch(
                input_batch, position, torch.cat((anchor, small_tokens))
            )
            before = self.state.snapshot() if self.check_preverify else None
            predictions, hidden, aux = self._verify(batch, metadata, slots)
            accepted = accepted_prefix(small_tokens, predictions)
            margin = (
                float(self.last_margins[accepted].item())
                if hasattr(self, "last_margins")
                and (
                    getattr(self.state, "execution_optimized", False) is not True
                    or (
                        round_idx < self.rounds - 1
                        and accepted < width - 1
                        and self.config.hierarchical_stop_policy != "none"
                    )
                )
                else float("inf")
            )
            if before is not None:
                batch_logits = self.last_logits.clone()
                batch_layers = self.layer_outputs.copy()
                after = self.state.snapshot()
                tokens = batch.input_ids.clone()
                predictions = predictions.clone()
                hidden = hidden.clone()
                aux = [x.clone() for x in aux] if aux else None
                self.state.restore(before)
                sequential = []
                sequential_logits = []
                sequential_layers = []
                for offset in range(width):
                    ref_batch, ref_metadata, ref_slots = self._batch(
                        input_batch, position + offset, tokens[offset : offset + 1]
                    )
                    ref_predictions, _, _ = self._verify_eager(
                        ref_batch, ref_metadata, ref_slots
                    )
                    sequential.append(ref_predictions[:1].clone())
                    sequential_logits.append(self.last_logits[:1].clone())
                    sequential_layers.append(self.layer_outputs.copy())
                reference = torch.cat(sequential)
                if not torch.equal(
                    predictions[: accepted + 1], reference[: accepted + 1]
                ):
                    delta = (
                        (batch_logits.float() - torch.cat(sequential_logits).float())
                        .abs()
                        .amax(-1)
                    )
                    layer_delta = {
                        index: (
                            values[0].float()
                            - torch.cat(
                                [row[index][0] for row in sequential_layers]
                            ).float()
                        )
                        .abs()
                        .amax(-1)
                        .tolist()
                        for index, values in batch_layers.items()
                    }
                    raise AssertionError(
                        f"Pre-verify batch/sequence mismatch at {position}: "
                        f"draft={small_tokens.tolist()} accepted={accepted} "
                        f"batch={predictions.tolist()} "
                        f"sequential={reference.tolist()} "
                        f"max_logit_diff={delta.tolist()} layers={layer_delta}"
                    )
                self.state.restore(after)
                batch, metadata, slots = self._batch(input_batch, position, tokens)
            emitted = accepted + 1
            self.draft_tokens[0, count : count + accepted] = small_tokens[:accepted]
            self.draft_tokens[0, count + accepted] = predictions[accepted]
            self.last_trace.append(
                {
                    "inner_round": round_idx,
                    "proposed": width - 1,
                    "accepted": accepted,
                    "emitted": emitted,
                    "offset": count,
                    "stop_reason": 2
                    if round_idx == self.rounds - 1
                    else (3 if width < self.depth + 1 else 0),
                }
            )
            count += emitted
            if hasattr(self, "policy_metrics"):
                self.policy_metrics["inner_rounds"] += 1
                self.policy_metrics["batch_round_calls"] += 1
                self.policy_metrics["inner_proposed"] += width - 1
                self.policy_metrics["inner_accepted"] += accepted
            if round_idx < self.rounds - 1 and should_stop_inner(
                getattr(self.config, "hierarchical_stop_policy", "none"),
                accepted,
                width - 1,
                margin,
            ):
                self.policy_metrics["early_stops"] += 1
                self.policy_metrics["skipped_rounds"] += self.rounds - round_idx - 1
                self.last_trace[-1]["stop_reason"] = 1
                break
            if getattr(
                self.config, "preverify_gdn_update_policy", "exact"
            ) != "exact" and (round_idx == self.rounds - 1 or width < self.depth + 1):
                break
            anchor = predictions[accepted : accepted + 1].clone()
            position += emitted
            self.state.advance(accepted)
            if getattr(self.state, "execution_optimized", False) is True:
                num_sampled = self.state.counts[emitted : emitted + 1]
                rejected = width - emitted
                num_rejected = self.state.counts[rejected : rejected + 1]
            else:
                num_sampled = torch.tensor(
                    [emitted], dtype=torch.int32, device=self.device
                )
                num_rejected = torch.tensor(
                    [width - emitted], dtype=torch.int32, device=self.device
                )
            if width < self.depth + 1:
                break
        self.draft_lengths.fill_(count)
        self.pending_req_id = input_batch.req_ids[0]
        if getattr(self.state, "windowed", False) is True:
            self.state.invalidate()
        return self.draft_tokens
