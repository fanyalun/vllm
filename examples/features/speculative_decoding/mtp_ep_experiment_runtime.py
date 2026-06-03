# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from mtp_ep_load_balance_utils import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    FinishedRequestStatTotals,
    SCHEMA_VERSION,
    StepTiming,
    aggregate_global_step_time_components,
    aggregate_worker_step_timings,
    average_step_histograms,
    classify_step_capture,
    compute_num_output_tokens_excluding_first,
    compute_decode_throughput_tok_s,
    compute_tpot_ms_from_finished_stats,
    count_layer_expert_histograms,
    merge_expert_to_ep_rank_maps,
    num_condition_rounds,
    shard_global_batch_indices,
    select_dataset_indices,
    TPOT_DEFINITION,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.metrics.loggers import StatLoggerBase

if TYPE_CHECKING:
    from argparse import Namespace


class FinishedRequestStatsLogger(StatLoggerBase):
    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.reset()

    def reset(self) -> None:
        self.decode_time_total_ms = 0.0
        self.num_generation_tokens_total = 0
        self.num_output_tokens_excl_first_total = 0

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int = 0,
    ) -> None:
        if iteration_stats is None:
            return
        for finished_req in iteration_stats.finished_requests:
            num_generation_tokens = int(finished_req.num_generation_tokens)
            self.decode_time_total_ms += float(finished_req.decode_time) * 1000.0
            self.num_generation_tokens_total += num_generation_tokens
            self.num_output_tokens_excl_first_total += max(
                num_generation_tokens - 1,
                0,
            )

    def log_engine_initialized(self) -> None:
        return

    def snapshot(self) -> FinishedRequestStatTotals:
        return FinishedRequestStatTotals(
            decode_time_total_ms=self.decode_time_total_ms,
            num_generation_tokens_total=self.num_generation_tokens_total,
            num_output_tokens_excl_first_total=(
                self.num_output_tokens_excl_first_total
            ),
        )


_FINISHED_REQUEST_STATS_LOGGER_ATTR = "_mtp_ep_finished_request_stats_logger"
_SCHEDULER_CAPACITY_CONFIG_ATTR = "_mtp_ep_scheduler_capacity_config"


@dataclass(frozen=True)
class SchedulerCapacityConfig:
    local_max_num_seqs: int
    configured_max_num_batched_tokens: int
    scheduler_max_num_seqs: int
    scheduler_max_num_batched_tokens: int
    scheduler_max_num_scheduled_tokens: int
    speculative_max_num_new_slots_for_drafting: int

    def to_log_dict(self) -> dict[str, int]:
        return {
            "local_max_num_seqs": self.local_max_num_seqs,
            "configured_max_num_batched_tokens": (
                self.configured_max_num_batched_tokens
            ),
            "scheduler_max_num_seqs": self.scheduler_max_num_seqs,
            "scheduler_max_num_batched_tokens": (
                self.scheduler_max_num_batched_tokens
            ),
            "scheduler_max_num_scheduled_tokens": (
                self.scheduler_max_num_scheduled_tokens
            ),
            "speculative_max_num_new_slots_for_drafting": (
                self.speculative_max_num_new_slots_for_drafting
            ),
        }


@dataclass
class ConditionRawData:
    batch_size: int
    draft_length: int
    data_parallel_size: int
    num_samples: int
    batch_size_scope: str
    local_max_num_seqs: int
    configured_max_num_batched_tokens: int
    scheduler_max_num_seqs: int
    scheduler_max_num_batched_tokens: int
    scheduler_max_num_scheduled_tokens: int
    speculative_max_num_new_slots_for_drafting: int
    mixed_step_policy: str
    tpot_definition: str
    selected_dataset_indices: np.ndarray
    prompt_lengths: np.ndarray
    output_lengths: np.ndarray
    condition_latency_ms: float
    decode_time_total_ms: float
    num_output_tokens_total: int
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int
    tpot_ms: float
    decode_throughput_tok_s: float
    step_histograms: np.ndarray
    step_total_tokens: np.ndarray
    step_total_ms: np.ndarray
    step_attention_ms: np.ndarray
    step_routing_ms: np.ndarray
    step_prepare_ms: np.ndarray
    step_finalize_ms: np.ndarray
    step_ffn_ms: np.ndarray
    captured_step_kinds: np.ndarray
    global_barrier_ids: np.ndarray
    barrier_first_ep_collective_seq_ids: np.ndarray
    barrier_last_ep_collective_seq_ids: np.ndarray
    barrier_num_ep_collectives: np.ndarray
    rank_barrier_first_ep_collective_seq_ids: np.ndarray
    rank_barrier_last_ep_collective_seq_ids: np.ndarray
    rank_barrier_num_ep_collectives: np.ndarray
    rank_step_kinds: np.ndarray
    rank_step_total_ms: np.ndarray
    rank_step_draft_ms: np.ndarray
    rank_layer_ffn_ms: np.ndarray
    rank_layer_local_routed_tokens: np.ndarray
    rank_layer_local_active_experts: np.ndarray
    global_step_indices: np.ndarray
    global_step_total_ms: np.ndarray
    global_draft_ms: np.ndarray
    global_step_ffn_ms: np.ndarray
    global_step_sorted_rank_ffn_ms: np.ndarray
    global_step_sorted_rank_local_routed_tokens: np.ndarray
    global_step_sorted_rank_local_active_experts: np.ndarray
    global_step_ffn_max_mean_ratio: np.ndarray
    global_step_other_ms: np.ndarray
    global_step_kinds: np.ndarray
    expert_to_ep_rank: np.ndarray
    layers: np.ndarray
    avg_histograms: np.ndarray
    num_forward_steps_total: int
    num_captured_steps: int
    num_global_candidate_steps: int
    num_global_captured_steps: int
    num_dropped_steps: int
    num_prefill_dropped_steps: int
    num_mixed_dropped_steps: int
    num_global_prefill_dropped_steps: int
    num_global_mixed_dropped_steps: int
    num_global_non_target_dropped_steps: int

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        step_all2all_ms = self.step_prepare_ms + self.step_finalize_ms
        return {
            "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
            "batch_size": np.asarray([self.batch_size], dtype=np.int64),
            "draft_length": np.asarray([self.draft_length], dtype=np.int64),
            "data_parallel_size": np.asarray([self.data_parallel_size], dtype=np.int64),
            "num_samples": np.asarray([self.num_samples], dtype=np.int64),
            "batch_size_scope": np.asarray([self.batch_size_scope]),
            "local_max_num_seqs": np.asarray(
                [self.local_max_num_seqs], dtype=np.int64
            ),
            "configured_max_num_batched_tokens": np.asarray(
                [self.configured_max_num_batched_tokens], dtype=np.int64
            ),
            "scheduler_max_num_seqs": np.asarray(
                [self.scheduler_max_num_seqs], dtype=np.int64
            ),
            "scheduler_max_num_batched_tokens": np.asarray(
                [self.scheduler_max_num_batched_tokens], dtype=np.int64
            ),
            "scheduler_max_num_scheduled_tokens": np.asarray(
                [self.scheduler_max_num_scheduled_tokens], dtype=np.int64
            ),
            "speculative_max_num_new_slots_for_drafting": np.asarray(
                [self.speculative_max_num_new_slots_for_drafting],
                dtype=np.int64,
            ),
            "mixed_step_policy": np.asarray([self.mixed_step_policy]),
            "tpot_definition": np.asarray([self.tpot_definition]),
            "selected_dataset_indices": self.selected_dataset_indices,
            "prompt_lengths": self.prompt_lengths,
            "output_lengths": self.output_lengths,
            "condition_latency_ms": np.asarray(
                [self.condition_latency_ms], dtype=np.float64
            ),
            "decode_time_total_ms": np.asarray(
                [self.decode_time_total_ms], dtype=np.float64
            ),
            "decode_only_total_ms": np.asarray(
                [self.decode_time_total_ms], dtype=np.float64
            ),
            "num_output_tokens_total": np.asarray(
                [self.num_output_tokens_total], dtype=np.int64
            ),
            "num_generation_tokens_total": np.asarray(
                [self.num_generation_tokens_total], dtype=np.int64
            ),
            "num_output_tokens_excl_first_total": np.asarray(
                [self.num_output_tokens_excl_first_total], dtype=np.int64
            ),
            "num_output_tokens_excl_first": np.asarray(
                [self.num_output_tokens_excl_first_total], dtype=np.int64
            ),
            "tpot_ms": np.asarray([self.tpot_ms], dtype=np.float64),
            "decode_throughput_tok_s": np.asarray(
                [self.decode_throughput_tok_s], dtype=np.float64
            ),
            "step_histograms": self.step_histograms,
            "step_total_tokens": self.step_total_tokens,
            "step_total_ms": self.step_total_ms,
            "step_attention_ms": self.step_attention_ms,
            "step_routing_ms": self.step_routing_ms,
            "step_prepare_ms": self.step_prepare_ms,
            "step_finalize_ms": self.step_finalize_ms,
            "step_all2all_ms": step_all2all_ms,
            "step_ffn_ms": self.step_ffn_ms,
            "captured_step_kinds": self.captured_step_kinds,
            "global_barrier_id": self.global_barrier_ids,
            "global_barrier_ids": self.global_barrier_ids,
            "barrier_first_ep_collective_seq_ids": (
                self.barrier_first_ep_collective_seq_ids
            ),
            "barrier_last_ep_collective_seq_ids": (
                self.barrier_last_ep_collective_seq_ids
            ),
            "barrier_num_ep_collectives": self.barrier_num_ep_collectives,
            "rank_barrier_first_ep_collective_seq_ids": (
                self.rank_barrier_first_ep_collective_seq_ids
            ),
            "rank_barrier_last_ep_collective_seq_ids": (
                self.rank_barrier_last_ep_collective_seq_ids
            ),
            "rank_barrier_num_ep_collectives": (
                self.rank_barrier_num_ep_collectives
            ),
            "rank_step_kinds": self.rank_step_kinds,
            "rank_step_total_ms": self.rank_step_total_ms,
            "rank_step_draft_ms": self.rank_step_draft_ms,
            "rank_layer_ffn_ms": self.rank_layer_ffn_ms,
            "rank_layer_local_routed_tokens": (
                self.rank_layer_local_routed_tokens
            ),
            "rank_layer_local_active_experts": (
                self.rank_layer_local_active_experts
            ),
            "global_step_indices": self.global_step_indices,
            "global_step_total_ms": self.global_step_total_ms,
            "global_draft_ms": self.global_draft_ms,
            "global_step_ffn_ms": self.global_step_ffn_ms,
            "global_step_ffn_phase_ms": self.global_step_ffn_ms,
            "global_step_sorted_rank_ffn_ms": self.global_step_sorted_rank_ffn_ms,
            "global_step_sorted_rank_local_routed_tokens": (
                self.global_step_sorted_rank_local_routed_tokens
            ),
            "global_step_sorted_rank_local_active_experts": (
                self.global_step_sorted_rank_local_active_experts
            ),
            "global_step_ffn_max_mean_ratio": self.global_step_ffn_max_mean_ratio,
            "global_step_other_ms": self.global_step_other_ms,
            "global_step_kinds": self.global_step_kinds,
            "expert_to_ep_rank": self.expert_to_ep_rank,
            "layers": self.layers,
            "avg_histograms": self.avg_histograms,
            "num_forward_steps_total": np.asarray(
                [self.num_forward_steps_total], dtype=np.int64
            ),
            "num_captured_steps": np.asarray(
                [self.num_captured_steps], dtype=np.int64
            ),
            "num_global_candidate_steps": np.asarray(
                [self.num_global_candidate_steps], dtype=np.int64
            ),
            "num_global_captured_steps": np.asarray(
                [self.num_global_captured_steps], dtype=np.int64
            ),
            "num_dropped_steps": np.asarray(
                [self.num_dropped_steps], dtype=np.int64
            ),
            "num_prefill_dropped_steps": np.asarray(
                [self.num_prefill_dropped_steps], dtype=np.int64
            ),
            "num_mixed_dropped_steps": np.asarray(
                [self.num_mixed_dropped_steps], dtype=np.int64
            ),
            "num_global_prefill_dropped_steps": np.asarray(
                [self.num_global_prefill_dropped_steps], dtype=np.int64
            ),
            "num_global_mixed_dropped_steps": np.asarray(
                [self.num_global_mixed_dropped_steps], dtype=np.int64
            ),
            "num_global_non_target_dropped_steps": np.asarray(
                [self.num_global_non_target_dropped_steps], dtype=np.int64
            ),
        }


@dataclass
class CollectedConditionSummary:
    batch_size: int
    draft_length: int
    raw_path: str
    local_max_num_seqs: int
    configured_max_num_batched_tokens: int
    scheduler_max_num_seqs: int
    scheduler_max_num_batched_tokens: int
    scheduler_max_num_scheduled_tokens: int
    speculative_max_num_new_slots_for_drafting: int
    condition_latency_ms: float
    decode_time_total_ms: float
    num_output_tokens_total: int
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int
    tpot_ms: float
    decode_throughput_tok_s: float
    num_forward_steps_total: int
    num_captured_steps: int
    num_global_candidate_steps: int
    num_global_captured_steps: int
    num_dropped_steps: int
    num_prefill_dropped_steps: int
    num_mixed_dropped_steps: int
    num_global_prefill_dropped_steps: int
    num_global_mixed_dropped_steps: int
    num_global_non_target_dropped_steps: int


@dataclass
class RankConditionData:
    selected_dataset_indices: np.ndarray
    prompt_lengths: np.ndarray
    output_lengths: np.ndarray
    step_histograms: np.ndarray
    step_total_tokens: np.ndarray
    step_total_ms: np.ndarray
    step_attention_ms: np.ndarray
    step_routing_ms: np.ndarray
    step_prepare_ms: np.ndarray
    step_finalize_ms: np.ndarray
    step_ffn_ms: np.ndarray
    captured_step_kinds: np.ndarray
    captured_step_indices: np.ndarray
    captured_step_start_time_ms: np.ndarray
    captured_step_end_time_ms: np.ndarray
    captured_prepare_start_time_ms: np.ndarray
    captured_finalize_end_time_ms: np.ndarray
    candidate_first_ep_collective_seq_ids: np.ndarray
    candidate_last_ep_collective_seq_ids: np.ndarray
    candidate_num_ep_collectives: np.ndarray
    candidate_step_kinds: np.ndarray
    candidate_drop_reasons: np.ndarray
    candidate_step_total_tokens: np.ndarray
    candidate_step_total_ms: np.ndarray
    candidate_step_draft_ms: np.ndarray
    candidate_step_ffn_ms: np.ndarray
    candidate_step_histograms: np.ndarray
    candidate_layer_ffn_ms: np.ndarray
    candidate_layer_local_routed_tokens: np.ndarray
    candidate_layer_local_active_experts: np.ndarray
    expert_to_ep_rank: np.ndarray
    condition_latency_ms: float
    decode_time_total_ms: float
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int
    num_forward_steps_total: int
    num_captured_steps: int
    num_dropped_steps: int
    num_prefill_dropped_steps: int
    num_mixed_dropped_steps: int
    local_max_num_seqs: int
    configured_max_num_batched_tokens: int
    scheduler_max_num_seqs: int
    scheduler_max_num_batched_tokens: int
    scheduler_max_num_scheduled_tokens: int
    speculative_max_num_new_slots_for_drafting: int
    trace_samples: list[dict[str, Any]] = field(default_factory=list)

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "selected_dataset_indices": self.selected_dataset_indices,
            "prompt_lengths": self.prompt_lengths,
            "output_lengths": self.output_lengths,
            "step_histograms": self.step_histograms,
            "step_total_tokens": self.step_total_tokens,
            "step_total_ms": self.step_total_ms,
            "step_attention_ms": self.step_attention_ms,
            "step_routing_ms": self.step_routing_ms,
            "step_prepare_ms": self.step_prepare_ms,
            "step_finalize_ms": self.step_finalize_ms,
            "step_ffn_ms": self.step_ffn_ms,
            "captured_step_kinds": self.captured_step_kinds,
            "captured_step_indices": self.captured_step_indices,
            "captured_step_start_time_ms": self.captured_step_start_time_ms,
            "captured_step_end_time_ms": self.captured_step_end_time_ms,
            "captured_prepare_start_time_ms": self.captured_prepare_start_time_ms,
            "captured_finalize_end_time_ms": self.captured_finalize_end_time_ms,
            "candidate_first_ep_collective_seq_ids": (
                self.candidate_first_ep_collective_seq_ids
            ),
            "candidate_last_ep_collective_seq_ids": (
                self.candidate_last_ep_collective_seq_ids
            ),
            "candidate_num_ep_collectives": self.candidate_num_ep_collectives,
            "candidate_step_kinds": self.candidate_step_kinds,
            "candidate_drop_reasons": self.candidate_drop_reasons,
            "candidate_step_total_tokens": self.candidate_step_total_tokens,
            "candidate_step_total_ms": self.candidate_step_total_ms,
            "candidate_step_draft_ms": self.candidate_step_draft_ms,
            "candidate_step_ffn_ms": self.candidate_step_ffn_ms,
            "candidate_step_histograms": self.candidate_step_histograms,
            "candidate_layer_ffn_ms": self.candidate_layer_ffn_ms,
            "candidate_layer_local_routed_tokens": (
                self.candidate_layer_local_routed_tokens
            ),
            "candidate_layer_local_active_experts": (
                self.candidate_layer_local_active_experts
            ),
            "expert_to_ep_rank": self.expert_to_ep_rank,
            "condition_latency_ms": np.asarray(
                [self.condition_latency_ms], dtype=np.float64
            ),
            "decode_time_total_ms": np.asarray(
                [self.decode_time_total_ms], dtype=np.float64
            ),
            "num_generation_tokens_total": np.asarray(
                [self.num_generation_tokens_total], dtype=np.int64
            ),
            "num_output_tokens_excl_first_total": np.asarray(
                [self.num_output_tokens_excl_first_total], dtype=np.int64
            ),
            "num_forward_steps_total": np.asarray(
                [self.num_forward_steps_total], dtype=np.int64
            ),
            "num_captured_steps": np.asarray(
                [self.num_captured_steps], dtype=np.int64
            ),
            "num_dropped_steps": np.asarray(
                [self.num_dropped_steps], dtype=np.int64
            ),
            "num_prefill_dropped_steps": np.asarray(
                [self.num_prefill_dropped_steps], dtype=np.int64
            ),
            "num_mixed_dropped_steps": np.asarray(
                [self.num_mixed_dropped_steps], dtype=np.int64
            ),
            "local_max_num_seqs": np.asarray(
                [self.local_max_num_seqs], dtype=np.int64
            ),
            "configured_max_num_batched_tokens": np.asarray(
                [self.configured_max_num_batched_tokens], dtype=np.int64
            ),
            "scheduler_max_num_seqs": np.asarray(
                [self.scheduler_max_num_seqs], dtype=np.int64
            ),
            "scheduler_max_num_batched_tokens": np.asarray(
                [self.scheduler_max_num_batched_tokens], dtype=np.int64
            ),
            "scheduler_max_num_scheduled_tokens": np.asarray(
                [self.scheduler_max_num_scheduled_tokens], dtype=np.int64
            ),
            "speculative_max_num_new_slots_for_drafting": np.asarray(
                [self.speculative_max_num_new_slots_for_drafting],
                dtype=np.int64,
            ),
        }


@dataclass
class StepAccumulator:
    step_start_time_ms: float = 0.0
    first_ep_collective_seq_id: int | None = None
    last_ep_collective_seq_id: int | None = None
    num_ep_collectives: int = 0
    attention_ms: float = 0.0
    routing_ms: float = 0.0
    prepare_ms: float = 0.0
    finalize_ms: float = 0.0
    ffn_ms: float = 0.0
    draft_ms: float = 0.0
    layer_ffn_ms: dict[int, float] = field(default_factory=dict)
    layer_stack: list[int] = field(default_factory=list)
    events: list[dict[str, float | str]] = field(default_factory=list)


@dataclass
class WorkerInstrumentationState:
    enabled: bool = False
    pending_step_records: deque[dict[str, Any]] = field(default_factory=deque)
    current_step: StepAccumulator | None = None
    draft_measure_depth: int = 0
    enter_step_logs: int = 0
    queued_step_logs: int = 0
    next_step_index: int = 0
    next_ep_collective_seq_id: int = 0


_WORKER_STATE = WorkerInstrumentationState()
_ORIGINAL_WORKER_EXECUTE_MODEL = None
_ORIGINAL_ROUTER_SELECT_EXPERTS = None
_ORIGINAL_QWEN2_MLP_FORWARD = None
_ORIGINAL_QWEN3_MLP_FORWARD = None
_ORIGINAL_QWEN3_SPARSE_MOE_FORWARD = None
_ORIGINAL_QWEN3_NEXT_ATTN_FORWARD = None
_ORIGINAL_QWEN_GDN_FORWARD = None
_ORIGINAL_MODULAR_PREPARE = None
_ORIGINAL_MODULAR_FUSED_EXPERTS = None
_ORIGINAL_MODULAR_FINALIZE = None
_ORIGINAL_MONOLITHIC_APPLY = None
_ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT = None
_ORIGINAL_EAGLE_SPECULATOR_PROPOSE = None
_ORIGINAL_SPEC_DECODE_BASE_PROPOSER_PROPOSE = None
_ORIGINAL_GPU_MODEL_RUNNER_RECORD_FUNCTION = None


def condition_name(batch_size: int, draft_length: int) -> str:
    return f"batch_{batch_size:03d}_draft_{draft_length:02d}"


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path("results") / f"qwen3_6_mtp_dp_ep_{timestamp}"


def ensure_collect_dirs(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "raw": raw_dir}


def prompt_cache_path(output_dir: Path) -> Path:
    return output_dir / "prompt_cache.json"


def save_run_metadata(output_dir: Path, args: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "batch_sizes": list(args.batch_sizes),
        "draft_lengths": list(args.draft_lengths),
        "data_parallel_size": args.data_parallel_size,
        "batch_size_scope": "global",
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "layers": list(args.layers),
        "num_experts": args.num_experts,
        "enforce_eager": args.enforce_eager,
        "warmup_rounds": args.warmup_rounds,
        "trace_steps_per_rank": args.trace_steps_per_rank,
        "enable_chunked_prefill": True,
        "mixed_step_policy": "include_all_global_barriers",
        "tpot_definition": TPOT_DEFINITION,
        "vllm_enable_v1_multiprocessing": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)


def save_collect_manifest(
    output_dir: Path,
    args: Any,
    condition_summaries: list[CollectedConditionSummary],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "batch_sizes": list(args.batch_sizes),
        "draft_lengths": list(args.draft_lengths),
        "data_parallel_size": args.data_parallel_size,
        "batch_size_scope": "global",
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "layers": list(args.layers),
        "num_experts": args.num_experts,
        "warmup_rounds": args.warmup_rounds,
        "trace_steps_per_rank": args.trace_steps_per_rank,
        "enable_chunked_prefill": True,
        "mixed_step_policy": "include_all_global_barriers",
        "tpot_definition": TPOT_DEFINITION,
        "conditions": [
            {
                "batch_size": summary.batch_size,
                "draft_length": summary.draft_length,
                "raw_path": summary.raw_path,
                "local_max_num_seqs": summary.local_max_num_seqs,
                "configured_max_num_batched_tokens": (
                    summary.configured_max_num_batched_tokens
                ),
                "scheduler_max_num_seqs": summary.scheduler_max_num_seqs,
                "scheduler_max_num_batched_tokens": (
                    summary.scheduler_max_num_batched_tokens
                ),
                "scheduler_max_num_scheduled_tokens": (
                    summary.scheduler_max_num_scheduled_tokens
                ),
                "speculative_max_num_new_slots_for_drafting": (
                    summary.speculative_max_num_new_slots_for_drafting
                ),
                "condition_latency_ms": summary.condition_latency_ms,
                "decode_time_total_ms": summary.decode_time_total_ms,
                "num_output_tokens_total": summary.num_output_tokens_total,
                "num_generation_tokens_total": (
                    summary.num_generation_tokens_total
                ),
                "num_output_tokens_excl_first_total": (
                    summary.num_output_tokens_excl_first_total
                ),
                "tpot_ms": summary.tpot_ms,
                "decode_throughput_tok_s": summary.decode_throughput_tok_s,
                "num_forward_steps_total": summary.num_forward_steps_total,
                "num_captured_steps": summary.num_captured_steps,
                "num_global_candidate_steps": summary.num_global_candidate_steps,
                "num_global_captured_steps": summary.num_global_captured_steps,
                "num_dropped_steps": summary.num_dropped_steps,
                "num_prefill_dropped_steps": summary.num_prefill_dropped_steps,
                "num_mixed_dropped_steps": summary.num_mixed_dropped_steps,
                "num_global_prefill_dropped_steps": (
                    summary.num_global_prefill_dropped_steps
                ),
                "num_global_mixed_dropped_steps": (
                    summary.num_global_mixed_dropped_steps
                ),
                "num_global_non_target_dropped_steps": (
                    summary.num_global_non_target_dropped_steps
                ),
            }
            for summary in condition_summaries
        ],
    }
    with (output_dir / "collect_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)


def load_condition_summary(raw_path: Path) -> CollectedConditionSummary:
    with np.load(raw_path, allow_pickle=False) as data:
        def read_optional_int(name: str) -> int:
            if name not in data:
                return -1
            return int(data[name][0])

        batch_size = int(data["batch_size"][0])
        draft_length = int(data["draft_length"][0])
        local_max_num_seqs = read_optional_int("local_max_num_seqs")
        configured_max_num_batched_tokens = read_optional_int(
            "configured_max_num_batched_tokens"
        )
        scheduler_max_num_seqs = read_optional_int("scheduler_max_num_seqs")
        scheduler_max_num_batched_tokens = read_optional_int(
            "scheduler_max_num_batched_tokens"
        )
        scheduler_max_num_scheduled_tokens = read_optional_int(
            "scheduler_max_num_scheduled_tokens"
        )
        speculative_max_num_new_slots_for_drafting = read_optional_int(
            "speculative_max_num_new_slots_for_drafting"
        )
        condition_latency_ms = float(data["condition_latency_ms"][0])
        decode_time_total_ms = float(data["decode_time_total_ms"][0])
        num_output_tokens_total = int(data["num_output_tokens_total"][0])
        num_generation_tokens_total = int(data["num_generation_tokens_total"][0])
        num_output_tokens_excl_first_total = int(
            data["num_output_tokens_excl_first_total"][0]
        )
        tpot_ms = float(data["tpot_ms"][0])
        decode_throughput_tok_s = float(data["decode_throughput_tok_s"][0])
        num_forward_steps_total = int(data["num_forward_steps_total"][0])
        num_captured_steps = int(data["num_captured_steps"][0])
        num_global_candidate_steps = int(data["num_global_candidate_steps"][0])
        num_global_captured_steps = int(data["num_global_captured_steps"][0])
        num_dropped_steps = int(data["num_dropped_steps"][0])
        num_prefill_dropped_steps = int(data["num_prefill_dropped_steps"][0])
        num_mixed_dropped_steps = int(data["num_mixed_dropped_steps"][0])
        num_global_prefill_dropped_steps = int(
            data["num_global_prefill_dropped_steps"][0]
        )
        num_global_mixed_dropped_steps = int(
            data["num_global_mixed_dropped_steps"][0]
        )
        num_global_non_target_dropped_steps = int(
            data["num_global_non_target_dropped_steps"][0]
        )
    return CollectedConditionSummary(
        batch_size=batch_size,
        draft_length=draft_length,
        raw_path=str(raw_path.relative_to(raw_path.parent.parent)),
        local_max_num_seqs=local_max_num_seqs,
        configured_max_num_batched_tokens=configured_max_num_batched_tokens,
        scheduler_max_num_seqs=scheduler_max_num_seqs,
        scheduler_max_num_batched_tokens=scheduler_max_num_batched_tokens,
        scheduler_max_num_scheduled_tokens=scheduler_max_num_scheduled_tokens,
        speculative_max_num_new_slots_for_drafting=(
            speculative_max_num_new_slots_for_drafting
        ),
        condition_latency_ms=condition_latency_ms,
        decode_time_total_ms=decode_time_total_ms,
        num_output_tokens_total=num_output_tokens_total,
        num_generation_tokens_total=num_generation_tokens_total,
        num_output_tokens_excl_first_total=num_output_tokens_excl_first_total,
        tpot_ms=tpot_ms,
        decode_throughput_tok_s=decode_throughput_tok_s,
        num_forward_steps_total=num_forward_steps_total,
        num_captured_steps=num_captured_steps,
        num_global_candidate_steps=num_global_candidate_steps,
        num_global_captured_steps=num_global_captured_steps,
        num_dropped_steps=num_dropped_steps,
        num_prefill_dropped_steps=num_prefill_dropped_steps,
        num_mixed_dropped_steps=num_mixed_dropped_steps,
        num_global_prefill_dropped_steps=num_global_prefill_dropped_steps,
        num_global_mixed_dropped_steps=num_global_mixed_dropped_steps,
        num_global_non_target_dropped_steps=num_global_non_target_dropped_steps,
    )


def load_prompt_items(args: Namespace) -> list[dict[str, list[int]]]:
    cache_path = getattr(args, "prompt_cache_path", None)
    if cache_path is not None:
        return load_prompt_items_from_cache(Path(cache_path))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
    )
    selected_indices = select_dataset_indices(args.num_samples, len(dataset))
    prompt_items: list[dict[str, list[int]]] = []
    for item in dataset.select(selected_indices.tolist()):
        message = item["instruction"].strip()
        input_code = item["input"].strip()
        if input_code:
            message = f"{message}\n\nInput Code:\n{input_code}"
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        ).input_ids
        prompt_items.append({"prompt_token_ids": token_ids})
    return prompt_items


def save_prompt_items_cache(
    prompt_items: list[dict[str, list[int]]],
    cache_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_token_ids": [item["prompt_token_ids"] for item in prompt_items],
    }
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp)


def load_prompt_items_from_cache(cache_path: Path) -> list[dict[str, list[int]]]:
    with cache_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    prompt_token_ids = payload["prompt_token_ids"]
    return [{"prompt_token_ids": list(token_ids)} for token_ids in prompt_token_ids]


def prepare_prompt_cache(args: Namespace, output_dir: Path) -> Path:
    cache_path = prompt_cache_path(output_dir)
    if cache_path.exists():
        return cache_path
    prompt_items = load_prompt_items(args)
    save_prompt_items_cache(prompt_items, cache_path)
    return cache_path


def validate_parallel_config(args: Namespace) -> None:
    if args.tensor_parallel_size != 1:
        raise ValueError(
            "This experiment only supports DP+EP with tensor_parallel_size=1."
        )
    if args.data_parallel_size < 1:
        raise ValueError("data_parallel_size must be >= 1.")


def get_local_max_num_seqs(batch_size: int, data_parallel_size: int) -> int:
    return math.ceil(batch_size / data_parallel_size)


def get_configured_max_num_batched_tokens(
    local_max_num_seqs: int,
    draft_length: int,
    max_num_batched_tokens_override: int | None = None,
) -> int:
    if max_num_batched_tokens_override is not None:
        if max_num_batched_tokens_override <= 0:
            raise ValueError("max_num_batched_tokens override must be positive.")
        return max_num_batched_tokens_override
    return DEFAULT_MAX_NUM_BATCHED_TOKENS


def _get_int_config_attr(config: Any | None, name: str) -> int:
    if config is None:
        return -1
    value = getattr(config, name, -1)
    if value is None:
        return -1
    return int(value)


def _snapshot_scheduler_capacity_config(
    llm: Any,
    *,
    local_max_num_seqs: int,
    configured_max_num_batched_tokens: int,
) -> SchedulerCapacityConfig:
    vllm_config = llm.llm_engine.vllm_config
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    return SchedulerCapacityConfig(
        local_max_num_seqs=local_max_num_seqs,
        configured_max_num_batched_tokens=configured_max_num_batched_tokens,
        scheduler_max_num_seqs=_get_int_config_attr(
            scheduler_config,
            "max_num_seqs",
        ),
        scheduler_max_num_batched_tokens=_get_int_config_attr(
            scheduler_config,
            "max_num_batched_tokens",
        ),
        scheduler_max_num_scheduled_tokens=_get_int_config_attr(
            scheduler_config,
            "max_num_scheduled_tokens",
        ),
        speculative_max_num_new_slots_for_drafting=_get_int_config_attr(
            speculative_config,
            "max_num_new_slots_for_drafting",
        ),
    )


def create_llm(args: Namespace, batch_size: int, draft_length: int):
    from vllm import LLM

    local_max_num_seqs = get_local_max_num_seqs(
        batch_size,
        args.data_parallel_size,
    )
    configured_max_num_batched_tokens = get_configured_max_num_batched_tokens(
        local_max_num_seqs,
        draft_length,
        getattr(args, "max_num_batched_tokens", None),
    )
    speculative_config = None
    if draft_length > 0:
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": draft_length,
            "max_model_len": args.max_model_len,
        }

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        async_scheduling=False,
        enable_expert_parallel=True,
        enable_return_routed_experts=True,
        enable_eplb=False,
        enable_chunked_prefill=True,
        max_model_len=args.max_model_len,
        max_num_seqs=local_max_num_seqs,
        max_num_batched_tokens=configured_max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
    )
    scheduler_capacity_config = _snapshot_scheduler_capacity_config(
        llm,
        local_max_num_seqs=local_max_num_seqs,
        configured_max_num_batched_tokens=configured_max_num_batched_tokens,
    )
    print(
        "[collect-rank] scheduler_capacity_config="
        f"{json.dumps(scheduler_capacity_config.to_log_dict(), sort_keys=True)}",
        flush=True,
    )
    setattr(
        llm,
        _SCHEDULER_CAPACITY_CONFIG_ATTR,
        scheduler_capacity_config,
    )
    logger = FinishedRequestStatsLogger(llm.llm_engine.vllm_config)
    llm.llm_engine.logger_manager.stat_loggers.append(logger)
    setattr(llm, _FINISHED_REQUEST_STATS_LOGGER_ATTR, logger)
    return llm


def get_scheduler_capacity_config(llm: Any) -> SchedulerCapacityConfig:
    config = getattr(llm, _SCHEDULER_CAPACITY_CONFIG_ATTR, None)
    if config is None:
        raise RuntimeError("Scheduler capacity config was not attached to LLM.")
    return config


def get_finished_request_stats_logger(llm: Any) -> FinishedRequestStatsLogger:
    logger = getattr(llm, _FINISHED_REQUEST_STATS_LOGGER_ATTR, None)
    if logger is None:
        raise RuntimeError("Finished-request stats logger was not attached to LLM.")
    return logger


def get_inproc_handles(llm: Any) -> tuple[Any, Any]:
    engine_core_client = llm.llm_engine.engine_core
    if not hasattr(engine_core_client, "engine_core"):
        raise RuntimeError(
            "This experiment requires in-proc V1 execution. "
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before running."
        )
    engine_core = engine_core_client.engine_core
    return engine_core.scheduler, engine_core.model_executor


def _synchronize_device() -> None:
    from vllm.platforms import current_platform

    synchronize = current_platform.synchronize
    if synchronize is not None:
        synchronize()


def _measure_worker_section(
    label: str,
    fn: Callable,
    *args: Any,
    add_to_step_ffn: bool = True,
    **kwargs: Any,
):
    current_step = _WORKER_STATE.current_step
    if not _WORKER_STATE.enabled or current_step is None:
        return fn(*args, **kwargs)
    is_outer_draft = False
    if label == "draft":
        if _WORKER_STATE.draft_measure_depth > 0:
            return fn(*args, **kwargs)
        _WORKER_STATE.draft_measure_depth += 1
        is_outer_draft = True

    try:
        ep_collective_seq_id = None
        if label in ("prepare", "finalize"):
            ep_collective_seq_id = _WORKER_STATE.next_ep_collective_seq_id
            _WORKER_STATE.next_ep_collective_seq_id += 1
            if current_step.first_ep_collective_seq_id is None:
                current_step.first_ep_collective_seq_id = ep_collective_seq_id
            current_step.last_ep_collective_seq_id = ep_collective_seq_id
            current_step.num_ep_collectives += 1

        _synchronize_device()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        _synchronize_device()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        start_ms = start * 1000.0 - current_step.step_start_time_ms
        end_ms = end * 1000.0 - current_step.step_start_time_ms
        current_step.events.append(
            {
                "label": label,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": elapsed_ms,
                "ep_collective_seq_id": ep_collective_seq_id,
            }
        )
        if label == "prepare":
            current_step.prepare_ms += elapsed_ms
        elif label == "finalize":
            current_step.finalize_ms += elapsed_ms
        elif label == "attention":
            current_step.attention_ms += elapsed_ms
        elif label == "routing":
            current_step.routing_ms += elapsed_ms
        elif label == "ffn":
            if add_to_step_ffn:
                current_step.ffn_ms += elapsed_ms
            if current_step.layer_stack:
                layer_idx = current_step.layer_stack[-1]
                current_step.layer_ffn_ms[layer_idx] = (
                    current_step.layer_ffn_ms.get(layer_idx, 0.0) + elapsed_ms
                )
        elif label == "draft":
            current_step.draft_ms += elapsed_ms
        else:
            raise ValueError(f"Unknown measured label: {label}")
        return result
    finally:
        if is_outer_draft:
            _WORKER_STATE.draft_measure_depth -= 1


def _measure_draft_context(original_context: Any):
    @contextmanager
    def measured_context():
        current_step = _WORKER_STATE.current_step
        pending_record = (
            _WORKER_STATE.pending_step_records[-1]
            if _WORKER_STATE.pending_step_records
            else None
        )
        if (
            not _WORKER_STATE.enabled
            or (current_step is None and pending_record is None)
            or _WORKER_STATE.draft_measure_depth > 0
        ):
            with original_context:
                yield
            return

        _WORKER_STATE.draft_measure_depth += 1
        _synchronize_device()
        start = time.perf_counter()
        try:
            with original_context:
                yield
        finally:
            _synchronize_device()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            if current_step is not None:
                start_ms = start * 1000.0 - current_step.step_start_time_ms
                end_ms = end * 1000.0 - current_step.step_start_time_ms
                current_step.events.append(
                    {
                        "label": "draft",
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": elapsed_ms,
                        "ep_collective_seq_id": None,
                    }
                )
                current_step.draft_ms += elapsed_ms
            elif pending_record is not None:
                timing = pending_record["timing"]
                trace = pending_record["trace"]
                timing["draft_ms"] = float(timing.get("draft_ms", 0.0)) + elapsed_ms
                step_start_time_ms = float(
                    trace.get("step_start_time_ms", start * 1000.0)
                )
                trace["events"].append(
                    {
                        "label": "draft",
                        "start_ms": start * 1000.0 - step_start_time_ms,
                        "end_ms": end * 1000.0 - step_start_time_ms,
                        "duration_ms": elapsed_ms,
                        "ep_collective_seq_id": None,
                    }
                )
            _WORKER_STATE.draft_measure_depth -= 1

    return measured_context()


def _extract_layer_index_from_module(module: Any) -> int | None:
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is not None:
        return int(layer_idx)
    layer_name = getattr(module, "layer_name", None)
    if layer_name is None:
        experts = getattr(module, "experts", None)
        layer_name = getattr(experts, "layer_name", None)
    if not layer_name:
        return None
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        return int(extract_layer_index(str(layer_name)))
    except AssertionError:
        return None


def _push_current_layer(layer_idx: int | None) -> bool:
    current_step = _WORKER_STATE.current_step
    if not _WORKER_STATE.enabled or current_step is None or layer_idx is None:
        return False
    current_step.layer_stack.append(layer_idx)
    return True


def _pop_current_layer(pushed: bool) -> None:
    current_step = _WORKER_STATE.current_step
    if pushed and current_step is not None:
        current_step.layer_stack.pop()


def _extract_input_batch_metadata(input_batch: Any) -> dict[str, Any]:
    req_ids = list(getattr(input_batch, "req_ids", ()))
    if hasattr(input_batch, "is_prefilling_np"):
        has_prefill = bool(np.any(input_batch.is_prefilling_np))
    elif hasattr(input_batch, "num_computed_tokens_cpu") and hasattr(
        input_batch, "num_prompt_tokens"
    ):
        num_reqs = int(getattr(input_batch, "num_reqs", len(req_ids)))
        has_prefill = bool(
            np.any(
                input_batch.num_computed_tokens_cpu[:num_reqs]
                < input_batch.num_prompt_tokens[:num_reqs]
            )
        )
    else:
        has_prefill = False
    return {
        "req_ids": req_ids,
        "has_prefill": has_prefill,
    }


def _extract_worker_step_metadata(worker: Any) -> dict[str, Any]:
    model_runner = getattr(worker, "model_runner", None)
    execute_model_state = getattr(model_runner, "execute_model_state", None)
    input_batch = getattr(execute_model_state, "input_batch", None)
    if input_batch is not None:
        return _extract_input_batch_metadata(input_batch)

    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is not None:
        return _extract_input_batch_metadata(input_batch)

    return {
        "req_ids": [],
        "has_prefill": False,
    }


def _install_worker_hooks() -> None:
    global _ORIGINAL_WORKER_EXECUTE_MODEL
    global _ORIGINAL_ROUTER_SELECT_EXPERTS
    global _ORIGINAL_QWEN2_MLP_FORWARD
    global _ORIGINAL_QWEN3_MLP_FORWARD
    global _ORIGINAL_QWEN3_SPARSE_MOE_FORWARD
    global _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD
    global _ORIGINAL_QWEN_GDN_FORWARD
    global _ORIGINAL_MODULAR_PREPARE
    global _ORIGINAL_MODULAR_FUSED_EXPERTS
    global _ORIGINAL_MODULAR_FINALIZE
    global _ORIGINAL_MONOLITHIC_APPLY
    global _ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT
    global _ORIGINAL_EAGLE_SPECULATOR_PROPOSE
    global _ORIGINAL_SPEC_DECODE_BASE_PROPOSER_PROPOSE
    global _ORIGINAL_GPU_MODEL_RUNNER_RECORD_FUNCTION

    if _ORIGINAL_WORKER_EXECUTE_MODEL is not None:
        return

    import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
        FusedMoEKernelMonolithicImpl,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeMLP,
        Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    _ORIGINAL_WORKER_EXECUTE_MODEL = Worker.execute_model
    _ORIGINAL_ROUTER_SELECT_EXPERTS = BaseRouter.select_experts
    _ORIGINAL_QWEN2_MLP_FORWARD = Qwen2MoeMLP.forward
    _ORIGINAL_QWEN3_MLP_FORWARD = Qwen3MoeMLP.forward
    _ORIGINAL_QWEN3_SPARSE_MOE_FORWARD = Qwen3MoeSparseMoeBlock.forward
    _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD = Qwen3NextAttention.forward
    _ORIGINAL_QWEN_GDN_FORWARD = QwenGatedDeltaNetAttention.forward
    _ORIGINAL_MODULAR_PREPARE = FusedMoEKernelModularImpl._prepare
    _ORIGINAL_MODULAR_FUSED_EXPERTS = FusedMoEKernelModularImpl._fused_experts
    _ORIGINAL_MODULAR_FINALIZE = FusedMoEKernelModularImpl._finalize
    _ORIGINAL_MONOLITHIC_APPLY = FusedMoEKernelMonolithicImpl.apply
    _ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT = GPUModelRunner.propose_draft_token_ids
    _ORIGINAL_EAGLE_SPECULATOR_PROPOSE = EagleSpeculator.propose
    _ORIGINAL_SPEC_DECODE_BASE_PROPOSER_PROPOSE = SpecDecodeBaseProposer.propose
    _ORIGINAL_GPU_MODEL_RUNNER_RECORD_FUNCTION = (
        gpu_model_runner_module.record_function_or_nullcontext
    )

    def patched_worker_execute_model(self, scheduler_output):
        if (
            not _WORKER_STATE.enabled
            or scheduler_output.total_num_scheduled_tokens <= 0
        ):
            return _ORIGINAL_WORKER_EXECUTE_MODEL(self, scheduler_output)

        if _WORKER_STATE.enter_step_logs < 3:
            print(
                "[worker_timing] execute begin "
                f"scheduled_tokens={scheduler_output.total_num_scheduled_tokens}",
                flush=True,
            )
            _WORKER_STATE.enter_step_logs += 1

        step_index = _WORKER_STATE.next_step_index
        _synchronize_device()
        start = time.perf_counter()
        _WORKER_STATE.current_step = StepAccumulator(
            step_start_time_ms=start * 1000.0
        )
        try:
            output = _ORIGINAL_WORKER_EXECUTE_MODEL(self, scheduler_output)
            _synchronize_device()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            assert _WORKER_STATE.current_step is not None
            _WORKER_STATE.pending_step_records.append(
                {
                    "timing": {
                        "total_ms": elapsed_ms,
                        "attention_ms": _WORKER_STATE.current_step.attention_ms,
                        "routing_ms": _WORKER_STATE.current_step.routing_ms,
                        "prepare_ms": _WORKER_STATE.current_step.prepare_ms,
                        "finalize_ms": _WORKER_STATE.current_step.finalize_ms,
                        "ffn_ms": _WORKER_STATE.current_step.ffn_ms,
                        "draft_ms": _WORKER_STATE.current_step.draft_ms,
                    },
                    "metadata": _extract_worker_step_metadata(self),
                    "trace": {
                        "step_index": step_index,
                        "first_ep_collective_seq_id": (
                            _WORKER_STATE.current_step.first_ep_collective_seq_id
                        ),
                        "last_ep_collective_seq_id": (
                            _WORKER_STATE.current_step.last_ep_collective_seq_id
                        ),
                        "num_ep_collectives": (
                            _WORKER_STATE.current_step.num_ep_collectives
                        ),
                        "step_start_time_ms": (
                            _WORKER_STATE.current_step.step_start_time_ms
                        ),
                        "step_end_time_ms": end * 1000.0,
                        "events": list(_WORKER_STATE.current_step.events),
                        "layer_ffn_ms": dict(_WORKER_STATE.current_step.layer_ffn_ms),
                    },
                }
            )
            _WORKER_STATE.next_step_index += 1
            if _WORKER_STATE.queued_step_logs < 3:
                print(
                    "[worker_timing] queued "
                    f"total_ms={elapsed_ms:.3f} "
                    f"attention_ms={_WORKER_STATE.current_step.attention_ms:.3f} "
                    f"routing_ms={_WORKER_STATE.current_step.routing_ms:.3f} "
                    f"prepare_ms={_WORKER_STATE.current_step.prepare_ms:.3f} "
                    f"finalize_ms={_WORKER_STATE.current_step.finalize_ms:.3f} "
                    f"ffn_ms={_WORKER_STATE.current_step.ffn_ms:.3f} "
                    f"draft_ms={_WORKER_STATE.current_step.draft_ms:.3f}",
                    flush=True,
                )
                _WORKER_STATE.queued_step_logs += 1
            return output
        finally:
            _WORKER_STATE.current_step = None

    def patched_router_select_experts(self, *args, **kwargs):
        return _measure_worker_section(
            "routing",
            _ORIGINAL_ROUTER_SELECT_EXPERTS,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen2_mlp_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "ffn",
            _ORIGINAL_QWEN2_MLP_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen3_mlp_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "ffn",
            _ORIGINAL_QWEN3_MLP_FORWARD,
            self,
            *args,
            add_to_step_ffn=False,
            **kwargs,
        )

    def patched_qwen3_sparse_moe_forward(self, *args, **kwargs):
        pushed = _push_current_layer(_extract_layer_index_from_module(self))
        try:
            return _ORIGINAL_QWEN3_SPARSE_MOE_FORWARD(self, *args, **kwargs)
        finally:
            _pop_current_layer(pushed)

    def patched_qwen3_next_attention_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "attention",
            _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen_gdn_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "attention",
            _ORIGINAL_QWEN_GDN_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_modular_prepare(self, *args, **kwargs):
        return _measure_worker_section(
            "prepare",
            _ORIGINAL_MODULAR_PREPARE,
            self,
            *args,
            **kwargs,
        )

    def patched_modular_fused_experts(self, *args, **kwargs):
        return _measure_worker_section(
            "ffn",
            _ORIGINAL_MODULAR_FUSED_EXPERTS,
            self,
            *args,
            **kwargs,
        )

    def patched_modular_finalize(self, *args, **kwargs):
        return _measure_worker_section(
            "finalize",
            _ORIGINAL_MODULAR_FINALIZE,
            self,
            *args,
            **kwargs,
        )

    def patched_monolithic_apply(
        self,
        hidden_states,
        w1,
        w2,
        router_logits,
        activation,
        global_num_experts,
        expert_map,
        apply_router_weight_on_input,
        num_expert_group=None,
        e_score_correction_bias=None,
        routed_scaling_factor=None,
        topk_group=None,
    ):
        if not _WORKER_STATE.enabled or _WORKER_STATE.current_step is None:
            return _ORIGINAL_MONOLITHIC_APPLY(
                self,
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                topk_group=topk_group,
            )

        a1q, a1q_scale, router_logits = _measure_worker_section(
            "prepare",
            self.prepare_finalize.prepare,
            hidden_states,
            router_logits=router_logits,
            quant_config=self.fused_experts.quant_config,
            defer_input_quant=self.fused_experts.expects_unquantized_inputs,
        )
        fused_out = _measure_worker_section(
            "ffn",
            self.fused_experts.apply,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            a1q_scale=a1q_scale,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
        )
        return _measure_worker_section(
            "finalize",
            self.prepare_finalize.finalize,
            fused_out,
        )

    def patched_propose_draft_token_ids(self, *args, **kwargs):
        return _measure_worker_section(
            "draft",
            _ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT,
            self,
            *args,
            **kwargs,
        )

    def patched_eagle_speculator_propose(self, *args, **kwargs):
        return _measure_worker_section(
            "draft",
            _ORIGINAL_EAGLE_SPECULATOR_PROPOSE,
            self,
            *args,
            **kwargs,
        )

    def patched_spec_decode_base_proposer_propose(self, *args, **kwargs):
        return _measure_worker_section(
            "draft",
            _ORIGINAL_SPEC_DECODE_BASE_PROPOSER_PROPOSE,
            self,
            *args,
            **kwargs,
        )

    def patched_record_function_or_nullcontext(name: str):
        original_context = _ORIGINAL_GPU_MODEL_RUNNER_RECORD_FUNCTION(name)
        if name != "gpu_model_runner: draft":
            return original_context
        return _measure_draft_context(original_context)

    Worker.execute_model = patched_worker_execute_model
    BaseRouter.select_experts = patched_router_select_experts
    Qwen2MoeMLP.forward = patched_qwen2_mlp_forward
    Qwen3MoeMLP.forward = patched_qwen3_mlp_forward
    Qwen3MoeSparseMoeBlock.forward = patched_qwen3_sparse_moe_forward
    Qwen3NextAttention.forward = patched_qwen3_next_attention_forward
    QwenGatedDeltaNetAttention.forward = patched_qwen_gdn_forward
    FusedMoEKernelModularImpl._prepare = patched_modular_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_modular_fused_experts
    FusedMoEKernelModularImpl._finalize = patched_modular_finalize
    FusedMoEKernelMonolithicImpl.apply = patched_monolithic_apply
    GPUModelRunner.propose_draft_token_ids = patched_propose_draft_token_ids
    EagleSpeculator.propose = patched_eagle_speculator_propose
    SpecDecodeBaseProposer.propose = patched_spec_decode_base_proposer_propose
    gpu_model_runner_module.record_function_or_nullcontext = (
        patched_record_function_or_nullcontext
    )


def install_experiment_hooks_worker(worker: Any) -> bool:
    _install_worker_hooks()
    _WORKER_STATE.enabled = False
    _WORKER_STATE.pending_step_records.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.draft_measure_depth = 0
    _WORKER_STATE.enter_step_logs = 0
    _WORKER_STATE.queued_step_logs = 0
    _WORKER_STATE.next_step_index = 0
    _WORKER_STATE.next_ep_collective_seq_id = 0
    return True


def start_condition_collection_worker(worker: Any) -> bool:
    _WORKER_STATE.enabled = True
    _WORKER_STATE.pending_step_records.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.draft_measure_depth = 0
    _WORKER_STATE.enter_step_logs = 0
    _WORKER_STATE.queued_step_logs = 0
    return True


def stop_condition_collection_worker(worker: Any) -> dict[str, int]:
    _WORKER_STATE.enabled = False
    pending = len(_WORKER_STATE.pending_step_records)
    _WORKER_STATE.pending_step_records.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.draft_measure_depth = 0
    return {"pending_timings": pending}


def pop_step_timing_worker(
    worker: Any,
    timeout_s: float = 5.0,
    poll_s: float = 0.01,
) -> dict[str, Any] | None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if _WORKER_STATE.pending_step_records:
            return _WORKER_STATE.pending_step_records.popleft()
        time.sleep(poll_s)
    if _WORKER_STATE.pending_step_records:
        return _WORKER_STATE.pending_step_records.popleft()
    return None


def collect_expert_to_ep_rank_worker(worker: Any) -> np.ndarray:
    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None:
        return np.empty((0,), dtype=np.int64)

    for module in model.modules():
        expert_map = getattr(module, "_expert_map", None)
        if expert_map is None:
            expert_map_manager = getattr(module, "expert_map_manager", None)
            expert_map = getattr(expert_map_manager, "expert_map", None)
        global_num_experts = getattr(module, "global_num_experts", None)
        ep_rank = getattr(module, "ep_rank", None)
        if expert_map is None or global_num_experts is None or ep_rank is None:
            continue
        expert_map_cpu = expert_map.detach().cpu().numpy()
        result = np.full((int(global_num_experts),), -1, dtype=np.int64)
        result[np.asarray(expert_map_cpu[: int(global_num_experts)]) >= 0] = int(
            ep_rank
        )
        return result
    return np.empty((0,), dtype=np.int64)


def _extract_step_phase_boundaries(
    worker_trace: dict[str, Any],
) -> tuple[float, float]:
    step_start_time_ms = float(worker_trace["step_start_time_ms"])
    prepare_events = [
        step_start_time_ms + float(event["start_ms"])
        for event in worker_trace["events"]
        if str(event["label"]) == "prepare"
    ]
    finalize_events = [
        step_start_time_ms + float(event["end_ms"])
        for event in worker_trace["events"]
        if str(event["label"]) == "finalize"
    ]
    if not prepare_events or not finalize_events:
        return float("nan"), float("nan")
    return min(prepare_events), max(finalize_events)


def _close_ffn_component(step_timing: StepTiming, tol_ms: float = 1e-3) -> float:
    unattributed_ms = step_timing.unattributed_ms
    if unattributed_ms < -tol_ms:
        raise RuntimeError(
            "Step timing components exceeded total step time: "
            f"total_ms={step_timing.total_ms:.6f}, "
            f"attention_ms={step_timing.attention_ms:.6f}, "
            f"routing_ms={step_timing.routing_ms:.6f}, "
            f"all2all_ms={step_timing.all2all_ms:.6f}, "
            f"ffn_ms={step_timing.ffn_ms:.6f}, "
            f"unattributed_ms={unattributed_ms:.6f}"
        )
    return step_timing.ffn_ms + max(unattributed_ms, 0.0)


class SchedulerStepRecorder:
    def __init__(
        self,
        scheduler: Any,
        model_executor: Any,
        *,
        use_spec_decode: bool,
        layers: tuple[int, ...],
        num_experts: int,
        expert_to_ep_rank: np.ndarray,
        local_ep_rank: int,
        trace_steps_limit: int = 0,
    ) -> None:
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.use_spec_decode = use_spec_decode
        self.layers = layers
        self.num_experts = num_experts
        self.expert_to_ep_rank = np.asarray(expert_to_ep_rank, dtype=np.int64)
        self.local_ep_rank = local_ep_rank
        self._original_update = None
        self.step_histograms: list[np.ndarray] = []
        self.step_total_tokens: list[int] = []
        self.step_total_ms: list[float] = []
        self.step_attention_ms: list[float] = []
        self.step_routing_ms: list[float] = []
        self.step_prepare_ms: list[float] = []
        self.step_finalize_ms: list[float] = []
        self.step_ffn_ms: list[float] = []
        self.step_kinds: list[str] = []
        self.step_indices: list[int] = []
        self.step_start_time_ms: list[float] = []
        self.step_end_time_ms: list[float] = []
        self.prepare_start_time_ms: list[float] = []
        self.finalize_end_time_ms: list[float] = []
        self.candidate_first_ep_collective_seq_ids: list[int] = []
        self.candidate_last_ep_collective_seq_ids: list[int] = []
        self.candidate_num_ep_collectives: list[int] = []
        self.candidate_step_kinds: list[str] = []
        self.candidate_drop_reasons: list[str] = []
        self.candidate_step_total_tokens: list[int] = []
        self.candidate_step_total_ms: list[float] = []
        self.candidate_step_draft_ms: list[float] = []
        self.candidate_step_ffn_ms: list[float] = []
        self.candidate_step_histograms: list[np.ndarray] = []
        self.candidate_layer_ffn_ms: list[np.ndarray] = []
        self.candidate_layer_local_routed_tokens: list[np.ndarray] = []
        self.candidate_layer_local_active_experts: list[np.ndarray] = []
        self.num_forward_steps_total = 0
        self.num_dropped_steps = 0
        self.num_prefill_dropped_steps = 0
        self.num_mixed_dropped_steps = 0
        self.debug_update_logs = 0
        self.trace_steps_limit = trace_steps_limit
        self.trace_samples: list[dict[str, Any]] = []

    def _build_layer_ffn_ms(self, worker_records: list[dict[str, Any] | None]):
        values = np.zeros((len(self.layers),), dtype=np.float64)
        layer_to_row = {layer: row for row, layer in enumerate(self.layers)}
        for record in worker_records:
            if record is None:
                continue
            trace = record.get("trace")
            if not trace:
                continue
            for raw_layer, elapsed_ms in trace.get("layer_ffn_ms", {}).items():
                layer = int(raw_layer)
                row = layer_to_row.get(layer)
                if row is not None:
                    values[row] = max(values[row], float(elapsed_ms))
        return values

    def _build_candidate_histogram(
        self,
        model_runner_output: Any,
    ) -> tuple[np.ndarray, int]:
        routed_experts = getattr(model_runner_output, "routed_experts", None)
        if routed_experts is None:
            return (
                np.zeros((len(self.layers), self.num_experts), dtype=np.int64),
                0,
            )
        routing_data = np.asarray(routed_experts.routing_data)
        if routing_data.size == 0:
            return (
                np.zeros((len(self.layers), self.num_experts), dtype=np.int64),
                0,
            )
        histograms = count_layer_expert_histograms(
            routing_data,
            layers=self.layers,
            num_experts=self.num_experts,
        )
        return histograms, int(routing_data.shape[0])

    def _build_local_routed_stats(
        self,
        histograms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.expert_to_ep_rank.shape != (self.num_experts,):
            raise ValueError(
                "expert_to_ep_rank must be shaped as "
                f"{(self.num_experts,)}; got {self.expert_to_ep_rank.shape}."
            )
        local_mask = self.expert_to_ep_rank == self.local_ep_rank
        local_counts = histograms[:, local_mask]
        return (
            np.sum(local_counts, axis=1).astype(np.int64),
            np.count_nonzero(local_counts > 0, axis=1).astype(np.int64),
        )

    def __enter__(self) -> "SchedulerStepRecorder":
        self._original_update = self.scheduler.update_from_output

        def wrapped_update(
            scheduler_self: Any,
            scheduler_output: Any,
            model_runner_output: Any,
        ) -> Any:
            if self.debug_update_logs < 5:
                print(
                    "[recorder] update begin "
                    f"use_spec_decode={self.use_spec_decode} "
                    f"num_forward_steps_total={self.num_forward_steps_total}",
                    flush=True,
                )
            worker_records = self.model_executor.collective_rpc(
                pop_step_timing_worker,
                timeout=30,
            )
            step_timing = aggregate_worker_step_timings(
                [
                    None if record is None else record["timing"]
                    for record in worker_records
                ]
            )
            draft_ms = max(
                (
                    float(record["timing"].get("draft_ms", 0.0))
                    for record in worker_records
                    if record is not None
                ),
                default=0.0,
            )
            layer_ffn_ms = self._build_layer_ffn_ms(worker_records)
            worker_metadata = next(
                (
                    record["metadata"]
                    for record in worker_records
                    if record is not None and record.get("metadata")
                ),
                None,
            )
            worker_trace = next(
                (
                    record["trace"]
                    for record in worker_records
                    if record is not None and record.get("trace") is not None
                ),
                None,
            )
            self.num_forward_steps_total += 1

            capture_decision = classify_step_capture(
                scheduler_output,
                model_runner_output,
                worker_step_metadata=worker_metadata,
                use_spec_decode=self.use_spec_decode,
            )
            captured_step = capture_decision.captured_step
            ffn_ms = float(np.sum(layer_ffn_ms))
            if ffn_ms <= 0.0:
                ffn_ms = _close_ffn_component(step_timing)
            first_ep_collective_seq_id = -1
            last_ep_collective_seq_id = -1
            num_ep_collectives = 0
            if worker_trace is not None:
                raw_ep_seq = worker_trace.get("first_ep_collective_seq_id")
                first_ep_collective_seq_id = (
                    -1 if raw_ep_seq is None else int(raw_ep_seq)
                )
                raw_last_ep_seq = worker_trace.get("last_ep_collective_seq_id")
                last_ep_collective_seq_id = (
                    -1 if raw_last_ep_seq is None else int(raw_last_ep_seq)
                )
                num_ep_collectives = int(worker_trace.get("num_ep_collectives", 0))
            candidate_histogram, candidate_total_tokens = (
                self._build_candidate_histogram(model_runner_output)
            )
            local_routed_tokens, local_active_experts = (
                self._build_local_routed_stats(candidate_histogram)
            )

            self.candidate_first_ep_collective_seq_ids.append(
                first_ep_collective_seq_id
            )
            self.candidate_last_ep_collective_seq_ids.append(
                last_ep_collective_seq_id
            )
            self.candidate_num_ep_collectives.append(num_ep_collectives)
            self.candidate_step_kinds.append(capture_decision.local_step_kind)
            self.candidate_drop_reasons.append(capture_decision.drop_reason or "")
            self.candidate_step_total_tokens.append(candidate_total_tokens)
            self.candidate_step_total_ms.append(step_timing.total_ms)
            self.candidate_step_draft_ms.append(draft_ms)
            self.candidate_step_ffn_ms.append(ffn_ms)
            self.candidate_step_histograms.append(candidate_histogram)
            self.candidate_layer_ffn_ms.append(layer_ffn_ms)
            self.candidate_layer_local_routed_tokens.append(local_routed_tokens)
            self.candidate_layer_local_active_experts.append(local_active_experts)

            if captured_step is None:
                self.num_dropped_steps += 1
                if capture_decision.drop_reason == "prefill":
                    self.num_prefill_dropped_steps += 1
                elif capture_decision.drop_reason == "mixed":
                    self.num_mixed_dropped_steps += 1
            else:
                self.step_histograms.append(candidate_histogram)
                self.step_total_tokens.append(captured_step.total_scheduled_tokens)
                self.step_total_ms.append(step_timing.total_ms)
                self.step_attention_ms.append(step_timing.attention_ms)
                self.step_routing_ms.append(step_timing.routing_ms)
                self.step_prepare_ms.append(step_timing.prepare_ms)
                self.step_finalize_ms.append(step_timing.finalize_ms)
                self.step_ffn_ms.append(ffn_ms)
                self.step_kinds.append(captured_step.step_kind)
                if worker_trace is None:
                    prepare_start_ms = float("nan")
                    finalize_end_ms = float("nan")
                    step_index = -1
                    step_start_time_ms = float("nan")
                    step_end_time_ms = float("nan")
                    trace_events = []
                else:
                    prepare_start_ms, finalize_end_ms = _extract_step_phase_boundaries(
                        worker_trace
                    )
                    step_index = int(worker_trace["step_index"])
                    step_start_time_ms = float(worker_trace["step_start_time_ms"])
                    step_end_time_ms = float(worker_trace["step_end_time_ms"])
                    trace_events = list(worker_trace["events"])
                self.step_indices.append(first_ep_collective_seq_id)
                self.step_start_time_ms.append(step_start_time_ms)
                self.step_end_time_ms.append(step_end_time_ms)
                self.prepare_start_time_ms.append(prepare_start_ms)
                self.finalize_end_time_ms.append(finalize_end_ms)
                if (
                    self.trace_steps_limit > 0
                    and len(self.trace_samples) < self.trace_steps_limit
                ):
                    self.trace_samples.append(
                        {
                            "step_index": step_index,
                            "first_ep_collective_seq_id": (
                                first_ep_collective_seq_id
                            ),
                            "step_start_time_ms": step_start_time_ms,
                            "step_end_time_ms": step_end_time_ms,
                            "step_total_ms": float(step_timing.total_ms),
                            "draft_ms": float(draft_ms),
                            "step_kind": captured_step.step_kind,
                            "total_scheduled_tokens": int(
                                captured_step.total_scheduled_tokens
                            ),
                            "request_ids": list(captured_step.request_ids),
                            "events": [
                                {
                                    "label": str(event["label"]),
                                    "start_ms": float(event["start_ms"]),
                                    "end_ms": float(event["end_ms"]),
                                    "duration_ms": float(event["duration_ms"]),
                                    "ep_collective_seq_id": (
                                        None
                                        if event.get("ep_collective_seq_id") is None
                                        else int(event["ep_collective_seq_id"])
                                    ),
                                }
                                for event in trace_events
                            ],
                            "phase_totals_ms": {
                                "attention": float(step_timing.attention_ms),
                                "routing": float(step_timing.routing_ms),
                                "prepare": float(step_timing.prepare_ms),
                                "finalize": float(step_timing.finalize_ms),
                                "ffn": float(ffn_ms),
                                "draft": float(draft_ms),
                            },
                        }
                    )

            if self.debug_update_logs < 5:
                print(
                    "[recorder] update end "
                    f"use_spec_decode={self.use_spec_decode} "
                    f"captured={captured_step is not None} "
                    f"drop_reason={capture_decision.drop_reason or 'none'} "
                    f"step_kind={capture_decision.local_step_kind} "
                    f"ep_seq={first_ep_collective_seq_id} "
                    f"total_ms={step_timing.total_ms:.3f}",
                    flush=True,
                )
                self.debug_update_logs += 1

            return self._original_update(scheduler_output, model_runner_output)

        self.scheduler.update_from_output = MethodType(wrapped_update, self.scheduler)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original_update is not None:
            self.scheduler.update_from_output = self._original_update


def _append_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _empty_step_histograms(args: Namespace) -> np.ndarray:
    return np.empty((0, len(args.layers), args.num_experts), dtype=np.int64)


def _empty_str_array() -> np.ndarray:
    return np.asarray([], dtype=np.str_)


def _empty_int_array() -> np.ndarray:
    return np.asarray([], dtype=np.int64)


def _empty_float_array() -> np.ndarray:
    return np.asarray([], dtype=np.float64)


def _empty_candidate_histograms(args: Namespace) -> np.ndarray:
    return np.empty((0, len(args.layers), args.num_experts), dtype=np.int64)


def _check_pending_timings(
    pending_counts: list[dict[str, int]],
    *,
    batch_size: int,
    draft_length: int,
    round_idx: int,
) -> None:
    pending_values = [item["pending_timings"] for item in pending_counts]
    if not any(pending_values):
        return
    if len(set(pending_values)) != 1:
        raise RuntimeError(
            "Worker timing queues ended with inconsistent leftover counts: "
            f"{pending_counts}"
        )
    print(
        "[collect-rank] warning leftover worker timings were discarded "
        f"batch_size={batch_size} draft_length={draft_length} "
        f"round={round_idx} pending_per_worker={pending_values[0]}",
        flush=True,
    )


def _run_recorded_round(
    llm: Any,
    scheduler: Any,
    model_executor: Any,
    sampling_params: Any,
    prompt_batch: list[dict[str, list[int]]],
    *,
    batch_size: int,
    draft_length: int,
    round_idx: int,
    use_spec_decode: bool,
    layers: tuple[int, ...],
    num_experts: int,
    expert_to_ep_rank: np.ndarray,
    local_ep_rank: int,
    trace_steps_limit: int,
) -> tuple[Any, SchedulerStepRecorder, float]:
    model_executor.collective_rpc(start_condition_collection_worker, timeout=30)
    try:
        with SchedulerStepRecorder(
            scheduler,
            model_executor,
            use_spec_decode=use_spec_decode,
            layers=layers,
            num_experts=num_experts,
            expert_to_ep_rank=expert_to_ep_rank,
            local_ep_rank=local_ep_rank,
            trace_steps_limit=trace_steps_limit,
        ) as recorder:
            start = time.perf_counter()
            outputs = llm.generate(
                prompt_batch,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            round_latency_ms = (time.perf_counter() - start) * 1000.0
    finally:
        pending_counts = model_executor.collective_rpc(stop_condition_collection_worker)
        _check_pending_timings(
            pending_counts,
            batch_size=batch_size,
            draft_length=draft_length,
            round_idx=round_idx,
        )
    return outputs, recorder, round_latency_ms


def _local_prompt_batch(
    prompt_items: list[dict[str, list[int]]],
    indices: np.ndarray,
) -> list[dict[str, list[int]]]:
    return [prompt_items[int(idx)] for idx in indices.tolist()]


def collect_condition_for_rank(args: Namespace) -> RankConditionData:
    from vllm import SamplingParams

    validate_parallel_config(args)
    prompt_items = load_prompt_items(args)
    if not prompt_items:
        raise RuntimeError("Prompt cache is empty.")

    dp_rank = args.dp_rank
    print(
        f"[collect-rank] start dp_rank={dp_rank} batch_size={args.batch_size} "
        f"draft_length={args.draft_length}",
        flush=True,
    )

    os.environ["VLLM_DP_RANK"] = str(args.dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(args.dp_local_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_IP"] = args.dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_master_port)

    llm = create_llm(args, args.batch_size, args.draft_length)
    scheduler_capacity_config = get_scheduler_capacity_config(llm)
    scheduler, model_executor = get_inproc_handles(llm)
    model_executor.collective_rpc(install_experiment_hooks_worker, timeout=30)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    use_spec_decode = args.draft_length > 0
    total_rounds = num_condition_rounds(args.num_samples, args.batch_size)

    warmup_indices = next(
        (
            shard_global_batch_indices(
                num_samples=args.num_samples,
                global_batch_size=args.batch_size,
                round_idx=round_idx,
                dp_size=args.data_parallel_size,
                dp_rank=dp_rank,
            )
            for round_idx in range(total_rounds)
            if len(
                shard_global_batch_indices(
                    num_samples=args.num_samples,
                    global_batch_size=args.batch_size,
                    round_idx=round_idx,
                    dp_size=args.data_parallel_size,
                    dp_rank=dp_rank,
                )
            )
            > 0
        ),
        np.empty((0,), dtype=np.int64),
    )
    warmup_batch = (
        _local_prompt_batch(prompt_items, warmup_indices)
        if len(warmup_indices) > 0
        else [prompt_items[0]]
    )

    if args.warmup_rounds > 0:
        for _ in range(args.warmup_rounds):
            llm.generate(warmup_batch, sampling_params=sampling_params, use_tqdm=False)
    finished_stats_logger = get_finished_request_stats_logger(llm)
    finished_stats_logger.reset()
    rank_expert_maps = model_executor.collective_rpc(
        collect_expert_to_ep_rank_worker,
        timeout=30,
    )
    valid_rank_maps = [
        np.asarray(rank_map, dtype=np.int64)
        for rank_map in rank_expert_maps
        if np.asarray(rank_map).shape == (args.num_experts,)
    ]
    expert_to_ep_rank = merge_expert_to_ep_rank_maps(
        valid_rank_maps,
        num_experts=args.num_experts,
        ep_size=args.data_parallel_size,
    )

    selected_indices_parts: list[np.ndarray] = []
    prompt_lengths_parts: list[np.ndarray] = []
    output_lengths_parts: list[np.ndarray] = []
    step_histograms_parts: list[np.ndarray] = []
    step_total_tokens_parts: list[np.ndarray] = []
    step_total_ms_parts: list[np.ndarray] = []
    step_attention_ms_parts: list[np.ndarray] = []
    step_routing_ms_parts: list[np.ndarray] = []
    step_prepare_ms_parts: list[np.ndarray] = []
    step_finalize_ms_parts: list[np.ndarray] = []
    step_ffn_ms_parts: list[np.ndarray] = []
    step_kinds_parts: list[np.ndarray] = []
    step_indices_parts: list[np.ndarray] = []
    step_start_time_ms_parts: list[np.ndarray] = []
    step_end_time_ms_parts: list[np.ndarray] = []
    prepare_start_time_ms_parts: list[np.ndarray] = []
    finalize_end_time_ms_parts: list[np.ndarray] = []
    candidate_first_ep_collective_seq_id_parts: list[np.ndarray] = []
    candidate_last_ep_collective_seq_id_parts: list[np.ndarray] = []
    candidate_num_ep_collective_parts: list[np.ndarray] = []
    candidate_step_kind_parts: list[np.ndarray] = []
    candidate_drop_reason_parts: list[np.ndarray] = []
    candidate_step_total_tokens_parts: list[np.ndarray] = []
    candidate_step_total_ms_parts: list[np.ndarray] = []
    candidate_step_draft_ms_parts: list[np.ndarray] = []
    candidate_step_ffn_ms_parts: list[np.ndarray] = []
    candidate_step_histogram_parts: list[np.ndarray] = []
    candidate_layer_ffn_ms_parts: list[np.ndarray] = []
    candidate_layer_local_routed_tokens_parts: list[np.ndarray] = []
    candidate_layer_local_active_experts_parts: list[np.ndarray] = []
    trace_samples: list[dict[str, Any]] = []
    condition_latency_ms = 0.0
    num_forward_steps_total = 0
    num_captured_steps = 0
    num_dropped_steps = 0
    num_prefill_dropped_steps = 0
    num_mixed_dropped_steps = 0
    finished_stats = FinishedRequestStatTotals(0.0, 0, 0)

    try:
        for round_idx in range(total_rounds):
            local_indices = shard_global_batch_indices(
                num_samples=args.num_samples,
                global_batch_size=args.batch_size,
                round_idx=round_idx,
                dp_size=args.data_parallel_size,
                dp_rank=dp_rank,
            )
            capture_round = len(local_indices) > 0
            prompt_batch = (
                _local_prompt_batch(prompt_items, local_indices)
                if capture_round
                else [prompt_items[0]]
            )
            start = time.perf_counter()
            if capture_round:
                outputs, recorder, _ = _run_recorded_round(
                    llm,
                    scheduler,
                    model_executor,
                    sampling_params,
                    prompt_batch,
                    batch_size=args.batch_size,
                    draft_length=args.draft_length,
                    round_idx=round_idx,
                    use_spec_decode=use_spec_decode,
                    layers=tuple(args.layers),
                    num_experts=args.num_experts,
                    expert_to_ep_rank=expert_to_ep_rank,
                    local_ep_rank=dp_rank,
                    trace_steps_limit=args.trace_steps_per_rank,
                )
            else:
                outputs = llm.generate(
                    prompt_batch,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                recorder = None
            condition_latency_ms += (time.perf_counter() - start) * 1000.0

            if not capture_round:
                continue
            if recorder is None:
                raise AssertionError("Recorder must exist for captured rounds.")

            selected_indices_parts.append(local_indices.astype(np.int64, copy=False))
            prompt_lengths_parts.append(
                np.asarray(
                    [len(item["prompt_token_ids"]) for item in prompt_batch],
                    dtype=np.int64,
                )
            )
            output_lengths_parts.append(
                np.asarray(
                    [len(output.outputs[0].token_ids) for output in outputs],
                    dtype=np.int64,
                )
            )
            candidate_first_ep_collective_seq_id_parts.append(
                np.asarray(
                    recorder.candidate_first_ep_collective_seq_ids,
                    dtype=np.int64,
                )
            )
            candidate_last_ep_collective_seq_id_parts.append(
                np.asarray(
                    recorder.candidate_last_ep_collective_seq_ids,
                    dtype=np.int64,
                )
            )
            candidate_num_ep_collective_parts.append(
                np.asarray(recorder.candidate_num_ep_collectives, dtype=np.int64)
            )
            candidate_step_kind_parts.append(
                np.asarray(recorder.candidate_step_kinds, dtype=np.str_)
            )
            candidate_drop_reason_parts.append(
                np.asarray(recorder.candidate_drop_reasons, dtype=np.str_)
            )
            candidate_step_total_tokens_parts.append(
                np.asarray(recorder.candidate_step_total_tokens, dtype=np.int64)
            )
            candidate_step_total_ms_parts.append(
                np.asarray(recorder.candidate_step_total_ms, dtype=np.float64)
            )
            candidate_step_draft_ms_parts.append(
                np.asarray(recorder.candidate_step_draft_ms, dtype=np.float64)
            )
            candidate_step_ffn_ms_parts.append(
                np.asarray(recorder.candidate_step_ffn_ms, dtype=np.float64)
            )
            if recorder.candidate_step_histograms:
                candidate_step_histogram_parts.append(
                    np.stack(recorder.candidate_step_histograms, axis=0).astype(
                        np.int64
                    )
                )
            if recorder.candidate_layer_ffn_ms:
                candidate_layer_ffn_ms_parts.append(
                    np.stack(recorder.candidate_layer_ffn_ms, axis=0).astype(
                        np.float64
                    )
                )
                candidate_layer_local_routed_tokens_parts.append(
                    np.stack(
                        recorder.candidate_layer_local_routed_tokens, axis=0
                    ).astype(np.int64)
                )
                candidate_layer_local_active_experts_parts.append(
                    np.stack(
                        recorder.candidate_layer_local_active_experts, axis=0
                    ).astype(np.int64)
                )
            if recorder.step_histograms:
                step_histograms_parts.append(
                    np.stack(recorder.step_histograms, axis=0).astype(np.int64)
                )
                step_total_tokens_parts.append(
                    np.asarray(recorder.step_total_tokens, dtype=np.int64)
                )
                step_total_ms_parts.append(
                    np.asarray(recorder.step_total_ms, dtype=np.float64)
                )
                step_attention_ms_parts.append(
                    np.asarray(recorder.step_attention_ms, dtype=np.float64)
                )
                step_routing_ms_parts.append(
                    np.asarray(recorder.step_routing_ms, dtype=np.float64)
                )
                step_prepare_ms_parts.append(
                    np.asarray(recorder.step_prepare_ms, dtype=np.float64)
                )
                step_finalize_ms_parts.append(
                    np.asarray(recorder.step_finalize_ms, dtype=np.float64)
                )
                step_ffn_ms_parts.append(
                    np.asarray(recorder.step_ffn_ms, dtype=np.float64)
                )
                step_kinds_parts.append(np.asarray(recorder.step_kinds, dtype=np.str_))
                step_indices_parts.append(
                    np.asarray(recorder.step_indices, dtype=np.int64)
                )
                step_start_time_ms_parts.append(
                    np.asarray(recorder.step_start_time_ms, dtype=np.float64)
                )
                step_end_time_ms_parts.append(
                    np.asarray(recorder.step_end_time_ms, dtype=np.float64)
                )
                prepare_start_time_ms_parts.append(
                    np.asarray(recorder.prepare_start_time_ms, dtype=np.float64)
                )
                finalize_end_time_ms_parts.append(
                    np.asarray(recorder.finalize_end_time_ms, dtype=np.float64)
                )

            num_forward_steps_total += recorder.num_forward_steps_total
            num_captured_steps += len(recorder.step_histograms)
            num_dropped_steps += recorder.num_dropped_steps
            num_prefill_dropped_steps += recorder.num_prefill_dropped_steps
            num_mixed_dropped_steps += recorder.num_mixed_dropped_steps
            if args.trace_steps_per_rank > 0 and recorder.trace_samples:
                remaining = args.trace_steps_per_rank - len(trace_samples)
                if remaining > 0:
                    trace_samples.extend(recorder.trace_samples[:remaining])
    finally:
        finished_stats = finished_stats_logger.snapshot()
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm

    candidate_first_ep_collective_seq_ids = (
        np.concatenate(candidate_first_ep_collective_seq_id_parts, axis=0)
        if candidate_first_ep_collective_seq_id_parts
        else _empty_int_array()
    )
    candidate_last_ep_collective_seq_ids = (
        np.concatenate(candidate_last_ep_collective_seq_id_parts, axis=0)
        if candidate_last_ep_collective_seq_id_parts
        else _empty_int_array()
    )
    candidate_num_ep_collectives = (
        np.concatenate(candidate_num_ep_collective_parts, axis=0)
        if candidate_num_ep_collective_parts
        else _empty_int_array()
    )
    candidate_step_kinds = (
        np.concatenate(candidate_step_kind_parts, axis=0)
        if candidate_step_kind_parts
        else _empty_str_array()
    )
    candidate_drop_reasons = (
        np.concatenate(candidate_drop_reason_parts, axis=0)
        if candidate_drop_reason_parts
        else _empty_str_array()
    )
    candidate_step_total_tokens = (
        np.concatenate(candidate_step_total_tokens_parts, axis=0)
        if candidate_step_total_tokens_parts
        else _empty_int_array()
    )
    candidate_step_total_ms = (
        np.concatenate(candidate_step_total_ms_parts, axis=0)
        if candidate_step_total_ms_parts
        else _empty_float_array()
    )
    candidate_step_draft_ms = (
        np.concatenate(candidate_step_draft_ms_parts, axis=0)
        if candidate_step_draft_ms_parts
        else _empty_float_array()
    )
    candidate_step_ffn_ms = (
        np.concatenate(candidate_step_ffn_ms_parts, axis=0)
        if candidate_step_ffn_ms_parts
        else _empty_float_array()
    )
    candidate_step_histograms = (
        np.concatenate(candidate_step_histogram_parts, axis=0)
        if candidate_step_histogram_parts
        else _empty_candidate_histograms(args)
    )
    candidate_layer_ffn_ms = (
        np.concatenate(candidate_layer_ffn_ms_parts, axis=0)
        if candidate_layer_ffn_ms_parts
        else np.empty((0, len(args.layers)), dtype=np.float64)
    )
    candidate_layer_local_routed_tokens = (
        np.concatenate(candidate_layer_local_routed_tokens_parts, axis=0)
        if candidate_layer_local_routed_tokens_parts
        else np.empty((0, len(args.layers)), dtype=np.int64)
    )
    candidate_layer_local_active_experts = (
        np.concatenate(candidate_layer_local_active_experts_parts, axis=0)
        if candidate_layer_local_active_experts_parts
        else np.empty((0, len(args.layers)), dtype=np.int64)
    )

    if not step_histograms_parts:
        return RankConditionData(
            selected_dataset_indices=np.concatenate(selected_indices_parts, axis=0)
            if selected_indices_parts
            else np.empty((0,), dtype=np.int64),
            prompt_lengths=np.concatenate(prompt_lengths_parts, axis=0)
            if prompt_lengths_parts
            else np.empty((0,), dtype=np.int64),
            output_lengths=np.concatenate(output_lengths_parts, axis=0)
            if output_lengths_parts
            else np.empty((0,), dtype=np.int64),
            step_histograms=_empty_step_histograms(args),
            step_total_tokens=np.empty((0,), dtype=np.int64),
            step_total_ms=np.empty((0,), dtype=np.float64),
            step_attention_ms=np.empty((0,), dtype=np.float64),
            step_routing_ms=np.empty((0,), dtype=np.float64),
            step_prepare_ms=np.empty((0,), dtype=np.float64),
            step_finalize_ms=np.empty((0,), dtype=np.float64),
            step_ffn_ms=np.empty((0,), dtype=np.float64),
            captured_step_kinds=_empty_str_array(),
            captured_step_indices=_empty_int_array(),
            captured_step_start_time_ms=_empty_float_array(),
            captured_step_end_time_ms=_empty_float_array(),
            captured_prepare_start_time_ms=_empty_float_array(),
            captured_finalize_end_time_ms=_empty_float_array(),
            candidate_first_ep_collective_seq_ids=(
                candidate_first_ep_collective_seq_ids
            ),
            candidate_last_ep_collective_seq_ids=(
                candidate_last_ep_collective_seq_ids
            ),
            candidate_num_ep_collectives=candidate_num_ep_collectives,
            candidate_step_kinds=candidate_step_kinds,
            candidate_drop_reasons=candidate_drop_reasons,
            candidate_step_total_tokens=candidate_step_total_tokens,
            candidate_step_total_ms=candidate_step_total_ms,
            candidate_step_draft_ms=candidate_step_draft_ms,
            candidate_step_ffn_ms=candidate_step_ffn_ms,
            candidate_step_histograms=candidate_step_histograms,
            candidate_layer_ffn_ms=candidate_layer_ffn_ms,
            candidate_layer_local_routed_tokens=candidate_layer_local_routed_tokens,
            candidate_layer_local_active_experts=(
                candidate_layer_local_active_experts
            ),
            expert_to_ep_rank=expert_to_ep_rank,
            condition_latency_ms=condition_latency_ms,
            decode_time_total_ms=finished_stats.decode_time_total_ms,
            num_generation_tokens_total=(
                finished_stats.num_generation_tokens_total
            ),
            num_output_tokens_excl_first_total=(
                finished_stats.num_output_tokens_excl_first_total
            ),
            num_forward_steps_total=num_forward_steps_total,
            num_captured_steps=num_captured_steps,
            num_dropped_steps=num_dropped_steps,
            num_prefill_dropped_steps=num_prefill_dropped_steps,
            num_mixed_dropped_steps=num_mixed_dropped_steps,
            local_max_num_seqs=scheduler_capacity_config.local_max_num_seqs,
            configured_max_num_batched_tokens=(
                scheduler_capacity_config.configured_max_num_batched_tokens
            ),
            scheduler_max_num_seqs=(
                scheduler_capacity_config.scheduler_max_num_seqs
            ),
            scheduler_max_num_batched_tokens=(
                scheduler_capacity_config.scheduler_max_num_batched_tokens
            ),
            scheduler_max_num_scheduled_tokens=(
                scheduler_capacity_config.scheduler_max_num_scheduled_tokens
            ),
            speculative_max_num_new_slots_for_drafting=(
                scheduler_capacity_config.speculative_max_num_new_slots_for_drafting
            ),
            trace_samples=trace_samples,
        )

    return RankConditionData(
        selected_dataset_indices=np.concatenate(selected_indices_parts, axis=0),
        prompt_lengths=np.concatenate(prompt_lengths_parts, axis=0),
        output_lengths=np.concatenate(output_lengths_parts, axis=0),
        step_histograms=np.concatenate(step_histograms_parts, axis=0),
        step_total_tokens=np.concatenate(step_total_tokens_parts, axis=0),
        step_total_ms=np.concatenate(step_total_ms_parts, axis=0),
        step_attention_ms=np.concatenate(step_attention_ms_parts, axis=0),
        step_routing_ms=np.concatenate(step_routing_ms_parts, axis=0),
        step_prepare_ms=np.concatenate(step_prepare_ms_parts, axis=0),
        step_finalize_ms=np.concatenate(step_finalize_ms_parts, axis=0),
        step_ffn_ms=np.concatenate(step_ffn_ms_parts, axis=0),
        captured_step_kinds=np.concatenate(step_kinds_parts, axis=0),
        captured_step_indices=np.concatenate(step_indices_parts, axis=0),
        captured_step_start_time_ms=np.concatenate(step_start_time_ms_parts, axis=0),
        captured_step_end_time_ms=np.concatenate(step_end_time_ms_parts, axis=0),
        captured_prepare_start_time_ms=np.concatenate(
            prepare_start_time_ms_parts, axis=0
        ),
        captured_finalize_end_time_ms=np.concatenate(
            finalize_end_time_ms_parts, axis=0
        ),
        candidate_first_ep_collective_seq_ids=candidate_first_ep_collective_seq_ids,
        candidate_last_ep_collective_seq_ids=candidate_last_ep_collective_seq_ids,
        candidate_num_ep_collectives=candidate_num_ep_collectives,
        candidate_step_kinds=candidate_step_kinds,
        candidate_drop_reasons=candidate_drop_reasons,
        candidate_step_total_tokens=candidate_step_total_tokens,
        candidate_step_total_ms=candidate_step_total_ms,
        candidate_step_draft_ms=candidate_step_draft_ms,
        candidate_step_ffn_ms=candidate_step_ffn_ms,
        candidate_step_histograms=candidate_step_histograms,
        candidate_layer_ffn_ms=candidate_layer_ffn_ms,
        candidate_layer_local_routed_tokens=candidate_layer_local_routed_tokens,
        candidate_layer_local_active_experts=candidate_layer_local_active_experts,
        expert_to_ep_rank=expert_to_ep_rank,
        condition_latency_ms=condition_latency_ms,
        decode_time_total_ms=finished_stats.decode_time_total_ms,
        num_generation_tokens_total=finished_stats.num_generation_tokens_total,
        num_output_tokens_excl_first_total=(
            finished_stats.num_output_tokens_excl_first_total
        ),
        num_forward_steps_total=num_forward_steps_total,
        num_captured_steps=num_captured_steps,
        num_dropped_steps=num_dropped_steps,
        num_prefill_dropped_steps=num_prefill_dropped_steps,
        num_mixed_dropped_steps=num_mixed_dropped_steps,
        local_max_num_seqs=scheduler_capacity_config.local_max_num_seqs,
        configured_max_num_batched_tokens=(
            scheduler_capacity_config.configured_max_num_batched_tokens
        ),
        scheduler_max_num_seqs=scheduler_capacity_config.scheduler_max_num_seqs,
        scheduler_max_num_batched_tokens=(
            scheduler_capacity_config.scheduler_max_num_batched_tokens
        ),
        scheduler_max_num_scheduled_tokens=(
            scheduler_capacity_config.scheduler_max_num_scheduled_tokens
        ),
        speculative_max_num_new_slots_for_drafting=(
            scheduler_capacity_config.speculative_max_num_new_slots_for_drafting
        ),
        trace_samples=trace_samples,
    )


def save_rank_condition_data(path: Path, data: RankConditionData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data.to_npz_payload())


def rank_trace_path(rank_output_path: Path) -> Path:
    return rank_output_path.with_suffix(".trace.json")


def save_rank_trace_samples(path: Path, args: Any, data: RankConditionData) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "batch_size": args.batch_size,
        "draft_length": args.draft_length,
        "data_parallel_size": args.data_parallel_size,
        "dp_rank": args.dp_rank,
        "trace_steps_per_rank": args.trace_steps_per_rank,
        "trace_samples": data.trace_samples,
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_rank_condition_data(path: Path) -> RankConditionData:
    with np.load(path, allow_pickle=False) as data:
        def read_optional_int(name: str) -> int:
            if name not in data:
                return -1
            return int(data[name][0])

        return RankConditionData(
            selected_dataset_indices=np.asarray(data["selected_dataset_indices"]),
            prompt_lengths=np.asarray(data["prompt_lengths"]),
            output_lengths=np.asarray(data["output_lengths"]),
            step_histograms=np.asarray(data["step_histograms"]),
            step_total_tokens=np.asarray(data["step_total_tokens"]),
            step_total_ms=np.asarray(data["step_total_ms"]),
            step_attention_ms=np.asarray(data["step_attention_ms"]),
            step_routing_ms=np.asarray(data["step_routing_ms"]),
            step_prepare_ms=np.asarray(data["step_prepare_ms"]),
            step_finalize_ms=np.asarray(data["step_finalize_ms"]),
            step_ffn_ms=np.asarray(data["step_ffn_ms"]),
            captured_step_kinds=np.asarray(data["captured_step_kinds"]),
            captured_step_indices=np.asarray(data["captured_step_indices"]),
            captured_step_start_time_ms=np.asarray(
                data["captured_step_start_time_ms"]
            ),
            captured_step_end_time_ms=np.asarray(data["captured_step_end_time_ms"]),
            captured_prepare_start_time_ms=np.asarray(
                data["captured_prepare_start_time_ms"]
            ),
            captured_finalize_end_time_ms=np.asarray(
                data["captured_finalize_end_time_ms"]
            ),
            candidate_first_ep_collective_seq_ids=np.asarray(
                data["candidate_first_ep_collective_seq_ids"]
            ),
            candidate_last_ep_collective_seq_ids=np.asarray(
                data["candidate_last_ep_collective_seq_ids"]
            ),
            candidate_num_ep_collectives=np.asarray(
                data["candidate_num_ep_collectives"]
            ),
            candidate_step_kinds=np.asarray(data["candidate_step_kinds"]),
            candidate_drop_reasons=np.asarray(data["candidate_drop_reasons"]),
            candidate_step_total_tokens=np.asarray(
                data["candidate_step_total_tokens"]
            ),
            candidate_step_total_ms=np.asarray(data["candidate_step_total_ms"]),
            candidate_step_draft_ms=np.asarray(data["candidate_step_draft_ms"]),
            candidate_step_ffn_ms=np.asarray(data["candidate_step_ffn_ms"]),
            candidate_step_histograms=np.asarray(data["candidate_step_histograms"]),
            candidate_layer_ffn_ms=np.asarray(data["candidate_layer_ffn_ms"]),
            candidate_layer_local_routed_tokens=np.asarray(
                data["candidate_layer_local_routed_tokens"]
            ),
            candidate_layer_local_active_experts=np.asarray(
                data["candidate_layer_local_active_experts"]
            ),
            expert_to_ep_rank=np.asarray(data["expert_to_ep_rank"]),
            condition_latency_ms=float(data["condition_latency_ms"][0]),
            decode_time_total_ms=float(data["decode_time_total_ms"][0]),
            num_generation_tokens_total=int(data["num_generation_tokens_total"][0]),
            num_output_tokens_excl_first_total=int(
                data["num_output_tokens_excl_first_total"][0]
            ),
            num_forward_steps_total=int(data["num_forward_steps_total"][0]),
            num_captured_steps=int(data["num_captured_steps"][0]),
            num_dropped_steps=int(data["num_dropped_steps"][0]),
            num_prefill_dropped_steps=int(data["num_prefill_dropped_steps"][0]),
            num_mixed_dropped_steps=int(data["num_mixed_dropped_steps"][0]),
            local_max_num_seqs=read_optional_int("local_max_num_seqs"),
            configured_max_num_batched_tokens=read_optional_int(
                "configured_max_num_batched_tokens"
            ),
            scheduler_max_num_seqs=read_optional_int("scheduler_max_num_seqs"),
            scheduler_max_num_batched_tokens=read_optional_int(
                "scheduler_max_num_batched_tokens"
            ),
            scheduler_max_num_scheduled_tokens=read_optional_int(
                "scheduler_max_num_scheduled_tokens"
            ),
            speculative_max_num_new_slots_for_drafting=read_optional_int(
                "speculative_max_num_new_slots_for_drafting"
            ),
            trace_samples=[],
        )


def collect_one_rank(args: Namespace) -> None:
    data = collect_condition_for_rank(args)
    save_rank_condition_data(args.rank_output_path, data)
    if args.trace_steps_per_rank > 0 and data.trace_samples:
        save_rank_trace_samples(rank_trace_path(args.rank_output_path), args, data)


def _aggregate_rank_condition_data(
    args: Namespace,
    partials: list[RankConditionData],
    *,
    condition_latency_ms: float,
) -> ConditionRawData:
    selected_indices = np.concatenate(
        [partial.selected_dataset_indices for partial in partials],
        axis=0,
    )
    expected_indices = select_dataset_indices(args.num_samples, args.num_samples)
    if selected_indices.shape != expected_indices.shape or not np.array_equal(
        np.sort(selected_indices, kind="stable"),
        expected_indices,
    ):
        raise RuntimeError("DP shard aggregation dropped or duplicated dataset items.")

    scheduler_capacity_config = SchedulerCapacityConfig(
        local_max_num_seqs=partials[0].local_max_num_seqs,
        configured_max_num_batched_tokens=(
            partials[0].configured_max_num_batched_tokens
        ),
        scheduler_max_num_seqs=partials[0].scheduler_max_num_seqs,
        scheduler_max_num_batched_tokens=(
            partials[0].scheduler_max_num_batched_tokens
        ),
        scheduler_max_num_scheduled_tokens=(
            partials[0].scheduler_max_num_scheduled_tokens
        ),
        speculative_max_num_new_slots_for_drafting=(
            partials[0].speculative_max_num_new_slots_for_drafting
        ),
    )
    for partial in partials[1:]:
        other_config = SchedulerCapacityConfig(
            local_max_num_seqs=partial.local_max_num_seqs,
            configured_max_num_batched_tokens=(
                partial.configured_max_num_batched_tokens
            ),
            scheduler_max_num_seqs=partial.scheduler_max_num_seqs,
            scheduler_max_num_batched_tokens=partial.scheduler_max_num_batched_tokens,
            scheduler_max_num_scheduled_tokens=(
                partial.scheduler_max_num_scheduled_tokens
            ),
            speculative_max_num_new_slots_for_drafting=(
                partial.speculative_max_num_new_slots_for_drafting
            ),
        )
        if other_config != scheduler_capacity_config:
            raise RuntimeError(
                "DP ranks used different scheduler capacity configs: "
                f"{scheduler_capacity_config.to_log_dict()} vs "
                f"{other_config.to_log_dict()}."
            )

    order = np.argsort(selected_indices, kind="stable")
    selected_indices = selected_indices[order]
    prompt_lengths = np.concatenate(
        [partial.prompt_lengths for partial in partials], axis=0
    )[order]
    output_lengths = np.concatenate(
        [partial.output_lengths for partial in partials], axis=0
    )[order]

    expert_to_ep_rank = merge_expert_to_ep_rank_maps(
        [partial.expert_to_ep_rank for partial in partials],
        num_experts=args.num_experts,
        ep_size=args.data_parallel_size,
    )
    global_steps = aggregate_global_step_time_components(
        [
            {
                "candidate_first_ep_collective_seq_ids": (
                    partial.candidate_first_ep_collective_seq_ids
                ),
                "candidate_last_ep_collective_seq_ids": (
                    partial.candidate_last_ep_collective_seq_ids
                ),
                "candidate_num_ep_collectives": (
                    partial.candidate_num_ep_collectives
                ),
                "candidate_step_kinds": partial.candidate_step_kinds,
                "candidate_step_total_ms": partial.candidate_step_total_ms,
                "candidate_step_draft_ms": partial.candidate_step_draft_ms,
                "candidate_step_ffn_ms": partial.candidate_step_ffn_ms,
                "candidate_step_total_tokens": partial.candidate_step_total_tokens,
                "candidate_step_histograms": partial.candidate_step_histograms,
                "candidate_layer_ffn_ms": partial.candidate_layer_ffn_ms,
                "candidate_layer_local_routed_tokens": (
                    partial.candidate_layer_local_routed_tokens
                ),
                "candidate_layer_local_active_experts": (
                    partial.candidate_layer_local_active_experts
                ),
            }
            for partial in partials
        ],
        data_parallel_size=args.data_parallel_size,
        layers=tuple(args.layers),
        num_experts=args.num_experts,
    )
    if global_steps.global_step_indices.size == 0:
        raise RuntimeError(
            "No globally aligned barriers were collected for "
            f"batch_size={args.batch_size}, draft_length={args.draft_length}. "
            "Re-run collect and inspect per-rank barrier span coverage."
        )

    decode_time_total_ms = sum(partial.decode_time_total_ms for partial in partials)
    num_output_tokens_total = int(output_lengths.sum())
    num_generation_tokens_total = sum(
        partial.num_generation_tokens_total for partial in partials
    )
    num_output_tokens_excl_first_total = sum(
        partial.num_output_tokens_excl_first_total for partial in partials
    )
    if num_output_tokens_excl_first_total == 0:
        num_output_tokens_excl_first_total = compute_num_output_tokens_excluding_first(
            output_lengths
        )
    tpot_ms = compute_tpot_ms_from_finished_stats(
        decode_time_total_ms,
        num_output_tokens_excl_first_total,
    )
    decode_throughput_tok_s = compute_decode_throughput_tok_s(
        num_generation_tokens_total,
        decode_time_total_ms,
    )

    return ConditionRawData(
        batch_size=args.batch_size,
        draft_length=args.draft_length,
        data_parallel_size=args.data_parallel_size,
        num_samples=args.num_samples,
        batch_size_scope="global",
        local_max_num_seqs=scheduler_capacity_config.local_max_num_seqs,
        configured_max_num_batched_tokens=(
            scheduler_capacity_config.configured_max_num_batched_tokens
        ),
        scheduler_max_num_seqs=scheduler_capacity_config.scheduler_max_num_seqs,
        scheduler_max_num_batched_tokens=(
            scheduler_capacity_config.scheduler_max_num_batched_tokens
        ),
        scheduler_max_num_scheduled_tokens=(
            scheduler_capacity_config.scheduler_max_num_scheduled_tokens
        ),
        speculative_max_num_new_slots_for_drafting=(
            scheduler_capacity_config.speculative_max_num_new_slots_for_drafting
        ),
        mixed_step_policy="include_all_global_barriers",
        tpot_definition=TPOT_DEFINITION,
        selected_dataset_indices=selected_indices,
        prompt_lengths=prompt_lengths,
        output_lengths=output_lengths,
        condition_latency_ms=condition_latency_ms,
        decode_time_total_ms=decode_time_total_ms,
        num_output_tokens_total=num_output_tokens_total,
        num_generation_tokens_total=num_generation_tokens_total,
        num_output_tokens_excl_first_total=num_output_tokens_excl_first_total,
        tpot_ms=tpot_ms,
        decode_throughput_tok_s=decode_throughput_tok_s,
        step_histograms=global_steps.global_step_histograms,
        step_total_tokens=global_steps.global_step_total_tokens,
        step_total_ms=global_steps.global_step_total_ms,
        step_attention_ms=np.empty((0,), dtype=np.float64),
        step_routing_ms=np.empty((0,), dtype=np.float64),
        step_prepare_ms=np.empty((0,), dtype=np.float64),
        step_finalize_ms=np.empty((0,), dtype=np.float64),
        step_ffn_ms=global_steps.global_step_ffn_ms,
        captured_step_kinds=global_steps.global_step_kinds,
        global_barrier_ids=global_steps.global_barrier_ids,
        barrier_first_ep_collective_seq_ids=(
            global_steps.barrier_first_ep_collective_seq_ids
        ),
        barrier_last_ep_collective_seq_ids=(
            global_steps.barrier_last_ep_collective_seq_ids
        ),
        barrier_num_ep_collectives=global_steps.barrier_num_ep_collectives,
        rank_barrier_first_ep_collective_seq_ids=(
            global_steps.rank_barrier_first_ep_collective_seq_ids
        ),
        rank_barrier_last_ep_collective_seq_ids=(
            global_steps.rank_barrier_last_ep_collective_seq_ids
        ),
        rank_barrier_num_ep_collectives=(
            global_steps.rank_barrier_num_ep_collectives
        ),
        rank_step_kinds=global_steps.rank_step_kinds,
        rank_step_total_ms=global_steps.rank_step_total_ms,
        rank_step_draft_ms=global_steps.rank_step_draft_ms,
        rank_layer_ffn_ms=global_steps.rank_layer_ffn_ms,
        rank_layer_local_routed_tokens=(
            global_steps.rank_layer_local_routed_tokens
        ),
        rank_layer_local_active_experts=(
            global_steps.rank_layer_local_active_experts
        ),
        global_step_indices=global_steps.global_step_indices,
        global_step_total_ms=global_steps.global_step_total_ms,
        global_draft_ms=global_steps.global_draft_ms,
        global_step_ffn_ms=global_steps.global_step_ffn_ms,
        global_step_sorted_rank_ffn_ms=(
            global_steps.global_step_sorted_rank_ffn_ms
        ),
        global_step_sorted_rank_local_routed_tokens=(
            global_steps.global_step_sorted_rank_local_routed_tokens
        ),
        global_step_sorted_rank_local_active_experts=(
            global_steps.global_step_sorted_rank_local_active_experts
        ),
        global_step_ffn_max_mean_ratio=(
            global_steps.global_step_ffn_max_mean_ratio
        ),
        global_step_other_ms=global_steps.global_step_other_ms,
        global_step_kinds=global_steps.global_step_kinds,
        expert_to_ep_rank=expert_to_ep_rank,
        layers=np.asarray(args.layers, dtype=np.int64),
        avg_histograms=average_step_histograms(global_steps.global_step_histograms),
        num_forward_steps_total=sum(
            partial.num_forward_steps_total for partial in partials
        ),
        num_captured_steps=sum(partial.num_captured_steps for partial in partials),
        num_global_candidate_steps=global_steps.num_global_candidate_steps,
        num_global_captured_steps=global_steps.num_global_captured_steps,
        num_dropped_steps=sum(partial.num_dropped_steps for partial in partials),
        num_prefill_dropped_steps=sum(
            partial.num_prefill_dropped_steps for partial in partials
        ),
        num_mixed_dropped_steps=sum(
            partial.num_mixed_dropped_steps for partial in partials
        ),
        num_global_prefill_dropped_steps=(
            global_steps.num_global_prefill_dropped_steps
        ),
        num_global_mixed_dropped_steps=global_steps.num_global_mixed_dropped_steps,
        num_global_non_target_dropped_steps=(
            global_steps.num_global_non_target_dropped_steps
        ),
    )


def collect_one_condition(
    args: Namespace,
    output_dir: Path,
) -> CollectedConditionSummary:
    validate_parallel_config(args)
    dirs = ensure_collect_dirs(output_dir)
    if getattr(args, "prompt_cache_path", None) is None:
        args.prompt_cache_path = prepare_prompt_cache(args, dirs["root"])
    partial_dir = dirs["root"] / "_dp_partials" / condition_name(
        args.batch_size, args.draft_length
    )
    partial_dir.mkdir(parents=True, exist_ok=True)
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    processes: list[subprocess.Popen[str]] = []
    partial_paths: list[Path] = []
    cwd = Path(__file__).resolve().parent.parent.parent.parent
    start = time.perf_counter()
    for dp_rank in range(args.data_parallel_size):
        partial_path = partial_dir / f"rank_{dp_rank:02d}.npz"
        partial_paths.append(partial_path)
        command = _build_collect_one_rank_command(
            args,
            output_dir,
            Path(__file__).resolve().parent
            / "qwen3_6_mtp_ep_load_balance_experiment.py",
            dp_rank=dp_rank,
            dp_local_rank=dp_rank,
            dp_master_ip=dp_master_ip,
            dp_master_port=dp_master_port,
            rank_output_path=partial_path,
        )
        processes.append(
            subprocess.Popen(
                command,
                cwd=cwd,
                env=os.environ.copy(),
                text=True,
            )
        )
    exit_code = 0
    for proc in processes:
        proc.wait()
        if proc.returncode:
            exit_code = proc.returncode
    if exit_code:
        raise subprocess.CalledProcessError(exit_code, "collect-one-rank")

    partials = [load_rank_condition_data(path) for path in partial_paths]
    raw_data = _aggregate_rank_condition_data(
        args,
        partials,
        condition_latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    raw_path = dirs["raw"] / f"{condition_name(args.batch_size, args.draft_length)}.npz"
    np.savez_compressed(raw_path, **raw_data.to_npz_payload())
    return load_condition_summary(raw_path)


def _build_collect_one_command(
    args: Namespace,
    output_dir: Path,
    entrypoint: Path,
    *,
    batch_size: int,
    draft_length: int,
    prompt_cache: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(entrypoint),
        "collect-one",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--dataset-split",
        args.dataset_split,
        "--batch-size",
        str(batch_size),
        "--draft-length",
        str(draft_length),
        "--num-samples",
        str(args.num_samples),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--num-experts",
        str(args.num_experts),
        "--output-dir",
        str(output_dir),
        "--prompt-cache-path",
        str(prompt_cache),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--trace-steps-per-rank",
        str(getattr(args, "trace_steps_per_rank", 0)),
    ]
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    _append_optional_arg(
        command,
        "--max-num-batched-tokens",
        getattr(args, "max_num_batched_tokens", None),
    )
    command.extend(["--layers", *(str(layer) for layer in args.layers)])
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    return command


def _build_collect_one_rank_command(
    args: Namespace,
    output_dir: Path,
    entrypoint: Path,
    *,
    dp_rank: int,
    dp_local_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    rank_output_path: Path,
) -> list[str]:
    command = _build_collect_one_command(
        args,
        output_dir,
        entrypoint,
        batch_size=args.batch_size,
        draft_length=args.draft_length,
        prompt_cache=Path(args.prompt_cache_path),
    )
    command[2] = "collect-one-rank"
    command.extend(
        [
            "--dp-rank",
            str(dp_rank),
            "--dp-local-rank",
            str(dp_local_rank),
            "--dp-master-ip",
            dp_master_ip,
            "--dp-master-port",
            str(dp_master_port),
            "--rank-output-path",
            str(rank_output_path),
        ]
    )
    return command


def collect_experiment(args: Namespace, output_dir: Path, entrypoint: Path) -> None:
    validate_parallel_config(args)
    dirs = ensure_collect_dirs(output_dir)
    save_run_metadata(dirs["root"], args)
    prompts_cache = prepare_prompt_cache(args, dirs["root"])
    condition_summaries: list[CollectedConditionSummary] = []
    for batch_size in args.batch_sizes:
        for draft_length in args.draft_lengths:
            print(
                f"[collect-parent] launching batch_size={batch_size} "
                f"draft_length={draft_length} dp={args.data_parallel_size}",
                flush=True,
            )
            command = _build_collect_one_command(
                args,
                dirs["root"],
                entrypoint,
                batch_size=batch_size,
                draft_length=draft_length,
                prompt_cache=prompts_cache,
            )
            subprocess.run(
                command,
                check=True,
                cwd=entrypoint.parent.parent.parent.parent,
                env=os.environ.copy(),
            )
            raw_path = dirs["raw"] / f"{condition_name(batch_size, draft_length)}.npz"
            summary = load_condition_summary(raw_path)
            condition_summaries.append(summary)

    save_collect_manifest(dirs["root"], args, condition_summaries)
