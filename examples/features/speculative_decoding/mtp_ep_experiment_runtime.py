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
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from mtp_ep_load_balance_utils import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    SCHEMA_VERSION,
    TIMING_BACKEND,
    TIMING_SCOPE,
    TPOT_DEFINITION,
    FinishedRequestStatTotals,
    StepTiming,
    aggregate_global_step_time_components,
    average_step_histograms,
    build_token_layer_destination_assignments,
    classify_step_capture,
    compute_decode_throughput_tok_s,
    compute_num_output_tokens_excluding_first,
    compute_tpot_ms_from_finished_stats,
    count_layer_expert_histograms,
    count_position_layer_local_routed_tokens,
    interval_set_overlap_duration_ms,
    interval_union_duration_ms,
    merge_expert_to_ep_rank_maps,
    num_condition_rounds,
    select_dataset_indices,
    shard_global_batch_indices,
    subtract_interval_overlap_ms,
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
        self.vllm_generation_tokens_total = 0
        self.vllm_request_tpot_total_ms = 0.0
        self.vllm_request_tpot_count = 0
        self.spec_num_drafts = 0
        self.spec_num_draft_tokens = 0
        self.spec_num_accepted_tokens = 0

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is not None:
            spec_stats = getattr(scheduler_stats, "spec_decoding_stats", None)
            if spec_stats is not None:
                self.spec_num_drafts += int(spec_stats.num_drafts)
                self.spec_num_draft_tokens += int(spec_stats.num_draft_tokens)
                self.spec_num_accepted_tokens += int(spec_stats.num_accepted_tokens)

        if iteration_stats is None:
            return
        self.vllm_generation_tokens_total += int(iteration_stats.num_generation_tokens)
        for finished_req in iteration_stats.finished_requests:
            num_generation_tokens = int(finished_req.num_generation_tokens)
            self.decode_time_total_ms += float(finished_req.decode_time) * 1000.0
            self.num_generation_tokens_total += num_generation_tokens
            self.num_output_tokens_excl_first_total += max(
                num_generation_tokens - 1,
                0,
            )
            self.vllm_request_tpot_total_ms += (
                float(finished_req.mean_time_per_output_token) * 1000.0
            )
            self.vllm_request_tpot_count += 1

    def log_engine_initialized(self) -> None:
        return

    def snapshot(self) -> FinishedRequestStatTotals:
        return FinishedRequestStatTotals(
            decode_time_total_ms=self.decode_time_total_ms,
            num_generation_tokens_total=self.num_generation_tokens_total,
            num_output_tokens_excl_first_total=(
                self.num_output_tokens_excl_first_total
            ),
            vllm_generation_tokens_total=self.vllm_generation_tokens_total,
            vllm_request_tpot_total_ms=self.vllm_request_tpot_total_ms,
            vllm_request_tpot_count=self.vllm_request_tpot_count,
            spec_num_drafts=self.spec_num_drafts,
            spec_num_draft_tokens=self.spec_num_draft_tokens,
            spec_num_accepted_tokens=self.spec_num_accepted_tokens,
        )


_FINISHED_REQUEST_STATS_LOGGER_ATTR = "_mtp_ep_finished_request_stats_logger"
_SCHEDULER_CAPACITY_CONFIG_ATTR = "_mtp_ep_scheduler_capacity_config"
HYBRID_PREDICTION_TRACE_SCHEMA_VERSION = 1
HYBRID_PREDICTION_TRACE_MODE_CHOICES = ("off", "record", "replay")
HYBRID_PREDICTION_SIM_MODE_CHOICES = ("exact_upper_bound",)

RTX_5090_NCCL_ENV_DEFAULTS = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_COLLNET_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_P2P_NET_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
}


def apply_rtx_5090_nccl_env_defaults(env: dict[str, str]) -> dict[str, str]:
    for name, value in RTX_5090_NCCL_ENV_DEFAULTS.items():
        env.setdefault(name, value)
    return env


def parse_local_gpu_ids(
    local_gpu_ids: str | None,
    data_parallel_size: int,
) -> list[str]:
    if local_gpu_ids:
        gpu_ids = [gpu_id.strip() for gpu_id in local_gpu_ids.split(",")]
    else:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = (
            [gpu_id.strip() for gpu_id in visible_devices.split(",")]
            if visible_devices
            else [str(gpu_id) for gpu_id in range(data_parallel_size)]
        )
    gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id]
    if not gpu_ids:
        raise ValueError("At least one local GPU ID must be available.")
    return gpu_ids


def validate_local_gpu_binding(args: Any) -> list[str]:
    gpu_ids = parse_local_gpu_ids(
        getattr(args, "local_gpu_ids", None),
        args.data_parallel_size,
    )
    if args.data_parallel_size > len(gpu_ids):
        raise ValueError(
            f"data_parallel_size={args.data_parallel_size} requires one visible "
            f"local GPU per DP rank, but only {len(gpu_ids)} GPU ID(s) were "
            f"provided: {gpu_ids}. For a 4-card RTX 5090 host, pass "
            "--data-parallel-size 4 --local-gpu-ids 0,1,2,3."
        )
    return gpu_ids


def normalize_local_gpu_binding(args: Any) -> list[str]:
    gpu_ids = validate_local_gpu_binding(args)
    args.local_gpu_ids = ",".join(gpu_ids)
    return gpu_ids


def get_rank_gpu_id(args: Any, dp_rank: int) -> str:
    gpu_ids = validate_local_gpu_binding(args)
    return gpu_ids[dp_rank]


def build_collect_subprocess_env(
    args: Any,
    *,
    dp_rank: int | None = None,
) -> dict[str, str]:
    env = apply_rtx_5090_nccl_env_defaults(os.environ.copy())
    if dp_rank is not None:
        env["CUDA_VISIBLE_DEVICES"] = get_rank_gpu_id(args, dp_rank)
        env["VLLM_DP_RANK_LOCAL"] = "0"
    return env


def configure_rank_process_environment(args: Any) -> int:
    apply_rtx_5090_nccl_env_defaults(os.environ)
    os.environ["CUDA_VISIBLE_DEVICES"] = get_rank_gpu_id(args, args.dp_rank)
    os.environ["VLLM_DP_RANK"] = str(args.dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = "0"
    os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_IP"] = args.dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_master_port)
    return 0


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


def _empty_hybrid_prediction_stats() -> dict[str, int]:
    return {
        "total_predictions": 0,
        "exact_match_count": 0,
        "within_one_count": 0,
        "abs_error_sum": 0,
        "signed_error_sum": 0,
        "predicted_accept_len_sum": 0,
        "accepted_len_sum": 0,
    }


def should_collect_hybrid_prediction_trace(args: Any) -> bool:
    return (
        getattr(args, "hybrid_prediction_trace_mode", "off") != "off"
    )


def _empty_hybrid_reload_timing_stats() -> dict[str, float | int]:
    return {
        "preload_total_ms": 0.0,
        "preload_call_count": 0,
        "preload_req_count": 0,
        "preloaded_total_ms": 0.0,
        "preloaded_row_count": 0,
        "fallback_total_ms": 0.0,
        "fallback_row_count": 0,
        "prepare_copy_ms": 0.0,
        "repair_compute_ms": 0.0,
        "repair_row_count": 0,
        "repair_from_start_count": 0,
        "repair_from_resident_count": 0,
        "verify_attention_ms": 0.0,
        "spill_copy_ms": 0.0,
        "layer_total_ms": 0.0,
        "verify_call_count": 0,
        "checkpoint_save_ms": 0.0,
        "post_replay_state_gather_ms": 0.0,
        "capture_materialize_ms": 0.0,
        "segment_start_save_ms": 0.0,
        "segment_start_wait_ms": 0.0,
        "tape_save_ms": 0.0,
    }


def _accumulate_hybrid_reload_timing_stats(
    total: dict[str, float | int],
    worker_stats: dict[str, float | int],
) -> None:
    total["preload_total_ms"] += float(worker_stats.get("preload_total_ms", 0.0))
    total["preload_call_count"] += int(worker_stats.get("preload_call_count", 0))
    total["preload_req_count"] += int(worker_stats.get("preload_req_count", 0))
    total["preloaded_total_ms"] += float(
        worker_stats.get("preloaded_total_ms", 0.0)
    )
    total["preloaded_row_count"] += int(
        worker_stats.get("preloaded_row_count", 0)
    )
    total["fallback_total_ms"] += float(worker_stats.get("fallback_total_ms", 0.0))
    total["fallback_row_count"] += int(worker_stats.get("fallback_row_count", 0))
    total["prepare_copy_ms"] += float(worker_stats.get("prepare_copy_ms", 0.0))
    total["repair_compute_ms"] += float(
        worker_stats.get("repair_compute_ms", 0.0)
    )
    total["verify_attention_ms"] += float(
        worker_stats.get("verify_attention_ms", 0.0)
    )
    total["spill_copy_ms"] += float(worker_stats.get("spill_copy_ms", 0.0))
    total["layer_total_ms"] += float(worker_stats.get("layer_total_ms", 0.0))
    total["verify_call_count"] += int(worker_stats.get("verify_call_count", 0))
    total["repair_row_count"] += int(worker_stats.get("repair_row_count", 0))
    total["repair_from_start_count"] += int(
        worker_stats.get("repair_from_start_count", 0)
    )
    total["repair_from_resident_count"] += int(
        worker_stats.get("repair_from_resident_count", 0)
    )
    total["checkpoint_save_ms"] += float(
        worker_stats.get("checkpoint_save_ms", 0.0)
    )
    total["post_replay_state_gather_ms"] += float(
        worker_stats.get("post_replay_state_gather_ms", 0.0)
    )
    total["capture_materialize_ms"] += float(
        worker_stats.get("capture_materialize_ms", 0.0)
    )
    total["segment_start_save_ms"] += float(
        worker_stats.get("segment_start_save_ms", 0.0)
    )
    total["segment_start_wait_ms"] += float(
        worker_stats.get("segment_start_wait_ms", 0.0)
    )
    total["tape_save_ms"] += float(worker_stats.get("tape_save_ms", 0.0))


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
    timing_backend: str
    timing_scope: str
    hybrid_spec_state_offload_mode: str
    hybrid_spec_state_ewma_alpha: float
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
    vllm_generation_elapsed_ms: float
    vllm_request_tpot_ms: float
    vllm_generation_throughput_tok_s: float
    spec_num_drafts: int
    spec_num_draft_tokens: int
    spec_num_accepted_tokens: int
    spec_acceptance_rate: float
    spec_mean_acceptance_length: float
    hybrid_prediction_total: int
    hybrid_prediction_exact_match: int
    hybrid_prediction_within_one: int
    hybrid_prediction_abs_error_sum: int
    hybrid_prediction_signed_error_sum: int
    hybrid_prediction_predicted_sum: int
    hybrid_prediction_accepted_sum: int
    hybrid_reload_preload_total_ms: float
    hybrid_reload_preload_call_count: int
    hybrid_reload_preload_req_count: int
    hybrid_reload_preloaded_total_ms: float
    hybrid_reload_preloaded_row_count: int
    hybrid_reload_fallback_total_ms: float
    hybrid_reload_fallback_row_count: int
    hybrid_replay_prepare_copy_ms: float
    hybrid_replay_repair_compute_ms: float
    hybrid_replay_verify_attention_ms: float
    hybrid_replay_spill_copy_ms: float
    hybrid_replay_layer_total_ms: float
    hybrid_replay_verify_call_count: int
    hybrid_replay_checkpoint_save_ms: float
    hybrid_replay_post_replay_state_gather_ms: float
    hybrid_replay_capture_materialize_ms: float
    hybrid_replay_segment_start_save_ms: float
    hybrid_replay_segment_start_wait_ms: float
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
    rank_execute_wall_ms: np.ndarray
    rank_verification_wall_ms: np.ndarray
    rank_draft_wall_ms: np.ndarray
    rank_iteration_wall_ms: np.ndarray
    rank_execute_gpu_ms: np.ndarray
    rank_verification_gpu_ms: np.ndarray
    rank_draft_gpu_ms: np.ndarray
    rank_iteration_gpu_ms: np.ndarray
    rank_attention_gpu_ms: np.ndarray
    rank_moe_gpu_ms: np.ndarray
    rank_gpu_other_ms: np.ndarray
    rank_timing_complete: np.ndarray
    rank_step_total_ms: np.ndarray
    rank_step_draft_ms: np.ndarray
    rank_layer_moe_gpu_ms: np.ndarray
    rank_layer_routed_expert_gpu_ms: np.ndarray
    rank_layer_shared_expert_gpu_ms: np.ndarray
    rank_layer_routing_gpu_ms: np.ndarray
    rank_layer_prepare_gpu_ms: np.ndarray
    rank_layer_finalize_gpu_ms: np.ndarray
    rank_layer_ffn_ms: np.ndarray
    rank_layer_local_routed_tokens: np.ndarray
    rank_layer_local_active_experts: np.ndarray
    rank_position_layer_ffn_ms: np.ndarray
    rank_position_layer_local_routed_tokens: np.ndarray
    global_step_indices: np.ndarray
    global_step_total_ms: np.ndarray
    global_draft_ms: np.ndarray
    global_step_ffn_ms: np.ndarray
    global_critical_rank_indices: np.ndarray
    global_verification_wall_ms: np.ndarray
    global_iteration_wall_ms: np.ndarray
    global_draft_wall_ms: np.ndarray
    global_verification_gpu_total_ms: np.ndarray
    global_attention_gpu_ms: np.ndarray
    global_moe_gpu_ms: np.ndarray
    global_gpu_other_ms: np.ndarray
    global_step_sorted_rank_routed_expert_gpu_ms: np.ndarray
    global_step_sorted_rank_moe_gpu_ms: np.ndarray
    global_step_routed_expert_max_mean_ratio: np.ndarray
    global_step_moe_max_mean_ratio: np.ndarray
    global_step_sorted_rank_ffn_ms: np.ndarray
    global_step_sorted_rank_local_routed_tokens: np.ndarray
    global_step_sorted_rank_local_active_experts: np.ndarray
    global_step_position_sorted_rank_ffn_ms: np.ndarray
    global_step_position_sorted_rank_local_routed_tokens: np.ndarray
    global_step_ffn_max_mean_ratio: np.ndarray
    global_step_other_ms: np.ndarray
    global_step_kinds: np.ndarray
    global_token_barrier_offsets: np.ndarray
    global_token_source_ranks: np.ndarray
    global_token_request_ids: np.ndarray
    global_token_position_ids: np.ndarray
    global_token_layer_destination_assignment_counts: np.ndarray
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
            "timing_backend": np.asarray([self.timing_backend]),
            "timing_scope": np.asarray([self.timing_scope]),
            "hybrid_spec_state_offload_mode": np.asarray(
                [self.hybrid_spec_state_offload_mode]
            ),
            "hybrid_spec_state_ewma_alpha": np.asarray(
                [self.hybrid_spec_state_ewma_alpha], dtype=np.float64
            ),
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
            "vllm_generation_elapsed_ms": np.asarray(
                [self.vllm_generation_elapsed_ms], dtype=np.float64
            ),
            "vllm_request_tpot_ms": np.asarray(
                [self.vllm_request_tpot_ms], dtype=np.float64
            ),
            "vllm_generation_throughput_tok_s": np.asarray(
                [self.vllm_generation_throughput_tok_s], dtype=np.float64
            ),
            "spec_num_drafts": np.asarray(
                [self.spec_num_drafts], dtype=np.int64
            ),
            "spec_num_draft_tokens": np.asarray(
                [self.spec_num_draft_tokens], dtype=np.int64
            ),
            "spec_num_accepted_tokens": np.asarray(
                [self.spec_num_accepted_tokens], dtype=np.int64
            ),
            "spec_acceptance_rate": np.asarray(
                [self.spec_acceptance_rate], dtype=np.float64
            ),
            "spec_mean_acceptance_length": np.asarray(
                [self.spec_mean_acceptance_length], dtype=np.float64
            ),
            "hybrid_prediction_total": np.asarray(
                [self.hybrid_prediction_total], dtype=np.int64
            ),
            "hybrid_prediction_exact_match": np.asarray(
                [self.hybrid_prediction_exact_match], dtype=np.int64
            ),
            "hybrid_prediction_within_one": np.asarray(
                [self.hybrid_prediction_within_one], dtype=np.int64
            ),
            "hybrid_prediction_abs_error_sum": np.asarray(
                [self.hybrid_prediction_abs_error_sum], dtype=np.int64
            ),
            "hybrid_prediction_signed_error_sum": np.asarray(
                [self.hybrid_prediction_signed_error_sum], dtype=np.int64
            ),
            "hybrid_prediction_predicted_sum": np.asarray(
                [self.hybrid_prediction_predicted_sum], dtype=np.int64
            ),
            "hybrid_prediction_accepted_sum": np.asarray(
                [self.hybrid_prediction_accepted_sum], dtype=np.int64
            ),
            "hybrid_reload_preload_total_ms": np.asarray(
                [self.hybrid_reload_preload_total_ms], dtype=np.float64
            ),
            "hybrid_reload_preload_call_count": np.asarray(
                [self.hybrid_reload_preload_call_count], dtype=np.int64
            ),
            "hybrid_reload_preload_req_count": np.asarray(
                [self.hybrid_reload_preload_req_count], dtype=np.int64
            ),
            "hybrid_reload_preloaded_total_ms": np.asarray(
                [self.hybrid_reload_preloaded_total_ms], dtype=np.float64
            ),
            "hybrid_reload_preloaded_row_count": np.asarray(
                [self.hybrid_reload_preloaded_row_count], dtype=np.int64
            ),
            "hybrid_reload_fallback_total_ms": np.asarray(
                [self.hybrid_reload_fallback_total_ms], dtype=np.float64
            ),
            "hybrid_reload_fallback_row_count": np.asarray(
                [self.hybrid_reload_fallback_row_count], dtype=np.int64
            ),
            "hybrid_replay_prepare_copy_ms": np.asarray(
                [self.hybrid_replay_prepare_copy_ms], dtype=np.float64
            ),
            "hybrid_replay_repair_compute_ms": np.asarray(
                [self.hybrid_replay_repair_compute_ms], dtype=np.float64
            ),
            "hybrid_replay_verify_attention_ms": np.asarray(
                [self.hybrid_replay_verify_attention_ms], dtype=np.float64
            ),
            "hybrid_replay_spill_copy_ms": np.asarray(
                [self.hybrid_replay_spill_copy_ms], dtype=np.float64
            ),
            "hybrid_replay_layer_total_ms": np.asarray(
                [self.hybrid_replay_layer_total_ms], dtype=np.float64
            ),
            "hybrid_replay_verify_call_count": np.asarray(
                [self.hybrid_replay_verify_call_count], dtype=np.int64
            ),
            "hybrid_replay_checkpoint_save_ms": np.asarray(
                [self.hybrid_replay_checkpoint_save_ms], dtype=np.float64
            ),
            "hybrid_replay_post_replay_state_gather_ms": np.asarray(
                [self.hybrid_replay_post_replay_state_gather_ms],
                dtype=np.float64,
            ),
            "hybrid_replay_capture_materialize_ms": np.asarray(
                [self.hybrid_replay_capture_materialize_ms],
                dtype=np.float64,
            ),
            "hybrid_replay_segment_start_save_ms": np.asarray(
                [self.hybrid_replay_segment_start_save_ms],
                dtype=np.float64,
            ),
            "hybrid_replay_segment_start_wait_ms": np.asarray(
                [self.hybrid_replay_segment_start_wait_ms],
                dtype=np.float64,
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
            "rank_execute_wall_ms": self.rank_execute_wall_ms,
            "rank_verification_wall_ms": self.rank_verification_wall_ms,
            "rank_draft_wall_ms": self.rank_draft_wall_ms,
            "rank_iteration_wall_ms": self.rank_iteration_wall_ms,
            "rank_execute_gpu_ms": self.rank_execute_gpu_ms,
            "rank_verification_gpu_ms": self.rank_verification_gpu_ms,
            "rank_draft_gpu_ms": self.rank_draft_gpu_ms,
            "rank_iteration_gpu_ms": self.rank_iteration_gpu_ms,
            "rank_attention_gpu_ms": self.rank_attention_gpu_ms,
            "rank_moe_gpu_ms": self.rank_moe_gpu_ms,
            "rank_gpu_other_ms": self.rank_gpu_other_ms,
            "rank_timing_complete": self.rank_timing_complete,
            "rank_step_total_ms": self.rank_step_total_ms,
            "rank_step_draft_ms": self.rank_step_draft_ms,
            "rank_layer_moe_gpu_ms": self.rank_layer_moe_gpu_ms,
            "rank_layer_routed_expert_gpu_ms": (
                self.rank_layer_routed_expert_gpu_ms
            ),
            "rank_layer_shared_expert_gpu_ms": (
                self.rank_layer_shared_expert_gpu_ms
            ),
            "rank_layer_routing_gpu_ms": self.rank_layer_routing_gpu_ms,
            "rank_layer_prepare_gpu_ms": self.rank_layer_prepare_gpu_ms,
            "rank_layer_finalize_gpu_ms": self.rank_layer_finalize_gpu_ms,
            "rank_layer_ffn_ms": self.rank_layer_ffn_ms,
            "rank_layer_local_routed_tokens": (
                self.rank_layer_local_routed_tokens
            ),
            "rank_layer_local_active_experts": (
                self.rank_layer_local_active_experts
            ),
            "rank_position_layer_ffn_ms": self.rank_position_layer_ffn_ms,
            "rank_position_layer_local_routed_tokens": (
                self.rank_position_layer_local_routed_tokens
            ),
            "global_step_indices": self.global_step_indices,
            "global_step_total_ms": self.global_step_total_ms,
            "global_draft_ms": self.global_draft_ms,
            "global_step_ffn_ms": self.global_step_ffn_ms,
            "global_step_ffn_phase_ms": self.global_step_ffn_ms,
            "global_critical_rank_indices": self.global_critical_rank_indices,
            "global_verification_wall_ms": self.global_verification_wall_ms,
            "global_iteration_wall_ms": self.global_iteration_wall_ms,
            "global_draft_wall_ms": self.global_draft_wall_ms,
            "global_verification_gpu_total_ms": (
                self.global_verification_gpu_total_ms
            ),
            "global_attention_gpu_ms": self.global_attention_gpu_ms,
            "global_moe_gpu_ms": self.global_moe_gpu_ms,
            "global_gpu_other_ms": self.global_gpu_other_ms,
            "global_step_sorted_rank_routed_expert_gpu_ms": (
                self.global_step_sorted_rank_routed_expert_gpu_ms
            ),
            "global_step_sorted_rank_moe_gpu_ms": (
                self.global_step_sorted_rank_moe_gpu_ms
            ),
            "global_step_routed_expert_max_mean_ratio": (
                self.global_step_routed_expert_max_mean_ratio
            ),
            "global_step_moe_max_mean_ratio": (
                self.global_step_moe_max_mean_ratio
            ),
            "global_step_sorted_rank_ffn_ms": self.global_step_sorted_rank_ffn_ms,
            "global_step_sorted_rank_local_routed_tokens": (
                self.global_step_sorted_rank_local_routed_tokens
            ),
            "global_step_sorted_rank_local_active_experts": (
                self.global_step_sorted_rank_local_active_experts
            ),
            "global_step_position_sorted_rank_ffn_ms": (
                self.global_step_position_sorted_rank_ffn_ms
            ),
            "global_step_position_sorted_rank_local_routed_tokens": (
                self.global_step_position_sorted_rank_local_routed_tokens
            ),
            "global_step_ffn_max_mean_ratio": self.global_step_ffn_max_mean_ratio,
            "global_step_other_ms": self.global_step_other_ms,
            "global_step_kinds": self.global_step_kinds,
            "global_token_barrier_offsets": self.global_token_barrier_offsets,
            "global_token_source_ranks": self.global_token_source_ranks,
            "global_token_request_ids": self.global_token_request_ids,
            "global_token_position_ids": self.global_token_position_ids,
            "global_token_layer_destination_assignment_counts": (
                self.global_token_layer_destination_assignment_counts
            ),
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
    hybrid_spec_state_offload_mode: str
    hybrid_spec_state_ewma_alpha: float
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
    vllm_generation_elapsed_ms: float
    vllm_request_tpot_ms: float
    vllm_generation_throughput_tok_s: float
    spec_num_drafts: int
    spec_num_draft_tokens: int
    spec_num_accepted_tokens: int
    spec_acceptance_rate: float
    spec_mean_acceptance_length: float
    hybrid_prediction_total: int
    hybrid_prediction_exact_match: int
    hybrid_prediction_within_one: int
    hybrid_prediction_abs_error_sum: int
    hybrid_prediction_signed_error_sum: int
    hybrid_prediction_predicted_sum: int
    hybrid_prediction_accepted_sum: int
    hybrid_reload_preload_total_ms: float
    hybrid_reload_preload_call_count: int
    hybrid_reload_preload_req_count: int
    hybrid_reload_preloaded_total_ms: float
    hybrid_reload_preloaded_row_count: int
    hybrid_reload_fallback_total_ms: float
    hybrid_reload_fallback_row_count: int
    hybrid_replay_prepare_copy_ms: float
    hybrid_replay_repair_compute_ms: float
    hybrid_replay_verify_attention_ms: float
    hybrid_replay_spill_copy_ms: float
    hybrid_replay_layer_total_ms: float
    hybrid_replay_verify_call_count: int
    hybrid_replay_checkpoint_save_ms: float
    hybrid_replay_post_replay_state_gather_ms: float
    hybrid_replay_capture_materialize_ms: float
    hybrid_replay_segment_start_save_ms: float
    hybrid_replay_segment_start_wait_ms: float
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
    candidate_execute_wall_ms: np.ndarray
    candidate_verification_wall_ms: np.ndarray
    candidate_draft_wall_ms: np.ndarray
    candidate_iteration_wall_ms: np.ndarray
    candidate_execute_gpu_ms: np.ndarray
    candidate_verification_gpu_ms: np.ndarray
    candidate_draft_gpu_ms: np.ndarray
    candidate_iteration_gpu_ms: np.ndarray
    candidate_attention_gpu_ms: np.ndarray
    candidate_moe_gpu_ms: np.ndarray
    candidate_gpu_other_ms: np.ndarray
    candidate_timing_complete: np.ndarray
    candidate_step_histograms: np.ndarray
    candidate_layer_ffn_ms: np.ndarray
    candidate_layer_moe_gpu_ms: np.ndarray
    candidate_layer_routed_expert_gpu_ms: np.ndarray
    candidate_layer_shared_expert_gpu_ms: np.ndarray
    candidate_layer_routing_gpu_ms: np.ndarray
    candidate_layer_prepare_gpu_ms: np.ndarray
    candidate_layer_finalize_gpu_ms: np.ndarray
    candidate_layer_local_routed_tokens: np.ndarray
    candidate_layer_local_active_experts: np.ndarray
    candidate_position_layer_ffn_ms: np.ndarray
    candidate_position_layer_local_routed_tokens: np.ndarray
    candidate_token_offsets: np.ndarray
    candidate_token_request_ids: np.ndarray
    candidate_token_position_ids: np.ndarray
    candidate_token_layer_destination_assignment_counts: np.ndarray
    expert_to_ep_rank: np.ndarray
    condition_latency_ms: float
    decode_time_total_ms: float
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int
    vllm_generation_tokens_total: int
    vllm_request_tpot_total_ms: float
    vllm_request_tpot_count: int
    spec_num_drafts: int
    spec_num_draft_tokens: int
    spec_num_accepted_tokens: int
    hybrid_prediction_total: int
    hybrid_prediction_exact_match: int
    hybrid_prediction_within_one: int
    hybrid_prediction_abs_error_sum: int
    hybrid_prediction_signed_error_sum: int
    hybrid_prediction_predicted_sum: int
    hybrid_prediction_accepted_sum: int
    hybrid_reload_preload_total_ms: float
    hybrid_reload_preload_call_count: int
    hybrid_reload_preload_req_count: int
    hybrid_reload_preloaded_total_ms: float
    hybrid_reload_preloaded_row_count: int
    hybrid_reload_fallback_total_ms: float
    hybrid_reload_fallback_row_count: int
    hybrid_replay_prepare_copy_ms: float
    hybrid_replay_repair_compute_ms: float
    hybrid_replay_verify_attention_ms: float
    hybrid_replay_spill_copy_ms: float
    hybrid_replay_layer_total_ms: float
    hybrid_replay_verify_call_count: int
    hybrid_replay_checkpoint_save_ms: float
    hybrid_replay_post_replay_state_gather_ms: float
    hybrid_replay_capture_materialize_ms: float
    hybrid_replay_segment_start_save_ms: float
    hybrid_replay_segment_start_wait_ms: float
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
    hybrid_prediction_trace_events: list[dict[str, int | str]] = field(
        default_factory=list
    )

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
            "candidate_execute_wall_ms": self.candidate_execute_wall_ms,
            "candidate_verification_wall_ms": (
                self.candidate_verification_wall_ms
            ),
            "candidate_draft_wall_ms": self.candidate_draft_wall_ms,
            "candidate_iteration_wall_ms": self.candidate_iteration_wall_ms,
            "candidate_execute_gpu_ms": self.candidate_execute_gpu_ms,
            "candidate_verification_gpu_ms": (
                self.candidate_verification_gpu_ms
            ),
            "candidate_draft_gpu_ms": self.candidate_draft_gpu_ms,
            "candidate_iteration_gpu_ms": self.candidate_iteration_gpu_ms,
            "candidate_attention_gpu_ms": self.candidate_attention_gpu_ms,
            "candidate_moe_gpu_ms": self.candidate_moe_gpu_ms,
            "candidate_gpu_other_ms": self.candidate_gpu_other_ms,
            "candidate_timing_complete": self.candidate_timing_complete,
            "candidate_step_histograms": self.candidate_step_histograms,
            "candidate_layer_ffn_ms": self.candidate_layer_ffn_ms,
            "candidate_layer_moe_gpu_ms": self.candidate_layer_moe_gpu_ms,
            "candidate_layer_routed_expert_gpu_ms": (
                self.candidate_layer_routed_expert_gpu_ms
            ),
            "candidate_layer_shared_expert_gpu_ms": (
                self.candidate_layer_shared_expert_gpu_ms
            ),
            "candidate_layer_routing_gpu_ms": (
                self.candidate_layer_routing_gpu_ms
            ),
            "candidate_layer_prepare_gpu_ms": (
                self.candidate_layer_prepare_gpu_ms
            ),
            "candidate_layer_finalize_gpu_ms": (
                self.candidate_layer_finalize_gpu_ms
            ),
            "candidate_layer_local_routed_tokens": (
                self.candidate_layer_local_routed_tokens
            ),
            "candidate_layer_local_active_experts": (
                self.candidate_layer_local_active_experts
            ),
            "candidate_position_layer_ffn_ms": (
                self.candidate_position_layer_ffn_ms
            ),
            "candidate_position_layer_local_routed_tokens": (
                self.candidate_position_layer_local_routed_tokens
            ),
            "candidate_token_offsets": self.candidate_token_offsets,
            "candidate_token_request_ids": self.candidate_token_request_ids,
            "candidate_token_position_ids": self.candidate_token_position_ids,
            "candidate_token_layer_destination_assignment_counts": (
                self.candidate_token_layer_destination_assignment_counts
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
            "vllm_generation_tokens_total": np.asarray(
                [self.vllm_generation_tokens_total], dtype=np.int64
            ),
            "vllm_request_tpot_total_ms": np.asarray(
                [self.vllm_request_tpot_total_ms], dtype=np.float64
            ),
            "vllm_request_tpot_count": np.asarray(
                [self.vllm_request_tpot_count], dtype=np.int64
            ),
            "spec_num_drafts": np.asarray(
                [self.spec_num_drafts], dtype=np.int64
            ),
            "spec_num_draft_tokens": np.asarray(
                [self.spec_num_draft_tokens], dtype=np.int64
            ),
            "spec_num_accepted_tokens": np.asarray(
                [self.spec_num_accepted_tokens], dtype=np.int64
            ),
            "hybrid_prediction_total": np.asarray(
                [self.hybrid_prediction_total], dtype=np.int64
            ),
            "hybrid_prediction_exact_match": np.asarray(
                [self.hybrid_prediction_exact_match], dtype=np.int64
            ),
            "hybrid_prediction_within_one": np.asarray(
                [self.hybrid_prediction_within_one], dtype=np.int64
            ),
            "hybrid_prediction_abs_error_sum": np.asarray(
                [self.hybrid_prediction_abs_error_sum], dtype=np.int64
            ),
            "hybrid_prediction_signed_error_sum": np.asarray(
                [self.hybrid_prediction_signed_error_sum], dtype=np.int64
            ),
            "hybrid_prediction_predicted_sum": np.asarray(
                [self.hybrid_prediction_predicted_sum], dtype=np.int64
            ),
            "hybrid_prediction_accepted_sum": np.asarray(
                [self.hybrid_prediction_accepted_sum], dtype=np.int64
            ),
            "hybrid_reload_preload_total_ms": np.asarray(
                [self.hybrid_reload_preload_total_ms], dtype=np.float64
            ),
            "hybrid_reload_preload_call_count": np.asarray(
                [self.hybrid_reload_preload_call_count], dtype=np.int64
            ),
            "hybrid_reload_preload_req_count": np.asarray(
                [self.hybrid_reload_preload_req_count], dtype=np.int64
            ),
            "hybrid_reload_preloaded_total_ms": np.asarray(
                [self.hybrid_reload_preloaded_total_ms], dtype=np.float64
            ),
            "hybrid_reload_preloaded_row_count": np.asarray(
                [self.hybrid_reload_preloaded_row_count], dtype=np.int64
            ),
            "hybrid_reload_fallback_total_ms": np.asarray(
                [self.hybrid_reload_fallback_total_ms], dtype=np.float64
            ),
            "hybrid_reload_fallback_row_count": np.asarray(
                [self.hybrid_reload_fallback_row_count], dtype=np.int64
            ),
            "hybrid_replay_prepare_copy_ms": np.asarray(
                [self.hybrid_replay_prepare_copy_ms], dtype=np.float64
            ),
            "hybrid_replay_repair_compute_ms": np.asarray(
                [self.hybrid_replay_repair_compute_ms], dtype=np.float64
            ),
            "hybrid_replay_verify_attention_ms": np.asarray(
                [self.hybrid_replay_verify_attention_ms], dtype=np.float64
            ),
            "hybrid_replay_spill_copy_ms": np.asarray(
                [self.hybrid_replay_spill_copy_ms], dtype=np.float64
            ),
            "hybrid_replay_layer_total_ms": np.asarray(
                [self.hybrid_replay_layer_total_ms], dtype=np.float64
            ),
            "hybrid_replay_verify_call_count": np.asarray(
                [self.hybrid_replay_verify_call_count], dtype=np.int64
            ),
            "hybrid_replay_checkpoint_save_ms": np.asarray(
                [self.hybrid_replay_checkpoint_save_ms], dtype=np.float64
            ),
            "hybrid_replay_post_replay_state_gather_ms": np.asarray(
                [self.hybrid_replay_post_replay_state_gather_ms],
                dtype=np.float64,
            ),
            "hybrid_replay_capture_materialize_ms": np.asarray(
                [self.hybrid_replay_capture_materialize_ms],
                dtype=np.float64,
            ),
            "hybrid_replay_segment_start_save_ms": np.asarray(
                [self.hybrid_replay_segment_start_save_ms],
                dtype=np.float64,
            ),
            "hybrid_replay_segment_start_wait_ms": np.asarray(
                [self.hybrid_replay_segment_start_wait_ms],
                dtype=np.float64,
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
class CudaEventInterval:
    label: str
    start_event: Any
    end_event: Any
    wall_start_s: float
    wall_end_s: float
    layer_idx: int | None
    ep_collective_seq_id: int | None
    in_draft_section: bool


class CudaEventPool:
    def __init__(self, event_factory: Callable[[], Any] | None = None) -> None:
        self._event_factory = event_factory
        self._available: list[Any] = []
        self.created = 0

    def _create_event(self) -> Any:
        if self._event_factory is not None:
            return self._event_factory()
        import torch

        return torch.cuda.Event(enable_timing=True)

    def acquire(self) -> Any:
        if self._available:
            return self._available.pop()
        self.created += 1
        return self._create_event()

    def release(self, event: Any) -> None:
        self._available.append(event)

    @property
    def available(self) -> int:
        return len(self._available)


@dataclass
class StepAccumulator:
    step_index: int
    execute_wall_start_s: float
    execute_start_event: Any
    execute_wall_end_s: float = 0.0
    execute_end_event: Any | None = None
    completion_event: Any | None = None
    first_ep_collective_seq_id: int | None = None
    last_ep_collective_seq_id: int | None = None
    num_ep_collectives: int = 0
    draft_depth: int = 0
    layer_stack: list[int] = field(default_factory=list)
    event_intervals: list[CudaEventInterval] = field(default_factory=list)
    draft_wall_intervals: list[tuple[float, float]] = field(default_factory=list)
    owned_events: list[Any] = field(default_factory=list)


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
    enable_nvtx_ranges: bool = False
    event_pool: CudaEventPool = field(default_factory=CudaEventPool)
    synchronize_count: int = 0


_WORKER_STATE = WorkerInstrumentationState()
_ORIGINAL_WORKER_EXECUTE_MODEL = None
_ORIGINAL_ROUTER_SELECT_EXPERTS = None
_ORIGINAL_QWEN2_MLP_FORWARD = None
_ORIGINAL_QWEN3_MLP_FORWARD = None
_ORIGINAL_QWEN3_SPARSE_MOE_FORWARD = None
_ORIGINAL_QWEN3_NEXT_SPARSE_MOE_FORWARD = None
_ORIGINAL_FUSED_MOE_FORWARD = None
_ORIGINAL_QWEN3_NEXT_ATTN_FORWARD = None
_ORIGINAL_QWEN_GDN_FORWARD = None
_ORIGINAL_MODULAR_PREPARE = None
_ORIGINAL_MODULAR_FUSED_EXPERTS = None
_ORIGINAL_MODULAR_FINALIZE = None
_ORIGINAL_MONOLITHIC_APPLY = None
_ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT = None


def condition_name(batch_size: int, draft_length: int) -> str:
    return f"batch_{batch_size:03d}_draft_{draft_length:02d}"


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path("results") / f"qwen3_6_mtp_dp_ep_{timestamp}"


def ensure_collect_dirs(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    prediction_trace_dir = output_dir / "_prediction_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_trace_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_dir,
        "raw": raw_dir,
        "prediction_traces": prediction_trace_dir,
    }


def condition_prediction_trace_dir(
    output_dir: Path,
    batch_size: int,
    draft_length: int,
) -> Path:
    return ensure_collect_dirs(output_dir)["prediction_traces"] / condition_name(
        batch_size, draft_length
    )


def rank_prediction_trace_path(
    output_dir: Path,
    *,
    batch_size: int,
    draft_length: int,
    dp_rank: int,
) -> Path:
    trace_dir = condition_prediction_trace_dir(
        output_dir,
        batch_size,
        draft_length,
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"rank_{dp_rank:02d}.npz"


def hybrid_prediction_trace_exists(
    output_dir: Path,
    *,
    batch_size: int,
    draft_length: int,
    data_parallel_size: int,
) -> bool:
    return all(
        rank_prediction_trace_path(
            output_dir,
            batch_size=batch_size,
            draft_length=draft_length,
            dp_rank=dp_rank,
        ).exists()
        for dp_rank in range(data_parallel_size)
    )


def save_rank_hybrid_prediction_trace(
    path: Path,
    args: Any,
    data: RankConditionData,
) -> None:
    trace_events = data.hybrid_prediction_trace_events
    payload = {
        "schema_version": np.asarray(
            [HYBRID_PREDICTION_TRACE_SCHEMA_VERSION],
            dtype=np.int64,
        ),
        "batch_size": np.asarray([args.batch_size], dtype=np.int64),
        "draft_length": np.asarray([args.draft_length], dtype=np.int64),
        "data_parallel_size": np.asarray([args.data_parallel_size], dtype=np.int64),
        "dp_rank": np.asarray([args.dp_rank], dtype=np.int64),
        "event_index": np.asarray(
            [int(event["event_index"]) for event in trace_events],
            dtype=np.int64,
        ),
        "req_event_index": np.asarray(
            [int(event["req_event_index"]) for event in trace_events],
            dtype=np.int64,
        ),
        "req_id": np.asarray(
            [str(event["req_id"]) for event in trace_events],
            dtype=np.str_,
        ),
        "accepted_len": np.asarray(
            [int(event["accepted_len"]) for event in trace_events],
            dtype=np.int64,
        ),
        "baseline_predicted_len": np.asarray(
            [int(event["baseline_predicted_len"]) for event in trace_events],
            dtype=np.int64,
        ),
        "effective_predicted_len": np.asarray(
            [int(event["effective_predicted_len"]) for event in trace_events],
            dtype=np.int64,
        ),
        "req_max_accept_len": np.asarray(
            [int(event["req_max_accept_len"]) for event in trace_events],
            dtype=np.int64,
        ),
        "draft_len": np.asarray(
            [int(event["draft_len"]) for event in trace_events],
            dtype=np.int64,
        ),
    }
    max_trace_tokens = max(
        (
            len(tuple(int(token_id) for token_id in event.get("output_token_ids", ())))
            for event in trace_events
        ),
        default=max(0, int(getattr(args, "draft_length", 0)) + 1),
    )
    if max_trace_tokens > 0:
        output_token_ids = np.full(
            (len(trace_events), max_trace_tokens),
            fill_value=-1,
            dtype=np.int64,
        )
        for row_idx, event in enumerate(trace_events):
            token_ids = tuple(
                int(token_id) for token_id in event.get("output_token_ids", ())
            )
            if token_ids:
                output_token_ids[row_idx, : len(token_ids)] = token_ids
        payload["output_token_ids"] = output_token_ids
    np.savez_compressed(path, **payload)


def load_rank_hybrid_prediction_trace(path: Path) -> list[dict[str, int | str]]:
    with np.load(path, allow_pickle=False) as data:
        schema_version = int(data["schema_version"][0])
        if schema_version != HYBRID_PREDICTION_TRACE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported hybrid prediction trace schema_version="
                f"{schema_version} for {path}."
            )
        event_indices = np.asarray(data["event_index"], dtype=np.int64)
        req_event_indices_arr = data.get("req_event_index")
        req_ids = np.asarray(data["req_id"], dtype=np.str_)
        accepted_lens = np.asarray(data["accepted_len"], dtype=np.int64)
        baseline_predicted_lens = np.asarray(
            data["baseline_predicted_len"],
            dtype=np.int64,
        )
        effective_predicted_lens = np.asarray(
            data["effective_predicted_len"],
            dtype=np.int64,
        )
        req_max_accept_lens = np.asarray(
            data["req_max_accept_len"],
            dtype=np.int64,
        )
        draft_lens = np.asarray(data["draft_len"], dtype=np.int64)
        output_token_ids_arr = data.get("output_token_ids")
        output_token_ids = (
            np.asarray(output_token_ids_arr, dtype=np.int64)
            if output_token_ids_arr is not None
            else None
        )
        if req_event_indices_arr is None:
            req_event_indices = []
            req_counts: dict[str, int] = {}
            for req_id in req_ids.tolist():
                req_event_index = int(req_counts.get(str(req_id), 0))
                req_event_indices.append(req_event_index)
                req_counts[str(req_id)] = req_event_index + 1
        else:
            req_event_indices = np.asarray(
                req_event_indices_arr,
                dtype=np.int64,
            ).tolist()
        return [
            {
                "event_index": int(event_index),
                "req_event_index": int(req_event_index),
                "req_id": str(req_id),
                "accepted_len": int(accepted_len),
                "baseline_predicted_len": int(baseline_predicted_len),
                "effective_predicted_len": int(effective_predicted_len),
                "req_max_accept_len": int(req_max_accept_len),
                "draft_len": int(draft_len),
                "output_token_ids": (
                    tuple(int(token_id) for token_id in output_token_ids[row_idx])
                    if output_token_ids is not None
                    else ()
                ),
            }
            for row_idx, (
                event_index,
                req_event_index,
                req_id,
                accepted_len,
                baseline_predicted_len,
                effective_predicted_len,
                req_max_accept_len,
                draft_len,
            ) in enumerate(
                zip(
                    event_indices,
                    req_event_indices,
                    req_ids,
                    accepted_lens,
                    baseline_predicted_lens,
                    effective_predicted_lens,
                    req_max_accept_lens,
                    draft_lens,
                    strict=True,
                )
            )
        ]


def _safe_collective_rpc(
    model_executor: Any,
    method: str | Callable[..., Any],
    *,
    timeout: float,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any | None:
    collective_rpc = getattr(model_executor, "collective_rpc", None)
    if collective_rpc is None:
        return None
    if getattr(model_executor, "rpc_broadcast_mq", None) is None:
        return None
    if getattr(model_executor, "is_failed", False):
        return None
    if getattr(model_executor, "shutting_down", False):
        return None
    try:
        return collective_rpc(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )
    except (AssertionError, RuntimeError, TimeoutError):
        return None


def load_oracle_trace_for_rank(
    args: Any,
    *,
    dp_rank: int,
) -> list[dict[str, int | str]]:
    oracle_root = getattr(args, "hybrid_prediction_oracle_trace_root", None)
    if oracle_root is None:
        raise ValueError(
            "Replay simulation requires --hybrid-prediction-oracle-trace-root."
        )
    trace_path = rank_prediction_trace_path(
        Path(oracle_root),
        batch_size=args.batch_size,
        draft_length=args.draft_length,
        dp_rank=dp_rank,
    )
    if not trace_path.exists():
        raise FileNotFoundError(
            "Missing oracle trace for replay simulation: "
            f"{trace_path}"
        )
    return load_rank_hybrid_prediction_trace(trace_path)


def prompt_cache_path(output_dir: Path) -> Path:
    return output_dir / "prompt_cache.json"


def add_hybrid_prediction_trace_args(parser: Any) -> None:
    parser.add_argument(
        "--hybrid-prediction-trace-mode",
        choices=HYBRID_PREDICTION_TRACE_MODE_CHOICES,
        default="off",
    )
    parser.add_argument(
        "--hybrid-prediction-oracle-trace-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--hybrid-prediction-target-accuracy",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--hybrid-prediction-sim-mode",
        choices=HYBRID_PREDICTION_SIM_MODE_CHOICES,
        default="exact_upper_bound",
    )
    parser.add_argument(
        "--hybrid-prediction-sim-seed",
        type=int,
        default=0,
    )


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
        "enable_nvtx_ranges": bool(
            getattr(args, "enable_nvtx_ranges", False)
        ),
        "hybrid_spec_state_offload_mode": (
            args.hybrid_spec_state_offload_mode
        ),
        "hybrid_spec_state_ewma_alpha": args.hybrid_spec_state_ewma_alpha,
        "timing_backend": TIMING_BACKEND,
        "timing_scope": TIMING_SCOPE,
        "local_gpu_ids": getattr(args, "local_gpu_ids", None),
        "rtx_5090_nccl_env_defaults": RTX_5090_NCCL_ENV_DEFAULTS,
        "enable_chunked_prefill": True,
        "mixed_step_policy": "include_all_global_barriers",
        "tpot_definition": TPOT_DEFINITION,
        "vllm_enable_v1_multiprocessing": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ),
        "hybrid_prediction_trace_mode": getattr(
            args,
            "hybrid_prediction_trace_mode",
            "off",
        ),
        "hybrid_prediction_oracle_trace_root": (
            None
            if getattr(args, "hybrid_prediction_oracle_trace_root", None) is None
            else str(args.hybrid_prediction_oracle_trace_root)
        ),
        "hybrid_prediction_target_accuracy": float(
            getattr(args, "hybrid_prediction_target_accuracy", 1.0)
        ),
        "hybrid_prediction_sim_mode": getattr(
            args,
            "hybrid_prediction_sim_mode",
            "exact_upper_bound",
        ),
        "hybrid_prediction_sim_seed": int(
            getattr(args, "hybrid_prediction_sim_seed", 0)
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
        "enable_nvtx_ranges": bool(
            getattr(args, "enable_nvtx_ranges", False)
        ),
        "hybrid_spec_state_offload_mode": (
            args.hybrid_spec_state_offload_mode
        ),
        "hybrid_spec_state_ewma_alpha": args.hybrid_spec_state_ewma_alpha,
        "timing_backend": TIMING_BACKEND,
        "timing_scope": TIMING_SCOPE,
        "local_gpu_ids": getattr(args, "local_gpu_ids", None),
        "rtx_5090_nccl_env_defaults": RTX_5090_NCCL_ENV_DEFAULTS,
        "enable_chunked_prefill": True,
        "mixed_step_policy": "include_all_global_barriers",
        "tpot_definition": TPOT_DEFINITION,
        "conditions": [
            {
                "batch_size": summary.batch_size,
                "draft_length": summary.draft_length,
                "raw_path": summary.raw_path,
                "hybrid_spec_state_offload_mode": (
                    summary.hybrid_spec_state_offload_mode
                ),
                "hybrid_spec_state_ewma_alpha": (
                    summary.hybrid_spec_state_ewma_alpha
                ),
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
                "vllm_generation_elapsed_ms": summary.vllm_generation_elapsed_ms,
                "vllm_request_tpot_ms": summary.vllm_request_tpot_ms,
                "vllm_generation_throughput_tok_s": (
                    summary.vllm_generation_throughput_tok_s
                ),
                "spec_num_drafts": summary.spec_num_drafts,
                "spec_num_draft_tokens": summary.spec_num_draft_tokens,
                "spec_num_accepted_tokens": summary.spec_num_accepted_tokens,
                "spec_acceptance_rate": summary.spec_acceptance_rate,
                "spec_mean_acceptance_length": (
                    summary.spec_mean_acceptance_length
                ),
                "hybrid_prediction_total": summary.hybrid_prediction_total,
                "hybrid_prediction_exact_match": (
                    summary.hybrid_prediction_exact_match
                ),
                "hybrid_prediction_within_one": (
                    summary.hybrid_prediction_within_one
                ),
                "hybrid_prediction_abs_error_sum": (
                    summary.hybrid_prediction_abs_error_sum
                ),
                "hybrid_prediction_signed_error_sum": (
                    summary.hybrid_prediction_signed_error_sum
                ),
                "hybrid_prediction_predicted_sum": (
                    summary.hybrid_prediction_predicted_sum
                ),
                "hybrid_prediction_accepted_sum": (
                    summary.hybrid_prediction_accepted_sum
                ),
                "hybrid_reload_preload_total_ms": (
                    summary.hybrid_reload_preload_total_ms
                ),
                "hybrid_reload_preload_call_count": (
                    summary.hybrid_reload_preload_call_count
                ),
                "hybrid_reload_preload_req_count": (
                    summary.hybrid_reload_preload_req_count
                ),
                "hybrid_reload_preloaded_total_ms": (
                    summary.hybrid_reload_preloaded_total_ms
                ),
                "hybrid_reload_preloaded_row_count": (
                    summary.hybrid_reload_preloaded_row_count
                ),
                "hybrid_reload_fallback_total_ms": (
                    summary.hybrid_reload_fallback_total_ms
                ),
                "hybrid_reload_fallback_row_count": (
                    summary.hybrid_reload_fallback_row_count
                ),
                "hybrid_replay_prepare_copy_ms": (
                    summary.hybrid_replay_prepare_copy_ms
                ),
                "hybrid_replay_repair_compute_ms": (
                    summary.hybrid_replay_repair_compute_ms
                ),
                "hybrid_replay_verify_attention_ms": (
                    summary.hybrid_replay_verify_attention_ms
                ),
                "hybrid_replay_spill_copy_ms": (
                    summary.hybrid_replay_spill_copy_ms
                ),
                "hybrid_replay_layer_total_ms": (
                    summary.hybrid_replay_layer_total_ms
                ),
                "hybrid_replay_verify_call_count": (
                    summary.hybrid_replay_verify_call_count
                ),
                "hybrid_replay_checkpoint_save_ms": (
                    summary.hybrid_replay_checkpoint_save_ms
                ),
                "hybrid_replay_post_replay_state_gather_ms": (
                    summary.hybrid_replay_post_replay_state_gather_ms
                ),
                "hybrid_replay_capture_materialize_ms": (
                    summary.hybrid_replay_capture_materialize_ms
                ),
                "hybrid_replay_segment_start_save_ms": (
                    summary.hybrid_replay_segment_start_save_ms
                ),
                "hybrid_replay_segment_start_wait_ms": (
                    summary.hybrid_replay_segment_start_wait_ms
                ),
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

        def read_optional_float(name: str) -> float:
            if name not in data:
                return 0.0
            return float(data[name][0])

        batch_size = int(data["batch_size"][0])
        draft_length = int(data["draft_length"][0])
        hybrid_spec_state_offload_mode = str(
            data["hybrid_spec_state_offload_mode"][0]
        )
        hybrid_spec_state_ewma_alpha = float(
            data["hybrid_spec_state_ewma_alpha"][0]
        )
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
        vllm_generation_elapsed_ms = float(data["vllm_generation_elapsed_ms"][0])
        vllm_request_tpot_ms = float(data["vllm_request_tpot_ms"][0])
        vllm_generation_throughput_tok_s = float(
            data["vllm_generation_throughput_tok_s"][0]
        )
        spec_num_drafts = int(data["spec_num_drafts"][0])
        spec_num_draft_tokens = int(data["spec_num_draft_tokens"][0])
        spec_num_accepted_tokens = int(data["spec_num_accepted_tokens"][0])
        spec_acceptance_rate = float(data["spec_acceptance_rate"][0])
        spec_mean_acceptance_length = float(
            data["spec_mean_acceptance_length"][0]
        )
        hybrid_prediction_total = read_optional_int("hybrid_prediction_total")
        hybrid_prediction_exact_match = read_optional_int(
            "hybrid_prediction_exact_match"
        )
        hybrid_prediction_within_one = read_optional_int(
            "hybrid_prediction_within_one"
        )
        hybrid_prediction_abs_error_sum = read_optional_int(
            "hybrid_prediction_abs_error_sum"
        )
        hybrid_prediction_signed_error_sum = read_optional_int(
            "hybrid_prediction_signed_error_sum"
        )
        hybrid_prediction_predicted_sum = read_optional_int(
            "hybrid_prediction_predicted_sum"
        )
        hybrid_prediction_accepted_sum = read_optional_int(
            "hybrid_prediction_accepted_sum"
        )
        hybrid_reload_preload_total_ms = read_optional_float(
            "hybrid_reload_preload_total_ms"
        )
        hybrid_reload_preload_call_count = read_optional_int(
            "hybrid_reload_preload_call_count"
        )
        hybrid_reload_preload_req_count = read_optional_int(
            "hybrid_reload_preload_req_count"
        )
        hybrid_reload_preloaded_total_ms = read_optional_float(
            "hybrid_reload_preloaded_total_ms"
        )
        hybrid_reload_preloaded_row_count = read_optional_int(
            "hybrid_reload_preloaded_row_count"
        )
        hybrid_reload_fallback_total_ms = read_optional_float(
            "hybrid_reload_fallback_total_ms"
        )
        hybrid_reload_fallback_row_count = read_optional_int(
            "hybrid_reload_fallback_row_count"
        )
        hybrid_replay_prepare_copy_ms = read_optional_float(
            "hybrid_replay_prepare_copy_ms"
        )
        hybrid_replay_repair_compute_ms = read_optional_float(
            "hybrid_replay_repair_compute_ms"
        )
        hybrid_replay_verify_attention_ms = read_optional_float(
            "hybrid_replay_verify_attention_ms"
        )
        hybrid_replay_spill_copy_ms = read_optional_float(
            "hybrid_replay_spill_copy_ms"
        )
        hybrid_replay_layer_total_ms = read_optional_float(
            "hybrid_replay_layer_total_ms"
        )
        hybrid_replay_verify_call_count = read_optional_int(
            "hybrid_replay_verify_call_count"
        )
        hybrid_replay_checkpoint_save_ms = read_optional_float(
            "hybrid_replay_checkpoint_save_ms"
        )
        hybrid_replay_post_replay_state_gather_ms = read_optional_float(
            "hybrid_replay_post_replay_state_gather_ms"
        )
        hybrid_replay_capture_materialize_ms = read_optional_float(
            "hybrid_replay_capture_materialize_ms"
        )
        hybrid_replay_segment_start_save_ms = read_optional_float(
            "hybrid_replay_segment_start_save_ms"
        )
        hybrid_replay_segment_start_wait_ms = read_optional_float(
            "hybrid_replay_segment_start_wait_ms"
        )
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
        hybrid_spec_state_offload_mode=hybrid_spec_state_offload_mode,
        hybrid_spec_state_ewma_alpha=hybrid_spec_state_ewma_alpha,
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
        vllm_generation_elapsed_ms=vllm_generation_elapsed_ms,
        vllm_request_tpot_ms=vllm_request_tpot_ms,
        vllm_generation_throughput_tok_s=vllm_generation_throughput_tok_s,
        spec_num_drafts=spec_num_drafts,
        spec_num_draft_tokens=spec_num_draft_tokens,
        spec_num_accepted_tokens=spec_num_accepted_tokens,
        spec_acceptance_rate=spec_acceptance_rate,
        spec_mean_acceptance_length=spec_mean_acceptance_length,
        hybrid_prediction_total=hybrid_prediction_total,
        hybrid_prediction_exact_match=hybrid_prediction_exact_match,
        hybrid_prediction_within_one=hybrid_prediction_within_one,
        hybrid_prediction_abs_error_sum=hybrid_prediction_abs_error_sum,
        hybrid_prediction_signed_error_sum=hybrid_prediction_signed_error_sum,
        hybrid_prediction_predicted_sum=hybrid_prediction_predicted_sum,
        hybrid_prediction_accepted_sum=hybrid_prediction_accepted_sum,
        hybrid_reload_preload_total_ms=hybrid_reload_preload_total_ms,
        hybrid_reload_preload_call_count=hybrid_reload_preload_call_count,
        hybrid_reload_preload_req_count=hybrid_reload_preload_req_count,
        hybrid_reload_preloaded_total_ms=hybrid_reload_preloaded_total_ms,
        hybrid_reload_preloaded_row_count=hybrid_reload_preloaded_row_count,
        hybrid_reload_fallback_total_ms=hybrid_reload_fallback_total_ms,
        hybrid_reload_fallback_row_count=hybrid_reload_fallback_row_count,
        hybrid_replay_prepare_copy_ms=hybrid_replay_prepare_copy_ms,
        hybrid_replay_repair_compute_ms=hybrid_replay_repair_compute_ms,
        hybrid_replay_verify_attention_ms=hybrid_replay_verify_attention_ms,
        hybrid_replay_spill_copy_ms=hybrid_replay_spill_copy_ms,
        hybrid_replay_layer_total_ms=hybrid_replay_layer_total_ms,
        hybrid_replay_verify_call_count=hybrid_replay_verify_call_count,
        hybrid_replay_checkpoint_save_ms=hybrid_replay_checkpoint_save_ms,
        hybrid_replay_post_replay_state_gather_ms=(
            hybrid_replay_post_replay_state_gather_ms
        ),
        hybrid_replay_capture_materialize_ms=(
            hybrid_replay_capture_materialize_ms
        ),
        hybrid_replay_segment_start_save_ms=(
            hybrid_replay_segment_start_save_ms
        ),
        hybrid_replay_segment_start_wait_ms=(
            hybrid_replay_segment_start_wait_ms
        ),
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
    if not args.enforce_eager:
        raise ValueError(
            "CUDA Event timing only supports eager execution. CUDA Graph mode "
            "is not supported; use the default --enforce-eager."
        )


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
            "hybrid_spec_state_offload_mode": (
                args.hybrid_spec_state_offload_mode
            ),
            "hybrid_spec_state_ewma_alpha": (
                args.hybrid_spec_state_ewma_alpha
            ),
        }
    mamba_cache_mode = (
        "align"
        if draft_length > 0
        and args.hybrid_spec_state_offload_mode != "disabled"
        else "none"
    )
    enable_prefix_caching = mamba_cache_mode == "align"

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
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode=mamba_cache_mode,
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


def _record_cuda_event(step: StepAccumulator) -> Any:
    import torch

    event = _WORKER_STATE.event_pool.acquire()
    event.record(torch.cuda.current_stream())
    step.owned_events.append(event)
    return event


def _nvtx_range_name(
    label: str,
    step: StepAccumulator,
    layer_idx: int | None,
) -> str:
    parts = [f"step={step.step_index}", label]
    if layer_idx is not None:
        parts.insert(1, f"layer={layer_idx}")
    return "/".join(parts)


def _push_nvtx_range(
    label: str,
    step: StepAccumulator,
    layer_idx: int | None,
) -> bool:
    if not _WORKER_STATE.enable_nvtx_ranges:
        return False
    import torch

    torch.cuda.nvtx.range_push(_nvtx_range_name(label, step, layer_idx))
    return True


def _pop_nvtx_range(pushed: bool) -> None:
    if not pushed:
        return
    import torch

    torch.cuda.nvtx.range_pop()


def _event_elapsed_ms(start_event: Any, end_event: Any) -> float:
    return float(start_event.elapsed_time(end_event))


def _measure_worker_section(
    label: str,
    fn: Callable,
    *args: Any,
    layer_idx: int | None = None,
    **kwargs: Any,
):
    current_step = _WORKER_STATE.current_step
    pending_record = (
        _WORKER_STATE.pending_step_records[-1]
        if _WORKER_STATE.pending_step_records
        else None
    )
    can_measure_pending_draft = (
        label == "draft" and current_step is None and pending_record is not None
    )
    if not _WORKER_STATE.enabled or (
        current_step is None and not can_measure_pending_draft
    ):
        return fn(*args, **kwargs)

    is_draft_section = label == "draft"
    if is_draft_section:
        if _WORKER_STATE.draft_measure_depth > 0:
            return fn(*args, **kwargs)
        _WORKER_STATE.draft_measure_depth += 1
    if can_measure_pending_draft:
        accumulator = pending_record["_accumulator"]
        try:
            start = time.perf_counter()
            start_event = _record_cuda_event(accumulator)
            pushed_nvtx = _push_nvtx_range("Draft", accumulator, None)
            result = fn(*args, **kwargs)
            end = time.perf_counter()
            _pop_nvtx_range(pushed_nvtx)
            end_event = _record_cuda_event(accumulator)
            accumulator.event_intervals.append(
                CudaEventInterval(
                    label="draft",
                    start_event=start_event,
                    end_event=end_event,
                    wall_start_s=start,
                    wall_end_s=end,
                    layer_idx=None,
                    ep_collective_seq_id=None,
                    in_draft_section=False,
                )
            )
            accumulator.draft_wall_intervals.append((start, end))
            accumulator.completion_event = end_event
            return result
        finally:
            _WORKER_STATE.draft_measure_depth -= 1

    assert current_step is not None
    if is_draft_section:
        current_step.draft_depth += 1
    try:
        in_draft_section = current_step.draft_depth > 0 and not is_draft_section
        ep_collective_seq_id = None
        if label in ("prepare", "finalize") and not in_draft_section:
            ep_collective_seq_id = _WORKER_STATE.next_ep_collective_seq_id
            _WORKER_STATE.next_ep_collective_seq_id += 1
            if current_step.first_ep_collective_seq_id is None:
                current_step.first_ep_collective_seq_id = ep_collective_seq_id
            current_step.last_ep_collective_seq_id = ep_collective_seq_id
            current_step.num_ep_collectives += 1

        start = time.perf_counter()
        start_event = _record_cuda_event(current_step)
        effective_layer_idx = (
            layer_idx
            if layer_idx is not None
            else (current_step.layer_stack[-1] if current_step.layer_stack else None)
        )
        pushed_nvtx = _push_nvtx_range(
            label.replace("_", " ").title().replace(" ", ""),
            current_step,
            effective_layer_idx,
        )
        try:
            result = fn(*args, **kwargs)
        finally:
            end = time.perf_counter()
            _pop_nvtx_range(pushed_nvtx)
            end_event = _record_cuda_event(current_step)
    finally:
        if is_draft_section:
            current_step.draft_depth -= 1
            _WORKER_STATE.draft_measure_depth -= 1

    current_step.event_intervals.append(
        CudaEventInterval(
            label=label,
            start_event=start_event,
            end_event=end_event,
            wall_start_s=start,
            wall_end_s=end,
            layer_idx=effective_layer_idx,
            ep_collective_seq_id=ep_collective_seq_id,
            in_draft_section=in_draft_section,
        )
    )
    if label == "draft":
        current_step.draft_wall_intervals.append((start, end))
        current_step.completion_event = end_event
    return result


def _resolve_step_accumulator(step: StepAccumulator) -> dict[str, Any]:
    completion_event = step.completion_event or step.execute_end_event
    if completion_event is None or step.execute_end_event is None:
        raise RuntimeError("Worker timing record is missing a completion Event.")
    completion_event.synchronize()
    _WORKER_STATE.synchronize_count += 1

    try:
        execute_gpu_interval = (
            0.0,
            _event_elapsed_ms(step.execute_start_event, step.execute_end_event),
        )
        parsed_events: list[dict[str, Any]] = []
        gpu_intervals_by_label: dict[str, list[tuple[float, float]]] = {}
        layer_gpu_ms: dict[str, dict[int, float]] = {}
        for interval in step.event_intervals:
            start_ms = _event_elapsed_ms(
                step.execute_start_event,
                interval.start_event,
            )
            end_ms = _event_elapsed_ms(
                step.execute_start_event,
                interval.end_event,
            )
            duration_ms = max(end_ms - start_ms, 0.0)
            parsed_events.append(
                {
                    "label": interval.label,
                    "layer_idx": interval.layer_idx,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                    "wall_start_ms": (
                        interval.wall_start_s - step.execute_wall_start_s
                    )
                    * 1000.0,
                    "wall_end_ms": (
                        interval.wall_end_s - step.execute_wall_start_s
                    )
                    * 1000.0,
                    "ep_collective_seq_id": interval.ep_collective_seq_id,
                    "in_draft_section": interval.in_draft_section,
                }
            )
            if interval.in_draft_section:
                continue
            gpu_intervals_by_label.setdefault(interval.label, []).append(
                (start_ms, end_ms)
            )
            if interval.layer_idx is not None:
                layer_values = layer_gpu_ms.setdefault(interval.label, {})
                layer_values[interval.layer_idx] = (
                    layer_values.get(interval.layer_idx, 0.0) + duration_ms
                )

        draft_gpu_intervals = gpu_intervals_by_label.get("draft", [])
        attention_intervals = gpu_intervals_by_label.get("attention", [])
        moe_intervals = gpu_intervals_by_label.get("moe", [])
        attention_moe_overlap_ms = interval_set_overlap_duration_ms(
            attention_intervals,
            moe_intervals,
        )
        if attention_moe_overlap_ms > 0.05:
            raise RuntimeError(
                "Attention and MoE top-level CUDA Event ranges overlap by "
                f"{attention_moe_overlap_ms:.6f} ms in step {step.step_index}."
            )

        execute_wall_interval = (
            step.execute_wall_start_s * 1000.0,
            step.execute_wall_end_s * 1000.0,
        )
        draft_wall_intervals_ms = [
            (start * 1000.0, end * 1000.0)
            for start, end in step.draft_wall_intervals
        ]
        execute_wall_ms = max(
            execute_wall_interval[1] - execute_wall_interval[0],
            0.0,
        )
        verification_wall_ms = subtract_interval_overlap_ms(
            execute_wall_interval,
            draft_wall_intervals_ms,
        )
        draft_wall_ms = interval_union_duration_ms(draft_wall_intervals_ms)
        iteration_wall_ms = interval_union_duration_ms(
            [execute_wall_interval, *draft_wall_intervals_ms]
        )
        execute_gpu_ms = execute_gpu_interval[1]
        verification_gpu_ms = subtract_interval_overlap_ms(
            execute_gpu_interval,
            draft_gpu_intervals,
        )
        draft_gpu_ms = interval_union_duration_ms(draft_gpu_intervals)
        iteration_gpu_ms = interval_union_duration_ms(
            [execute_gpu_interval, *draft_gpu_intervals]
        )
        attention_gpu_ms = interval_union_duration_ms(attention_intervals)
        moe_gpu_ms = interval_union_duration_ms(moe_intervals)
        gpu_other_ms = verification_gpu_ms - interval_union_duration_ms(
            [*attention_intervals, *moe_intervals]
        )
        if gpu_other_ms < -1e-3:
            raise RuntimeError(
                "Attention and MoE CUDA Event ranges exceed verification GPU "
                f"time in step {step.step_index}: other={gpu_other_ms:.6f} ms."
            )
        gpu_other_ms = max(gpu_other_ms, 0.0)

        label_totals = {
            label: interval_union_duration_ms(intervals)
            for label, intervals in gpu_intervals_by_label.items()
        }
        return {
            "timing": {
                "timing_backend": TIMING_BACKEND,
                "execute_wall_ms": execute_wall_ms,
                "verification_wall_ms": verification_wall_ms,
                "draft_wall_ms": draft_wall_ms,
                "iteration_wall_ms": iteration_wall_ms,
                "execute_gpu_ms": execute_gpu_ms,
                "verification_gpu_ms": verification_gpu_ms,
                "draft_gpu_ms": draft_gpu_ms,
                "iteration_gpu_ms": iteration_gpu_ms,
                "attention_gpu_ms": attention_gpu_ms,
                "moe_gpu_ms": moe_gpu_ms,
                "gpu_other_ms": gpu_other_ms,
                "routing_gpu_ms": label_totals.get("routing", 0.0),
                "prepare_gpu_ms": label_totals.get("prepare", 0.0),
                "routed_expert_gpu_ms": label_totals.get(
                    "routed_expert",
                    0.0,
                ),
                "shared_expert_gpu_ms": label_totals.get(
                    "shared_expert",
                    0.0,
                ),
                "finalize_gpu_ms": label_totals.get("finalize", 0.0),
                "timing_complete": True,
                # v9-compatible aliases. v10 analysis does not use these names.
                "total_ms": verification_wall_ms,
                "attention_ms": attention_gpu_ms,
                "routing_ms": label_totals.get("routing", 0.0),
                "prepare_ms": label_totals.get("prepare", 0.0),
                "finalize_ms": label_totals.get("finalize", 0.0),
                "ffn_ms": label_totals.get("routed_expert", 0.0),
                "draft_ms": draft_wall_ms,
            },
            "trace": {
                "step_index": step.step_index,
                "first_ep_collective_seq_id": step.first_ep_collective_seq_id,
                "last_ep_collective_seq_id": step.last_ep_collective_seq_id,
                "num_ep_collectives": step.num_ep_collectives,
                "step_start_time_ms": step.execute_wall_start_s * 1000.0,
                "step_end_time_ms": step.execute_wall_end_s * 1000.0,
                "events": parsed_events,
                "layer_gpu_ms": layer_gpu_ms,
            },
        }
    finally:
        for event in step.owned_events:
            _WORKER_STATE.event_pool.release(event)


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
    global _ORIGINAL_QWEN3_NEXT_SPARSE_MOE_FORWARD
    global _ORIGINAL_FUSED_MOE_FORWARD
    global _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD
    global _ORIGINAL_QWEN_GDN_FORWARD
    global _ORIGINAL_MODULAR_PREPARE
    global _ORIGINAL_MODULAR_FUSED_EXPERTS
    global _ORIGINAL_MODULAR_FINALIZE
    global _ORIGINAL_MONOLITHIC_APPLY
    global _ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT

    if _ORIGINAL_WORKER_EXECUTE_MODEL is not None:
        return

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
        FusedMoEKernelMonolithicImpl,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeMLP,
        Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextSparseMoeBlock,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    _ORIGINAL_WORKER_EXECUTE_MODEL = Worker.execute_model
    _ORIGINAL_ROUTER_SELECT_EXPERTS = BaseRouter.select_experts
    _ORIGINAL_QWEN2_MLP_FORWARD = Qwen2MoeMLP.forward
    _ORIGINAL_QWEN3_MLP_FORWARD = Qwen3MoeMLP.forward
    _ORIGINAL_QWEN3_SPARSE_MOE_FORWARD = Qwen3MoeSparseMoeBlock.forward
    _ORIGINAL_QWEN3_NEXT_SPARSE_MOE_FORWARD = Qwen3NextSparseMoeBlock.forward
    _ORIGINAL_FUSED_MOE_FORWARD = FusedMoE.forward
    _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD = Qwen3NextAttention.forward
    _ORIGINAL_QWEN_GDN_FORWARD = QwenGatedDeltaNetAttention.forward
    _ORIGINAL_MODULAR_PREPARE = FusedMoEKernelModularImpl._prepare
    _ORIGINAL_MODULAR_FUSED_EXPERTS = FusedMoEKernelModularImpl._fused_experts
    _ORIGINAL_MODULAR_FINALIZE = FusedMoEKernelModularImpl._finalize
    _ORIGINAL_MONOLITHIC_APPLY = FusedMoEKernelMonolithicImpl.apply
    _ORIGINAL_GPU_MODEL_RUNNER_PROPOSE_DRAFT = GPUModelRunner.propose_draft_token_ids

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
        start = time.perf_counter()
        import torch

        start_event = _WORKER_STATE.event_pool.acquire()
        start_event.record(torch.cuda.current_stream())
        _WORKER_STATE.current_step = StepAccumulator(
            step_index=step_index,
            execute_wall_start_s=start,
            execute_start_event=start_event,
            owned_events=[start_event],
        )
        pushed_nvtx = _push_nvtx_range(
            "Execute",
            _WORKER_STATE.current_step,
            None,
        )
        try:
            output = _ORIGINAL_WORKER_EXECUTE_MODEL(self, scheduler_output)
            end = time.perf_counter()
            assert _WORKER_STATE.current_step is not None
            _pop_nvtx_range(pushed_nvtx)
            pushed_nvtx = False
            end_event = _record_cuda_event(_WORKER_STATE.current_step)
            _WORKER_STATE.current_step.execute_wall_end_s = end
            _WORKER_STATE.current_step.execute_end_event = end_event
            _WORKER_STATE.current_step.completion_event = end_event
            _WORKER_STATE.pending_step_records.append(
                {
                    "_accumulator": _WORKER_STATE.current_step,
                    "metadata": _extract_worker_step_metadata(self),
                }
            )
            _WORKER_STATE.next_step_index += 1
            if _WORKER_STATE.queued_step_logs < 3:
                print(
                    "[worker_timing] queued "
                    f"step={step_index} backend={TIMING_BACKEND}",
                    flush=True,
                )
                _WORKER_STATE.queued_step_logs += 1
            return output
        finally:
            if _WORKER_STATE.current_step is not None and pushed_nvtx:
                _pop_nvtx_range(True)
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
            "shared_expert",
            _ORIGINAL_QWEN2_MLP_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen3_mlp_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "shared_expert",
            _ORIGINAL_QWEN3_MLP_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen3_sparse_moe_forward(self, *args, **kwargs):
        layer_idx = _extract_layer_index_from_module(self)
        pushed = _push_current_layer(layer_idx)
        try:
            return _measure_worker_section(
                "moe",
                _ORIGINAL_QWEN3_SPARSE_MOE_FORWARD,
                self,
                *args,
                layer_idx=layer_idx,
                **kwargs,
            )
        finally:
            _pop_current_layer(pushed)

    def patched_qwen3_next_sparse_moe_forward(self, *args, **kwargs):
        layer_idx = _extract_layer_index_from_module(self)
        pushed = _push_current_layer(layer_idx)
        try:
            return _measure_worker_section(
                "moe",
                _ORIGINAL_QWEN3_NEXT_SPARSE_MOE_FORWARD,
                self,
                *args,
                layer_idx=layer_idx,
                **kwargs,
            )
        finally:
            _pop_current_layer(pushed)

    def patched_fused_moe_forward(self, *args, **kwargs):
        pushed = _push_current_layer(_extract_layer_index_from_module(self))
        try:
            return _ORIGINAL_FUSED_MOE_FORWARD(self, *args, **kwargs)
        finally:
            _pop_current_layer(pushed)

    def patched_qwen3_next_attention_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "attention",
            _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD,
            self,
            *args,
            layer_idx=_extract_layer_index_from_module(self),
            **kwargs,
        )

    def patched_qwen_gdn_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "attention",
            _ORIGINAL_QWEN_GDN_FORWARD,
            self,
            *args,
            layer_idx=_extract_layer_index_from_module(self),
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
            "routed_expert",
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
            "routed_expert",
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

    Worker.execute_model = patched_worker_execute_model
    BaseRouter.select_experts = patched_router_select_experts
    Qwen2MoeMLP.forward = patched_qwen2_mlp_forward
    Qwen3MoeMLP.forward = patched_qwen3_mlp_forward
    Qwen3MoeSparseMoeBlock.forward = patched_qwen3_sparse_moe_forward
    Qwen3NextSparseMoeBlock.forward = patched_qwen3_next_sparse_moe_forward
    FusedMoE.forward = patched_fused_moe_forward
    Qwen3NextAttention.forward = patched_qwen3_next_attention_forward
    QwenGatedDeltaNetAttention.forward = patched_qwen_gdn_forward
    FusedMoEKernelModularImpl._prepare = patched_modular_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_modular_fused_experts
    FusedMoEKernelModularImpl._finalize = patched_modular_finalize
    FusedMoEKernelMonolithicImpl.apply = patched_monolithic_apply
    GPUModelRunner.propose_draft_token_ids = patched_propose_draft_token_ids


def install_experiment_hooks_worker(
    worker: Any,
    enable_nvtx_ranges: bool = False,
) -> bool:
    _install_worker_hooks()
    _WORKER_STATE.enabled = False
    _WORKER_STATE.pending_step_records.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.draft_measure_depth = 0
    _WORKER_STATE.enter_step_logs = 0
    _WORKER_STATE.queued_step_logs = 0
    _WORKER_STATE.next_step_index = 0
    _WORKER_STATE.next_ep_collective_seq_id = 0
    _WORKER_STATE.enable_nvtx_ranges = enable_nvtx_ranges
    _WORKER_STATE.synchronize_count = 0
    return True


def start_condition_collection_worker(worker: Any) -> bool:
    _WORKER_STATE.enabled = True
    _WORKER_STATE.pending_step_records.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.draft_measure_depth = 0
    _WORKER_STATE.enter_step_logs = 0
    _WORKER_STATE.queued_step_logs = 0
    _WORKER_STATE.synchronize_count = 0
    return True


def stop_condition_collection_worker(worker: Any) -> dict[str, int]:
    _WORKER_STATE.enabled = False
    pending = len(_WORKER_STATE.pending_step_records)
    while _WORKER_STATE.pending_step_records:
        pending_record = _WORKER_STATE.pending_step_records.popleft()
        accumulator = pending_record.get("_accumulator")
        if accumulator is None:
            continue
        completion_event = accumulator.completion_event
        if completion_event is not None:
            completion_event.synchronize()
        for event in accumulator.owned_events:
            _WORKER_STATE.event_pool.release(event)
    _WORKER_STATE.current_step = None
    _WORKER_STATE.draft_measure_depth = 0
    return {"pending_timings": pending}


def reset_hybrid_prediction_stats_worker(worker: Any) -> bool:
    model_runner = getattr(worker, "model_runner", None)
    reset = getattr(model_runner, "reset_hybrid_spec_prediction_stats", None)
    if reset is None:
        return False
    reset()
    return True


def collect_hybrid_prediction_stats_worker(worker: Any) -> dict[str, int]:
    model_runner = getattr(worker, "model_runner", None)
    snapshot = getattr(model_runner, "snapshot_hybrid_spec_prediction_stats", None)
    if snapshot is None:
        return _empty_hybrid_prediction_stats()
    stats = snapshot()
    return {
        "total_predictions": int(stats.total_predictions),
        "exact_match_count": int(stats.exact_match_count),
        "within_one_count": int(stats.within_one_count),
        "abs_error_sum": int(stats.abs_error_sum),
        "signed_error_sum": int(stats.signed_error_sum),
        "predicted_accept_len_sum": int(stats.predicted_accept_len_sum),
        "accepted_len_sum": int(stats.accepted_len_sum),
    }


def configure_hybrid_prediction_trace_worker(
    worker: Any,
    trace_mode: str,
    oracle_trace: list[dict[str, int | str]] | None = None,
    target_accuracy: float = 1.0,
    sim_mode: str = "exact_upper_bound",
    sim_seed: int = 0,
) -> dict[str, int | float | str]:
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return {
            "trace_mode": trace_mode,
            "trace_events": 0,
            "exact_match_events": 0,
        }
    reset_trace = getattr(model_runner, "reset_hybrid_spec_prediction_trace", None)
    if reset_trace is not None:
        reset_trace()
    disable_override = getattr(
        model_runner,
        "disable_hybrid_spec_prediction_override",
        None,
    )
    if disable_override is not None:
        disable_override()
    if trace_mode == "off":
        return {
            "trace_mode": trace_mode,
            "trace_events": 0,
            "exact_match_events": 0,
        }
    enable_trace = getattr(model_runner, "enable_hybrid_spec_prediction_trace", None)
    if enable_trace is not None:
        enable_trace()
    if trace_mode != "replay":
        return {
            "trace_mode": trace_mode,
            "trace_events": 0,
            "exact_match_events": 0,
        }
    if sim_mode != "exact_upper_bound":
        raise ValueError(f"Unsupported hybrid prediction sim_mode={sim_mode!r}.")
    oracle_trace = [] if oracle_trace is None else list(oracle_trace)
    total_events = len(oracle_trace)
    exact_match_events = int(round(float(target_accuracy) * total_events))
    exact_match_events = max(0, min(total_events, exact_match_events))
    if exact_match_events == total_events:
        exact_match_event_indices = set(range(total_events))
    elif exact_match_events == 0:
        exact_match_event_indices = set()
    else:
        rng = np.random.default_rng(int(sim_seed))
        exact_match_event_indices = {
            int(event_index)
            for event_index in rng.permutation(total_events)[:exact_match_events]
        }
    configure_override = getattr(
        model_runner,
        "configure_hybrid_spec_prediction_override",
        None,
    )
    if configure_override is None:
        raise RuntimeError(
            "GPUModelRunner does not expose hybrid prediction override hooks."
        )
    configure_override(
        mode=sim_mode,
        oracle_trace=oracle_trace,
        exact_match_event_indices=exact_match_event_indices,
    )
    return {
        "trace_mode": trace_mode,
        "trace_events": total_events,
        "exact_match_events": exact_match_events,
    }


def collect_hybrid_prediction_trace_worker(
    worker: Any,
) -> list[dict[str, int | str]]:
    model_runner = getattr(worker, "model_runner", None)
    snapshot = getattr(model_runner, "snapshot_hybrid_spec_prediction_trace", None)
    if snapshot is None:
        return []
    return list(snapshot())


def reset_hybrid_reload_timing_stats_worker(worker: Any) -> bool:
    model_runner = getattr(worker, "model_runner", None)
    reset = getattr(model_runner, "reset_hybrid_spec_reload_timing_stats", None)
    if reset is None:
        return False
    reset()
    return True


def collect_hybrid_reload_timing_stats_worker(
    worker: Any,
) -> dict[str, float | int]:
    model_runner = getattr(worker, "model_runner", None)
    snapshot = getattr(model_runner, "snapshot_hybrid_spec_reload_timing_stats", None)
    if snapshot is None:
        return _empty_hybrid_reload_timing_stats()
    stats = snapshot()
    if hasattr(stats, "preload_total_ms"):
        return {
            "preload_total_ms": float(stats.preload_total_ms),
            "preload_call_count": int(stats.preload_call_count),
            "preload_req_count": int(stats.preload_req_count),
            "preloaded_total_ms": float(stats.preloaded_total_ms),
            "preloaded_row_count": int(stats.preloaded_row_count),
            "fallback_total_ms": float(stats.fallback_total_ms),
            "fallback_row_count": int(stats.fallback_row_count),
            "prepare_copy_ms": 0.0,
            "repair_compute_ms": 0.0,
            "verify_attention_ms": 0.0,
            "spill_copy_ms": 0.0,
            "layer_total_ms": 0.0,
            "verify_call_count": 0,
            "checkpoint_save_ms": 0.0,
            "post_replay_state_gather_ms": 0.0,
            "capture_materialize_ms": 0.0,
            "segment_start_save_ms": 0.0,
            "segment_start_wait_ms": 0.0,
            "tape_save_ms": 0.0,
        }

    # Replay-based predict_last no longer reports the old preload/fallback
    # breakdown. Map it into the legacy experiment schema so existing collect
    # and analysis code can keep running while preserving the replay-native
    # counters for future consumers.
    repair_copy_ms = float(getattr(stats, "repair_copy_ms", 0.0))
    repair_compute_ms = float(getattr(stats, "repair_compute_ms", 0.0))
    repair_row_count = int(getattr(stats, "repair_row_count", 0))
    repair_from_start_count = int(
        getattr(stats, "repair_from_start_count", 0)
    )
    repair_from_resident_count = int(
        getattr(stats, "repair_from_resident_count", 0)
    )
    verify_attention_ms = float(getattr(stats, "verify_attention_ms", 0.0))
    layer_total_ms = float(getattr(stats, "layer_total_ms", 0.0))
    verify_call_count = int(getattr(stats, "verify_call_count", 0))
    checkpoint_save_ms = float(getattr(stats, "checkpoint_save_ms", 0.0))
    post_replay_state_gather_ms = float(
        getattr(stats, "post_replay_state_gather_ms", 0.0)
    )
    capture_materialize_ms = float(
        getattr(stats, "capture_materialize_ms", 0.0)
    )
    segment_start_save_ms = float(
        getattr(stats, "segment_start_save_ms", 0.0)
    )
    segment_start_wait_ms = float(
        getattr(stats, "segment_start_wait_ms", 0.0)
    )
    tape_save_ms = float(getattr(stats, "tape_save_ms", 0.0))
    return {
        "preload_total_ms": checkpoint_save_ms + tape_save_ms,
        "preload_call_count": 0,
        "preload_req_count": 0,
        "preloaded_total_ms": repair_copy_ms + repair_compute_ms,
        "preloaded_row_count": repair_row_count,
        "fallback_total_ms": 0.0,
        "fallback_row_count": 0,
        "repair_copy_ms": repair_copy_ms,
        "repair_compute_ms": repair_compute_ms,
        "repair_row_count": repair_row_count,
        "repair_from_start_count": repair_from_start_count,
        "repair_from_resident_count": repair_from_resident_count,
        "prepare_copy_ms": repair_copy_ms,
        "checkpoint_save_ms": checkpoint_save_ms,
        "post_replay_state_gather_ms": post_replay_state_gather_ms,
        "capture_materialize_ms": capture_materialize_ms,
        "segment_start_save_ms": segment_start_save_ms,
        "segment_start_wait_ms": segment_start_wait_ms,
        "tape_save_ms": tape_save_ms,
        "verify_attention_ms": verify_attention_ms,
        "spill_copy_ms": tape_save_ms,
        "layer_total_ms": layer_total_ms,
        "verify_call_count": verify_call_count,
    }


def pop_step_timing_worker(
    worker: Any,
    timeout_s: float = 5.0,
    poll_s: float = 0.01,
) -> dict[str, Any] | None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if _WORKER_STATE.pending_step_records:
            pending_record = _WORKER_STATE.pending_step_records.popleft()
            resolved = _resolve_step_accumulator(
                pending_record.pop("_accumulator")
            )
            resolved["metadata"] = pending_record.get("metadata")
            return resolved
        time.sleep(poll_s)
    if _WORKER_STATE.pending_step_records:
        pending_record = _WORKER_STATE.pending_step_records.popleft()
        resolved = _resolve_step_accumulator(pending_record.pop("_accumulator"))
        resolved["metadata"] = pending_record.get("metadata")
        return resolved
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
        step_start_time_ms + float(event.get("wall_start_ms", event["start_ms"]))
        for event in worker_trace["events"]
        if str(event["label"]) == "prepare"
    ]
    finalize_events = [
        step_start_time_ms + float(event.get("wall_end_ms", event["end_ms"]))
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
        draft_length: int,
        layers: tuple[int, ...],
        num_experts: int,
        expert_to_ep_rank: np.ndarray,
        local_ep_rank: int,
        trace_steps_limit: int = 0,
    ) -> None:
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.use_spec_decode = use_spec_decode
        self.max_verification_positions = draft_length + 1 if use_spec_decode else 1
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
        self.candidate_execute_wall_ms: list[float] = []
        self.candidate_verification_wall_ms: list[float] = []
        self.candidate_draft_wall_ms: list[float] = []
        self.candidate_iteration_wall_ms: list[float] = []
        self.candidate_execute_gpu_ms: list[float] = []
        self.candidate_verification_gpu_ms: list[float] = []
        self.candidate_draft_gpu_ms: list[float] = []
        self.candidate_iteration_gpu_ms: list[float] = []
        self.candidate_attention_gpu_ms: list[float] = []
        self.candidate_moe_gpu_ms: list[float] = []
        self.candidate_gpu_other_ms: list[float] = []
        self.candidate_timing_complete: list[bool] = []
        self.candidate_step_histograms: list[np.ndarray] = []
        self.candidate_layer_ffn_ms: list[np.ndarray] = []
        self.candidate_layer_moe_gpu_ms: list[np.ndarray] = []
        self.candidate_layer_routed_expert_gpu_ms: list[np.ndarray] = []
        self.candidate_layer_shared_expert_gpu_ms: list[np.ndarray] = []
        self.candidate_layer_routing_gpu_ms: list[np.ndarray] = []
        self.candidate_layer_prepare_gpu_ms: list[np.ndarray] = []
        self.candidate_layer_finalize_gpu_ms: list[np.ndarray] = []
        self.candidate_layer_local_routed_tokens: list[np.ndarray] = []
        self.candidate_layer_local_active_experts: list[np.ndarray] = []
        self.candidate_position_layer_ffn_ms: list[np.ndarray] = []
        self.candidate_position_layer_local_routed_tokens: list[np.ndarray] = []
        self.candidate_token_request_ids: list[np.ndarray] = []
        self.candidate_token_position_ids: list[np.ndarray] = []
        self.candidate_token_layer_destination_assignment_counts: list[
            np.ndarray
        ] = []
        self.num_forward_steps_total = 0
        self.num_dropped_steps = 0
        self.num_prefill_dropped_steps = 0
        self.num_mixed_dropped_steps = 0
        self.debug_update_logs = 0
        self.trace_steps_limit = trace_steps_limit
        self.trace_samples: list[dict[str, Any]] = []

    def _build_layer_gpu_ms(
        self,
        worker_records: list[dict[str, Any] | None],
        label: str,
    ) -> np.ndarray:
        values = np.zeros((len(self.layers),), dtype=np.float64)
        layer_to_row = {layer: row for row, layer in enumerate(self.layers)}
        for record in worker_records:
            if record is None:
                continue
            trace = record.get("trace")
            if not trace:
                continue
            layer_values = trace.get("layer_gpu_ms", {}).get(label, {})
            for raw_layer, elapsed_ms in layer_values.items():
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

    def _build_position_local_routed_tokens(
        self,
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> np.ndarray:
        routed_experts = getattr(model_runner_output, "routed_experts", None)
        if routed_experts is None:
            return np.zeros(
                (self.max_verification_positions, len(self.layers)),
                dtype=np.int64,
            )
        routing_data = np.asarray(routed_experts.routing_data)
        if routing_data.size == 0:
            return np.zeros(
                (self.max_verification_positions, len(self.layers)),
                dtype=np.int64,
            )
        request_ids = tuple(getattr(model_runner_output, "req_ids", ()))
        num_scheduled_tokens = dict(
            getattr(scheduler_output, "num_scheduled_tokens", {})
        )
        return count_position_layer_local_routed_tokens(
            routing_data,
            request_ids,
            num_scheduled_tokens,
            max_positions=self.max_verification_positions,
            expert_to_ep_rank=self.expert_to_ep_rank,
            local_ep_rank=self.local_ep_rank,
            layers=self.layers,
        )

    def _attribute_position_layer_ffn_ms(
        self,
        layer_ffn_ms: np.ndarray,
        position_layer_tokens: np.ndarray,
    ) -> np.ndarray:
        position_layer_ffn_ms = np.zeros_like(
            position_layer_tokens,
            dtype=np.float64,
        )
        routed_totals = position_layer_tokens.sum(axis=0)
        nonzero = routed_totals > 0
        position_layer_ffn_ms[:, nonzero] = (
            position_layer_tokens[:, nonzero].astype(np.float64)
            / routed_totals[nonzero].astype(np.float64)
        ) * layer_ffn_ms[nonzero]
        return position_layer_ffn_ms

    def _build_token_assignments(
        self,
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        routed_experts = getattr(model_runner_output, "routed_experts", None)
        if routed_experts is None:
            return (
                np.empty((0,), dtype=np.str_),
                np.empty((0,), dtype=np.int16),
                np.empty(
                    (0, len(self.layers), int(self.expert_to_ep_rank.max()) + 1),
                    dtype=np.int16,
                ),
            )
        routing_data = np.asarray(routed_experts.routing_data)
        request_ids = tuple(getattr(model_runner_output, "req_ids", ()))
        num_scheduled_tokens = dict(
            getattr(scheduler_output, "num_scheduled_tokens", {})
        )
        ep_size = int(self.expert_to_ep_rank.max()) + 1
        return build_token_layer_destination_assignments(
            routing_data,
            request_ids,
            num_scheduled_tokens,
            expert_to_ep_rank=self.expert_to_ep_rank,
            ep_size=ep_size,
            layers=self.layers,
        )

    def __enter__(self) -> SchedulerStepRecorder:
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
            valid_records = [
                record for record in worker_records if record is not None
            ]
            if not valid_records:
                raise RuntimeError("No CUDA Event timing record was returned.")
            timing_record = max(
                valid_records,
                key=lambda record: float(
                    record["timing"]["verification_wall_ms"]
                ),
            )
            timing = timing_record["timing"]
            draft_ms = float(timing["draft_wall_ms"])
            step_timing = StepTiming(
                total_ms=float(timing["verification_wall_ms"]),
                attention_ms=float(timing["attention_gpu_ms"]),
                routing_ms=float(timing["routing_gpu_ms"]),
                prepare_ms=float(timing["prepare_gpu_ms"]),
                finalize_ms=float(timing["finalize_gpu_ms"]),
                ffn_ms=float(timing["routed_expert_gpu_ms"]),
            )
            layer_moe_gpu_ms = self._build_layer_gpu_ms(
                worker_records,
                "moe",
            )
            layer_routed_expert_gpu_ms = self._build_layer_gpu_ms(
                worker_records,
                "routed_expert",
            )
            layer_shared_expert_gpu_ms = self._build_layer_gpu_ms(
                worker_records,
                "shared_expert",
            )
            layer_routing_gpu_ms = self._build_layer_gpu_ms(
                worker_records,
                "routing",
            )
            layer_prepare_gpu_ms = self._build_layer_gpu_ms(
                worker_records,
                "prepare",
            )
            layer_finalize_gpu_ms = self._build_layer_gpu_ms(
                worker_records,
                "finalize",
            )
            layer_ffn_ms = layer_routed_expert_gpu_ms
            timing_complete = bool(timing["timing_complete"]) and bool(
                np.all(layer_moe_gpu_ms > 0.0)
                and np.all(layer_routed_expert_gpu_ms > 0.0)
            )
            worker_metadata = next(
                (
                    record["metadata"]
                    for record in valid_records
                    if record is not None and record.get("metadata")
                ),
                None,
            )
            worker_trace = next(
                (
                    record["trace"]
                    for record in valid_records
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
            position_local_routed_tokens = (
                self._build_position_local_routed_tokens(
                    scheduler_output,
                    model_runner_output,
                )
            )
            position_layer_ffn_ms = self._attribute_position_layer_ffn_ms(
                layer_ffn_ms,
                position_local_routed_tokens,
            )
            (
                token_request_ids,
                token_position_ids,
                token_assignments,
            ) = self._build_token_assignments(
                scheduler_output,
                model_runner_output,
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
            self.candidate_execute_wall_ms.append(
                float(timing["execute_wall_ms"])
            )
            self.candidate_verification_wall_ms.append(
                float(timing["verification_wall_ms"])
            )
            self.candidate_draft_wall_ms.append(
                float(timing["draft_wall_ms"])
            )
            self.candidate_iteration_wall_ms.append(
                float(timing["iteration_wall_ms"])
            )
            self.candidate_execute_gpu_ms.append(
                float(timing["execute_gpu_ms"])
            )
            self.candidate_verification_gpu_ms.append(
                float(timing["verification_gpu_ms"])
            )
            self.candidate_draft_gpu_ms.append(float(timing["draft_gpu_ms"]))
            self.candidate_iteration_gpu_ms.append(
                float(timing["iteration_gpu_ms"])
            )
            self.candidate_attention_gpu_ms.append(
                float(timing["attention_gpu_ms"])
            )
            self.candidate_moe_gpu_ms.append(float(timing["moe_gpu_ms"]))
            self.candidate_gpu_other_ms.append(float(timing["gpu_other_ms"]))
            self.candidate_timing_complete.append(timing_complete)
            self.candidate_step_histograms.append(candidate_histogram)
            self.candidate_layer_ffn_ms.append(layer_ffn_ms)
            self.candidate_layer_moe_gpu_ms.append(layer_moe_gpu_ms)
            self.candidate_layer_routed_expert_gpu_ms.append(
                layer_routed_expert_gpu_ms
            )
            self.candidate_layer_shared_expert_gpu_ms.append(
                layer_shared_expert_gpu_ms
            )
            self.candidate_layer_routing_gpu_ms.append(layer_routing_gpu_ms)
            self.candidate_layer_prepare_gpu_ms.append(layer_prepare_gpu_ms)
            self.candidate_layer_finalize_gpu_ms.append(layer_finalize_gpu_ms)
            self.candidate_layer_local_routed_tokens.append(local_routed_tokens)
            self.candidate_layer_local_active_experts.append(local_active_experts)
            self.candidate_position_layer_ffn_ms.append(position_layer_ffn_ms)
            self.candidate_position_layer_local_routed_tokens.append(
                position_local_routed_tokens
            )
            self.candidate_token_request_ids.append(token_request_ids)
            self.candidate_token_position_ids.append(token_position_ids)
            self.candidate_token_layer_destination_assignment_counts.append(
                token_assignments
            )

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
                            "timing_backend": TIMING_BACKEND,
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
                                    "layer_idx": event.get("layer_idx"),
                                    "ep_collective_seq_id": (
                                        None
                                        if event.get("ep_collective_seq_id") is None
                                        else int(event["ep_collective_seq_id"])
                                    ),
                                }
                                for event in trace_events
                            ],
                            "phase_totals_ms": {
                                "attention_gpu": float(
                                    timing["attention_gpu_ms"]
                                ),
                                "moe_gpu": float(timing["moe_gpu_ms"]),
                                "gpu_other": float(timing["gpu_other_ms"]),
                                "routing_gpu": float(
                                    timing["routing_gpu_ms"]
                                ),
                                "prepare_gpu": float(
                                    timing["prepare_gpu_ms"]
                                ),
                                "routed_expert_gpu": float(
                                    timing["routed_expert_gpu_ms"]
                                ),
                                "shared_expert_gpu": float(
                                    timing["shared_expert_gpu_ms"]
                                ),
                                "finalize_gpu": float(
                                    timing["finalize_gpu_ms"]
                                ),
                                "draft_wall": float(draft_ms),
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
    pending_counts: list[dict[str, int] | None],
    *,
    batch_size: int,
    draft_length: int,
    round_idx: int,
) -> None:
    valid_counts = [item for item in pending_counts if item is not None]
    missing_worker_count = len(pending_counts) - len(valid_counts)
    if not valid_counts:
        if missing_worker_count:
            print(
                "[collect-rank] warning cleanup did not receive pending timing "
                f"counts from any worker batch_size={batch_size} "
                f"draft_length={draft_length} round={round_idx} "
                f"missing_workers={missing_worker_count}",
                flush=True,
            )
        return
    if missing_worker_count:
        print(
            "[collect-rank] warning cleanup received partial pending timing "
            f"counts batch_size={batch_size} draft_length={draft_length} "
            f"round={round_idx} missing_workers={missing_worker_count} "
            f"returned={valid_counts}",
            flush=True,
        )
        return

    pending_values = [int(item.get("pending_timings", 0)) for item in valid_counts]
    if not any(pending_values):
        return
    if len(set(pending_values)) != 1:
        raise RuntimeError(
            "Worker timing queues ended with inconsistent leftover counts: "
            f"{valid_counts}"
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
    round_error: BaseException | None = None
    try:
        with SchedulerStepRecorder(
            scheduler,
            model_executor,
            use_spec_decode=use_spec_decode,
            draft_length=draft_length,
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
    except BaseException as exc:
        round_error = exc
        raise
    finally:
        try:
            pending_counts = model_executor.collective_rpc(
                stop_condition_collection_worker
            )
            _check_pending_timings(
                pending_counts,
                batch_size=batch_size,
                draft_length=draft_length,
                round_idx=round_idx,
            )
        except Exception as cleanup_exc:
            if round_error is None:
                raise
            print(
                "[collect-rank] warning cleanup after round failure also "
                f"failed batch_size={batch_size} draft_length={draft_length} "
                f"round={round_idx} cleanup_error={cleanup_exc!r}",
                flush=True,
            )
    return outputs, recorder, round_latency_ms


def _local_prompt_batch(
    prompt_items: list[dict[str, list[int]]],
    indices: np.ndarray,
) -> list[dict[str, list[int]]]:
    return [prompt_items[int(idx)] for idx in indices.tolist()]


def collect_condition_for_rank(args: Namespace) -> RankConditionData:
    validate_parallel_config(args)
    dp_local_rank = configure_rank_process_environment(args)

    from vllm import SamplingParams

    prompt_items = load_prompt_items(args)
    if not prompt_items:
        raise RuntimeError("Prompt cache is empty.")

    dp_rank = args.dp_rank
    print(
        f"[collect-rank] start dp_rank={dp_rank} batch_size={args.batch_size} "
        f"draft_length={args.draft_length}",
        flush=True,
    )

    print(
        "[collect-rank] device_binding="
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"dp_local_rank={dp_local_rank}",
        flush=True,
    )

    llm = create_llm(args, args.batch_size, args.draft_length)
    scheduler_capacity_config = get_scheduler_capacity_config(llm)
    scheduler, model_executor = get_inproc_handles(llm)
    model_executor.collective_rpc(
        install_experiment_hooks_worker,
        kwargs={
            "enable_nvtx_ranges": bool(
                getattr(args, "enable_nvtx_ranges", False)
            )
        },
        timeout=30,
    )

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
    model_executor.collective_rpc(
        reset_hybrid_prediction_stats_worker,
        timeout=30,
    )
    model_executor.collective_rpc(
        reset_hybrid_reload_timing_stats_worker,
        timeout=30,
    )
    oracle_trace = None
    if args.hybrid_prediction_trace_mode == "replay":
        oracle_trace = load_oracle_trace_for_rank(args, dp_rank=dp_rank)
    model_executor.collective_rpc(
        configure_hybrid_prediction_trace_worker,
        kwargs={
            "trace_mode": args.hybrid_prediction_trace_mode,
            "oracle_trace": oracle_trace,
            "target_accuracy": args.hybrid_prediction_target_accuracy,
            "sim_mode": args.hybrid_prediction_sim_mode,
            "sim_seed": args.hybrid_prediction_sim_seed,
        },
        timeout=30,
    )
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
    candidate_execute_wall_ms_parts: list[np.ndarray] = []
    candidate_verification_wall_ms_parts: list[np.ndarray] = []
    candidate_draft_wall_ms_parts: list[np.ndarray] = []
    candidate_iteration_wall_ms_parts: list[np.ndarray] = []
    candidate_execute_gpu_ms_parts: list[np.ndarray] = []
    candidate_verification_gpu_ms_parts: list[np.ndarray] = []
    candidate_draft_gpu_ms_parts: list[np.ndarray] = []
    candidate_iteration_gpu_ms_parts: list[np.ndarray] = []
    candidate_attention_gpu_ms_parts: list[np.ndarray] = []
    candidate_moe_gpu_ms_parts: list[np.ndarray] = []
    candidate_gpu_other_ms_parts: list[np.ndarray] = []
    candidate_timing_complete_parts: list[np.ndarray] = []
    candidate_step_histogram_parts: list[np.ndarray] = []
    candidate_layer_ffn_ms_parts: list[np.ndarray] = []
    candidate_layer_moe_gpu_ms_parts: list[np.ndarray] = []
    candidate_layer_routed_expert_gpu_ms_parts: list[np.ndarray] = []
    candidate_layer_shared_expert_gpu_ms_parts: list[np.ndarray] = []
    candidate_layer_routing_gpu_ms_parts: list[np.ndarray] = []
    candidate_layer_prepare_gpu_ms_parts: list[np.ndarray] = []
    candidate_layer_finalize_gpu_ms_parts: list[np.ndarray] = []
    candidate_layer_local_routed_tokens_parts: list[np.ndarray] = []
    candidate_layer_local_active_experts_parts: list[np.ndarray] = []
    candidate_position_layer_ffn_ms_parts: list[np.ndarray] = []
    candidate_position_layer_local_routed_tokens_parts: list[np.ndarray] = []
    candidate_token_count_parts: list[np.ndarray] = []
    candidate_token_request_id_parts: list[np.ndarray] = []
    candidate_token_position_id_parts: list[np.ndarray] = []
    candidate_token_assignment_parts: list[np.ndarray] = []
    trace_samples: list[dict[str, Any]] = []
    condition_latency_ms = 0.0
    num_forward_steps_total = 0
    num_captured_steps = 0
    num_dropped_steps = 0
    num_prefill_dropped_steps = 0
    num_mixed_dropped_steps = 0
    finished_stats = FinishedRequestStatTotals(0.0, 0, 0)
    hybrid_prediction_stats = _empty_hybrid_prediction_stats()
    hybrid_reload_timing_stats = _empty_hybrid_reload_timing_stats()
    hybrid_prediction_trace_events: list[dict[str, int | str]] = []

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
            for parts, values, dtype in (
                (
                    candidate_execute_wall_ms_parts,
                    recorder.candidate_execute_wall_ms,
                    np.float64,
                ),
                (
                    candidate_verification_wall_ms_parts,
                    recorder.candidate_verification_wall_ms,
                    np.float64,
                ),
                (
                    candidate_draft_wall_ms_parts,
                    recorder.candidate_draft_wall_ms,
                    np.float64,
                ),
                (
                    candidate_iteration_wall_ms_parts,
                    recorder.candidate_iteration_wall_ms,
                    np.float64,
                ),
                (
                    candidate_execute_gpu_ms_parts,
                    recorder.candidate_execute_gpu_ms,
                    np.float64,
                ),
                (
                    candidate_verification_gpu_ms_parts,
                    recorder.candidate_verification_gpu_ms,
                    np.float64,
                ),
                (
                    candidate_draft_gpu_ms_parts,
                    recorder.candidate_draft_gpu_ms,
                    np.float64,
                ),
                (
                    candidate_iteration_gpu_ms_parts,
                    recorder.candidate_iteration_gpu_ms,
                    np.float64,
                ),
                (
                    candidate_attention_gpu_ms_parts,
                    recorder.candidate_attention_gpu_ms,
                    np.float64,
                ),
                (
                    candidate_moe_gpu_ms_parts,
                    recorder.candidate_moe_gpu_ms,
                    np.float64,
                ),
                (
                    candidate_gpu_other_ms_parts,
                    recorder.candidate_gpu_other_ms,
                    np.float64,
                ),
                (
                    candidate_timing_complete_parts,
                    recorder.candidate_timing_complete,
                    np.bool_,
                ),
            ):
                parts.append(np.asarray(values, dtype=dtype))
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
                for parts, values in (
                    (
                        candidate_layer_moe_gpu_ms_parts,
                        recorder.candidate_layer_moe_gpu_ms,
                    ),
                    (
                        candidate_layer_routed_expert_gpu_ms_parts,
                        recorder.candidate_layer_routed_expert_gpu_ms,
                    ),
                    (
                        candidate_layer_shared_expert_gpu_ms_parts,
                        recorder.candidate_layer_shared_expert_gpu_ms,
                    ),
                    (
                        candidate_layer_routing_gpu_ms_parts,
                        recorder.candidate_layer_routing_gpu_ms,
                    ),
                    (
                        candidate_layer_prepare_gpu_ms_parts,
                        recorder.candidate_layer_prepare_gpu_ms,
                    ),
                    (
                        candidate_layer_finalize_gpu_ms_parts,
                        recorder.candidate_layer_finalize_gpu_ms,
                    ),
                ):
                    parts.append(
                        np.stack(values, axis=0).astype(np.float64)
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
                candidate_position_layer_ffn_ms_parts.append(
                    np.stack(
                        recorder.candidate_position_layer_ffn_ms, axis=0
                    ).astype(np.float64)
                )
                candidate_position_layer_local_routed_tokens_parts.append(
                    np.stack(
                        recorder.candidate_position_layer_local_routed_tokens,
                        axis=0,
                    ).astype(np.int64)
                )
                candidate_token_count_parts.append(
                    np.asarray(
                        [
                            positions.shape[0]
                            for positions in recorder.candidate_token_position_ids
                        ],
                        dtype=np.int64,
                    )
                )
                candidate_token_request_id_parts.extend(
                    recorder.candidate_token_request_ids
                )
                candidate_token_position_id_parts.extend(
                    recorder.candidate_token_position_ids
                )
                candidate_token_assignment_parts.extend(
                    recorder.candidate_token_layer_destination_assignment_counts
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
        worker_prediction_stats = _safe_collective_rpc(
            model_executor,
            collect_hybrid_prediction_stats_worker,
            timeout=30,
        )
        for worker_stats in worker_prediction_stats or []:
            for key in hybrid_prediction_stats:
                hybrid_prediction_stats[key] += int(worker_stats.get(key, 0))
        worker_reload_timing_stats = _safe_collective_rpc(
            model_executor,
            collect_hybrid_reload_timing_stats_worker,
            timeout=30,
        )
        for worker_stats in worker_reload_timing_stats or []:
            _accumulate_hybrid_reload_timing_stats(
                hybrid_reload_timing_stats, worker_stats
            )
        if should_collect_hybrid_prediction_trace(args):
            worker_prediction_traces = _safe_collective_rpc(
                model_executor,
                collect_hybrid_prediction_trace_worker,
                timeout=30,
            )
            hybrid_prediction_trace_events = sorted(
                (
                    event
                    for worker_trace in (worker_prediction_traces or [])
                    for event in worker_trace
                ),
                key=lambda event: int(event["event_index"]),
            )
        with suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
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
    candidate_execute_wall_ms = (
        np.concatenate(candidate_execute_wall_ms_parts, axis=0)
        if candidate_execute_wall_ms_parts
        else _empty_float_array()
    )
    candidate_verification_wall_ms = (
        np.concatenate(candidate_verification_wall_ms_parts, axis=0)
        if candidate_verification_wall_ms_parts
        else _empty_float_array()
    )
    candidate_draft_wall_ms = (
        np.concatenate(candidate_draft_wall_ms_parts, axis=0)
        if candidate_draft_wall_ms_parts
        else _empty_float_array()
    )
    candidate_iteration_wall_ms = (
        np.concatenate(candidate_iteration_wall_ms_parts, axis=0)
        if candidate_iteration_wall_ms_parts
        else _empty_float_array()
    )
    candidate_execute_gpu_ms = (
        np.concatenate(candidate_execute_gpu_ms_parts, axis=0)
        if candidate_execute_gpu_ms_parts
        else _empty_float_array()
    )
    candidate_verification_gpu_ms = (
        np.concatenate(candidate_verification_gpu_ms_parts, axis=0)
        if candidate_verification_gpu_ms_parts
        else _empty_float_array()
    )
    candidate_draft_gpu_ms = (
        np.concatenate(candidate_draft_gpu_ms_parts, axis=0)
        if candidate_draft_gpu_ms_parts
        else _empty_float_array()
    )
    candidate_iteration_gpu_ms = (
        np.concatenate(candidate_iteration_gpu_ms_parts, axis=0)
        if candidate_iteration_gpu_ms_parts
        else _empty_float_array()
    )
    candidate_attention_gpu_ms = (
        np.concatenate(candidate_attention_gpu_ms_parts, axis=0)
        if candidate_attention_gpu_ms_parts
        else _empty_float_array()
    )
    candidate_moe_gpu_ms = (
        np.concatenate(candidate_moe_gpu_ms_parts, axis=0)
        if candidate_moe_gpu_ms_parts
        else _empty_float_array()
    )
    candidate_gpu_other_ms = (
        np.concatenate(candidate_gpu_other_ms_parts, axis=0)
        if candidate_gpu_other_ms_parts
        else _empty_float_array()
    )
    candidate_timing_complete = (
        np.concatenate(candidate_timing_complete_parts, axis=0)
        if candidate_timing_complete_parts
        else np.empty((0,), dtype=np.bool_)
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
    def concat_layer_timing(parts: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(parts, axis=0)
            if parts
            else np.empty((0, len(args.layers)), dtype=np.float64)
        )

    candidate_layer_moe_gpu_ms = concat_layer_timing(
        candidate_layer_moe_gpu_ms_parts
    )
    candidate_layer_routed_expert_gpu_ms = concat_layer_timing(
        candidate_layer_routed_expert_gpu_ms_parts
    )
    candidate_layer_shared_expert_gpu_ms = concat_layer_timing(
        candidate_layer_shared_expert_gpu_ms_parts
    )
    candidate_layer_routing_gpu_ms = concat_layer_timing(
        candidate_layer_routing_gpu_ms_parts
    )
    candidate_layer_prepare_gpu_ms = concat_layer_timing(
        candidate_layer_prepare_gpu_ms_parts
    )
    candidate_layer_finalize_gpu_ms = concat_layer_timing(
        candidate_layer_finalize_gpu_ms_parts
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
    candidate_position_layer_ffn_ms = (
        np.concatenate(candidate_position_layer_ffn_ms_parts, axis=0)
        if candidate_position_layer_ffn_ms_parts
        else np.empty((0, 0, len(args.layers)), dtype=np.float64)
    )
    candidate_position_layer_local_routed_tokens = (
        np.concatenate(
            candidate_position_layer_local_routed_tokens_parts, axis=0
        )
        if candidate_position_layer_local_routed_tokens_parts
        else np.empty((0, 0, len(args.layers)), dtype=np.int64)
    )
    candidate_token_counts = (
        np.concatenate(candidate_token_count_parts, axis=0)
        if candidate_token_count_parts
        else np.empty((0,), dtype=np.int64)
    )
    candidate_token_offsets = np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(candidate_token_counts, dtype=np.int64),
        )
    )
    candidate_token_request_ids = (
        np.concatenate(candidate_token_request_id_parts, axis=0)
        if candidate_token_request_id_parts
        else np.empty((0,), dtype=np.str_)
    )
    candidate_token_position_ids = (
        np.concatenate(candidate_token_position_id_parts, axis=0)
        if candidate_token_position_id_parts
        else np.empty((0,), dtype=np.int16)
    )
    candidate_token_layer_destination_assignment_counts = (
        np.concatenate(candidate_token_assignment_parts, axis=0)
        if candidate_token_assignment_parts
        else np.empty(
            (0, len(args.layers), args.data_parallel_size),
            dtype=np.int16,
        )
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
            candidate_execute_wall_ms=candidate_execute_wall_ms,
            candidate_verification_wall_ms=candidate_verification_wall_ms,
            candidate_draft_wall_ms=candidate_draft_wall_ms,
            candidate_iteration_wall_ms=candidate_iteration_wall_ms,
            candidate_execute_gpu_ms=candidate_execute_gpu_ms,
            candidate_verification_gpu_ms=candidate_verification_gpu_ms,
            candidate_draft_gpu_ms=candidate_draft_gpu_ms,
            candidate_iteration_gpu_ms=candidate_iteration_gpu_ms,
            candidate_attention_gpu_ms=candidate_attention_gpu_ms,
            candidate_moe_gpu_ms=candidate_moe_gpu_ms,
            candidate_gpu_other_ms=candidate_gpu_other_ms,
            candidate_timing_complete=candidate_timing_complete,
            candidate_step_histograms=candidate_step_histograms,
            candidate_layer_ffn_ms=candidate_layer_ffn_ms,
            candidate_layer_moe_gpu_ms=candidate_layer_moe_gpu_ms,
            candidate_layer_routed_expert_gpu_ms=(
                candidate_layer_routed_expert_gpu_ms
            ),
            candidate_layer_shared_expert_gpu_ms=(
                candidate_layer_shared_expert_gpu_ms
            ),
            candidate_layer_routing_gpu_ms=candidate_layer_routing_gpu_ms,
            candidate_layer_prepare_gpu_ms=candidate_layer_prepare_gpu_ms,
            candidate_layer_finalize_gpu_ms=candidate_layer_finalize_gpu_ms,
            candidate_layer_local_routed_tokens=candidate_layer_local_routed_tokens,
            candidate_layer_local_active_experts=(
                candidate_layer_local_active_experts
            ),
            candidate_position_layer_ffn_ms=candidate_position_layer_ffn_ms,
            candidate_position_layer_local_routed_tokens=(
                candidate_position_layer_local_routed_tokens
            ),
            candidate_token_offsets=candidate_token_offsets,
            candidate_token_request_ids=candidate_token_request_ids,
            candidate_token_position_ids=candidate_token_position_ids,
            candidate_token_layer_destination_assignment_counts=(
                candidate_token_layer_destination_assignment_counts
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
            vllm_generation_tokens_total=(
                finished_stats.vllm_generation_tokens_total
            ),
            vllm_request_tpot_total_ms=(
                finished_stats.vllm_request_tpot_total_ms
            ),
            vllm_request_tpot_count=finished_stats.vllm_request_tpot_count,
            spec_num_drafts=finished_stats.spec_num_drafts,
            spec_num_draft_tokens=finished_stats.spec_num_draft_tokens,
            spec_num_accepted_tokens=finished_stats.spec_num_accepted_tokens,
            hybrid_prediction_total=hybrid_prediction_stats["total_predictions"],
            hybrid_prediction_exact_match=(
                hybrid_prediction_stats["exact_match_count"]
            ),
            hybrid_prediction_within_one=(
                hybrid_prediction_stats["within_one_count"]
            ),
            hybrid_prediction_abs_error_sum=(
                hybrid_prediction_stats["abs_error_sum"]
            ),
            hybrid_prediction_signed_error_sum=(
                hybrid_prediction_stats["signed_error_sum"]
            ),
            hybrid_prediction_predicted_sum=(
                hybrid_prediction_stats["predicted_accept_len_sum"]
            ),
            hybrid_prediction_accepted_sum=(
                hybrid_prediction_stats["accepted_len_sum"]
            ),
            hybrid_reload_preload_total_ms=float(
                hybrid_reload_timing_stats["preload_total_ms"]
            ),
            hybrid_reload_preload_call_count=int(
                hybrid_reload_timing_stats["preload_call_count"]
            ),
            hybrid_reload_preload_req_count=int(
                hybrid_reload_timing_stats["preload_req_count"]
            ),
            hybrid_reload_preloaded_total_ms=float(
                hybrid_reload_timing_stats["preloaded_total_ms"]
            ),
            hybrid_reload_preloaded_row_count=int(
                hybrid_reload_timing_stats["preloaded_row_count"]
            ),
            hybrid_reload_fallback_total_ms=float(
                hybrid_reload_timing_stats["fallback_total_ms"]
            ),
            hybrid_reload_fallback_row_count=int(
                hybrid_reload_timing_stats["fallback_row_count"]
            ),
            hybrid_replay_prepare_copy_ms=float(
                hybrid_reload_timing_stats["prepare_copy_ms"]
            ),
            hybrid_replay_repair_compute_ms=float(
                hybrid_reload_timing_stats["repair_compute_ms"]
            ),
            hybrid_replay_verify_attention_ms=float(
                hybrid_reload_timing_stats["verify_attention_ms"]
            ),
            hybrid_replay_spill_copy_ms=float(
                hybrid_reload_timing_stats["spill_copy_ms"]
            ),
            hybrid_replay_layer_total_ms=float(
                hybrid_reload_timing_stats["layer_total_ms"]
            ),
            hybrid_replay_verify_call_count=int(
                hybrid_reload_timing_stats["verify_call_count"]
            ),
            hybrid_replay_checkpoint_save_ms=float(
                hybrid_reload_timing_stats["checkpoint_save_ms"]
            ),
            hybrid_replay_post_replay_state_gather_ms=float(
                hybrid_reload_timing_stats["post_replay_state_gather_ms"]
            ),
            hybrid_replay_capture_materialize_ms=float(
                hybrid_reload_timing_stats["capture_materialize_ms"]
            ),
            hybrid_replay_segment_start_save_ms=float(
                hybrid_reload_timing_stats["segment_start_save_ms"]
            ),
            hybrid_replay_segment_start_wait_ms=float(
                hybrid_reload_timing_stats["segment_start_wait_ms"]
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
            hybrid_prediction_trace_events=hybrid_prediction_trace_events,
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
        candidate_execute_wall_ms=candidate_execute_wall_ms,
        candidate_verification_wall_ms=candidate_verification_wall_ms,
        candidate_draft_wall_ms=candidate_draft_wall_ms,
        candidate_iteration_wall_ms=candidate_iteration_wall_ms,
        candidate_execute_gpu_ms=candidate_execute_gpu_ms,
        candidate_verification_gpu_ms=candidate_verification_gpu_ms,
        candidate_draft_gpu_ms=candidate_draft_gpu_ms,
        candidate_iteration_gpu_ms=candidate_iteration_gpu_ms,
        candidate_attention_gpu_ms=candidate_attention_gpu_ms,
        candidate_moe_gpu_ms=candidate_moe_gpu_ms,
        candidate_gpu_other_ms=candidate_gpu_other_ms,
        candidate_timing_complete=candidate_timing_complete,
        candidate_step_histograms=candidate_step_histograms,
        candidate_layer_ffn_ms=candidate_layer_ffn_ms,
        candidate_layer_moe_gpu_ms=candidate_layer_moe_gpu_ms,
        candidate_layer_routed_expert_gpu_ms=(
            candidate_layer_routed_expert_gpu_ms
        ),
        candidate_layer_shared_expert_gpu_ms=(
            candidate_layer_shared_expert_gpu_ms
        ),
        candidate_layer_routing_gpu_ms=candidate_layer_routing_gpu_ms,
        candidate_layer_prepare_gpu_ms=candidate_layer_prepare_gpu_ms,
        candidate_layer_finalize_gpu_ms=candidate_layer_finalize_gpu_ms,
        candidate_layer_local_routed_tokens=candidate_layer_local_routed_tokens,
        candidate_layer_local_active_experts=candidate_layer_local_active_experts,
        candidate_position_layer_ffn_ms=candidate_position_layer_ffn_ms,
        candidate_position_layer_local_routed_tokens=(
            candidate_position_layer_local_routed_tokens
        ),
        candidate_token_offsets=candidate_token_offsets,
        candidate_token_request_ids=candidate_token_request_ids,
        candidate_token_position_ids=candidate_token_position_ids,
        candidate_token_layer_destination_assignment_counts=(
            candidate_token_layer_destination_assignment_counts
        ),
        expert_to_ep_rank=expert_to_ep_rank,
        condition_latency_ms=condition_latency_ms,
        decode_time_total_ms=finished_stats.decode_time_total_ms,
        num_generation_tokens_total=finished_stats.num_generation_tokens_total,
        num_output_tokens_excl_first_total=(
            finished_stats.num_output_tokens_excl_first_total
        ),
        vllm_generation_tokens_total=(
            finished_stats.vllm_generation_tokens_total
        ),
        vllm_request_tpot_total_ms=finished_stats.vllm_request_tpot_total_ms,
        vllm_request_tpot_count=finished_stats.vllm_request_tpot_count,
        spec_num_drafts=finished_stats.spec_num_drafts,
        spec_num_draft_tokens=finished_stats.spec_num_draft_tokens,
        spec_num_accepted_tokens=finished_stats.spec_num_accepted_tokens,
        hybrid_prediction_total=hybrid_prediction_stats["total_predictions"],
        hybrid_prediction_exact_match=(
            hybrid_prediction_stats["exact_match_count"]
        ),
        hybrid_prediction_within_one=(
            hybrid_prediction_stats["within_one_count"]
        ),
        hybrid_prediction_abs_error_sum=(
            hybrid_prediction_stats["abs_error_sum"]
        ),
        hybrid_prediction_signed_error_sum=(
            hybrid_prediction_stats["signed_error_sum"]
        ),
        hybrid_prediction_predicted_sum=(
            hybrid_prediction_stats["predicted_accept_len_sum"]
        ),
        hybrid_prediction_accepted_sum=(
            hybrid_prediction_stats["accepted_len_sum"]
        ),
        hybrid_reload_preload_total_ms=float(
            hybrid_reload_timing_stats["preload_total_ms"]
        ),
        hybrid_reload_preload_call_count=int(
            hybrid_reload_timing_stats["preload_call_count"]
        ),
        hybrid_reload_preload_req_count=int(
            hybrid_reload_timing_stats["preload_req_count"]
        ),
        hybrid_reload_preloaded_total_ms=float(
            hybrid_reload_timing_stats["preloaded_total_ms"]
        ),
        hybrid_reload_preloaded_row_count=int(
            hybrid_reload_timing_stats["preloaded_row_count"]
        ),
        hybrid_reload_fallback_total_ms=float(
            hybrid_reload_timing_stats["fallback_total_ms"]
        ),
        hybrid_reload_fallback_row_count=int(
            hybrid_reload_timing_stats["fallback_row_count"]
        ),
        hybrid_replay_prepare_copy_ms=float(
            hybrid_reload_timing_stats["prepare_copy_ms"]
        ),
        hybrid_replay_repair_compute_ms=float(
            hybrid_reload_timing_stats["repair_compute_ms"]
        ),
        hybrid_replay_verify_attention_ms=float(
            hybrid_reload_timing_stats["verify_attention_ms"]
        ),
        hybrid_replay_spill_copy_ms=float(
            hybrid_reload_timing_stats["spill_copy_ms"]
        ),
        hybrid_replay_layer_total_ms=float(
            hybrid_reload_timing_stats["layer_total_ms"]
        ),
        hybrid_replay_verify_call_count=int(
            hybrid_reload_timing_stats["verify_call_count"]
        ),
        hybrid_replay_checkpoint_save_ms=float(
            hybrid_reload_timing_stats["checkpoint_save_ms"]
        ),
        hybrid_replay_post_replay_state_gather_ms=float(
            hybrid_reload_timing_stats["post_replay_state_gather_ms"]
        ),
        hybrid_replay_capture_materialize_ms=float(
            hybrid_reload_timing_stats["capture_materialize_ms"]
        ),
        hybrid_replay_segment_start_save_ms=float(
            hybrid_reload_timing_stats["segment_start_save_ms"]
        ),
        hybrid_replay_segment_start_wait_ms=float(
            hybrid_reload_timing_stats["segment_start_wait_ms"]
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
        hybrid_prediction_trace_events=hybrid_prediction_trace_events,
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

        def read_optional_float(name: str) -> float:
            if name not in data:
                return 0.0
            return float(data[name][0])

        candidate_layer_ffn_ms = np.asarray(data["candidate_layer_ffn_ms"])
        candidate_position_shape = (
            candidate_layer_ffn_ms.shape[0],
            0,
            candidate_layer_ffn_ms.shape[1],
        )

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
            candidate_execute_wall_ms=np.asarray(
                data["candidate_execute_wall_ms"]
            ),
            candidate_verification_wall_ms=np.asarray(
                data["candidate_verification_wall_ms"]
            ),
            candidate_draft_wall_ms=np.asarray(
                data["candidate_draft_wall_ms"]
            ),
            candidate_iteration_wall_ms=np.asarray(
                data["candidate_iteration_wall_ms"]
            ),
            candidate_execute_gpu_ms=np.asarray(
                data["candidate_execute_gpu_ms"]
            ),
            candidate_verification_gpu_ms=np.asarray(
                data["candidate_verification_gpu_ms"]
            ),
            candidate_draft_gpu_ms=np.asarray(data["candidate_draft_gpu_ms"]),
            candidate_iteration_gpu_ms=np.asarray(
                data["candidate_iteration_gpu_ms"]
            ),
            candidate_attention_gpu_ms=np.asarray(
                data["candidate_attention_gpu_ms"]
            ),
            candidate_moe_gpu_ms=np.asarray(data["candidate_moe_gpu_ms"]),
            candidate_gpu_other_ms=np.asarray(
                data["candidate_gpu_other_ms"]
            ),
            candidate_timing_complete=np.asarray(
                data["candidate_timing_complete"],
                dtype=np.bool_,
            ),
            candidate_step_histograms=np.asarray(data["candidate_step_histograms"]),
            candidate_layer_ffn_ms=candidate_layer_ffn_ms,
            candidate_layer_moe_gpu_ms=np.asarray(
                data["candidate_layer_moe_gpu_ms"]
            ),
            candidate_layer_routed_expert_gpu_ms=np.asarray(
                data["candidate_layer_routed_expert_gpu_ms"]
            ),
            candidate_layer_shared_expert_gpu_ms=np.asarray(
                data["candidate_layer_shared_expert_gpu_ms"]
            ),
            candidate_layer_routing_gpu_ms=np.asarray(
                data["candidate_layer_routing_gpu_ms"]
            ),
            candidate_layer_prepare_gpu_ms=np.asarray(
                data["candidate_layer_prepare_gpu_ms"]
            ),
            candidate_layer_finalize_gpu_ms=np.asarray(
                data["candidate_layer_finalize_gpu_ms"]
            ),
            candidate_layer_local_routed_tokens=np.asarray(
                data["candidate_layer_local_routed_tokens"]
            ),
            candidate_layer_local_active_experts=np.asarray(
                data["candidate_layer_local_active_experts"]
            ),
            candidate_position_layer_ffn_ms=np.asarray(
                data["candidate_position_layer_ffn_ms"]
            )
            if "candidate_position_layer_ffn_ms" in data
            else np.empty(candidate_position_shape, dtype=np.float64),
            candidate_position_layer_local_routed_tokens=np.asarray(
                data["candidate_position_layer_local_routed_tokens"]
            )
            if "candidate_position_layer_local_routed_tokens" in data
            else np.empty(candidate_position_shape, dtype=np.int64),
            candidate_token_offsets=np.asarray(
                data["candidate_token_offsets"],
                dtype=np.int64,
            )
            if "candidate_token_offsets" in data
            else np.zeros(
                (candidate_layer_ffn_ms.shape[0] + 1,),
                dtype=np.int64,
            ),
            candidate_token_request_ids=np.asarray(
                data["candidate_token_request_ids"],
                dtype=np.str_,
            )
            if "candidate_token_request_ids" in data
            else np.empty((0,), dtype=np.str_),
            candidate_token_position_ids=np.asarray(
                data["candidate_token_position_ids"],
                dtype=np.int16,
            )
            if "candidate_token_position_ids" in data
            else np.empty((0,), dtype=np.int16),
            candidate_token_layer_destination_assignment_counts=np.asarray(
                data["candidate_token_layer_destination_assignment_counts"],
                dtype=np.int16,
            )
            if "candidate_token_layer_destination_assignment_counts" in data
            else np.empty(
                (
                    0,
                    candidate_layer_ffn_ms.shape[1],
                    int(np.max(data["expert_to_ep_rank"])) + 1,
                ),
                dtype=np.int16,
            ),
            expert_to_ep_rank=np.asarray(data["expert_to_ep_rank"]),
            condition_latency_ms=float(data["condition_latency_ms"][0]),
            decode_time_total_ms=float(data["decode_time_total_ms"][0]),
            num_generation_tokens_total=int(data["num_generation_tokens_total"][0]),
            num_output_tokens_excl_first_total=int(
                data["num_output_tokens_excl_first_total"][0]
            ),
            vllm_generation_tokens_total=int(
                data["vllm_generation_tokens_total"][0]
            ),
            vllm_request_tpot_total_ms=float(
                data["vllm_request_tpot_total_ms"][0]
            ),
            vllm_request_tpot_count=int(data["vllm_request_tpot_count"][0]),
            spec_num_drafts=int(data["spec_num_drafts"][0]),
            spec_num_draft_tokens=int(data["spec_num_draft_tokens"][0]),
            spec_num_accepted_tokens=int(data["spec_num_accepted_tokens"][0]),
            hybrid_prediction_total=read_optional_int("hybrid_prediction_total"),
            hybrid_prediction_exact_match=read_optional_int(
                "hybrid_prediction_exact_match"
            ),
            hybrid_prediction_within_one=read_optional_int(
                "hybrid_prediction_within_one"
            ),
            hybrid_prediction_abs_error_sum=read_optional_int(
                "hybrid_prediction_abs_error_sum"
            ),
            hybrid_prediction_signed_error_sum=read_optional_int(
                "hybrid_prediction_signed_error_sum"
            ),
            hybrid_prediction_predicted_sum=read_optional_int(
                "hybrid_prediction_predicted_sum"
            ),
            hybrid_prediction_accepted_sum=read_optional_int(
                "hybrid_prediction_accepted_sum"
            ),
            hybrid_reload_preload_total_ms=read_optional_float(
                "hybrid_reload_preload_total_ms"
            ),
            hybrid_reload_preload_call_count=read_optional_int(
                "hybrid_reload_preload_call_count"
            ),
            hybrid_reload_preload_req_count=read_optional_int(
                "hybrid_reload_preload_req_count"
            ),
            hybrid_reload_preloaded_total_ms=read_optional_float(
                "hybrid_reload_preloaded_total_ms"
            ),
            hybrid_reload_preloaded_row_count=read_optional_int(
                "hybrid_reload_preloaded_row_count"
            ),
            hybrid_reload_fallback_total_ms=read_optional_float(
                "hybrid_reload_fallback_total_ms"
            ),
            hybrid_reload_fallback_row_count=read_optional_int(
                "hybrid_reload_fallback_row_count"
            ),
            hybrid_replay_prepare_copy_ms=read_optional_float(
                "hybrid_replay_prepare_copy_ms"
            ),
            hybrid_replay_repair_compute_ms=read_optional_float(
                "hybrid_replay_repair_compute_ms"
            ),
            hybrid_replay_verify_attention_ms=read_optional_float(
                "hybrid_replay_verify_attention_ms"
            ),
            hybrid_replay_spill_copy_ms=read_optional_float(
                "hybrid_replay_spill_copy_ms"
            ),
            hybrid_replay_layer_total_ms=read_optional_float(
                "hybrid_replay_layer_total_ms"
            ),
            hybrid_replay_verify_call_count=read_optional_int(
                "hybrid_replay_verify_call_count"
            ),
            hybrid_replay_checkpoint_save_ms=read_optional_float(
                "hybrid_replay_checkpoint_save_ms"
            ),
            hybrid_replay_post_replay_state_gather_ms=read_optional_float(
                "hybrid_replay_post_replay_state_gather_ms"
            ),
            hybrid_replay_capture_materialize_ms=read_optional_float(
                "hybrid_replay_capture_materialize_ms"
            ),
            hybrid_replay_segment_start_save_ms=read_optional_float(
                "hybrid_replay_segment_start_save_ms"
            ),
            hybrid_replay_segment_start_wait_ms=read_optional_float(
                "hybrid_replay_segment_start_wait_ms"
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
    if should_collect_hybrid_prediction_trace(args):
        save_rank_hybrid_prediction_trace(
            rank_prediction_trace_path(
                args.output_dir,
                batch_size=args.batch_size,
                draft_length=args.draft_length,
                dp_rank=args.dp_rank,
            ),
            args,
            data,
        )


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
                "candidate_execute_wall_ms": partial.candidate_execute_wall_ms,
                "candidate_verification_wall_ms": (
                    partial.candidate_verification_wall_ms
                ),
                "candidate_draft_wall_ms": partial.candidate_draft_wall_ms,
                "candidate_iteration_wall_ms": (
                    partial.candidate_iteration_wall_ms
                ),
                "candidate_execute_gpu_ms": partial.candidate_execute_gpu_ms,
                "candidate_verification_gpu_ms": (
                    partial.candidate_verification_gpu_ms
                ),
                "candidate_draft_gpu_ms": partial.candidate_draft_gpu_ms,
                "candidate_iteration_gpu_ms": (
                    partial.candidate_iteration_gpu_ms
                ),
                "candidate_attention_gpu_ms": partial.candidate_attention_gpu_ms,
                "candidate_moe_gpu_ms": partial.candidate_moe_gpu_ms,
                "candidate_gpu_other_ms": partial.candidate_gpu_other_ms,
                "candidate_timing_complete": partial.candidate_timing_complete,
                "candidate_step_total_tokens": partial.candidate_step_total_tokens,
                "candidate_step_histograms": partial.candidate_step_histograms,
                "candidate_layer_ffn_ms": partial.candidate_layer_ffn_ms,
                "candidate_layer_moe_gpu_ms": (
                    partial.candidate_layer_moe_gpu_ms
                ),
                "candidate_layer_routed_expert_gpu_ms": (
                    partial.candidate_layer_routed_expert_gpu_ms
                ),
                "candidate_layer_shared_expert_gpu_ms": (
                    partial.candidate_layer_shared_expert_gpu_ms
                ),
                "candidate_layer_routing_gpu_ms": (
                    partial.candidate_layer_routing_gpu_ms
                ),
                "candidate_layer_prepare_gpu_ms": (
                    partial.candidate_layer_prepare_gpu_ms
                ),
                "candidate_layer_finalize_gpu_ms": (
                    partial.candidate_layer_finalize_gpu_ms
                ),
                "candidate_layer_local_routed_tokens": (
                    partial.candidate_layer_local_routed_tokens
                ),
                "candidate_layer_local_active_experts": (
                    partial.candidate_layer_local_active_experts
                ),
                "candidate_position_layer_ffn_ms": (
                    partial.candidate_position_layer_ffn_ms
                ),
                "candidate_position_layer_local_routed_tokens": (
                    partial.candidate_position_layer_local_routed_tokens
                ),
                "candidate_token_offsets": partial.candidate_token_offsets,
                "candidate_token_request_ids": (
                    partial.candidate_token_request_ids
                ),
                "candidate_token_position_ids": (
                    partial.candidate_token_position_ids
                ),
                "candidate_token_layer_destination_assignment_counts": (
                    partial.candidate_token_layer_destination_assignment_counts
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
    vllm_generation_tokens_total = sum(
        partial.vllm_generation_tokens_total for partial in partials
    )
    vllm_generation_elapsed_ms = max(
        (partial.condition_latency_ms for partial in partials),
        default=0.0,
    )
    vllm_request_tpot_total_ms = sum(
        partial.vllm_request_tpot_total_ms for partial in partials
    )
    vllm_request_tpot_count = sum(
        partial.vllm_request_tpot_count for partial in partials
    )
    vllm_request_tpot_ms = (
        vllm_request_tpot_total_ms / vllm_request_tpot_count
        if vllm_request_tpot_count > 0
        else 0.0
    )
    vllm_generation_throughput_tok_s = compute_decode_throughput_tok_s(
        vllm_generation_tokens_total,
        vllm_generation_elapsed_ms,
    )
    spec_num_drafts = sum(partial.spec_num_drafts for partial in partials)
    spec_num_draft_tokens = sum(
        partial.spec_num_draft_tokens for partial in partials
    )
    spec_num_accepted_tokens = sum(
        partial.spec_num_accepted_tokens for partial in partials
    )
    hybrid_prediction_total = sum(
        partial.hybrid_prediction_total for partial in partials
    )
    hybrid_prediction_exact_match = sum(
        partial.hybrid_prediction_exact_match for partial in partials
    )
    hybrid_prediction_within_one = sum(
        partial.hybrid_prediction_within_one for partial in partials
    )
    hybrid_prediction_abs_error_sum = sum(
        partial.hybrid_prediction_abs_error_sum for partial in partials
    )
    hybrid_prediction_signed_error_sum = sum(
        partial.hybrid_prediction_signed_error_sum for partial in partials
    )
    hybrid_prediction_predicted_sum = sum(
        partial.hybrid_prediction_predicted_sum for partial in partials
    )
    hybrid_prediction_accepted_sum = sum(
        partial.hybrid_prediction_accepted_sum for partial in partials
    )
    hybrid_reload_preload_total_ms = sum(
        partial.hybrid_reload_preload_total_ms for partial in partials
    )
    hybrid_reload_preload_call_count = sum(
        partial.hybrid_reload_preload_call_count for partial in partials
    )
    hybrid_reload_preload_req_count = sum(
        partial.hybrid_reload_preload_req_count for partial in partials
    )
    hybrid_reload_preloaded_total_ms = sum(
        partial.hybrid_reload_preloaded_total_ms for partial in partials
    )
    hybrid_reload_preloaded_row_count = sum(
        partial.hybrid_reload_preloaded_row_count for partial in partials
    )
    hybrid_reload_fallback_total_ms = sum(
        partial.hybrid_reload_fallback_total_ms for partial in partials
    )
    hybrid_reload_fallback_row_count = sum(
        partial.hybrid_reload_fallback_row_count for partial in partials
    )
    hybrid_replay_prepare_copy_ms = sum(
        partial.hybrid_replay_prepare_copy_ms for partial in partials
    )
    hybrid_replay_repair_compute_ms = sum(
        partial.hybrid_replay_repair_compute_ms for partial in partials
    )
    hybrid_replay_verify_attention_ms = sum(
        partial.hybrid_replay_verify_attention_ms for partial in partials
    )
    hybrid_replay_spill_copy_ms = sum(
        partial.hybrid_replay_spill_copy_ms for partial in partials
    )
    hybrid_replay_layer_total_ms = sum(
        partial.hybrid_replay_layer_total_ms for partial in partials
    )
    hybrid_replay_verify_call_count = sum(
        partial.hybrid_replay_verify_call_count for partial in partials
    )
    hybrid_replay_checkpoint_save_ms = sum(
        partial.hybrid_replay_checkpoint_save_ms for partial in partials
    )
    hybrid_replay_post_replay_state_gather_ms = sum(
        partial.hybrid_replay_post_replay_state_gather_ms
        for partial in partials
    )
    hybrid_replay_capture_materialize_ms = sum(
        partial.hybrid_replay_capture_materialize_ms for partial in partials
    )
    hybrid_replay_segment_start_save_ms = sum(
        partial.hybrid_replay_segment_start_save_ms for partial in partials
    )
    hybrid_replay_segment_start_wait_ms = sum(
        partial.hybrid_replay_segment_start_wait_ms for partial in partials
    )
    spec_acceptance_rate = (
        spec_num_accepted_tokens / spec_num_draft_tokens
        if spec_num_draft_tokens > 0
        else math.nan
    )
    spec_mean_acceptance_length = (
        1.0 + spec_num_accepted_tokens / spec_num_drafts
        if spec_num_drafts > 0
        else math.nan
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
        timing_backend=TIMING_BACKEND,
        timing_scope=TIMING_SCOPE,
        hybrid_spec_state_offload_mode=args.hybrid_spec_state_offload_mode,
        hybrid_spec_state_ewma_alpha=args.hybrid_spec_state_ewma_alpha,
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
        vllm_generation_elapsed_ms=vllm_generation_elapsed_ms,
        vllm_request_tpot_ms=vllm_request_tpot_ms,
        vllm_generation_throughput_tok_s=vllm_generation_throughput_tok_s,
        spec_num_drafts=spec_num_drafts,
        spec_num_draft_tokens=spec_num_draft_tokens,
        spec_num_accepted_tokens=spec_num_accepted_tokens,
        spec_acceptance_rate=spec_acceptance_rate,
        spec_mean_acceptance_length=spec_mean_acceptance_length,
        hybrid_prediction_total=hybrid_prediction_total,
        hybrid_prediction_exact_match=hybrid_prediction_exact_match,
        hybrid_prediction_within_one=hybrid_prediction_within_one,
        hybrid_prediction_abs_error_sum=hybrid_prediction_abs_error_sum,
        hybrid_prediction_signed_error_sum=hybrid_prediction_signed_error_sum,
        hybrid_prediction_predicted_sum=hybrid_prediction_predicted_sum,
        hybrid_prediction_accepted_sum=hybrid_prediction_accepted_sum,
        hybrid_reload_preload_total_ms=hybrid_reload_preload_total_ms,
        hybrid_reload_preload_call_count=hybrid_reload_preload_call_count,
        hybrid_reload_preload_req_count=hybrid_reload_preload_req_count,
        hybrid_reload_preloaded_total_ms=hybrid_reload_preloaded_total_ms,
        hybrid_reload_preloaded_row_count=hybrid_reload_preloaded_row_count,
        hybrid_reload_fallback_total_ms=hybrid_reload_fallback_total_ms,
        hybrid_reload_fallback_row_count=hybrid_reload_fallback_row_count,
        hybrid_replay_prepare_copy_ms=hybrid_replay_prepare_copy_ms,
        hybrid_replay_repair_compute_ms=hybrid_replay_repair_compute_ms,
        hybrid_replay_verify_attention_ms=hybrid_replay_verify_attention_ms,
        hybrid_replay_spill_copy_ms=hybrid_replay_spill_copy_ms,
        hybrid_replay_layer_total_ms=hybrid_replay_layer_total_ms,
        hybrid_replay_verify_call_count=hybrid_replay_verify_call_count,
        hybrid_replay_checkpoint_save_ms=hybrid_replay_checkpoint_save_ms,
        hybrid_replay_post_replay_state_gather_ms=(
            hybrid_replay_post_replay_state_gather_ms
        ),
        hybrid_replay_capture_materialize_ms=(
            hybrid_replay_capture_materialize_ms
        ),
        hybrid_replay_segment_start_save_ms=(
            hybrid_replay_segment_start_save_ms
        ),
        hybrid_replay_segment_start_wait_ms=(
            hybrid_replay_segment_start_wait_ms
        ),
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
        rank_execute_wall_ms=global_steps.rank_execute_wall_ms,
        rank_verification_wall_ms=global_steps.rank_verification_wall_ms,
        rank_draft_wall_ms=global_steps.rank_draft_wall_ms,
        rank_iteration_wall_ms=global_steps.rank_iteration_wall_ms,
        rank_execute_gpu_ms=global_steps.rank_execute_gpu_ms,
        rank_verification_gpu_ms=global_steps.rank_verification_gpu_ms,
        rank_draft_gpu_ms=global_steps.rank_draft_gpu_ms,
        rank_iteration_gpu_ms=global_steps.rank_iteration_gpu_ms,
        rank_attention_gpu_ms=global_steps.rank_attention_gpu_ms,
        rank_moe_gpu_ms=global_steps.rank_moe_gpu_ms,
        rank_gpu_other_ms=global_steps.rank_gpu_other_ms,
        rank_timing_complete=global_steps.rank_timing_complete,
        rank_step_total_ms=global_steps.rank_step_total_ms,
        rank_step_draft_ms=global_steps.rank_step_draft_ms,
        rank_layer_moe_gpu_ms=global_steps.rank_layer_moe_gpu_ms,
        rank_layer_routed_expert_gpu_ms=(
            global_steps.rank_layer_routed_expert_gpu_ms
        ),
        rank_layer_shared_expert_gpu_ms=(
            global_steps.rank_layer_shared_expert_gpu_ms
        ),
        rank_layer_routing_gpu_ms=global_steps.rank_layer_routing_gpu_ms,
        rank_layer_prepare_gpu_ms=global_steps.rank_layer_prepare_gpu_ms,
        rank_layer_finalize_gpu_ms=global_steps.rank_layer_finalize_gpu_ms,
        rank_layer_ffn_ms=global_steps.rank_layer_ffn_ms,
        rank_layer_local_routed_tokens=(
            global_steps.rank_layer_local_routed_tokens
        ),
        rank_layer_local_active_experts=(
            global_steps.rank_layer_local_active_experts
        ),
        rank_position_layer_ffn_ms=global_steps.rank_position_layer_ffn_ms,
        rank_position_layer_local_routed_tokens=(
            global_steps.rank_position_layer_local_routed_tokens
        ),
        global_step_indices=global_steps.global_step_indices,
        global_step_total_ms=global_steps.global_step_total_ms,
        global_draft_ms=global_steps.global_draft_ms,
        global_step_ffn_ms=global_steps.global_step_ffn_ms,
        global_critical_rank_indices=global_steps.global_critical_rank_indices,
        global_verification_wall_ms=global_steps.global_verification_wall_ms,
        global_iteration_wall_ms=global_steps.global_iteration_wall_ms,
        global_draft_wall_ms=global_steps.global_draft_wall_ms,
        global_verification_gpu_total_ms=(
            global_steps.global_verification_gpu_total_ms
        ),
        global_attention_gpu_ms=global_steps.global_attention_gpu_ms,
        global_moe_gpu_ms=global_steps.global_moe_gpu_ms,
        global_gpu_other_ms=global_steps.global_gpu_other_ms,
        global_step_sorted_rank_routed_expert_gpu_ms=(
            global_steps.global_step_sorted_rank_routed_expert_gpu_ms
        ),
        global_step_sorted_rank_moe_gpu_ms=(
            global_steps.global_step_sorted_rank_moe_gpu_ms
        ),
        global_step_routed_expert_max_mean_ratio=(
            global_steps.global_step_routed_expert_max_mean_ratio
        ),
        global_step_moe_max_mean_ratio=(
            global_steps.global_step_moe_max_mean_ratio
        ),
        global_step_sorted_rank_ffn_ms=(
            global_steps.global_step_sorted_rank_ffn_ms
        ),
        global_step_sorted_rank_local_routed_tokens=(
            global_steps.global_step_sorted_rank_local_routed_tokens
        ),
        global_step_sorted_rank_local_active_experts=(
            global_steps.global_step_sorted_rank_local_active_experts
        ),
        global_step_position_sorted_rank_ffn_ms=(
            global_steps.global_step_position_sorted_rank_ffn_ms
        ),
        global_step_position_sorted_rank_local_routed_tokens=(
            global_steps.global_step_position_sorted_rank_local_routed_tokens
        ),
        global_step_ffn_max_mean_ratio=(
            global_steps.global_step_ffn_max_mean_ratio
        ),
        global_step_other_ms=global_steps.global_step_other_ms,
        global_step_kinds=global_steps.global_step_kinds,
        global_token_barrier_offsets=global_steps.global_token_barrier_offsets,
        global_token_source_ranks=global_steps.global_token_source_ranks,
        global_token_request_ids=global_steps.global_token_request_ids,
        global_token_position_ids=global_steps.global_token_position_ids,
        global_token_layer_destination_assignment_counts=(
            global_steps.global_token_layer_destination_assignment_counts
        ),
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
    entrypoint: Path | None = None,
) -> CollectedConditionSummary:
    validate_parallel_config(args)
    normalize_local_gpu_binding(args)
    dirs = ensure_collect_dirs(output_dir)
    if getattr(args, "prompt_cache_path", None) is None:
        args.prompt_cache_path = prepare_prompt_cache(args, dirs["root"])
    partial_dir = dirs["root"] / "_dp_partials" / condition_name(
        args.batch_size, args.draft_length
    )
    partial_dir.mkdir(parents=True, exist_ok=True)
    dp_master_ip = "127.0.0.1"
    partial_paths: list[Path] = []
    cwd = Path(__file__).resolve().parent.parent.parent.parent
    rank_entrypoint = (
        entrypoint
        if entrypoint is not None
        else Path(__file__).resolve().parent
        / "qwen3_6_mtp_ep_load_balance_experiment.py"
    )
    start = time.perf_counter()
    partial_paths = [
        partial_dir / f"rank_{dp_rank:02d}.npz"
        for dp_rank in range(args.data_parallel_size)
    ]
    max_port_retries = 3
    exit_code = 0
    for port_attempt in range(max_port_retries):
        dp_master_port = get_open_port()
        processes: list[subprocess.Popen[str]] = []
        for partial_path in partial_paths:
            partial_path.unlink(missing_ok=True)
        if should_collect_hybrid_prediction_trace(args):
            for dp_rank in range(args.data_parallel_size):
                rank_prediction_trace_path(
                    dirs["root"],
                    batch_size=args.batch_size,
                    draft_length=args.draft_length,
                    dp_rank=dp_rank,
                ).unlink(missing_ok=True)
        for dp_rank, partial_path in enumerate(partial_paths):
            command = _build_collect_one_rank_command(
                args,
                output_dir,
                rank_entrypoint,
                dp_rank=dp_rank,
                dp_local_rank=0,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                rank_output_path=partial_path,
            )
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=build_collect_subprocess_env(args, dp_rank=dp_rank),
                    text=True,
                )
            )
        exit_code = 0
        for proc in processes:
            proc.wait()
            if proc.returncode:
                exit_code = proc.returncode
        if exit_code == 0:
            break
        if any(path.exists() for path in partial_paths):
            raise subprocess.CalledProcessError(
                exit_code,
                "collect --internal-stage rank",
            )
        if port_attempt + 1 >= max_port_retries:
            raise subprocess.CalledProcessError(
                exit_code,
                "collect --internal-stage rank",
            )
        print(
            "[collect-condition] retrying rank launch after early startup "
            f"failure batch_size={args.batch_size} draft_length={args.draft_length} "
            f"attempt={port_attempt + 1}/{max_port_retries} "
            f"previous_dp_master_port={dp_master_port}",
            flush=True,
        )
        time.sleep(1.0)

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
        "collect",
        "--internal-stage",
        "condition",
        "--model",
        args.model,
        "--hybrid-spec-state-offload-mode",
        getattr(args, "hybrid_spec_state_offload_mode", "disabled"),
        "--hybrid-spec-state-ewma-alpha",
        str(getattr(args, "hybrid_spec_state_ewma_alpha", 0.5)),
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
    _append_optional_arg(
        command,
        "--local-gpu-ids",
        getattr(args, "local_gpu_ids", None),
    )
    _append_optional_arg(
        command,
        "--hybrid-prediction-oracle-trace-root",
        getattr(args, "hybrid_prediction_oracle_trace_root", None),
    )
    command.extend(["--layers", *(str(layer) for layer in args.layers)])
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    if getattr(args, "enable_nvtx_ranges", False):
        command.append("--enable-nvtx-ranges")
    command.extend(
        [
            "--hybrid-prediction-trace-mode",
            getattr(args, "hybrid_prediction_trace_mode", "off"),
            "--hybrid-prediction-target-accuracy",
            str(getattr(args, "hybrid_prediction_target_accuracy", 1.0)),
            "--hybrid-prediction-sim-mode",
            getattr(args, "hybrid_prediction_sim_mode", "exact_upper_bound"),
            "--hybrid-prediction-sim-seed",
            str(getattr(args, "hybrid_prediction_sim_seed", 0)),
        ]
    )
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
    internal_stage_index = command.index("--internal-stage") + 1
    command[internal_stage_index] = "rank"
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
    normalize_local_gpu_binding(args)
    dirs = ensure_collect_dirs(output_dir)
    save_run_metadata(dirs["root"], args)
    prompts_cache = prepare_prompt_cache(args, dirs["root"])
    condition_summaries: list[CollectedConditionSummary] = []
    for batch_size in args.batch_sizes:
        for draft_length in args.draft_lengths:
            raw_path = dirs["raw"] / f"{condition_name(batch_size, draft_length)}.npz"
            trace_ready = (
                not should_collect_hybrid_prediction_trace(args)
                or hybrid_prediction_trace_exists(
                    dirs["root"],
                    batch_size=batch_size,
                    draft_length=draft_length,
                    data_parallel_size=args.data_parallel_size,
                )
            )
            if raw_path.exists() and trace_ready:
                print(
                    f"[collect-parent] skipping existing batch_size={batch_size} "
                    f"draft_length={draft_length}: {raw_path}",
                    flush=True,
                )
                condition_summaries.append(load_condition_summary(raw_path))
                continue
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
                env=build_collect_subprocess_env(args),
            )
            raw_path = dirs["raw"] / f"{condition_name(batch_size, draft_length)}.npz"
            summary = load_condition_summary(raw_path)
            condition_summaries.append(summary)

    save_collect_manifest(dirs["root"], args, condition_summaries)
