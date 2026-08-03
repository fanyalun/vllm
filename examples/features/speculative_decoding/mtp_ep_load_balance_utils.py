# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 10
TIMING_BACKEND = "cuda_event_deferred"
TIMING_SCOPE = "wall_critical_rank_cuda_event"

DEFAULT_HF_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_LOCAL_MODEL = (
    Path.home()
    / ".cache"
    / "modelscope"
    / "hub"
    / "models"
    / "Qwen"
    / "Qwen3.6-35B-A3B"
)


def resolve_default_model() -> str:
    if DEFAULT_LOCAL_MODEL.exists():
        return str(DEFAULT_LOCAL_MODEL)
    return DEFAULT_HF_MODEL


DEFAULT_MODEL = resolve_default_model()
DEFAULT_DATASET = "likaixin/InstructCoder"
DEFAULT_DATASET_CONFIG = None
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_BATCH_SIZES = (8, 16)
DEFAULT_DRAFT_LENGTHS = (0, 2, 4, 6)
DEFAULT_DATA_PARALLEL_SIZE = 4
DEFAULT_LAYERS = tuple(range(40))
DEFAULT_NUM_EXPERTS = 256
DEFAULT_MAX_TOKENS = 128
DEFAULT_MAX_MODEL_LEN = 768
DEFAULT_NUM_SAMPLES = 512
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
TPOT_DEFINITION = (
    "legacy_tpot_ms = sum(finished_request.decode_time) / "
    "sum(max(num_generation_tokens - 1, 0)); "
    "vllm_request_tpot_ms = avg(finished_request.mean_time_per_output_token)"
)


@dataclass(frozen=True)
class CapturedStep:
    step_kind: str
    routing_data: np.ndarray
    total_scheduled_tokens: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepCaptureDecision:
    captured_step: CapturedStep | None
    local_step_kind: str
    drop_reason: str | None = None


@dataclass(frozen=True)
class StepTiming:
    total_ms: float
    attention_ms: float
    routing_ms: float
    prepare_ms: float
    finalize_ms: float
    ffn_ms: float

    @property
    def all2all_ms(self) -> float:
        return self.prepare_ms + self.finalize_ms

    @property
    def prepare_finalize_ms(self) -> float:
        return self.all2all_ms

    @property
    def unattributed_ms(self) -> float:
        return (
            self.total_ms
            - self.attention_ms
            - self.routing_ms
            - self.all2all_ms
            - self.ffn_ms
        )


@dataclass(frozen=True)
class GlobalStepTimingAggregation:
    global_barrier_ids: np.ndarray
    global_step_indices: np.ndarray
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
    global_step_histograms: np.ndarray
    global_step_total_tokens: np.ndarray
    global_token_barrier_offsets: np.ndarray
    global_token_source_ranks: np.ndarray
    global_token_request_ids: np.ndarray
    global_token_position_ids: np.ndarray
    global_token_layer_destination_assignment_counts: np.ndarray
    num_global_candidate_steps: int
    num_global_captured_steps: int
    num_global_prefill_dropped_steps: int
    num_global_mixed_dropped_steps: int
    num_global_non_target_dropped_steps: int


@dataclass(frozen=True)
class FinishedRequestStatTotals:
    decode_time_total_ms: float
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int
    vllm_generation_tokens_total: int = 0
    vllm_request_tpot_total_ms: float = 0.0
    vllm_request_tpot_count: int = 0
    spec_num_drafts: int = 0
    spec_num_draft_tokens: int = 0
    spec_num_accepted_tokens: int = 0


def merge_intervals(
    intervals: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    tol_ms: float = 1e-9,
) -> list[tuple[float, float]]:
    normalized = sorted(
        (float(start), float(end))
        for start, end in intervals
        if math.isfinite(start) and math.isfinite(end) and end >= start
    )
    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + tol_ms:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_union_duration_ms(
    intervals: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> float:
    return float(sum(end - start for start, end in merge_intervals(intervals)))


def interval_overlap_duration_ms(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return max(min(first[1], second[1]) - max(first[0], second[0]), 0.0)


def interval_set_overlap_duration_ms(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    first_merged = merge_intervals(first)
    second_merged = merge_intervals(second)
    first_idx = 0
    second_idx = 0
    overlap_ms = 0.0
    while first_idx < len(first_merged) and second_idx < len(second_merged):
        first_interval = first_merged[first_idx]
        second_interval = second_merged[second_idx]
        overlap_ms += interval_overlap_duration_ms(
            first_interval,
            second_interval,
        )
        if first_interval[1] <= second_interval[1]:
            first_idx += 1
        else:
            second_idx += 1
    return float(overlap_ms)


def subtract_interval_overlap_ms(
    base: tuple[float, float],
    excluded: list[tuple[float, float]],
) -> float:
    base_duration_ms = max(float(base[1]) - float(base[0]), 0.0)
    clipped = [
        (max(base[0], start), min(base[1], end))
        for start, end in excluded
        if end > base[0] and start < base[1]
    ]
    return max(base_duration_ms - interval_union_duration_ms(clipped), 0.0)


def classify_step_capture(
    scheduler_output: Any,
    model_runner_output: Any,
    worker_step_metadata: dict[str, Any] | None,
    use_spec_decode: bool,
) -> StepCaptureDecision:
    routed_experts = getattr(model_runner_output, "routed_experts", None)
    if routed_experts is None:
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="missing_routing",
            drop_reason="missing_routing",
        )

    routing_data = np.asarray(routed_experts.routing_data)
    req_ids = tuple(getattr(model_runner_output, "req_ids", ()))
    num_scheduled_tokens = dict(getattr(scheduler_output, "num_scheduled_tokens", {}))
    expected_rows = sum(num_scheduled_tokens.get(req_id, 0) for req_id in req_ids)
    if routing_data.shape[0] != expected_rows:
        raise ValueError(
            "routing_data rows do not match the scheduled token layout: "
            f"{routing_data.shape[0]} vs {expected_rows}."
        )

    scheduled_spec_tokens = dict(
        getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
    )
    metadata_req_ids = req_ids
    has_prefill = False
    if worker_step_metadata is not None:
        metadata_req_ids = tuple(worker_step_metadata.get("req_ids", ()))
        if metadata_req_ids and metadata_req_ids != req_ids:
            raise ValueError(
                "worker step metadata req_ids do not match model runner output: "
                f"{metadata_req_ids} vs {req_ids}."
            )
        has_prefill = bool(worker_step_metadata.get("has_prefill", False))

    if has_prefill:
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="prefill",
            drop_reason="prefill",
        )

    if use_spec_decode:
        if not scheduled_spec_tokens:
            return StepCaptureDecision(
                captured_step=None,
                local_step_kind="non_target",
                drop_reason="non_target",
            )
        if set(req_ids) != set(scheduled_spec_tokens):
            return StepCaptureDecision(
                captured_step=None,
                local_step_kind="mixed",
                drop_reason="mixed",
            )
        return StepCaptureDecision(
            captured_step=CapturedStep(
                step_kind="verification_only",
                routing_data=routing_data,
                total_scheduled_tokens=expected_rows,
                request_ids=req_ids,
            ),
            local_step_kind="verification_only",
        )

    if scheduled_spec_tokens:
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="mixed",
            drop_reason="mixed",
        )
    if not num_scheduled_tokens or not all(
        num_tokens == 1 for num_tokens in num_scheduled_tokens.values()
    ):
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="non_target",
            drop_reason="non_target",
        )
    return StepCaptureDecision(
        captured_step=CapturedStep(
            step_kind="decode_only",
            routing_data=routing_data,
            total_scheduled_tokens=routing_data.shape[0],
            request_ids=req_ids,
        ),
        local_step_kind="decode_only",
    )


def should_capture_baseline_decode_step(scheduler_output: Any) -> bool:
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
    scheduled_spec_tokens = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        {},
    )
    return (
        bool(num_scheduled_tokens)
        and not scheduled_spec_tokens
        and all(num_tokens == 1 for num_tokens in num_scheduled_tokens.values())
    )


def should_capture_mtp_verification_step(scheduler_output: Any) -> bool:
    scheduled_spec_tokens = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        {},
    )
    return bool(scheduled_spec_tokens)


def select_step_routing_data(
    scheduler_output: Any,
    model_runner_output: Any,
    use_spec_decode: bool,
) -> CapturedStep | None:
    return classify_step_capture(
        scheduler_output,
        model_runner_output,
        worker_step_metadata=None,
        use_spec_decode=use_spec_decode,
    ).captured_step


def count_layer_expert_histograms(
    routing_data: np.ndarray,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    num_experts: int = DEFAULT_NUM_EXPERTS,
) -> np.ndarray:
    if routing_data.ndim != 3:
        raise ValueError(
            "routing_data must be a rank-3 array shaped as "
            "(num_tokens, num_layers, topk)."
        )

    histograms = np.zeros((len(layers), num_experts), dtype=np.int64)
    for row_idx, layer_idx in enumerate(layers):
        layer_assignments = routing_data[:, layer_idx, :].reshape(-1)
        histograms[row_idx] = np.bincount(
            layer_assignments,
            minlength=num_experts,
        )[:num_experts]
    return histograms


def count_position_layer_local_routed_tokens(
    routing_data: np.ndarray,
    request_ids: tuple[str, ...],
    num_scheduled_tokens: dict[str, int],
    *,
    max_positions: int,
    expert_to_ep_rank: np.ndarray,
    local_ep_rank: int,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> np.ndarray:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive.")
    if routing_data.ndim != 3:
        raise ValueError(
            "routing_data must be a rank-3 array shaped as "
            "(num_tokens, num_layers, topk)."
        )
    if expert_to_ep_rank.ndim != 1:
        raise ValueError("expert_to_ep_rank must be a rank-1 array.")

    local_mask = np.asarray(expert_to_ep_rank) == local_ep_rank
    position_layer_tokens = np.zeros(
        (max_positions, len(layers)),
        dtype=np.int64,
    )
    row_offset = 0
    for request_id in request_ids:
        scheduled_count = int(num_scheduled_tokens.get(request_id, 0))
        for position in range(min(scheduled_count, max_positions)):
            token_routes = routing_data[row_offset + position]
            for layer_row, layer_idx in enumerate(layers):
                assignments = np.asarray(token_routes[layer_idx], dtype=np.int64)
                position_layer_tokens[position, layer_row] += int(
                    np.count_nonzero(local_mask[assignments])
                )
        row_offset += scheduled_count

    if row_offset != routing_data.shape[0]:
        raise ValueError(
            "routing_data rows do not match the request scheduling layout: "
            f"{routing_data.shape[0]} vs {row_offset}."
        )
    return position_layer_tokens


def build_token_layer_destination_assignments(
    routing_data: np.ndarray,
    request_ids: tuple[str, ...],
    num_scheduled_tokens: dict[str, int],
    *,
    expert_to_ep_rank: np.ndarray,
    ep_size: int,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if routing_data.ndim != 3:
        raise ValueError(
            "routing_data must be a rank-3 array shaped as "
            "(num_tokens, num_layers, topk)."
        )
    if expert_to_ep_rank.ndim != 1:
        raise ValueError("expert_to_ep_rank must be a rank-1 array.")
    if ep_size <= 0:
        raise ValueError("ep_size must be positive.")

    token_request_ids: list[str] = []
    token_position_ids: list[int] = []
    for request_id in request_ids:
        scheduled_count = int(num_scheduled_tokens.get(request_id, 0))
        token_request_ids.extend([str(request_id)] * scheduled_count)
        token_position_ids.extend(range(scheduled_count))
    if len(token_request_ids) != routing_data.shape[0]:
        raise ValueError(
            "routing_data rows do not match the request scheduling layout: "
            f"{routing_data.shape[0]} vs {len(token_request_ids)}."
        )

    selected_routes = np.asarray(routing_data[:, layers, :], dtype=np.int64)
    if selected_routes.size:
        if np.min(selected_routes) < 0 or np.max(selected_routes) >= len(
            expert_to_ep_rank
        ):
            raise ValueError("routing_data contains an out-of-range expert ID.")
        destination_ranks = expert_to_ep_rank[selected_routes]
    else:
        destination_ranks = np.empty(selected_routes.shape, dtype=np.int64)
    assignments = np.zeros(
        (routing_data.shape[0], len(layers), ep_size),
        dtype=np.int16,
    )
    for destination_rank in range(ep_size):
        assignments[:, :, destination_rank] = np.count_nonzero(
            destination_ranks == destination_rank,
            axis=2,
        )
    return (
        np.asarray(token_request_ids, dtype=np.str_),
        np.asarray(token_position_ids, dtype=np.int16),
        assignments,
    )


def average_step_histograms(step_histograms: np.ndarray) -> np.ndarray:
    if step_histograms.size == 0:
        raise ValueError("step_histograms must contain at least one captured step.")
    return step_histograms.mean(axis=0)


def sort_experts_desc(avg_histograms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_expert_ids = np.argsort(-avg_histograms, axis=1, kind="stable")
    sorted_histograms = np.take_along_axis(
        avg_histograms,
        sorted_expert_ids,
        axis=1,
    )
    return sorted_histograms, sorted_expert_ids


def reorder_histograms_by_expert_order(
    avg_histograms: np.ndarray,
    expert_order: np.ndarray,
) -> np.ndarray:
    return np.take_along_axis(avg_histograms, expert_order, axis=1)


def aggregate_worker_step_timings(
    worker_timings: list[dict[str, float] | None],
) -> StepTiming:
    valid_timings = [timing for timing in worker_timings if timing is not None]
    if not valid_timings:
        raise ValueError("Expected at least one worker timing bundle.")
    return StepTiming(
        total_ms=max(timing["total_ms"] for timing in valid_timings),
        attention_ms=max(timing["attention_ms"] for timing in valid_timings),
        routing_ms=max(timing["routing_ms"] for timing in valid_timings),
        prepare_ms=max(timing["prepare_ms"] for timing in valid_timings),
        finalize_ms=max(timing["finalize_ms"] for timing in valid_timings),
        ffn_ms=max(timing["ffn_ms"] for timing in valid_timings),
    )


def compute_critical_rank_step_time_components(
    rank_step_total_ms: np.ndarray,
    rank_layer_ffn_ms: np.ndarray,
    *,
    tol_ms: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rank_step_total_ms = np.asarray(rank_step_total_ms, dtype=np.float64)
    rank_layer_ffn_ms = np.asarray(rank_layer_ffn_ms, dtype=np.float64)
    if rank_step_total_ms.ndim != 2:
        raise ValueError("rank_step_total_ms must have shape [step, rank].")
    if rank_layer_ffn_ms.ndim != 3:
        raise ValueError(
            "rank_layer_ffn_ms must have shape [step, rank, layer]."
        )
    if rank_layer_ffn_ms.shape[:2] != rank_step_total_ms.shape:
        raise ValueError(
            "rank_step_total_ms and rank_layer_ffn_ms must have matching "
            "step and rank dimensions."
        )
    if rank_step_total_ms.shape[1] == 0:
        raise ValueError("Timing arrays must contain at least one rank.")

    critical_rank_indices = np.argmax(rank_step_total_ms, axis=1)
    step_indices = np.arange(rank_step_total_ms.shape[0])
    total_ms = rank_step_total_ms[step_indices, critical_rank_indices]
    ffn_ms = np.sum(
        rank_layer_ffn_ms[step_indices, critical_rank_indices],
        axis=1,
    )
    other_ms = total_ms - ffn_ms
    if np.any(ffn_ms < 0):
        raise ValueError("Critical-rank FFN time must be non-negative.")
    if np.any(other_ms < -tol_ms):
        bad_step = int(np.flatnonzero(other_ms < -tol_ms)[0])
        raise ValueError(
            "Critical-rank timing produced negative Other time at step "
            f"{bad_step}: total={total_ms[bad_step]:.6f} ms, "
            f"ffn={ffn_ms[bad_step]:.6f} ms."
        )
    return (
        critical_rank_indices.astype(np.int64),
        total_ms,
        ffn_ms,
        np.maximum(other_ms, 0.0),
    )


def compute_balancedness(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    max_count = counts.max(initial=0.0)
    if max_count <= 0:
        return 1.0
    return float(counts.mean() / max_count)


def compute_gini(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    total = counts.sum()
    if total <= 0:
        return 0.0

    n = counts.size
    index = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * index - n - 1) * counts)
    return float(numerator / (n * total))


def classify_imbalance_change(
    balancedness_delta: float,
    gini_delta: float,
    tol: float = 1e-12,
) -> str:
    if abs(balancedness_delta) <= tol and abs(gini_delta) <= tol:
        return "unchanged"
    if balancedness_delta < -tol and gini_delta >= -tol:
        return "worsened"
    if balancedness_delta > tol and gini_delta <= tol:
        return "improved"
    return "mixed"


def build_condition_metrics(
    *,
    batch_size: int,
    draft_length: int,
    num_steps: int,
    layers: tuple[int, ...],
    avg_histograms: np.ndarray,
    baseline_histograms: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_row, layer_idx in enumerate(layers):
        avg_counts = avg_histograms[layer_row]
        baseline_counts = baseline_histograms[layer_row]

        avg_total = float(avg_counts.sum())
        baseline_total = float(baseline_counts.sum())
        balancedness = compute_balancedness(avg_counts)
        baseline_balancedness = compute_balancedness(baseline_counts)
        gini = compute_gini(avg_counts)
        baseline_gini = compute_gini(baseline_counts)

        rows.append(
            {
                "batch_size": batch_size,
                "draft_length": draft_length,
                "layer": layer_idx,
                "num_steps": num_steps,
                "avg_total_routed_assignments_per_step": avg_total,
                "baseline_avg_total_routed_assignments_per_step": baseline_total,
                "avg_total_routed_assignments_delta": avg_total - baseline_total,
                "balancedness": balancedness,
                "baseline_balancedness": baseline_balancedness,
                "balancedness_delta": balancedness - baseline_balancedness,
                "balancedness_relative_change": (
                    balancedness / baseline_balancedness - 1.0
                    if baseline_balancedness > 0
                    else 0.0
                ),
                "gini": gini,
                "baseline_gini": baseline_gini,
                "gini_delta": gini - baseline_gini,
                "imbalance_change": classify_imbalance_change(
                    balancedness - baseline_balancedness,
                    gini - baseline_gini,
                ),
            }
        )
    return rows


def select_dataset_indices(batch_size: int, available_items: int) -> np.ndarray:
    if batch_size > available_items:
        raise ValueError(
            f"Requested num_samples={batch_size}, but only {available_items} "
            "dataset items are available."
        )
    return np.arange(batch_size, dtype=np.int64)


def num_condition_rounds(num_samples: int, global_batch_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive.")
    return math.ceil(num_samples / global_batch_size)


def shard_global_batch_indices(
    *,
    num_samples: int,
    global_batch_size: int,
    round_idx: int,
    dp_size: int,
    dp_rank: int,
) -> np.ndarray:
    if dp_size <= 0:
        raise ValueError("dp_size must be positive.")
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank={dp_rank} must be in [0, {dp_size}).")
    start = round_idx * global_batch_size
    stop = min(start + global_batch_size, num_samples)
    if start >= stop:
        return np.empty((0,), dtype=np.int64)
    round_indices = np.arange(start, stop, dtype=np.int64)
    floor = len(round_indices) // dp_size
    remainder = len(round_indices) % dp_size
    local_start = dp_rank * floor + min(dp_rank, remainder)
    local_len = floor + (1 if dp_rank < remainder else 0)
    return round_indices[local_start : local_start + local_len]


def build_speedup_rows(
    decode_time_ms_by_condition: dict[tuple[int, int], float],
    generation_tokens_by_condition: dict[tuple[int, int], int],
    output_tokens_excl_first_by_condition: dict[tuple[int, int], int],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
    *,
    vllm_request_tpot_ms_by_condition: dict[tuple[int, int], float] | None = None,
    vllm_generation_throughput_tok_s_by_condition: (
        dict[tuple[int, int], float] | None
    ) = None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size in batch_sizes:
        baseline_decode_time_ms = decode_time_ms_by_condition[(batch_size, 0)]
        baseline_generation_tokens = generation_tokens_by_condition[(batch_size, 0)]
        baseline_output_tokens_excl_first = output_tokens_excl_first_by_condition[
            (batch_size, 0)
        ]
        baseline_tpot = compute_tpot_ms_from_finished_stats(
            baseline_decode_time_ms,
            baseline_output_tokens_excl_first,
        )
        baseline_decode_throughput = compute_decode_throughput_tok_s(
            baseline_generation_tokens,
            baseline_decode_time_ms,
        )
        baseline_vllm_request_tpot = (
            vllm_request_tpot_ms_by_condition[(batch_size, 0)]
            if vllm_request_tpot_ms_by_condition is not None
            else baseline_tpot
        )
        baseline_vllm_generation_throughput = (
            vllm_generation_throughput_tok_s_by_condition[(batch_size, 0)]
            if vllm_generation_throughput_tok_s_by_condition is not None
            else baseline_decode_throughput
        )
        for draft_length in draft_lengths:
            decode_time_total_ms = decode_time_ms_by_condition[
                (batch_size, draft_length)
            ]
            num_generation_tokens_total = generation_tokens_by_condition[
                (batch_size, draft_length)
            ]
            num_output_tokens_excl_first_total = (
                output_tokens_excl_first_by_condition[(batch_size, draft_length)]
            )
            tpot_ms = compute_tpot_ms_from_finished_stats(
                decode_time_total_ms,
                num_output_tokens_excl_first_total,
            )
            decode_throughput = compute_decode_throughput_tok_s(
                num_generation_tokens_total,
                decode_time_total_ms,
            )
            vllm_request_tpot = (
                vllm_request_tpot_ms_by_condition[(batch_size, draft_length)]
                if vllm_request_tpot_ms_by_condition is not None
                else tpot_ms
            )
            vllm_generation_throughput = (
                vllm_generation_throughput_tok_s_by_condition[
                    (batch_size, draft_length)
                ]
                if vllm_generation_throughput_tok_s_by_condition is not None
                else decode_throughput
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "decode_time_total_ms": decode_time_total_ms,
                    "num_generation_tokens_total": num_generation_tokens_total,
                    "num_output_tokens_excl_first_total": (
                        num_output_tokens_excl_first_total
                    ),
                    "tpot_ms": tpot_ms,
                    "baseline_tpot_ms": baseline_tpot,
                    "tpot_speedup": (
                        baseline_tpot / tpot_ms if tpot_ms > 0 else 0.0
                    ),
                    "decode_throughput_tok_s": decode_throughput,
                    "baseline_decode_throughput_tok_s": baseline_decode_throughput,
                    "decode_throughput_speedup": (
                        decode_throughput / baseline_decode_throughput
                        if baseline_decode_throughput > 0
                        else 0.0
                    ),
                    "vllm_request_tpot_ms": vllm_request_tpot,
                    "baseline_vllm_request_tpot_ms": baseline_vllm_request_tpot,
                    "vllm_request_tpot_speedup": (
                        baseline_vllm_request_tpot / vllm_request_tpot
                        if vllm_request_tpot > 0
                        else 0.0
                    ),
                    "vllm_generation_throughput_tok_s": (
                        vllm_generation_throughput
                    ),
                    "baseline_vllm_generation_throughput_tok_s": (
                        baseline_vllm_generation_throughput
                    ),
                    "vllm_generation_throughput_speedup": (
                        vllm_generation_throughput
                        / baseline_vllm_generation_throughput
                        if baseline_vllm_generation_throughput > 0
                        else 0.0
                    ),
                }
            )
    return rows


def aggregate_global_step_time_components(
    rank_step_data: list[dict[str, np.ndarray]],
    *,
    data_parallel_size: int,
    layers: tuple[int, ...],
    num_experts: int,
    tol_ms: float = 1e-3,
) -> GlobalStepTimingAggregation:
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive.")
    if len(rank_step_data) != data_parallel_size:
        raise ValueError(
            "rank_step_data length must match data_parallel_size: "
            f"{len(rank_step_data)} vs {data_parallel_size}."
        )

    rank_records: list[list[dict[str, Any]]] = []
    for rank_idx, rank_data in enumerate(rank_step_data):
        first_seq_ids = np.asarray(
            rank_data["candidate_first_ep_collective_seq_ids"], dtype=np.int64
        )
        last_seq_ids = np.asarray(
            rank_data["candidate_last_ep_collective_seq_ids"], dtype=np.int64
        )
        num_ep_collectives = np.asarray(
            rank_data["candidate_num_ep_collectives"], dtype=np.int64
        )
        step_kinds = np.asarray(rank_data["candidate_step_kinds"], dtype=np.str_)
        size = first_seq_ids.shape[0]
        step_total_ms = np.asarray(
            rank_data["candidate_step_total_ms"], dtype=np.float64
        )
        step_draft_ms = np.asarray(
            rank_data["candidate_step_draft_ms"], dtype=np.float64
        )
        step_ffn_ms = np.asarray(
            rank_data["candidate_step_ffn_ms"], dtype=np.float64
        )
        step_total_tokens = np.asarray(
            rank_data["candidate_step_total_tokens"], dtype=np.int64
        )
        step_histograms = np.asarray(rank_data["candidate_step_histograms"])
        layer_ffn_ms = np.asarray(
            rank_data["candidate_layer_ffn_ms"], dtype=np.float64
        )
        execute_wall_ms = np.asarray(
            rank_data.get("candidate_execute_wall_ms", step_total_ms),
            dtype=np.float64,
        )
        verification_wall_ms = np.asarray(
            rank_data.get("candidate_verification_wall_ms", step_total_ms),
            dtype=np.float64,
        )
        draft_wall_ms = np.asarray(
            rank_data.get("candidate_draft_wall_ms", step_draft_ms),
            dtype=np.float64,
        )
        iteration_wall_ms = np.asarray(
            rank_data.get(
                "candidate_iteration_wall_ms",
                verification_wall_ms + draft_wall_ms,
            ),
            dtype=np.float64,
        )
        execute_gpu_ms = np.asarray(
            rank_data.get("candidate_execute_gpu_ms", step_total_ms),
            dtype=np.float64,
        )
        verification_gpu_ms = np.asarray(
            rank_data.get("candidate_verification_gpu_ms", step_total_ms),
            dtype=np.float64,
        )
        draft_gpu_ms = np.asarray(
            rank_data.get("candidate_draft_gpu_ms", step_draft_ms),
            dtype=np.float64,
        )
        iteration_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_iteration_gpu_ms",
                verification_gpu_ms + draft_gpu_ms,
            ),
            dtype=np.float64,
        )
        attention_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_attention_gpu_ms",
                np.zeros((size,), dtype=np.float64),
            ),
            dtype=np.float64,
        )
        moe_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_moe_gpu_ms",
                np.sum(layer_ffn_ms, axis=1, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        gpu_other_ms = np.asarray(
            rank_data.get(
                "candidate_gpu_other_ms",
                np.maximum(
                    verification_gpu_ms - attention_gpu_ms - moe_gpu_ms,
                    0.0,
                ),
            ),
            dtype=np.float64,
        )
        timing_complete = np.asarray(
            rank_data.get(
                "candidate_timing_complete",
                np.ones((size,), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        layer_moe_gpu_ms = np.asarray(
            rank_data.get("candidate_layer_moe_gpu_ms", layer_ffn_ms),
            dtype=np.float64,
        )
        layer_routed_expert_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_layer_routed_expert_gpu_ms",
                layer_ffn_ms,
            ),
            dtype=np.float64,
        )
        layer_shared_expert_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_layer_shared_expert_gpu_ms",
                np.zeros_like(layer_ffn_ms),
            ),
            dtype=np.float64,
        )
        layer_routing_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_layer_routing_gpu_ms",
                np.zeros_like(layer_ffn_ms),
            ),
            dtype=np.float64,
        )
        layer_prepare_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_layer_prepare_gpu_ms",
                np.zeros_like(layer_ffn_ms),
            ),
            dtype=np.float64,
        )
        layer_finalize_gpu_ms = np.asarray(
            rank_data.get(
                "candidate_layer_finalize_gpu_ms",
                np.zeros_like(layer_ffn_ms),
            ),
            dtype=np.float64,
        )
        layer_local_routed_tokens = np.asarray(
            rank_data["candidate_layer_local_routed_tokens"], dtype=np.int64
        )
        layer_local_active_experts = np.asarray(
            rank_data["candidate_layer_local_active_experts"], dtype=np.int64
        )
        position_layer_ffn_ms = np.asarray(
            rank_data.get(
                "candidate_position_layer_ffn_ms",
                np.empty((size, 0, len(layers)), dtype=np.float64),
            ),
            dtype=np.float64,
        )
        position_layer_local_routed_tokens = np.asarray(
            rank_data.get(
                "candidate_position_layer_local_routed_tokens",
                np.empty((size, 0, len(layers)), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        token_offsets = np.asarray(
            rank_data.get(
                "candidate_token_offsets",
                np.zeros((size + 1,), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        token_request_ids = np.asarray(
            rank_data.get("candidate_token_request_ids", np.empty((0,), dtype=np.str_)),
            dtype=np.str_,
        )
        token_position_ids = np.asarray(
            rank_data.get(
                "candidate_token_position_ids",
                np.empty((0,), dtype=np.int16),
            ),
            dtype=np.int16,
        )
        token_assignments = np.asarray(
            rank_data.get(
                "candidate_token_layer_destination_assignment_counts",
                np.empty(
                    (0, len(layers), data_parallel_size),
                    dtype=np.int16,
                ),
            ),
            dtype=np.int16,
        )
        for array_name, array in (
            ("candidate_last_ep_collective_seq_ids", last_seq_ids),
            ("candidate_num_ep_collectives", num_ep_collectives),
            ("candidate_step_kinds", step_kinds),
            ("candidate_step_total_ms", step_total_ms),
            ("candidate_step_draft_ms", step_draft_ms),
            ("candidate_step_ffn_ms", step_ffn_ms),
            ("candidate_step_total_tokens", step_total_tokens),
            ("candidate_execute_wall_ms", execute_wall_ms),
            ("candidate_verification_wall_ms", verification_wall_ms),
            ("candidate_draft_wall_ms", draft_wall_ms),
            ("candidate_iteration_wall_ms", iteration_wall_ms),
            ("candidate_execute_gpu_ms", execute_gpu_ms),
            ("candidate_verification_gpu_ms", verification_gpu_ms),
            ("candidate_draft_gpu_ms", draft_gpu_ms),
            ("candidate_iteration_gpu_ms", iteration_gpu_ms),
            ("candidate_attention_gpu_ms", attention_gpu_ms),
            ("candidate_moe_gpu_ms", moe_gpu_ms),
            ("candidate_gpu_other_ms", gpu_other_ms),
            ("candidate_timing_complete", timing_complete),
        ):
            if array.shape[0] != size:
                raise ValueError(
                    f"Rank {rank_idx} {array_name} has inconsistent length: "
                    f"{array.shape[0]} vs {size}."
                )
        if step_histograms.shape != (size, len(layers), num_experts):
            raise ValueError(
                f"Rank {rank_idx} candidate_step_histograms has shape "
                f"{step_histograms.shape}; expected "
                f"{(size, len(layers), num_experts)}."
            )
        for array_name, array in (
            ("candidate_layer_ffn_ms", layer_ffn_ms),
            ("candidate_layer_local_routed_tokens", layer_local_routed_tokens),
            ("candidate_layer_local_active_experts", layer_local_active_experts),
            ("candidate_layer_moe_gpu_ms", layer_moe_gpu_ms),
            (
                "candidate_layer_routed_expert_gpu_ms",
                layer_routed_expert_gpu_ms,
            ),
            (
                "candidate_layer_shared_expert_gpu_ms",
                layer_shared_expert_gpu_ms,
            ),
            ("candidate_layer_routing_gpu_ms", layer_routing_gpu_ms),
            ("candidate_layer_prepare_gpu_ms", layer_prepare_gpu_ms),
            ("candidate_layer_finalize_gpu_ms", layer_finalize_gpu_ms),
        ):
            if array.shape != (size, len(layers)):
                raise ValueError(
                    f"Rank {rank_idx} {array_name} has shape {array.shape}; "
                    f"expected {(size, len(layers))}."
                )
        max_positions = int(position_layer_ffn_ms.shape[1])
        for array_name, array, dtype_name in (
            (
                "candidate_position_layer_ffn_ms",
                position_layer_ffn_ms,
                "float",
            ),
            (
                "candidate_position_layer_local_routed_tokens",
                position_layer_local_routed_tokens,
                "int",
            ),
        ):
            expected_shape = (size, max_positions, len(layers))
            if array.shape != expected_shape:
                raise ValueError(
                    f"Rank {rank_idx} {array_name} has shape {array.shape}; "
                    f"expected {expected_shape}."
                )
            if dtype_name == "int" and not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"Rank {rank_idx} {array_name} must be integral.")
        if token_offsets.shape != (size + 1,):
            raise ValueError(
                f"Rank {rank_idx} candidate_token_offsets has shape "
                f"{token_offsets.shape}; expected {(size + 1,)}."
            )
        if token_offsets[0] != 0 or np.any(np.diff(token_offsets) < 0):
            raise ValueError(
                f"Rank {rank_idx} candidate_token_offsets must start at zero "
                "and be non-decreasing."
            )
        num_tokens = int(token_offsets[-1])
        if token_request_ids.shape != (num_tokens,):
            raise ValueError(
                f"Rank {rank_idx} candidate_token_request_ids has shape "
                f"{token_request_ids.shape}; expected {(num_tokens,)}."
            )
        if token_position_ids.shape != (num_tokens,):
            raise ValueError(
                f"Rank {rank_idx} candidate_token_position_ids has shape "
                f"{token_position_ids.shape}; expected {(num_tokens,)}."
            )
        expected_assignment_shape = (
            num_tokens,
            len(layers),
            data_parallel_size,
        )
        if token_assignments.shape != expected_assignment_shape:
            raise ValueError(
                f"Rank {rank_idx} token assignment tensor has shape "
                f"{token_assignments.shape}; expected "
                f"{expected_assignment_shape}."
            )

        seen_spans: set[tuple[int, int, int]] = set()
        records: list[dict[str, Any]] = []
        for idx in range(size):
            first_seq_id = int(first_seq_ids[idx])
            last_seq_id = int(last_seq_ids[idx])
            collective_count = int(num_ep_collectives[idx])
            span = (first_seq_id, last_seq_id, collective_count)
            if first_seq_id < 0 or last_seq_id < 0 or collective_count <= 0:
                raise ValueError(
                    f"Rank {rank_idx} produced an invalid EP collective span "
                    f"{span} at local barrier index {idx}."
                )
            expected_count = last_seq_id - first_seq_id + 1
            if collective_count != expected_count:
                raise ValueError(
                    f"Rank {rank_idx} produced inconsistent EP span {span}: "
                    f"count should be {expected_count}."
                )
            if span in seen_spans:
                raise ValueError(
                    f"Rank {rank_idx} produced duplicate EP collective span "
                    f"{span}."
                )
            seen_spans.add(span)

            token_start = int(token_offsets[idx])
            token_end = int(token_offsets[idx + 1])
            records.append(
                {
                    "rank_idx": rank_idx,
                    "span": span,
                    "step_kind": str(step_kinds[idx]),
                    "step_total_ms": float(step_total_ms[idx]),
                    "step_draft_ms": float(step_draft_ms[idx]),
                    "step_ffn_ms": float(step_ffn_ms[idx]),
                    "execute_wall_ms": float(execute_wall_ms[idx]),
                    "verification_wall_ms": float(verification_wall_ms[idx]),
                    "draft_wall_ms": float(draft_wall_ms[idx]),
                    "iteration_wall_ms": float(iteration_wall_ms[idx]),
                    "execute_gpu_ms": float(execute_gpu_ms[idx]),
                    "verification_gpu_ms": float(verification_gpu_ms[idx]),
                    "draft_gpu_ms": float(draft_gpu_ms[idx]),
                    "iteration_gpu_ms": float(iteration_gpu_ms[idx]),
                    "attention_gpu_ms": float(attention_gpu_ms[idx]),
                    "moe_gpu_ms": float(moe_gpu_ms[idx]),
                    "gpu_other_ms": float(gpu_other_ms[idx]),
                    "timing_complete": bool(timing_complete[idx]),
                    "step_total_tokens": int(step_total_tokens[idx]),
                    "step_histograms": step_histograms[idx],
                    "layer_ffn_ms": layer_ffn_ms[idx],
                    "layer_moe_gpu_ms": layer_moe_gpu_ms[idx],
                    "layer_routed_expert_gpu_ms": (
                        layer_routed_expert_gpu_ms[idx]
                    ),
                    "layer_shared_expert_gpu_ms": (
                        layer_shared_expert_gpu_ms[idx]
                    ),
                    "layer_routing_gpu_ms": layer_routing_gpu_ms[idx],
                    "layer_prepare_gpu_ms": layer_prepare_gpu_ms[idx],
                    "layer_finalize_gpu_ms": layer_finalize_gpu_ms[idx],
                    "layer_local_routed_tokens": layer_local_routed_tokens[idx],
                    "layer_local_active_experts": layer_local_active_experts[idx],
                    "position_layer_ffn_ms": position_layer_ffn_ms[idx],
                    "position_layer_local_routed_tokens": (
                        position_layer_local_routed_tokens[idx]
                    ),
                    "token_request_ids": token_request_ids[token_start:token_end],
                    "token_position_ids": token_position_ids[token_start:token_end],
                    "token_assignments": token_assignments[token_start:token_end],
                }
            )
        rank_records.append(records)

    barrier_counts = [len(records) for records in rank_records]
    num_barriers = min(barrier_counts, default=0)
    num_global_candidate_steps = max(barrier_counts, default=0)

    global_barrier_ids: list[int] = []
    global_step_indices: list[int] = []
    barrier_first_seq_ids: list[int] = []
    barrier_last_seq_ids: list[int] = []
    barrier_num_collectives: list[int] = []
    rank_barrier_first_seq_ids: list[np.ndarray] = []
    rank_barrier_last_seq_ids: list[np.ndarray] = []
    rank_barrier_num_collectives: list[np.ndarray] = []
    rank_step_kinds: list[np.ndarray] = []
    rank_execute_wall_ms: list[np.ndarray] = []
    rank_verification_wall_ms: list[np.ndarray] = []
    rank_draft_wall_ms: list[np.ndarray] = []
    rank_iteration_wall_ms: list[np.ndarray] = []
    rank_execute_gpu_ms: list[np.ndarray] = []
    rank_verification_gpu_ms: list[np.ndarray] = []
    rank_draft_gpu_ms: list[np.ndarray] = []
    rank_iteration_gpu_ms: list[np.ndarray] = []
    rank_attention_gpu_ms: list[np.ndarray] = []
    rank_moe_gpu_ms: list[np.ndarray] = []
    rank_gpu_other_ms: list[np.ndarray] = []
    rank_timing_complete: list[np.ndarray] = []
    rank_step_total_ms: list[np.ndarray] = []
    rank_step_draft_ms: list[np.ndarray] = []
    rank_layer_moe_gpu_ms: list[np.ndarray] = []
    rank_layer_routed_expert_gpu_ms: list[np.ndarray] = []
    rank_layer_shared_expert_gpu_ms: list[np.ndarray] = []
    rank_layer_routing_gpu_ms: list[np.ndarray] = []
    rank_layer_prepare_gpu_ms: list[np.ndarray] = []
    rank_layer_finalize_gpu_ms: list[np.ndarray] = []
    rank_layer_ffn_ms: list[np.ndarray] = []
    rank_layer_local_routed_tokens: list[np.ndarray] = []
    rank_layer_local_active_experts: list[np.ndarray] = []
    rank_position_layer_ffn_ms: list[np.ndarray] = []
    rank_position_layer_local_routed_tokens: list[np.ndarray] = []
    global_step_total_ms: list[float] = []
    global_draft_ms: list[float] = []
    global_step_ffn_ms: list[float] = []
    global_critical_rank_indices: list[int] = []
    global_verification_wall_ms: list[float] = []
    global_iteration_wall_ms: list[float] = []
    global_draft_wall_ms: list[float] = []
    global_verification_gpu_total_ms: list[float] = []
    global_attention_gpu_ms: list[float] = []
    global_moe_gpu_ms: list[float] = []
    global_gpu_other_ms: list[float] = []
    global_step_sorted_rank_routed_expert_gpu_ms: list[np.ndarray] = []
    global_step_sorted_rank_moe_gpu_ms: list[np.ndarray] = []
    global_step_routed_expert_max_mean_ratio: list[float] = []
    global_step_moe_max_mean_ratio: list[float] = []
    global_step_sorted_rank_ffn_ms: list[np.ndarray] = []
    global_step_sorted_rank_local_routed_tokens: list[np.ndarray] = []
    global_step_sorted_rank_local_active_experts: list[np.ndarray] = []
    global_step_position_sorted_rank_ffn_ms: list[np.ndarray] = []
    global_step_position_sorted_rank_local_routed_tokens: list[np.ndarray] = []
    global_step_ffn_max_mean_ratio: list[float] = []
    global_step_other_ms: list[float] = []
    global_step_kinds: list[str] = []
    global_step_histograms: list[np.ndarray] = []
    global_step_total_tokens: list[int] = []
    global_token_barrier_offsets = [0]
    global_token_source_ranks: list[np.ndarray] = []
    global_token_request_ids: list[np.ndarray] = []
    global_token_position_ids: list[np.ndarray] = []
    global_token_assignments: list[np.ndarray] = []
    num_global_prefill_dropped_steps = 0
    num_global_mixed_dropped_steps = 0
    num_global_non_target_dropped_steps = (
        num_global_candidate_steps - num_barriers
    )

    for barrier_id in range(num_barriers):
        records = [records[barrier_id] for records in rank_records]
        spans = [record["span"] for record in records]
        step_kind_set = {str(record["step_kind"]) for record in records}
        global_step_kind = (
            next(iter(step_kind_set)) if len(step_kind_set) == 1 else "mixed_rank"
        )
        if "prefill" in step_kind_set:
            num_global_prefill_dropped_steps += 1
        if "mixed" in step_kind_set or global_step_kind == "mixed_rank":
            num_global_mixed_dropped_steps += 1
        if global_step_kind not in {
            "decode_only",
            "verification_only",
            "prefill",
            "mixed",
            "mixed_rank",
        }:
            num_global_non_target_dropped_steps += 1

        per_rank_total_ms = np.asarray(
            [float(record["step_total_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_draft_ms = np.asarray(
            [float(record["step_draft_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_step_ffn_ms = np.asarray(
            [float(record["step_ffn_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_layer_ffn_ms = np.stack(
            [record["layer_ffn_ms"] for record in records], axis=0
        ).astype(np.float64)
        per_rank_execute_wall_ms = np.asarray(
            [float(record["execute_wall_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_verification_wall_ms = np.asarray(
            [float(record["verification_wall_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_draft_wall_ms = np.asarray(
            [float(record["draft_wall_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_iteration_wall_ms = np.asarray(
            [float(record["iteration_wall_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_execute_gpu_ms = np.asarray(
            [float(record["execute_gpu_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_verification_gpu_ms = np.asarray(
            [float(record["verification_gpu_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_draft_gpu_ms = np.asarray(
            [float(record["draft_gpu_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_iteration_gpu_ms = np.asarray(
            [float(record["iteration_gpu_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_attention_gpu_ms = np.asarray(
            [float(record["attention_gpu_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_moe_gpu_ms = np.asarray(
            [float(record["moe_gpu_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_gpu_other_ms = np.asarray(
            [float(record["gpu_other_ms"]) for record in records],
            dtype=np.float64,
        )
        per_rank_timing_complete = np.asarray(
            [bool(record["timing_complete"]) for record in records],
            dtype=np.bool_,
        )
        per_rank_layer_moe_gpu_ms = np.stack(
            [record["layer_moe_gpu_ms"] for record in records],
            axis=0,
        ).astype(np.float64)
        per_rank_layer_routed_expert_gpu_ms = np.stack(
            [record["layer_routed_expert_gpu_ms"] for record in records],
            axis=0,
        ).astype(np.float64)
        per_rank_layer_shared_expert_gpu_ms = np.stack(
            [record["layer_shared_expert_gpu_ms"] for record in records],
            axis=0,
        ).astype(np.float64)
        per_rank_layer_routing_gpu_ms = np.stack(
            [record["layer_routing_gpu_ms"] for record in records],
            axis=0,
        ).astype(np.float64)
        per_rank_layer_prepare_gpu_ms = np.stack(
            [record["layer_prepare_gpu_ms"] for record in records],
            axis=0,
        ).astype(np.float64)
        per_rank_layer_finalize_gpu_ms = np.stack(
            [record["layer_finalize_gpu_ms"] for record in records],
            axis=0,
        ).astype(np.float64)
        per_rank_layer_tokens = np.stack(
            [record["layer_local_routed_tokens"] for record in records], axis=0
        ).astype(np.int64)
        per_rank_layer_active = np.stack(
            [record["layer_local_active_experts"] for record in records], axis=0
        ).astype(np.int64)
        position_counts = {
            int(np.asarray(record["position_layer_ffn_ms"]).shape[0])
            for record in records
        }
        if len(position_counts) != 1:
            raise ValueError(
                f"EP collective span {spans} has inconsistent position counts: "
                f"{sorted(position_counts)}."
            )
        max_positions = next(iter(position_counts))
        barrier_token_request_ids = np.concatenate(
            [record["token_request_ids"] for record in records],
            axis=0,
        )
        barrier_token_position_ids = np.concatenate(
            [record["token_position_ids"] for record in records],
            axis=0,
        )
        barrier_token_assignments = np.concatenate(
            [record["token_assignments"] for record in records],
            axis=0,
        )
        barrier_token_source_ranks = np.concatenate(
            [
                np.full(
                    (record["token_position_ids"].shape[0],),
                    int(record["rank_idx"]),
                    dtype=np.int16,
                )
                for record in records
            ],
            axis=0,
        )
        if barrier_token_assignments.shape[0] > 0:
            complete_layer_destination_tokens = np.sum(
                barrier_token_assignments,
                axis=0,
                dtype=np.int64,
            )
            per_rank_layer_tokens = complete_layer_destination_tokens.T
            position_layer_destination_tokens = np.zeros(
                (max_positions, len(layers), data_parallel_size),
                dtype=np.int64,
            )
            for position in range(max_positions):
                position_layer_destination_tokens[position] = np.sum(
                    barrier_token_assignments[
                        barrier_token_position_ids == position
                    ],
                    axis=0,
                    dtype=np.int64,
                )
            per_rank_position_layer_tokens = np.transpose(
                position_layer_destination_tokens,
                (2, 0, 1),
            )
            per_rank_position_layer_ffn_ms = np.zeros_like(
                per_rank_position_layer_tokens,
                dtype=np.float64,
            )
            for rank_idx in range(data_parallel_size):
                totals = per_rank_position_layer_tokens[rank_idx].sum(axis=0)
                for layer_idx in np.flatnonzero(totals > 0):
                    per_rank_position_layer_ffn_ms[
                        rank_idx, :, layer_idx
                    ] = (
                        per_rank_position_layer_tokens[
                            rank_idx, :, layer_idx
                        ].astype(np.float64)
                        / float(totals[layer_idx])
                    ) * per_rank_layer_ffn_ms[rank_idx, layer_idx]
        else:
            per_rank_position_layer_ffn_ms = np.stack(
                [record["position_layer_ffn_ms"] for record in records], axis=0
            ).astype(np.float64)
            per_rank_position_layer_tokens = np.stack(
                [
                    record["position_layer_local_routed_tokens"]
                    for record in records
                ],
                axis=0,
            ).astype(np.int64)
        if (
            np.max(per_rank_layer_ffn_ms) <= tol_ms
            and np.max(per_rank_step_ffn_ms) > tol_ms
        ):
            raise RuntimeError(
                "Layer-level FFN timings are missing for EP collective span "
                f"{spans}: candidate_step_ffn_ms is non-zero but "
                "candidate_layer_ffn_ms is all zero. Check FusedMoE layer "
                "context hooks."
            )
        timing_arrays = (
            per_rank_execute_wall_ms,
            per_rank_verification_wall_ms,
            per_rank_draft_wall_ms,
            per_rank_iteration_wall_ms,
            per_rank_execute_gpu_ms,
            per_rank_verification_gpu_ms,
            per_rank_draft_gpu_ms,
            per_rank_iteration_gpu_ms,
            per_rank_attention_gpu_ms,
            per_rank_moe_gpu_ms,
            per_rank_gpu_other_ms,
            per_rank_layer_moe_gpu_ms,
            per_rank_layer_routed_expert_gpu_ms,
            per_rank_layer_shared_expert_gpu_ms,
            per_rank_layer_routing_gpu_ms,
            per_rank_layer_prepare_gpu_ms,
            per_rank_layer_finalize_gpu_ms,
        )
        if any(np.any(~np.isfinite(values)) for values in timing_arrays):
            raise ValueError(
                f"Barrier {barrier_id} contains non-finite timing values."
            )
        if any(np.any(values < -tol_ms) for values in timing_arrays):
            raise ValueError(f"Barrier {barrier_id} contains negative timing values.")
        decomposition_error = np.abs(
            per_rank_attention_gpu_ms
            + per_rank_moe_gpu_ms
            + per_rank_gpu_other_ms
            - per_rank_verification_gpu_ms
        )
        if np.any(decomposition_error > tol_ms):
            bad_rank = int(np.argmax(decomposition_error))
            raise ValueError(
                "CUDA Event timing decomposition is incomplete at barrier "
                f"{barrier_id}, rank {bad_rank}: attention + moe + other != "
                "verification total."
            )
        strict_target = global_step_kind in {"decode_only", "verification_only"}
        if strict_target and not np.all(per_rank_timing_complete):
            bad_ranks = np.flatnonzero(~per_rank_timing_complete).tolist()
            raise ValueError(
                f"Strict barrier {barrier_id} has incomplete timing on ranks "
                f"{bad_ranks}."
            )
        sorted_rank_ffn = []
        sorted_rank_moe = []
        sorted_rank_tokens = []
        sorted_rank_active = []
        position_sorted_rank_ffn = np.zeros(
            (max_positions, data_parallel_size),
            dtype=np.float64,
        )
        position_sorted_rank_tokens = np.zeros(
            (max_positions, data_parallel_size),
            dtype=np.int64,
        )
        for layer_idx in range(len(layers)):
            order = np.argsort(
                -per_rank_layer_routed_expert_gpu_ms[:, layer_idx],
                kind="stable",
            )
            sorted_rank_ffn.append(
                per_rank_layer_routed_expert_gpu_ms[order, layer_idx]
            )
            sorted_rank_tokens.append(per_rank_layer_tokens[order, layer_idx])
            sorted_rank_active.append(per_rank_layer_active[order, layer_idx])
            if max_positions:
                position_sorted_rank_ffn += (
                    per_rank_position_layer_ffn_ms[order, :, layer_idx].T
                )
                position_sorted_rank_tokens += (
                    per_rank_position_layer_tokens[order, :, layer_idx].T
                )
            moe_order = np.argsort(
                -per_rank_layer_moe_gpu_ms[:, layer_idx],
                kind="stable",
            )
            sorted_rank_moe.append(
                per_rank_layer_moe_gpu_ms[moe_order, layer_idx]
            )
        sorted_rank_ffn_ms = np.sum(np.stack(sorted_rank_ffn, axis=0), axis=0)
        sorted_rank_moe_ms = np.sum(np.stack(sorted_rank_moe, axis=0), axis=0)
        sorted_rank_routed_tokens = np.sum(
            np.stack(sorted_rank_tokens, axis=0), axis=0
        )
        sorted_rank_active_experts = np.sum(
            np.stack(sorted_rank_active, axis=0), axis=0
        )
        critical_rank_idx = int(np.argmax(per_rank_verification_wall_ms))
        total_ms = float(per_rank_verification_wall_ms[critical_rank_idx])
        draft_ms = float(per_rank_draft_wall_ms[critical_rank_idx])
        ffn_ms = float(
            np.sum(
                per_rank_layer_routed_expert_gpu_ms[critical_rank_idx],
                dtype=np.float64,
            )
        )
        other_ms = float(per_rank_gpu_other_ms[critical_rank_idx])
        mean_rank_ffn_ms = float(np.mean(sorted_rank_ffn_ms))
        max_rank_ffn_ms = (
            float(sorted_rank_ffn_ms[0]) if sorted_rank_ffn_ms.size else 0.0
        )
        ffn_max_mean_ratio = (
            max_rank_ffn_ms / mean_rank_ffn_ms
            if mean_rank_ffn_ms > 0
            else float("nan")
        )
        mean_rank_moe_ms = float(np.mean(sorted_rank_moe_ms))
        max_rank_moe_ms = (
            float(sorted_rank_moe_ms[0]) if sorted_rank_moe_ms.size else 0.0
        )
        moe_max_mean_ratio = (
            max_rank_moe_ms / mean_rank_moe_ms
            if mean_rank_moe_ms > 0
            else float("nan")
        )

        global_barrier_ids.append(barrier_id)
        global_step_indices.append(int(spans[0][0]))
        barrier_first_seq_ids.append(int(spans[0][0]))
        barrier_last_seq_ids.append(int(spans[0][1]))
        barrier_num_collectives.append(int(spans[0][2]))
        rank_barrier_first_seq_ids.append(
            np.asarray([span[0] for span in spans], dtype=np.int64)
        )
        rank_barrier_last_seq_ids.append(
            np.asarray([span[1] for span in spans], dtype=np.int64)
        )
        rank_barrier_num_collectives.append(
            np.asarray([span[2] for span in spans], dtype=np.int64)
        )
        rank_step_kinds.append(
            np.asarray([str(record["step_kind"]) for record in records], dtype=np.str_)
        )
        rank_execute_wall_ms.append(per_rank_execute_wall_ms)
        rank_verification_wall_ms.append(per_rank_verification_wall_ms)
        rank_draft_wall_ms.append(per_rank_draft_wall_ms)
        rank_iteration_wall_ms.append(per_rank_iteration_wall_ms)
        rank_execute_gpu_ms.append(per_rank_execute_gpu_ms)
        rank_verification_gpu_ms.append(per_rank_verification_gpu_ms)
        rank_draft_gpu_ms.append(per_rank_draft_gpu_ms)
        rank_iteration_gpu_ms.append(per_rank_iteration_gpu_ms)
        rank_attention_gpu_ms.append(per_rank_attention_gpu_ms)
        rank_moe_gpu_ms.append(per_rank_moe_gpu_ms)
        rank_gpu_other_ms.append(per_rank_gpu_other_ms)
        rank_timing_complete.append(per_rank_timing_complete)
        rank_step_total_ms.append(per_rank_total_ms)
        rank_step_draft_ms.append(per_rank_draft_ms)
        rank_layer_moe_gpu_ms.append(per_rank_layer_moe_gpu_ms)
        rank_layer_routed_expert_gpu_ms.append(
            per_rank_layer_routed_expert_gpu_ms
        )
        rank_layer_shared_expert_gpu_ms.append(
            per_rank_layer_shared_expert_gpu_ms
        )
        rank_layer_routing_gpu_ms.append(per_rank_layer_routing_gpu_ms)
        rank_layer_prepare_gpu_ms.append(per_rank_layer_prepare_gpu_ms)
        rank_layer_finalize_gpu_ms.append(per_rank_layer_finalize_gpu_ms)
        rank_layer_ffn_ms.append(per_rank_layer_ffn_ms)
        rank_layer_local_routed_tokens.append(per_rank_layer_tokens)
        rank_layer_local_active_experts.append(per_rank_layer_active)
        rank_position_layer_ffn_ms.append(per_rank_position_layer_ffn_ms)
        rank_position_layer_local_routed_tokens.append(
            per_rank_position_layer_tokens
        )
        global_step_total_ms.append(total_ms)
        global_draft_ms.append(draft_ms)
        global_step_ffn_ms.append(ffn_ms)
        global_critical_rank_indices.append(critical_rank_idx)
        global_verification_wall_ms.append(total_ms)
        global_iteration_wall_ms.append(
            float(per_rank_iteration_wall_ms[critical_rank_idx])
        )
        global_draft_wall_ms.append(draft_ms)
        global_verification_gpu_total_ms.append(
            float(per_rank_verification_gpu_ms[critical_rank_idx])
        )
        global_attention_gpu_ms.append(
            float(per_rank_attention_gpu_ms[critical_rank_idx])
        )
        global_moe_gpu_ms.append(float(per_rank_moe_gpu_ms[critical_rank_idx]))
        global_gpu_other_ms.append(other_ms)
        global_step_sorted_rank_routed_expert_gpu_ms.append(sorted_rank_ffn_ms)
        global_step_sorted_rank_moe_gpu_ms.append(sorted_rank_moe_ms)
        global_step_routed_expert_max_mean_ratio.append(ffn_max_mean_ratio)
        global_step_moe_max_mean_ratio.append(moe_max_mean_ratio)
        global_step_sorted_rank_ffn_ms.append(sorted_rank_ffn_ms)
        global_step_sorted_rank_local_routed_tokens.append(sorted_rank_routed_tokens)
        global_step_sorted_rank_local_active_experts.append(sorted_rank_active_experts)
        global_step_position_sorted_rank_ffn_ms.append(position_sorted_rank_ffn)
        global_step_position_sorted_rank_local_routed_tokens.append(
            position_sorted_rank_tokens
        )
        global_step_ffn_max_mean_ratio.append(ffn_max_mean_ratio)
        global_step_other_ms.append(other_ms)
        global_step_kinds.append(global_step_kind)
        global_step_histograms.append(
            np.sum([record["step_histograms"] for record in records], axis=0)
        )
        global_step_total_tokens.append(
            sum(int(record["step_total_tokens"]) for record in records)
        )
        global_token_source_ranks.append(barrier_token_source_ranks)
        global_token_request_ids.append(barrier_token_request_ids)
        global_token_position_ids.append(barrier_token_position_ids)
        global_token_assignments.append(barrier_token_assignments)
        global_token_barrier_offsets.append(
            global_token_barrier_offsets[-1]
            + int(barrier_token_position_ids.shape[0])
        )

    if global_step_histograms:
        histogram_array = np.stack(global_step_histograms, axis=0).astype(np.int64)
    else:
        histogram_array = np.empty((0, len(layers), num_experts), dtype=np.int64)

    if global_step_sorted_rank_ffn_ms:
        sorted_rank_ffn_array = np.stack(
            global_step_sorted_rank_ffn_ms, axis=0
        ).astype(np.float64)
        sorted_rank_moe_array = np.stack(
            global_step_sorted_rank_moe_gpu_ms, axis=0
        ).astype(np.float64)
        sorted_rank_tokens_array = np.stack(
            global_step_sorted_rank_local_routed_tokens, axis=0
        ).astype(np.int64)
        sorted_rank_active_array = np.stack(
            global_step_sorted_rank_local_active_experts, axis=0
        ).astype(np.int64)
        position_sorted_rank_ffn_array = np.stack(
            global_step_position_sorted_rank_ffn_ms, axis=0
        ).astype(np.float64)
        position_sorted_rank_tokens_array = np.stack(
            global_step_position_sorted_rank_local_routed_tokens, axis=0
        ).astype(np.int64)
    else:
        sorted_rank_ffn_array = np.empty(
            (0, data_parallel_size), dtype=np.float64
        )
        sorted_rank_moe_array = np.empty(
            (0, data_parallel_size), dtype=np.float64
        )
        sorted_rank_tokens_array = np.empty(
            (0, data_parallel_size), dtype=np.int64
        )
        sorted_rank_active_array = np.empty(
            (0, data_parallel_size), dtype=np.int64
        )
        position_sorted_rank_ffn_array = np.empty(
            (0, 0, data_parallel_size), dtype=np.float64
        )
        position_sorted_rank_tokens_array = np.empty(
            (0, 0, data_parallel_size), dtype=np.int64
        )

    return GlobalStepTimingAggregation(
        global_barrier_ids=np.asarray(global_barrier_ids, dtype=np.int64),
        global_step_indices=np.asarray(global_step_indices, dtype=np.int64),
        barrier_first_ep_collective_seq_ids=np.asarray(
            barrier_first_seq_ids, dtype=np.int64
        ),
        barrier_last_ep_collective_seq_ids=np.asarray(
            barrier_last_seq_ids, dtype=np.int64
        ),
        barrier_num_ep_collectives=np.asarray(
            barrier_num_collectives, dtype=np.int64
        ),
        rank_barrier_first_ep_collective_seq_ids=(
            np.stack(rank_barrier_first_seq_ids, axis=0)
            if rank_barrier_first_seq_ids
            else np.empty((0, data_parallel_size), dtype=np.int64)
        ),
        rank_barrier_last_ep_collective_seq_ids=(
            np.stack(rank_barrier_last_seq_ids, axis=0)
            if rank_barrier_last_seq_ids
            else np.empty((0, data_parallel_size), dtype=np.int64)
        ),
        rank_barrier_num_ep_collectives=(
            np.stack(rank_barrier_num_collectives, axis=0)
            if rank_barrier_num_collectives
            else np.empty((0, data_parallel_size), dtype=np.int64)
        ),
        rank_step_kinds=(
            np.stack(rank_step_kinds, axis=0).astype(np.str_)
            if rank_step_kinds
            else np.empty((0, data_parallel_size), dtype=np.str_)
        ),
        rank_execute_wall_ms=(
            np.stack(rank_execute_wall_ms, axis=0).astype(np.float64)
            if rank_execute_wall_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_verification_wall_ms=(
            np.stack(rank_verification_wall_ms, axis=0).astype(np.float64)
            if rank_verification_wall_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_draft_wall_ms=(
            np.stack(rank_draft_wall_ms, axis=0).astype(np.float64)
            if rank_draft_wall_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_iteration_wall_ms=(
            np.stack(rank_iteration_wall_ms, axis=0).astype(np.float64)
            if rank_iteration_wall_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_execute_gpu_ms=(
            np.stack(rank_execute_gpu_ms, axis=0).astype(np.float64)
            if rank_execute_gpu_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_verification_gpu_ms=(
            np.stack(rank_verification_gpu_ms, axis=0).astype(np.float64)
            if rank_verification_gpu_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_draft_gpu_ms=(
            np.stack(rank_draft_gpu_ms, axis=0).astype(np.float64)
            if rank_draft_gpu_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_iteration_gpu_ms=(
            np.stack(rank_iteration_gpu_ms, axis=0).astype(np.float64)
            if rank_iteration_gpu_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_attention_gpu_ms=(
            np.stack(rank_attention_gpu_ms, axis=0).astype(np.float64)
            if rank_attention_gpu_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_moe_gpu_ms=(
            np.stack(rank_moe_gpu_ms, axis=0).astype(np.float64)
            if rank_moe_gpu_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_gpu_other_ms=(
            np.stack(rank_gpu_other_ms, axis=0).astype(np.float64)
            if rank_gpu_other_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_timing_complete=(
            np.stack(rank_timing_complete, axis=0).astype(np.bool_)
            if rank_timing_complete
            else np.empty((0, data_parallel_size), dtype=np.bool_)
        ),
        rank_step_total_ms=(
            np.stack(rank_step_total_ms, axis=0).astype(np.float64)
            if rank_step_total_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_step_draft_ms=(
            np.stack(rank_step_draft_ms, axis=0).astype(np.float64)
            if rank_step_draft_ms
            else np.empty((0, data_parallel_size), dtype=np.float64)
        ),
        rank_layer_moe_gpu_ms=(
            np.stack(rank_layer_moe_gpu_ms, axis=0).astype(np.float64)
            if rank_layer_moe_gpu_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_routed_expert_gpu_ms=(
            np.stack(rank_layer_routed_expert_gpu_ms, axis=0).astype(np.float64)
            if rank_layer_routed_expert_gpu_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_shared_expert_gpu_ms=(
            np.stack(rank_layer_shared_expert_gpu_ms, axis=0).astype(np.float64)
            if rank_layer_shared_expert_gpu_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_routing_gpu_ms=(
            np.stack(rank_layer_routing_gpu_ms, axis=0).astype(np.float64)
            if rank_layer_routing_gpu_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_prepare_gpu_ms=(
            np.stack(rank_layer_prepare_gpu_ms, axis=0).astype(np.float64)
            if rank_layer_prepare_gpu_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_finalize_gpu_ms=(
            np.stack(rank_layer_finalize_gpu_ms, axis=0).astype(np.float64)
            if rank_layer_finalize_gpu_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_ffn_ms=(
            np.stack(rank_layer_ffn_ms, axis=0).astype(np.float64)
            if rank_layer_ffn_ms
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.float64)
        ),
        rank_layer_local_routed_tokens=(
            np.stack(rank_layer_local_routed_tokens, axis=0).astype(np.int64)
            if rank_layer_local_routed_tokens
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.int64)
        ),
        rank_layer_local_active_experts=(
            np.stack(rank_layer_local_active_experts, axis=0).astype(np.int64)
            if rank_layer_local_active_experts
            else np.empty((0, data_parallel_size, len(layers)), dtype=np.int64)
        ),
        rank_position_layer_ffn_ms=(
            np.stack(rank_position_layer_ffn_ms, axis=0).astype(np.float64)
            if rank_position_layer_ffn_ms
            else np.empty(
                (0, data_parallel_size, 0, len(layers)), dtype=np.float64
            )
        ),
        rank_position_layer_local_routed_tokens=(
            np.stack(
                rank_position_layer_local_routed_tokens, axis=0
            ).astype(np.int64)
            if rank_position_layer_local_routed_tokens
            else np.empty(
                (0, data_parallel_size, 0, len(layers)), dtype=np.int64
            )
        ),
        global_step_total_ms=np.asarray(global_step_total_ms, dtype=np.float64),
        global_draft_ms=np.asarray(global_draft_ms, dtype=np.float64),
        global_step_ffn_ms=np.asarray(global_step_ffn_ms, dtype=np.float64),
        global_critical_rank_indices=np.asarray(
            global_critical_rank_indices, dtype=np.int64
        ),
        global_verification_wall_ms=np.asarray(
            global_verification_wall_ms, dtype=np.float64
        ),
        global_iteration_wall_ms=np.asarray(
            global_iteration_wall_ms, dtype=np.float64
        ),
        global_draft_wall_ms=np.asarray(global_draft_wall_ms, dtype=np.float64),
        global_verification_gpu_total_ms=np.asarray(
            global_verification_gpu_total_ms, dtype=np.float64
        ),
        global_attention_gpu_ms=np.asarray(
            global_attention_gpu_ms, dtype=np.float64
        ),
        global_moe_gpu_ms=np.asarray(global_moe_gpu_ms, dtype=np.float64),
        global_gpu_other_ms=np.asarray(global_gpu_other_ms, dtype=np.float64),
        global_step_sorted_rank_routed_expert_gpu_ms=sorted_rank_ffn_array,
        global_step_sorted_rank_moe_gpu_ms=sorted_rank_moe_array,
        global_step_routed_expert_max_mean_ratio=np.asarray(
            global_step_routed_expert_max_mean_ratio,
            dtype=np.float64,
        ),
        global_step_moe_max_mean_ratio=np.asarray(
            global_step_moe_max_mean_ratio,
            dtype=np.float64,
        ),
        global_step_sorted_rank_ffn_ms=sorted_rank_ffn_array,
        global_step_sorted_rank_local_routed_tokens=sorted_rank_tokens_array,
        global_step_sorted_rank_local_active_experts=sorted_rank_active_array,
        global_step_position_sorted_rank_ffn_ms=(
            position_sorted_rank_ffn_array
        ),
        global_step_position_sorted_rank_local_routed_tokens=(
            position_sorted_rank_tokens_array
        ),
        global_step_ffn_max_mean_ratio=np.asarray(
            global_step_ffn_max_mean_ratio, dtype=np.float64
        ),
        global_step_other_ms=np.asarray(global_step_other_ms, dtype=np.float64),
        global_step_kinds=np.asarray(global_step_kinds, dtype=np.str_),
        global_step_histograms=histogram_array,
        global_step_total_tokens=np.asarray(global_step_total_tokens, dtype=np.int64),
        global_token_barrier_offsets=np.asarray(
            global_token_barrier_offsets,
            dtype=np.int64,
        ),
        global_token_source_ranks=(
            np.concatenate(global_token_source_ranks, axis=0)
            if global_token_source_ranks
            else np.empty((0,), dtype=np.int16)
        ),
        global_token_request_ids=(
            np.concatenate(global_token_request_ids, axis=0)
            if global_token_request_ids
            else np.empty((0,), dtype=np.str_)
        ),
        global_token_position_ids=(
            np.concatenate(global_token_position_ids, axis=0)
            if global_token_position_ids
            else np.empty((0,), dtype=np.int16)
        ),
        global_token_layer_destination_assignment_counts=(
            np.concatenate(global_token_assignments, axis=0)
            if global_token_assignments
            else np.empty(
                (0, len(layers), data_parallel_size),
                dtype=np.int16,
            )
        ),
        num_global_candidate_steps=num_global_candidate_steps,
        num_global_captured_steps=len(global_step_indices),
        num_global_prefill_dropped_steps=num_global_prefill_dropped_steps,
        num_global_mixed_dropped_steps=num_global_mixed_dropped_steps,
        num_global_non_target_dropped_steps=num_global_non_target_dropped_steps,
    )


def summarize_global_step_time_components(
    step_total_ms: np.ndarray,
    step_ffn_ms: np.ndarray,
    step_other_ms: np.ndarray,
) -> dict[str, float]:
    avg_total_ms = float(np.mean(step_total_ms))
    avg_ffn_ms = float(np.mean(step_ffn_ms))
    avg_other_ms = float(np.mean(step_other_ms))
    return {
        "avg_step_total_ms": avg_total_ms,
        "avg_ffn_ms": avg_ffn_ms,
        "avg_other_ms": avg_other_ms,
    }


def normalize_global_time_components(
    summary_row: dict[str, float | int],
    baseline_total_ms: float,
) -> dict[str, float]:
    if baseline_total_ms <= 0:
        raise ValueError("baseline_total_ms must be positive.")
    return {
        "normalized_ffn_ms": (
            float(summary_row["avg_ffn_ms"]) / baseline_total_ms
        ),
        "normalized_other_ms": (
            float(summary_row["avg_other_ms"]) / baseline_total_ms
        ),
        "ffn_share": (
            float(summary_row["avg_ffn_ms"])
            / float(summary_row["avg_step_total_ms"])
            if float(summary_row["avg_step_total_ms"]) > 0
            else 0.0
        ),
        "other_share": (
            float(summary_row["avg_other_ms"])
            / float(summary_row["avg_step_total_ms"])
            if float(summary_row["avg_step_total_ms"]) > 0
            else 0.0
        ),
    }


def compute_num_output_tokens_excluding_first(
    output_lengths: np.ndarray,
) -> int:
    output_lengths = np.asarray(output_lengths, dtype=np.int64)
    return int(np.maximum(output_lengths - 1, 0).sum())


def compute_tpot_ms(
    decode_only_total_ms: float,
    output_lengths: np.ndarray,
) -> float:
    num_output_tokens_excl_first = compute_num_output_tokens_excluding_first(
        output_lengths
    )
    if num_output_tokens_excl_first <= 0:
        return 0.0
    return decode_only_total_ms / num_output_tokens_excl_first


def compute_tpot_ms_from_finished_stats(
    decode_time_total_ms: float,
    num_output_tokens_excl_first_total: int,
) -> float:
    if num_output_tokens_excl_first_total <= 0:
        return 0.0
    return decode_time_total_ms / num_output_tokens_excl_first_total


def compute_decode_throughput_tok_s(
    num_generation_tokens_total: int,
    decode_time_total_ms: float,
) -> float:
    if decode_time_total_ms <= 0:
        return 0.0
    return num_generation_tokens_total / (decode_time_total_ms / 1000.0)


def build_expert_to_ep_rank(
    *,
    num_experts: int,
    ep_size: int,
    placement_strategy: str = "linear",
) -> np.ndarray:
    if ep_size <= 0:
        raise ValueError("ep_size must be positive.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    expert_to_rank = np.full((num_experts,), -1, dtype=np.int64)
    if placement_strategy == "linear":
        base = num_experts // ep_size
        remainder = num_experts % ep_size
        for ep_rank in range(ep_size):
            local_count = base + (1 if ep_rank < remainder else 0)
            start = ep_rank * base + min(ep_rank, remainder)
            expert_to_rank[start : start + local_count] = ep_rank
    elif placement_strategy == "round_robin":
        for expert_id in range(num_experts):
            expert_to_rank[expert_id] = expert_id % ep_size
    else:
        raise ValueError(f"Unsupported expert placement strategy: {placement_strategy}")
    if np.any(expert_to_rank < 0):
        raise ValueError("Expert to EP rank mapping is incomplete.")
    return expert_to_rank


def merge_expert_to_ep_rank_maps(
    rank_maps: list[np.ndarray],
    *,
    num_experts: int,
    ep_size: int,
) -> np.ndarray:
    if not rank_maps:
        return build_expert_to_ep_rank(num_experts=num_experts, ep_size=ep_size)
    merged = np.full((num_experts,), -1, dtype=np.int64)
    for rank_map in rank_maps:
        rank_map = np.asarray(rank_map, dtype=np.int64)
        if rank_map.shape != (num_experts,):
            raise ValueError(
                f"expert_to_ep_rank map has shape {rank_map.shape}; "
                f"expected {(num_experts,)}."
            )
        owned = rank_map >= 0
        conflicts = owned & (merged >= 0) & (merged != rank_map)
        if np.any(conflicts):
            conflict_ids = np.flatnonzero(conflicts)[:8].tolist()
            raise ValueError(
                "Conflicting expert to EP rank ownership for experts "
                f"{conflict_ids}."
            )
        merged[owned] = rank_map[owned]
    if np.any(merged < 0):
        fallback = build_expert_to_ep_rank(num_experts=num_experts, ep_size=ep_size)
        merged[merged < 0] = fallback[merged < 0]
    return merged


def build_rank_load_from_histograms(
    avg_histograms: np.ndarray,
    expert_to_ep_rank: np.ndarray,
    ep_size: int,
) -> np.ndarray:
    avg_histograms = np.asarray(avg_histograms, dtype=np.float64)
    expert_to_ep_rank = np.asarray(expert_to_ep_rank, dtype=np.int64)
    if avg_histograms.ndim != 2:
        raise ValueError("avg_histograms must be shaped as (layers, experts).")
    if expert_to_ep_rank.shape != (avg_histograms.shape[1],):
        raise ValueError(
            "expert_to_ep_rank length must match avg_histograms expert dimension."
        )
    rank_load = np.zeros((avg_histograms.shape[0], ep_size), dtype=np.float64)
    for expert_id, ep_rank in enumerate(expert_to_ep_rank):
        if not 0 <= int(ep_rank) < ep_size:
            raise ValueError(
                f"expert_id={expert_id} has invalid ep_rank={int(ep_rank)}."
            )
        rank_load[:, int(ep_rank)] += avg_histograms[:, expert_id]
    return rank_load
