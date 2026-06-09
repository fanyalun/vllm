# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mtp_ep_load_balance_utils import (
    SCHEMA_VERSION,
    TPOT_DEFINITION,
    build_condition_metrics,
    build_rank_load_from_histograms,
    build_speedup_rows,
    reorder_histograms_by_expert_order,
    sort_experts_desc,
)

PLOT_MODULE = None


@dataclass
class LoadedConditionData:
    schema_version: int
    batch_size: int
    draft_length: int
    data_parallel_size: int
    num_samples: int
    batch_size_scope: str
    mixed_step_policy: str
    tpot_definition: str
    timing_backend: str
    timing_scope: str
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


def import_plot_module():
    global PLOT_MODULE
    if PLOT_MODULE is None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        PLOT_MODULE = plt
    return PLOT_MODULE


def ensure_analysis_dirs(output_dir: Path) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    speedup_dir = plots_dir / "speedup"
    time_dir = plots_dir / "step_time_breakdown"
    expert_load_dir = plots_dir / "expert_load"
    rank_load_dir = plots_dir / "rank_load"
    sorted_rank_ffn_dir = plots_dir / "sorted_rank"
    rank_traces_dir = plots_dir / "rank_traces"
    draft_drop_dir = plots_dir / "draft_drop"
    position_ffn_dir = plots_dir / "position_ffn_breakdown"
    routing_time_audit_dir = plots_dir / "routing_time_audit"
    for path in (
        tables_dir,
        speedup_dir,
        time_dir,
        expert_load_dir,
        rank_load_dir,
        sorted_rank_ffn_dir,
        rank_traces_dir,
        draft_drop_dir,
        position_ffn_dir,
        routing_time_audit_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "tables": tables_dir,
        "speedup": speedup_dir,
        "time": time_dir,
        "expert_load": expert_load_dir,
        "rank_load": rank_load_dir,
        "rank_ffn_time_sorted": sorted_rank_ffn_dir,
        "rank_traces": rank_traces_dir,
        "draft_drop": draft_drop_dir,
        "position_ffn": position_ffn_dir,
        "routing_time_audit": routing_time_audit_dir,
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rank_trace_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rank_trace_rows(
    condition_name: str,
    trace_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not trace_payloads:
        return rows
    if not any(payload.get("trace_samples") for payload in trace_payloads):
        return rows

    origin_ms = min(
        sample["step_start_time_ms"]
        for payload in trace_payloads
        for sample in payload.get("trace_samples", [])
    )
    for payload in trace_payloads:
        dp_rank = int(payload["dp_rank"])
        for order_idx, sample in enumerate(payload.get("trace_samples", [])):
            phase_totals = sample["phase_totals_ms"]
            rows.append(
                {
                    "condition": condition_name,
                    "batch_size": int(payload["batch_size"]),
                    "draft_length": int(payload["draft_length"]),
                    "dp_rank": dp_rank,
                    "trace_order": order_idx,
                    "step_index": int(sample["step_index"]),
                    "step_kind": sample["step_kind"],
                    "total_scheduled_tokens": int(sample["total_scheduled_tokens"]),
                    "step_start_offset_ms": (
                        float(sample["step_start_time_ms"]) - origin_ms
                    ),
                    "step_end_offset_ms": (
                        float(sample["step_end_time_ms"]) - origin_ms
                    ),
                    "step_total_ms": float(sample["step_total_ms"]),
                    "attention_gpu_ms": float(
                        phase_totals["attention_gpu"]
                    ),
                    "moe_gpu_ms": float(phase_totals["moe_gpu"]),
                    "gpu_other_ms": float(phase_totals["gpu_other"]),
                    "routing_gpu_ms": float(phase_totals["routing_gpu"]),
                    "prepare_gpu_ms": float(phase_totals["prepare_gpu"]),
                    "routed_expert_gpu_ms": float(
                        phase_totals["routed_expert_gpu"]
                    ),
                    "shared_expert_gpu_ms": float(
                        phase_totals["shared_expert_gpu"]
                    ),
                    "finalize_gpu_ms": float(phase_totals["finalize_gpu"]),
                }
            )
    return rows


def plot_rank_trace_timeline(
    plot_dir: Path,
    condition_name: str,
    trace_payloads: list[dict[str, Any]],
) -> Path | None:
    samples_exist = any(payload.get("trace_samples") for payload in trace_payloads)
    if not samples_exist:
        return None

    plt = import_plot_module()
    label_colors = {
        "attention": "#4E79A7",
        "moe": "#B07AA1",
        "routing": "#F28E2B",
        "prepare": "#E15759",
        "routed_expert": "#59A14F",
        "shared_expert": "#EDC948",
        "finalize": "#76B7B2",
    }
    origin_ms = min(
        sample["step_start_time_ms"]
        for payload in trace_payloads
        for sample in payload.get("trace_samples", [])
    )
    max_end_ms = max(
        sample["step_end_time_ms"]
        for payload in trace_payloads
        for sample in payload.get("trace_samples", [])
    )

    fig, axes = plt.subplots(
        len(trace_payloads),
        1,
        figsize=(12, 2.8 * len(trace_payloads)),
        sharex=True,
    )
    if len(trace_payloads) == 1:
        axes = [axes]

    for ax, payload in zip(axes, trace_payloads):
        dp_rank = int(payload["dp_rank"])
        samples = sorted(
            payload.get("trace_samples", []),
            key=lambda item: float(item["step_start_time_ms"]),
        )
        for sample_idx, sample in enumerate(samples):
            y = len(samples) - sample_idx - 1
            start_ms = float(sample["step_start_time_ms"]) - origin_ms
            total_ms = float(sample["step_total_ms"])
            ax.broken_barh(
                [(start_ms, total_ms)],
                (y - 0.35, 0.7),
                facecolors="none",
                edgecolors="#444444",
                linewidth=0.8,
            )
            for event in sample["events"]:
                label = str(event["label"])
                event_start = start_ms + float(event["start_ms"])
                duration = float(event["duration_ms"])
                ax.broken_barh(
                    [(event_start, duration)],
                    (y - 0.35, 0.7),
                    facecolors=label_colors.get(label, "#999999"),
                    edgecolors="none",
                )
            ax.text(
                start_ms - 1.0,
                y,
                f"step={int(sample['step_index'])}",
                ha="right",
                va="center",
                fontsize=8,
            )
        ax.set_title(f"dp_rank={dp_rank}")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in label_colors.values()
    ]
    fig.legend(
        handles,
        list(label_colors.keys()),
        ncol=len(label_colors),
        loc="upper center",
    )
    axes[-1].set_xlabel("CUDA Event offset within traced samples (ms)")
    axes[-1].set_xlim(0.0, max_end_ms - origin_ms)
    fig.suptitle(f"{condition_name} DP rank event timeline", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = plot_dir / f"{condition_name}_timeline.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _scalar(npz: Any, key: str, cast: type) -> Any:
    value = npz[key]
    return cast(value.reshape(-1)[0])


def load_condition_data(path: Path) -> LoadedConditionData:
    with np.load(path, allow_pickle=False) as npz:
        schema_version = _scalar(npz, "schema_version", int)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported raw schema_version="
                f"{schema_version} for {path}. Schema v9 and older timing data "
                "used synchronized wall-clock instrumentation and cannot be "
                "analyzed as CUDA Event data. Re-run `collect`."
            )
        num_barriers = int(np.asarray(npz["global_barrier_ids"]).shape[0])
        data_parallel_size = _scalar(npz, "data_parallel_size", int)
        layers = np.asarray(npz["layers"])
        rank_step_total_ms = np.asarray(npz["rank_step_total_ms"], dtype=np.float64)
        rank_layer_ffn_ms = np.asarray(
            npz["rank_layer_ffn_ms"], dtype=np.float64
        )
        return LoadedConditionData(
            schema_version=schema_version,
            batch_size=_scalar(npz, "batch_size", int),
            draft_length=_scalar(npz, "draft_length", int),
            data_parallel_size=data_parallel_size,
            num_samples=_scalar(npz, "num_samples", int),
            batch_size_scope=_scalar(npz, "batch_size_scope", str),
            mixed_step_policy=_scalar(npz, "mixed_step_policy", str),
            tpot_definition=_scalar(npz, "tpot_definition", str),
            timing_backend=_scalar(npz, "timing_backend", str),
            timing_scope=_scalar(npz, "timing_scope", str),
            selected_dataset_indices=np.asarray(npz["selected_dataset_indices"]),
            prompt_lengths=np.asarray(npz["prompt_lengths"]),
            output_lengths=np.asarray(npz["output_lengths"]),
            condition_latency_ms=_scalar(npz, "condition_latency_ms", float),
            decode_time_total_ms=_scalar(npz, "decode_time_total_ms", float),
            num_output_tokens_total=_scalar(npz, "num_output_tokens_total", int),
            num_generation_tokens_total=_scalar(
                npz, "num_generation_tokens_total", int
            ),
            num_output_tokens_excl_first_total=_scalar(
                npz, "num_output_tokens_excl_first_total", int
            ),
            tpot_ms=_scalar(npz, "tpot_ms", float),
            decode_throughput_tok_s=_scalar(
                npz, "decode_throughput_tok_s", float
            ),
            vllm_generation_elapsed_ms=_scalar(
                npz, "vllm_generation_elapsed_ms", float
            ),
            vllm_request_tpot_ms=_scalar(npz, "vllm_request_tpot_ms", float),
            vllm_generation_throughput_tok_s=_scalar(
                npz, "vllm_generation_throughput_tok_s", float
            ),
            spec_num_drafts=_scalar(npz, "spec_num_drafts", int),
            spec_num_draft_tokens=_scalar(npz, "spec_num_draft_tokens", int),
            spec_num_accepted_tokens=_scalar(
                npz, "spec_num_accepted_tokens", int
            ),
            spec_acceptance_rate=_scalar(npz, "spec_acceptance_rate", float),
            spec_mean_acceptance_length=_scalar(
                npz, "spec_mean_acceptance_length", float
            ),
            step_histograms=np.asarray(npz["step_histograms"]),
            step_total_tokens=np.asarray(npz["step_total_tokens"]),
            step_total_ms=np.asarray(npz["step_total_ms"]),
            step_attention_ms=np.asarray(npz["step_attention_ms"]),
            step_routing_ms=np.asarray(npz["step_routing_ms"]),
            step_prepare_ms=np.asarray(npz["step_prepare_ms"]),
            step_finalize_ms=np.asarray(npz["step_finalize_ms"]),
            step_ffn_ms=np.asarray(npz["step_ffn_ms"]),
            captured_step_kinds=np.asarray(npz["captured_step_kinds"]),
            global_barrier_ids=np.asarray(npz["global_barrier_ids"]),
            barrier_first_ep_collective_seq_ids=np.asarray(
                npz["barrier_first_ep_collective_seq_ids"]
            ),
            barrier_last_ep_collective_seq_ids=np.asarray(
                npz["barrier_last_ep_collective_seq_ids"]
            ),
            barrier_num_ep_collectives=np.asarray(
                npz["barrier_num_ep_collectives"]
            ),
            rank_barrier_first_ep_collective_seq_ids=np.asarray(
                npz["rank_barrier_first_ep_collective_seq_ids"]
            )
            if "rank_barrier_first_ep_collective_seq_ids" in npz
            else np.empty((num_barriers, data_parallel_size), dtype=np.int64),
            rank_barrier_last_ep_collective_seq_ids=np.asarray(
                npz["rank_barrier_last_ep_collective_seq_ids"]
            )
            if "rank_barrier_last_ep_collective_seq_ids" in npz
            else np.empty((num_barriers, data_parallel_size), dtype=np.int64),
            rank_barrier_num_ep_collectives=np.asarray(
                npz["rank_barrier_num_ep_collectives"]
            )
            if "rank_barrier_num_ep_collectives" in npz
            else np.empty((num_barriers, data_parallel_size), dtype=np.int64),
            rank_step_kinds=np.asarray(npz["rank_step_kinds"]),
            rank_execute_wall_ms=np.asarray(npz["rank_execute_wall_ms"]),
            rank_verification_wall_ms=np.asarray(
                npz["rank_verification_wall_ms"]
            ),
            rank_draft_wall_ms=np.asarray(npz["rank_draft_wall_ms"]),
            rank_iteration_wall_ms=np.asarray(npz["rank_iteration_wall_ms"]),
            rank_execute_gpu_ms=np.asarray(npz["rank_execute_gpu_ms"]),
            rank_verification_gpu_ms=np.asarray(
                npz["rank_verification_gpu_ms"]
            ),
            rank_draft_gpu_ms=np.asarray(npz["rank_draft_gpu_ms"]),
            rank_iteration_gpu_ms=np.asarray(npz["rank_iteration_gpu_ms"]),
            rank_attention_gpu_ms=np.asarray(npz["rank_attention_gpu_ms"]),
            rank_moe_gpu_ms=np.asarray(npz["rank_moe_gpu_ms"]),
            rank_gpu_other_ms=np.asarray(npz["rank_gpu_other_ms"]),
            rank_timing_complete=np.asarray(
                npz["rank_timing_complete"],
                dtype=np.bool_,
            ),
            rank_step_total_ms=rank_step_total_ms,
            rank_step_draft_ms=np.asarray(npz["rank_step_draft_ms"]),
            rank_layer_moe_gpu_ms=np.asarray(npz["rank_layer_moe_gpu_ms"]),
            rank_layer_routed_expert_gpu_ms=np.asarray(
                npz["rank_layer_routed_expert_gpu_ms"]
            ),
            rank_layer_shared_expert_gpu_ms=np.asarray(
                npz["rank_layer_shared_expert_gpu_ms"]
            ),
            rank_layer_routing_gpu_ms=np.asarray(
                npz["rank_layer_routing_gpu_ms"]
            ),
            rank_layer_prepare_gpu_ms=np.asarray(
                npz["rank_layer_prepare_gpu_ms"]
            ),
            rank_layer_finalize_gpu_ms=np.asarray(
                npz["rank_layer_finalize_gpu_ms"]
            ),
            rank_layer_ffn_ms=rank_layer_ffn_ms,
            rank_layer_local_routed_tokens=np.asarray(
                npz["rank_layer_local_routed_tokens"]
            ),
            rank_layer_local_active_experts=np.asarray(
                npz["rank_layer_local_active_experts"]
            ),
            global_step_indices=np.asarray(npz["global_step_indices"]),
            global_step_total_ms=np.asarray(npz["global_step_total_ms"]),
            global_draft_ms=np.asarray(npz["global_draft_ms"]),
            global_step_ffn_ms=np.asarray(npz["global_step_ffn_ms"]),
            global_critical_rank_indices=np.asarray(
                npz["global_critical_rank_indices"]
            ),
            global_verification_wall_ms=np.asarray(
                npz["global_verification_wall_ms"]
            ),
            global_iteration_wall_ms=np.asarray(
                npz["global_iteration_wall_ms"]
            ),
            global_draft_wall_ms=np.asarray(npz["global_draft_wall_ms"]),
            global_verification_gpu_total_ms=np.asarray(
                npz["global_verification_gpu_total_ms"]
            ),
            global_attention_gpu_ms=np.asarray(
                npz["global_attention_gpu_ms"]
            ),
            global_moe_gpu_ms=np.asarray(npz["global_moe_gpu_ms"]),
            global_gpu_other_ms=np.asarray(npz["global_gpu_other_ms"]),
            global_step_sorted_rank_routed_expert_gpu_ms=np.asarray(
                npz["global_step_sorted_rank_routed_expert_gpu_ms"]
            ),
            global_step_sorted_rank_moe_gpu_ms=np.asarray(
                npz["global_step_sorted_rank_moe_gpu_ms"]
            ),
            global_step_routed_expert_max_mean_ratio=np.asarray(
                npz["global_step_routed_expert_max_mean_ratio"]
            ),
            global_step_moe_max_mean_ratio=np.asarray(
                npz["global_step_moe_max_mean_ratio"]
            ),
            global_step_sorted_rank_ffn_ms=np.asarray(
                npz["global_step_sorted_rank_ffn_ms"]
            ),
            global_step_sorted_rank_local_routed_tokens=np.asarray(
                npz["global_step_sorted_rank_local_routed_tokens"]
            ),
            global_step_sorted_rank_local_active_experts=np.asarray(
                npz["global_step_sorted_rank_local_active_experts"]
            ),
            global_step_position_sorted_rank_ffn_ms=np.asarray(
                npz["global_step_position_sorted_rank_ffn_ms"]
            ),
            global_step_position_sorted_rank_local_routed_tokens=np.asarray(
                npz["global_step_position_sorted_rank_local_routed_tokens"]
            ),
            global_step_ffn_max_mean_ratio=np.asarray(
                npz["global_step_ffn_max_mean_ratio"]
            ),
            global_step_other_ms=np.asarray(npz["global_gpu_other_ms"]),
            global_step_kinds=np.asarray(npz["global_step_kinds"]),
            global_token_barrier_offsets=np.asarray(
                npz["global_token_barrier_offsets"],
                dtype=np.int64,
            )
            if "global_token_barrier_offsets" in npz
            else np.zeros((num_barriers + 1,), dtype=np.int64),
            global_token_source_ranks=np.asarray(
                npz["global_token_source_ranks"],
                dtype=np.int16,
            )
            if "global_token_source_ranks" in npz
            else np.empty((0,), dtype=np.int16),
            global_token_request_ids=np.asarray(
                npz["global_token_request_ids"],
                dtype=np.str_,
            )
            if "global_token_request_ids" in npz
            else np.empty((0,), dtype=np.str_),
            global_token_position_ids=np.asarray(
                npz["global_token_position_ids"],
                dtype=np.int16,
            )
            if "global_token_position_ids" in npz
            else np.empty((0,), dtype=np.int16),
            global_token_layer_destination_assignment_counts=np.asarray(
                npz["global_token_layer_destination_assignment_counts"],
                dtype=np.int16,
            )
            if "global_token_layer_destination_assignment_counts" in npz
            else np.empty(
                (0, layers.shape[0], data_parallel_size),
                dtype=np.int16,
            ),
            expert_to_ep_rank=np.asarray(npz["expert_to_ep_rank"]),
            layers=layers,
            avg_histograms=np.asarray(npz["avg_histograms"]),
            num_forward_steps_total=_scalar(npz, "num_forward_steps_total", int),
            num_captured_steps=_scalar(npz, "num_captured_steps", int),
            num_global_candidate_steps=_scalar(
                npz, "num_global_candidate_steps", int
            ),
            num_global_captured_steps=_scalar(
                npz, "num_global_captured_steps", int
            ),
            num_dropped_steps=_scalar(npz, "num_dropped_steps", int),
            num_prefill_dropped_steps=_scalar(npz, "num_prefill_dropped_steps", int),
            num_mixed_dropped_steps=_scalar(npz, "num_mixed_dropped_steps", int),
            num_global_prefill_dropped_steps=_scalar(
                npz, "num_global_prefill_dropped_steps", int
            ),
            num_global_mixed_dropped_steps=_scalar(
                npz, "num_global_mixed_dropped_steps", int
            ),
            num_global_non_target_dropped_steps=_scalar(
                npz, "num_global_non_target_dropped_steps", int
            ),
        )


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "collect_manifest.json"
    if not manifest_path.exists():
        return synthesize_manifest(output_dir)
    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported manifest schema_version="
            f"{manifest['schema_version']}. Schema v10 raw data must be "
            "re-collected before analysis."
        )
    return manifest


def synthesize_manifest(output_dir: Path) -> dict[str, Any]:
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata: dict[str, Any] = {}
    if run_metadata_path.exists():
        with run_metadata_path.open("r", encoding="utf-8") as fp:
            run_metadata = json.load(fp)

    raw_paths = sorted((output_dir / "raw").glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No collect_manifest.json or raw/*.npz found under {output_dir}."
        )

    conditions: list[dict[str, Any]] = []
    batch_sizes: set[int] = set()
    draft_lengths: set[int] = set()
    first_data: LoadedConditionData | None = None
    for raw_path in raw_paths:
        data = load_condition_data(raw_path)
        if first_data is None:
            first_data = data
        batch_sizes.add(data.batch_size)
        draft_lengths.add(data.draft_length)
        conditions.append(
            {
                "batch_size": data.batch_size,
                "draft_length": data.draft_length,
                "raw_path": str(Path("raw") / raw_path.name),
                "condition_latency_ms": data.condition_latency_ms,
                "decode_time_total_ms": data.decode_time_total_ms,
                "num_output_tokens_total": data.num_output_tokens_total,
                "num_generation_tokens_total": data.num_generation_tokens_total,
                "num_output_tokens_excl_first_total": (
                    data.num_output_tokens_excl_first_total
                ),
                "tpot_ms": data.tpot_ms,
                "decode_throughput_tok_s": data.decode_throughput_tok_s,
                "vllm_generation_elapsed_ms": data.vllm_generation_elapsed_ms,
                "vllm_request_tpot_ms": data.vllm_request_tpot_ms,
                "vllm_generation_throughput_tok_s": (
                    data.vllm_generation_throughput_tok_s
                ),
                "spec_num_drafts": data.spec_num_drafts,
                "spec_num_draft_tokens": data.spec_num_draft_tokens,
                "spec_num_accepted_tokens": data.spec_num_accepted_tokens,
                "spec_acceptance_rate": data.spec_acceptance_rate,
                "spec_mean_acceptance_length": data.spec_mean_acceptance_length,
                "num_forward_steps_total": data.num_forward_steps_total,
                "num_captured_steps": data.num_captured_steps,
                "num_global_candidate_steps": data.num_global_candidate_steps,
                "num_global_captured_steps": data.num_global_captured_steps,
                "num_dropped_steps": data.num_dropped_steps,
                "num_prefill_dropped_steps": data.num_prefill_dropped_steps,
                "num_mixed_dropped_steps": data.num_mixed_dropped_steps,
                "num_global_prefill_dropped_steps": (
                    data.num_global_prefill_dropped_steps
                ),
                "num_global_mixed_dropped_steps": (
                    data.num_global_mixed_dropped_steps
                ),
                "num_global_non_target_dropped_steps": (
                    data.num_global_non_target_dropped_steps
                ),
            }
        )

    assert first_data is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "model": run_metadata.get("model", "unknown"),
        "dataset": run_metadata.get("dataset", "unknown"),
        "dataset_config": run_metadata.get("dataset_config"),
        "dataset_split": run_metadata.get("dataset_split", "unknown"),
        "batch_sizes": sorted(batch_sizes),
        "draft_lengths": sorted(draft_lengths),
        "data_parallel_size": int(
            run_metadata.get("data_parallel_size", first_data.data_parallel_size)
        ),
        "batch_size_scope": run_metadata.get(
            "batch_size_scope", first_data.batch_size_scope
        ),
        "num_samples": int(run_metadata.get("num_samples", first_data.num_samples)),
        "max_tokens": int(run_metadata.get("max_tokens", 0)),
        "layers": run_metadata.get("layers", first_data.layers.tolist()),
        "num_experts": int(
            run_metadata.get("num_experts", first_data.avg_histograms.shape[1])
        ),
        "warmup_rounds": int(run_metadata.get("warmup_rounds", 0)),
        "trace_steps_per_rank": int(run_metadata.get("trace_steps_per_rank", 0)),
        "mixed_step_policy": run_metadata.get(
            "mixed_step_policy", first_data.mixed_step_policy
        ),
        "tpot_definition": run_metadata.get(
            "tpot_definition", first_data.tpot_definition
        ),
        "conditions": conditions,
    }


def load_all_conditions(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], LoadedConditionData]]:
    manifest = load_manifest(output_dir)
    results: dict[tuple[int, int], LoadedConditionData] = {}
    for condition in manifest["conditions"]:
        path = output_dir / condition["raw_path"]
        data = load_condition_data(path)
        results[(data.batch_size, data.draft_length)] = data
    return manifest, results


def build_step_time_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, float | int]]:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    rows: list[dict[str, float | int]] = []
    baseline_wall_totals: dict[int, float] = {}
    baseline_gpu_totals: dict[int, float] = {}
    partial_rows: dict[tuple[int, int], dict[str, float | int]] = {}

    for batch_size in batch_sizes:
        for draft_length in draft_lengths:
            data = results[(batch_size, draft_length)]
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            if not np.any(target_mask):
                raise ValueError(
                    "No strict target barriers for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            verification_wall = data.global_verification_wall_ms[target_mask]
            iteration_wall = data.global_iteration_wall_ms[target_mask]
            draft_wall = data.global_draft_wall_ms[target_mask]
            verification_gpu = data.global_verification_gpu_total_ms[target_mask]
            attention_gpu = data.global_attention_gpu_ms[target_mask]
            moe_gpu = data.global_moe_gpu_ms[target_mask]
            other_gpu = data.global_gpu_other_ms[target_mask]
            routed_expert_gpu = data.global_step_ffn_ms[target_mask]
            if not np.allclose(
                attention_gpu + moe_gpu + other_gpu,
                verification_gpu,
                rtol=1e-6,
                atol=1e-3,
            ):
                raise ValueError(
                    "Attention + MoE + Other must equal verification GPU total "
                    f"for batch_size={batch_size}, draft_length={draft_length}."
                )
            avg_verification_wall_ms = float(np.mean(verification_wall))
            avg_verification_gpu_ms = float(np.mean(verification_gpu))
            avg_attention_gpu_ms = float(np.mean(attention_gpu))
            avg_moe_gpu_ms = float(np.mean(moe_gpu))
            avg_other_gpu_ms = float(np.mean(other_gpu))
            avg_routed_expert_gpu_ms = float(np.mean(routed_expert_gpu))
            row: dict[str, float | int] = {
                "batch_size": batch_size,
                "draft_length": draft_length,
                "timing_scope": data.timing_scope,
                "timing_backend": data.timing_backend,
                "decode_step_scope": (
                    "verification_only" if draft_length > 0 else "decode_only"
                ),
                "num_steps": int(np.count_nonzero(target_mask)),
                "num_forward_steps_total": data.num_forward_steps_total,
                "num_captured_steps": data.num_captured_steps,
                "num_global_candidate_steps": data.num_global_candidate_steps,
                "num_global_captured_steps": data.num_global_captured_steps,
                "num_dropped_steps": data.num_dropped_steps,
                "num_prefill_dropped_steps": data.num_prefill_dropped_steps,
                "num_mixed_dropped_steps": data.num_mixed_dropped_steps,
                "num_global_prefill_dropped_steps": (
                    data.num_global_prefill_dropped_steps
                ),
                "num_global_mixed_dropped_steps": (
                    data.num_global_mixed_dropped_steps
                ),
                "num_global_non_target_dropped_steps": (
                    data.num_global_non_target_dropped_steps
                ),
                "global_captured_step_ratio": (
                    data.num_global_captured_steps / data.num_global_candidate_steps
                    if data.num_global_candidate_steps > 0
                    else 0.0
                ),
                "avg_verification_wall_ms": avg_verification_wall_ms,
                "p50_verification_wall_ms": float(
                    np.percentile(verification_wall, 50)
                ),
                "p95_verification_wall_ms": float(
                    np.percentile(verification_wall, 95)
                ),
                "avg_iteration_wall_ms": float(np.mean(iteration_wall)),
                "avg_draft_wall_ms": float(np.mean(draft_wall)),
                "avg_verification_gpu_total_ms": avg_verification_gpu_ms,
                "avg_attention_gpu_ms": avg_attention_gpu_ms,
                "avg_moe_gpu_ms": avg_moe_gpu_ms,
                "avg_routed_expert_gpu_ms": avg_routed_expert_gpu_ms,
                "avg_gpu_other_ms": avg_other_gpu_ms,
                "attention_gpu_share": (
                    avg_attention_gpu_ms / avg_verification_gpu_ms
                    if avg_verification_gpu_ms > 0
                    else 0.0
                ),
                "moe_gpu_share": (
                    avg_moe_gpu_ms / avg_verification_gpu_ms
                    if avg_verification_gpu_ms > 0
                    else 0.0
                ),
                "avg_step_total_ms": avg_verification_wall_ms,
                "avg_ffn_ms": avg_routed_expert_gpu_ms,
                "avg_other_ms": avg_other_gpu_ms,
                "ffn_share": (
                    avg_routed_expert_gpu_ms / avg_verification_gpu_ms
                    if avg_verification_gpu_ms > 0
                    else 0.0
                ),
            }
            partial_rows[(batch_size, draft_length)] = row
            if draft_length == 0:
                baseline_wall_totals[batch_size] = avg_verification_wall_ms
                baseline_gpu_totals[batch_size] = avg_verification_gpu_ms

    for batch_size in batch_sizes:
        baseline_wall = baseline_wall_totals[batch_size]
        baseline_gpu = baseline_gpu_totals[batch_size]
        for draft_length in draft_lengths:
            row = dict(partial_rows[(batch_size, draft_length)])
            row["normalized_verification_wall"] = (
                float(row["avg_verification_wall_ms"]) / baseline_wall
                if baseline_wall > 0
                else 0.0
            )
            row["normalized_verification_gpu_total"] = (
                float(row["avg_verification_gpu_total_ms"]) / baseline_gpu
                if baseline_gpu > 0
                else 0.0
            )
            row["normalized_attention_gpu"] = (
                float(row["avg_attention_gpu_ms"]) / baseline_gpu
                if baseline_gpu > 0
                else 0.0
            )
            row["normalized_moe_gpu"] = (
                float(row["avg_moe_gpu_ms"]) / baseline_gpu
                if baseline_gpu > 0
                else 0.0
            )
            row["normalized_routed_expert_gpu"] = (
                float(row["avg_routed_expert_gpu_ms"]) / baseline_gpu
                if baseline_gpu > 0
                else 0.0
            )
            row["normalized_gpu_other"] = (
                float(row["avg_gpu_other_ms"]) / baseline_gpu
                if baseline_gpu > 0
                else 0.0
            )
            row["normalized_total_ms"] = row["normalized_verification_wall"]
            row["normalized_ffn_ms"] = row["normalized_routed_expert_gpu"]
            row["normalized_other_ms"] = row["normalized_gpu_other"]
            rows.append(row)
    return rows


def build_load_distribution_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    layers = tuple(results[(batch_sizes[0], draft_lengths[0])].layers.tolist())

    load_metric_rows: list[dict[str, Any]] = []
    condition_sorted_rows: list[dict[str, Any]] = []
    baseline_sorted_rows: list[dict[str, Any]] = []
    rank_load_rows: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        baseline_data = results[(batch_size, 0)]
        baseline_avg = strict_average_histograms(
            baseline_data,
            draft_length=0,
        )
        _, baseline_order = sort_experts_desc(baseline_avg)
        for draft_length in draft_lengths:
            data = results[(batch_size, draft_length)]
            condition_avg = strict_average_histograms(
                data,
                draft_length=draft_length,
            )
            num_steps = int(
                np.count_nonzero(
                    strict_target_barrier_mask(
                        data,
                        draft_length=draft_length,
                    )
                )
            )
            metrics_rows = build_condition_metrics(
                batch_size=batch_size,
                draft_length=draft_length,
                num_steps=num_steps,
                layers=layers,
                avg_histograms=condition_avg,
                baseline_histograms=baseline_avg,
            )
            load_metric_rows.extend(metrics_rows)
            rank_load = build_rank_load_from_histograms(
                condition_avg,
                data.expert_to_ep_rank,
                data.data_parallel_size,
            )

            condition_sorted_counts, condition_order = sort_experts_desc(
                condition_avg
            )
            baseline_sorted_counts = reorder_histograms_by_expert_order(
                condition_avg,
                baseline_order,
            )

            for layer_row, layer_idx in enumerate(layers):
                for expert_rank, expert_id in enumerate(condition_order[layer_row]):
                    condition_sorted_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "layer": layer_idx,
                            "expert_rank": expert_rank,
                            "expert_id": int(expert_id),
                            "avg_routed_assignments_per_step": float(
                                condition_sorted_counts[layer_row, expert_rank]
                            ),
                        }
                    )
                for expert_rank, expert_id in enumerate(baseline_order[layer_row]):
                    baseline_sorted_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "layer": layer_idx,
                            "expert_rank": expert_rank,
                            "expert_id": int(expert_id),
                            "avg_routed_assignments_per_step": float(
                                baseline_sorted_counts[layer_row, expert_rank]
                            ),
                        }
                    )
                for ep_rank in range(data.data_parallel_size):
                    rank_load_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "layer": layer_idx,
                            "ep_rank": ep_rank,
                            "avg_routed_assignments_per_step": float(
                                rank_load[layer_row, ep_rank]
                            ),
                        }
                    )
    return (
        load_metric_rows,
        condition_sorted_rows,
        baseline_sorted_rows,
        rank_load_rows,
    )


def _finite_mean(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.percentile(finite_values, percentile))


def _max_min_ratio(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if min_value <= 0.0:
        return float("nan")
    return max_value / min_value


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 2:
        return float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def _spearman_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 2:
        return float("nan")
    return _pearson_corr(_rankdata(left), _rankdata(right))


def _decode_step_scope(draft_length: int) -> str:
    return "verification_only" if draft_length > 0 else "decode_only"


def _validate_assignment_tensor(
    data: LoadedConditionData,
    *,
    batch_size: int,
    draft_length: int,
) -> None:
    if data.schema_version != SCHEMA_VERSION:
        raise RuntimeError(
            "Routing-time audit requires schema v10 raw data for "
            f"batch_size={batch_size}, draft_length={draft_length}; found "
            f"schema v{data.schema_version}."
        )
    assignments = np.asarray(
        data.global_token_layer_destination_assignment_counts,
        dtype=np.int64,
    )
    expected_shape = (
        data.global_token_position_ids.shape[0],
        data.layers.shape[0],
        data.data_parallel_size,
    )
    if assignments.shape != expected_shape:
        raise RuntimeError(
            "global_token_layer_destination_assignment_counts has shape "
            f"{assignments.shape}; expected {expected_shape} for "
            f"batch_size={batch_size}, draft_length={draft_length}."
        )
    offsets = np.asarray(data.global_token_barrier_offsets, dtype=np.int64)
    if offsets.shape != (data.global_barrier_ids.shape[0] + 1,):
        raise RuntimeError(
            "global_token_barrier_offsets must contain one boundary per "
            f"global barrier for batch_size={batch_size}, "
            f"draft_length={draft_length}."
        )
    if offsets.size and (offsets[0] != 0 or offsets[-1] != assignments.shape[0]):
        raise RuntimeError(
            "global_token_barrier_offsets must span all token assignment rows "
            f"for batch_size={batch_size}, draft_length={draft_length}."
        )


def _condition_assignment_summaries(
    data: LoadedConditionData,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(data.global_token_barrier_offsets, dtype=np.int64)
    assignments = np.asarray(
        data.global_token_layer_destination_assignment_counts,
        dtype=np.int64,
    )
    source_ranks = np.asarray(data.global_token_source_ranks, dtype=np.int64)
    num_barriers = data.global_barrier_ids.shape[0]
    destination_assignments = np.zeros(
        (
            num_barriers,
            data.layers.shape[0],
            data.data_parallel_size,
        ),
        dtype=np.int64,
    )
    self_assignments = np.zeros_like(destination_assignments)
    lengths = np.diff(offsets)
    nonempty = np.flatnonzero(lengths > 0)
    if nonempty.size == 0:
        return destination_assignments, self_assignments

    starts = offsets[nonempty]
    destination_assignments[nonempty] = np.add.reduceat(
        assignments,
        starts,
        axis=0,
    )
    for destination_rank in range(data.data_parallel_size):
        self_rank_assignments = (
            assignments[:, :, destination_rank]
            * (source_ranks == destination_rank)[:, None]
        )
        self_assignments[nonempty, :, destination_rank] = np.add.reduceat(
            self_rank_assignments,
            starts,
            axis=0,
        )
    return destination_assignments, self_assignments


def _ffn_sorted_destination_tokens_for_steps(
    data: LoadedConditionData,
    barrier_indices: np.ndarray,
    destination_assignments: np.ndarray,
) -> np.ndarray:
    sorted_tokens = np.zeros(
        (barrier_indices.shape[0], data.data_parallel_size),
        dtype=np.int64,
    )
    ffn = np.asarray(
        data.rank_layer_routed_expert_gpu_ms[barrier_indices],
        dtype=np.float64,
    )
    for layer_idx in range(data.layers.shape[0]):
        order = np.argsort(-ffn[:, :, layer_idx], axis=1, kind="stable")
        sorted_tokens += np.take_along_axis(
            destination_assignments[:, layer_idx, :],
            order,
            axis=1,
        )
    return sorted_tokens


def validate_routing_time_audit_invariants(
    data: LoadedConditionData,
    *,
    batch_size: int,
    draft_length: int,
    target_mask: np.ndarray,
    destination_assignments: np.ndarray,
) -> None:
    _validate_assignment_tensor(
        data,
        batch_size=batch_size,
        draft_length=draft_length,
    )
    if data.step_histograms.shape[0] != target_mask.shape[0]:
        raise RuntimeError(
            "step_histograms must align with global barriers for "
            f"batch_size={batch_size}, draft_length={draft_length}."
        )
    expected_layer_rank_shape = (
        data.global_barrier_ids.shape[0],
        data.data_parallel_size,
        data.layers.shape[0],
    )
    for name in (
        "rank_layer_local_routed_tokens",
        "rank_layer_routed_expert_gpu_ms",
        "rank_layer_local_active_experts",
    ):
        array = np.asarray(getattr(data, name))
        if array.shape != expected_layer_rank_shape:
            raise RuntimeError(
                f"{name} has shape {array.shape}; expected "
                f"{expected_layer_rank_shape} for batch_size={batch_size}, "
                f"draft_length={draft_length}."
            )
    sorted_tokens = np.asarray(
        data.global_step_sorted_rank_local_routed_tokens,
        dtype=np.int64,
    )
    expected_sorted_shape = (
        data.global_barrier_ids.shape[0],
        data.data_parallel_size,
    )
    if sorted_tokens.shape != expected_sorted_shape:
        raise RuntimeError(
            "global_step_sorted_rank_local_routed_tokens has shape "
            f"{sorted_tokens.shape}; expected {expected_sorted_shape} for "
            f"batch_size={batch_size}, draft_length={draft_length}."
        )

    target_indices = np.flatnonzero(target_mask)
    np.testing.assert_array_equal(
        np.transpose(destination_assignments[target_indices], (0, 2, 1)),
        data.rank_layer_local_routed_tokens[target_indices],
        err_msg=(
            "v10 token-level destination assignments must equal "
            "rank_layer_local_routed_tokens for "
            f"batch_size={batch_size}, draft_length={draft_length}."
        ),
    )
    histogram_layer_totals = np.sum(
        data.step_histograms[target_indices],
        axis=2,
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        np.sum(destination_assignments[target_indices], axis=2, dtype=np.int64),
        histogram_layer_totals,
        err_msg=(
            "v10 destination assignments must equal step_histograms layer "
            f"totals for batch_size={batch_size}, draft_length={draft_length}."
        ),
    )
    recomputed_sorted = _ffn_sorted_destination_tokens_for_steps(
        data,
        target_indices,
        destination_assignments[target_indices],
    )
    np.testing.assert_array_equal(
        recomputed_sorted,
        sorted_tokens[target_indices],
        err_msg=(
            "Current sorted-rank token data must equal recomputed v10 "
            "destination tokens after FFN sort for "
            f"batch_size={batch_size}, draft_length={draft_length}."
        ),
    )


def build_routing_time_audit_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    condition_rows: list[dict[str, Any]] = []
    layer_rank_rows: list[dict[str, Any]] = []
    sorted_rank_rows: list[dict[str, Any]] = []

    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            if not np.any(target_mask):
                raise RuntimeError(
                    "No strict target barriers for routing-time audit: "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            (
                destination_assignments_by_barrier,
                self_assignments_by_barrier,
            ) = _condition_assignment_summaries(data)
            validate_routing_time_audit_invariants(
                data,
                batch_size=batch_size,
                draft_length=draft_length,
                target_mask=target_mask,
                destination_assignments=destination_assignments_by_barrier,
            )

            num_layers = int(data.layers.shape[0])
            num_ranks = int(data.data_parallel_size)
            num_steps = int(np.count_nonzero(target_mask))
            decode_step_scope = _decode_step_scope(draft_length)
            target_indices = np.flatnonzero(target_mask)
            target_destination = destination_assignments_by_barrier[
                target_indices
            ].astype(np.float64)
            target_self = self_assignments_by_barrier[target_indices].astype(
                np.float64
            )
            target_ffn = np.transpose(
                np.asarray(
                    data.rank_layer_routed_expert_gpu_ms[target_indices],
                    dtype=np.float64,
                ),
                (0, 2, 1),
            )
            target_active = np.transpose(
                np.asarray(
                    data.rank_layer_local_active_experts[target_indices],
                    dtype=np.float64,
                ),
                (0, 2, 1),
            )
            ffn_sum = np.sum(target_ffn, axis=0)
            destination_sum = np.sum(target_destination, axis=0)
            self_sum = np.sum(target_self, axis=0)
            active_sum = np.sum(target_active, axis=0)
            physical_ffn_sum = np.sum(target_ffn, axis=(0, 1))
            physical_destination_sum = np.sum(target_destination, axis=(0, 1))
            physical_self_sum = np.sum(target_self, axis=(0, 1))
            ffn_sorted_ffn_sum = np.zeros((num_ranks,), dtype=np.float64)
            ffn_sorted_destination_sum = np.zeros((num_ranks,), dtype=np.float64)
            ffn_sorted_self_sum = np.zeros((num_ranks,), dtype=np.float64)
            ffn_sorted_active_sum = np.zeros((num_ranks,), dtype=np.float64)
            token_sorted_destination_sum = np.zeros((num_ranks,), dtype=np.float64)
            token_sorted_ffn_sum = np.zeros((num_ranks,), dtype=np.float64)
            token_sorted_self_sum = np.zeros((num_ranks,), dtype=np.float64)
            token_sorted_active_sum = np.zeros((num_ranks,), dtype=np.float64)

            for layer_idx in range(num_layers):
                ffn_order = np.argsort(
                    -target_ffn[:, layer_idx, :],
                    axis=1,
                    kind="stable",
                )
                ffn_sorted_ffn_sum += np.sum(
                    np.take_along_axis(
                        target_ffn[:, layer_idx, :],
                        ffn_order,
                        axis=1,
                    ),
                    axis=0,
                )
                ffn_sorted_destination_sum += np.sum(
                    np.take_along_axis(
                        target_destination[:, layer_idx, :],
                        ffn_order,
                        axis=1,
                    ),
                    axis=0,
                )
                ffn_sorted_self_sum += np.sum(
                    np.take_along_axis(
                        target_self[:, layer_idx, :],
                        ffn_order,
                        axis=1,
                    ),
                    axis=0,
                )
                ffn_sorted_active_sum += np.sum(
                    np.take_along_axis(
                        target_active[:, layer_idx, :],
                        ffn_order,
                        axis=1,
                    ),
                    axis=0,
                )

                token_order = np.argsort(
                    -target_destination[:, layer_idx, :],
                    axis=1,
                    kind="stable",
                )
                token_sorted_destination_sum += np.sum(
                    np.take_along_axis(
                        target_destination[:, layer_idx, :],
                        token_order,
                        axis=1,
                    ),
                    axis=0,
                )
                token_sorted_ffn_sum += np.sum(
                    np.take_along_axis(
                        target_ffn[:, layer_idx, :],
                        token_order,
                        axis=1,
                    ),
                    axis=0,
                )
                token_sorted_self_sum += np.sum(
                    np.take_along_axis(
                        target_self[:, layer_idx, :],
                        token_order,
                        axis=1,
                    ),
                    axis=0,
                )
                token_sorted_active_sum += np.sum(
                    np.take_along_axis(
                        target_active[:, layer_idx, :],
                        token_order,
                        axis=1,
                    ),
                    axis=0,
                )

            avg_ffn = ffn_sum / num_steps
            avg_destination = destination_sum / num_steps
            avg_self = self_sum / num_steps
            avg_active = active_sum / num_steps
            avg_physical_ffn = physical_ffn_sum / num_steps
            avg_physical_destination = physical_destination_sum / num_steps
            avg_physical_self = physical_self_sum / num_steps
            ms_per_destination = np.divide(
                avg_physical_ffn,
                avg_physical_destination,
                out=np.full_like(avg_physical_ffn, np.nan, dtype=np.float64),
                where=avg_physical_destination > 0.0,
            )
            ms_per_self = np.divide(
                avg_physical_ffn,
                avg_physical_self,
                out=np.full_like(avg_physical_ffn, np.nan, dtype=np.float64),
                where=avg_physical_self > 0.0,
            )
            ffn_destination_pearson = _pearson_corr(
                avg_ffn,
                avg_destination,
            )
            ffn_destination_spearman = _spearman_corr(
                avg_ffn,
                avg_destination,
            )
            ffn_self_pearson = _pearson_corr(avg_ffn, avg_self)
            ffn_self_spearman = _spearman_corr(avg_ffn, avg_self)
            destination_ratio = _max_min_ratio(avg_physical_destination)
            ffn_ratio = _max_min_ratio(avg_physical_ffn)
            self_ratio = _max_min_ratio(avg_physical_self)
            ms_per_destination_ratio = _max_min_ratio(ms_per_destination)
            ms_per_self_ratio = _max_min_ratio(ms_per_self)
            hypothesis_supported = (
                np.isfinite(destination_ratio)
                and np.isfinite(ffn_ratio)
                and np.isfinite(ffn_destination_pearson)
                and destination_ratio >= 1.10
                and ffn_destination_pearson >= 0.70
                and (
                    not np.isfinite(ms_per_destination_ratio)
                    or ms_per_destination_ratio < 1.50
                )
            )
            if hypothesis_supported:
                interpretation = (
                    "v10_destination_tokens_correlate_with_ffn_time"
                )
            elif (
                np.isfinite(ffn_self_pearson)
                and (
                    not np.isfinite(ffn_destination_pearson)
                    or ffn_self_pearson > ffn_destination_pearson
                )
            ):
                interpretation = (
                    "v8_self_tokens_match_better_than_v10_destination_tokens"
                )
            else:
                interpretation = (
                    "v10_data_does_not_explain_ffn_time_by_destination_tokens"
                )

            for layer_row, layer in enumerate(data.layers):
                for rank in range(num_ranks):
                    destination_assignments = float(
                        avg_destination[layer_row, rank]
                    )
                    self_assignments = float(avg_self[layer_row, rank])
                    routed_expert_ms = float(avg_ffn[layer_row, rank])
                    layer_rank_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "decode_step_scope": decode_step_scope,
                            "layer": int(layer),
                            "physical_rank": rank,
                            "avg_routed_expert_gpu_ms": routed_expert_ms,
                            "avg_v10_destination_assignments": (
                                destination_assignments
                            ),
                            "avg_v8_emulated_self_assignments": (
                                self_assignments
                            ),
                            "avg_active_experts": float(
                                avg_active[layer_row, rank]
                            ),
                            "ms_per_destination_assignment": (
                                routed_expert_ms / destination_assignments
                                if destination_assignments > 0.0
                                else float("nan")
                            ),
                            "ms_per_v8_self_assignment": (
                                routed_expert_ms / self_assignments
                                if self_assignments > 0.0
                                else float("nan")
                            ),
                            "num_global_barriers": num_steps,
                        }
                    )

            condition_rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "decode_step_scope": decode_step_scope,
                    "num_global_barriers": num_steps,
                    "num_layers": num_layers,
                    "data_parallel_size": num_ranks,
                    "ffn_max_min_ratio": ffn_ratio,
                    "v10_destination_token_max_min_ratio": destination_ratio,
                    "v8_emulated_self_token_max_min_ratio": self_ratio,
                    "pearson_ffn_v10_destination_tokens": (
                        ffn_destination_pearson
                    ),
                    "spearman_ffn_v10_destination_tokens": (
                        ffn_destination_spearman
                    ),
                    "pearson_ffn_v8_emulated_self_tokens": ffn_self_pearson,
                    "spearman_ffn_v8_emulated_self_tokens": ffn_self_spearman,
                    "physical_rank_ms_per_destination_assignment_max_min_ratio": (
                        ms_per_destination_ratio
                    ),
                    "physical_rank_ms_per_v8_self_assignment_max_min_ratio": (
                        ms_per_self_ratio
                    ),
                    "hypothesis_supported": bool(hypothesis_supported),
                    "interpretation": interpretation,
                }
            )

            for position in range(num_ranks):
                sorted_rank_rows.append(
                    {
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "decode_step_scope": decode_step_scope,
                        "sorted_rank_position": position,
                        "avg_ffn_sorted_routed_expert_gpu_ms": float(
                            ffn_sorted_ffn_sum[position] / num_steps
                        ),
                        "avg_ffn_sorted_v10_destination_tokens": float(
                            ffn_sorted_destination_sum[position] / num_steps
                        ),
                        "avg_ffn_sorted_v8_emulated_self_tokens": float(
                            ffn_sorted_self_sum[position] / num_steps
                        ),
                        "avg_ffn_sorted_active_experts": float(
                            ffn_sorted_active_sum[position] / num_steps
                        ),
                        "avg_token_sorted_v10_destination_tokens": float(
                            token_sorted_destination_sum[position] / num_steps
                        ),
                        "avg_token_sorted_routed_expert_gpu_ms": float(
                            token_sorted_ffn_sum[position] / num_steps
                        ),
                        "avg_token_sorted_v8_emulated_self_tokens": float(
                            token_sorted_self_sum[position] / num_steps
                        ),
                        "avg_token_sorted_active_experts": float(
                            token_sorted_active_sum[position] / num_steps
                        ),
                        "num_global_barriers": num_steps,
                    }
                )

    return condition_rows, layer_rank_rows, sorted_rank_rows


def validate_barrier_shapes(
    data: LoadedConditionData,
    *,
    batch_size: int,
    draft_length: int,
) -> None:
    num_barriers = data.global_barrier_ids.shape[0]
    num_ranks = data.data_parallel_size
    num_layers = data.layers.shape[0]
    expected_rank_shape = (num_barriers, num_ranks)
    expected_layer_shape = (num_barriers, num_ranks, num_layers)
    rank_names = (
        "rank_step_kinds",
        "rank_execute_wall_ms",
        "rank_verification_wall_ms",
        "rank_draft_wall_ms",
        "rank_iteration_wall_ms",
        "rank_execute_gpu_ms",
        "rank_verification_gpu_ms",
        "rank_draft_gpu_ms",
        "rank_iteration_gpu_ms",
        "rank_attention_gpu_ms",
        "rank_moe_gpu_ms",
        "rank_gpu_other_ms",
        "rank_timing_complete",
        "rank_step_total_ms",
        "rank_step_draft_ms",
    )
    layer_names = (
        "rank_layer_ffn_ms",
        "rank_layer_moe_gpu_ms",
        "rank_layer_routed_expert_gpu_ms",
        "rank_layer_shared_expert_gpu_ms",
        "rank_layer_routing_gpu_ms",
        "rank_layer_prepare_gpu_ms",
        "rank_layer_finalize_gpu_ms",
        "rank_layer_local_routed_tokens",
        "rank_layer_local_active_experts",
    )
    for name, expected_shape in (
        *((name, expected_rank_shape) for name in rank_names),
        *((name, expected_layer_shape) for name in layer_names),
    ):
        if not hasattr(data, name):
            continue
        array = getattr(data, name)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {expected_shape} "
                f"for batch_size={batch_size}, draft_length={draft_length}."
            )
    np.testing.assert_array_equal(
        data.global_barrier_ids,
        np.arange(num_barriers, dtype=np.int64),
    )


def build_barrier_rank_layer_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            validate_barrier_shapes(
                data, batch_size=batch_size, draft_length=draft_length
            )
            for barrier_row, global_barrier_id in enumerate(data.global_barrier_ids):
                for rank in range(data.data_parallel_size):
                    for layer_row, layer in enumerate(data.layers):
                        rows.append(
                            {
                                "batch_size": batch_size,
                                "draft_length": draft_length,
                                "global_barrier_id": int(global_barrier_id),
                                "first_ep_collective_seq_id": int(
                                    data.barrier_first_ep_collective_seq_ids[
                                        barrier_row
                                    ]
                                ),
                                "last_ep_collective_seq_id": int(
                                    data.barrier_last_ep_collective_seq_ids[
                                        barrier_row
                                    ]
                                ),
                                "num_ep_collectives": int(
                                    data.barrier_num_ep_collectives[barrier_row]
                                ),
                                "global_step_kind": str(
                                    data.global_step_kinds[barrier_row]
                                ),
                                "rank": rank,
                                "rank_step_kind": str(
                                    data.rank_step_kinds[barrier_row, rank]
                                ),
                                "rank_step_total_ms": float(
                                    data.rank_step_total_ms[barrier_row, rank]
                                ),
                                "execute_wall_ms": float(
                                    data.rank_execute_wall_ms[barrier_row, rank]
                                ),
                                "verification_wall_ms": float(
                                    data.rank_verification_wall_ms[
                                        barrier_row, rank
                                    ]
                                ),
                                "draft_wall_ms": float(
                                    data.rank_draft_wall_ms[barrier_row, rank]
                                ),
                                "iteration_wall_ms": float(
                                    data.rank_iteration_wall_ms[barrier_row, rank]
                                ),
                                "execute_gpu_ms": float(
                                    data.rank_execute_gpu_ms[barrier_row, rank]
                                ),
                                "verification_gpu_ms": float(
                                    data.rank_verification_gpu_ms[
                                        barrier_row, rank
                                    ]
                                ),
                                "draft_gpu_ms": float(
                                    data.rank_draft_gpu_ms[barrier_row, rank]
                                ),
                                "iteration_gpu_ms": float(
                                    data.rank_iteration_gpu_ms[barrier_row, rank]
                                ),
                                "attention_gpu_ms": float(
                                    data.rank_attention_gpu_ms[barrier_row, rank]
                                ),
                                "moe_gpu_ms": float(
                                    data.rank_moe_gpu_ms[barrier_row, rank]
                                ),
                                "gpu_other_ms": float(
                                    data.rank_gpu_other_ms[barrier_row, rank]
                                ),
                                "timing_complete": bool(
                                    data.rank_timing_complete[barrier_row, rank]
                                ),
                                "rank_step_draft_ms": float(
                                    data.rank_step_draft_ms[barrier_row, rank]
                                ),
                                "layer": int(layer),
                                "rank_layer_ffn_ms": float(
                                    data.rank_layer_ffn_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "layer_moe_gpu_ms": float(
                                    data.rank_layer_moe_gpu_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "layer_routed_expert_gpu_ms": float(
                                    data.rank_layer_routed_expert_gpu_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "layer_shared_expert_gpu_ms": float(
                                    data.rank_layer_shared_expert_gpu_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "layer_routing_gpu_ms": float(
                                    data.rank_layer_routing_gpu_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "layer_prepare_gpu_ms": float(
                                    data.rank_layer_prepare_gpu_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "layer_finalize_gpu_ms": float(
                                    data.rank_layer_finalize_gpu_ms[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "rank_layer_local_routed_tokens": int(
                                    data.rank_layer_local_routed_tokens[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                                "rank_layer_local_active_experts": int(
                                    data.rank_layer_local_active_experts[
                                        barrier_row, rank, layer_row
                                    ]
                                ),
                            }
                        )
    return rows


def build_timing_completeness_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            for barrier_row, barrier_id in enumerate(data.global_barrier_ids):
                for rank in range(data.data_parallel_size):
                    decomposition_error_ms = abs(
                        float(data.rank_attention_gpu_ms[barrier_row, rank])
                        + float(data.rank_moe_gpu_ms[barrier_row, rank])
                        + float(data.rank_gpu_other_ms[barrier_row, rank])
                        - float(data.rank_verification_gpu_ms[barrier_row, rank])
                    )
                    rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "global_barrier_id": int(barrier_id),
                            "rank": rank,
                            "strict_target_barrier": bool(
                                target_mask[barrier_row]
                            ),
                            "timing_complete": bool(
                                data.rank_timing_complete[barrier_row, rank]
                            ),
                            "missing_moe_layers": int(
                                np.count_nonzero(
                                    data.rank_layer_moe_gpu_ms[
                                        barrier_row, rank
                                    ]
                                    <= 0.0
                                )
                            ),
                            "missing_routed_expert_layers": int(
                                np.count_nonzero(
                                    data.rank_layer_routed_expert_gpu_ms[
                                        barrier_row, rank
                                    ]
                                    <= 0.0
                                )
                            ),
                            "decomposition_error_ms": decomposition_error_ms,
                        }
                    )
    return rows


def strict_target_barrier_mask(
    data: LoadedConditionData,
    *,
    draft_length: int,
) -> np.ndarray:
    target_kind = "verification_only" if draft_length > 0 else "decode_only"
    num_barriers = int(data.global_barrier_ids.shape[0])
    if data.global_step_kinds.shape != (num_barriers,):
        raise ValueError(
            "global_step_kinds must align with global barriers: "
            f"{data.global_step_kinds.shape} vs {(num_barriers,)}."
        )
    if data.rank_step_kinds.shape != (
        num_barriers,
        data.data_parallel_size,
    ):
        raise ValueError(
            "rank_step_kinds must align with global barriers and ranks: "
            f"{data.rank_step_kinds.shape} vs "
            f"{(num_barriers, data.data_parallel_size)}."
        )
    return (data.global_step_kinds == target_kind) & np.all(
        data.rank_step_kinds == target_kind,
        axis=1,
    )


def strict_average_histograms(
    data: LoadedConditionData,
    *,
    draft_length: int,
) -> np.ndarray:
    target_mask = strict_target_barrier_mask(
        data,
        draft_length=draft_length,
    )
    if data.step_histograms.shape[0] != target_mask.shape[0]:
        raise ValueError(
            "step_histograms must align with global barriers: "
            f"{data.step_histograms.shape[0]} vs {target_mask.shape[0]}."
        )
    if not np.any(target_mask):
        raise ValueError(
            f"No strict target barriers for draft_length={draft_length}."
        )
    return np.mean(data.step_histograms[target_mask], axis=0)


def build_position_breakdown_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            if draft_length <= 0:
                continue
            data = results[(batch_size, draft_length)]
            if data.schema_version < 9:
                raise RuntimeError(
                    "Position breakdown requires schema v9 complete "
                    "destination-rank routing. Re-run collect."
                )
            ffn = np.asarray(
                data.global_step_position_sorted_rank_ffn_ms,
                dtype=np.float64,
            )
            assignments = np.asarray(
                data.global_step_position_sorted_rank_local_routed_tokens,
                dtype=np.float64,
            )
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            position_count = draft_length + 1
            if (
                ffn.ndim != 3
                or ffn.shape[1] < position_count
                or assignments.shape != ffn.shape
            ):
                raise RuntimeError(
                    "Position arrays must be matching rank-3 arrays with at "
                    f"least {position_count} positions for batch_size="
                    f"{batch_size}, draft_length={draft_length}; got "
                    f"{ffn.shape} and {assignments.shape}."
                )
            if not np.any(target_mask):
                raise RuntimeError(
                    "No strict global verification-only barriers for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            ffn = ffn[target_mask, :position_count]
            assignments = assignments[target_mask, :position_count]
            avg_ffn = ffn.mean(axis=0)
            avg_assignments = assignments.mean(axis=0)
            for position in range(position_count):
                for sorted_rank in range(avg_ffn.shape[1]):
                    rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "verification_position": position,
                            "sorted_rank_position": sorted_rank,
                            "num_global_steps": int(ffn.shape[0]),
                            "avg_attributed_ffn_ms": float(
                                avg_ffn[position, sorted_rank]
                            ),
                            "avg_attributed_routed_expert_gpu_ms": float(
                                avg_ffn[position, sorted_rank]
                            ),
                            "avg_destination_routed_assignments": float(
                                avg_assignments[position, sorted_rank]
                            ),
                        }
                    )
    return rows


def _build_position_metric_matrix(
    selected: list[dict[str, Any]],
    verification_positions: list[int],
    rank_positions: list[int],
    metric_key: str,
) -> np.ndarray:
    values = np.zeros((len(verification_positions), len(rank_positions)))
    rank_indices = {
        rank_position: index
        for index, rank_position in enumerate(rank_positions)
    }
    for row in selected:
        position = int(row["verification_position"])
        rank_index = rank_indices[int(row["sorted_rank_position"])]
        values[position, rank_index] = float(row[metric_key])
    return values


def plot_position_breakdown(
    plot_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    plt = import_plot_module()
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    draft_lengths = sorted({int(row["draft_length"]) for row in rows})
    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
    ]
    plot_paths: list[Path] = []
    for draft_length in draft_lengths:
        fig, axes = plt.subplots(
            2,
            len(batch_sizes),
            figsize=(7.0 * len(batch_sizes), 8.6),
            squeeze=False,
            sharex="col",
            sharey="row",
        )
        for column, batch_size in enumerate(batch_sizes):
            ffn_axis = axes[0][column]
            assignment_axis = axes[1][column]
            selected = [
                row
                for row in rows
                if int(row["batch_size"]) == batch_size
                and int(row["draft_length"]) == draft_length
            ]
            rank_positions = sorted(
                {int(row["sorted_rank_position"]) for row in selected}
            )
            verification_positions = list(range(draft_length + 1))
            x = np.asarray(rank_positions, dtype=np.int64)
            ffn_values = _build_position_metric_matrix(
                selected,
                verification_positions,
                rank_positions,
                "avg_attributed_ffn_ms",
            )
            assignment_values = _build_position_metric_matrix(
                selected,
                verification_positions,
                rank_positions,
                "avg_destination_routed_assignments",
            )
            labels = [
                f"pos {position}" for position in verification_positions
            ]
            for axis, values in (
                (ffn_axis, ffn_values),
                (assignment_axis, assignment_values),
            ):
                axis.stackplot(
                    x,
                    values,
                    labels=labels,
                    colors=colors[: len(verification_positions)],
                    alpha=0.85,
                )
                axis.plot(
                    x,
                    values.sum(axis=0),
                    color="black",
                    linewidth=1.5,
                    marker="o",
                    label="total",
                )
                axis.grid(True, alpha=0.25)
                axis.set_xticks(rank_positions)
            ffn_axis.set_title(f"batch_size={batch_size}")
            assignment_axis.set_xlabel(
                "sorted destination rank position (0 = heaviest per layer)"
            )
        axes[0][0].set_ylabel(
            "avg attributed routed-expert/FFN GPU time "
            "after per-layer sort (ms)"
        )
        axes[1][0].set_ylabel(
            "avg destination-rank routed assignments after per-layer sort"
        )
        handles, labels = axes[0][-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(
            f"draft_length={draft_length} verification-position routed-expert/"
            "FFN GPU time and destination-rank assignment breakdown"
        )
        fig.tight_layout(rect=(0, 0, 0.9, 0.92))
        plot_path = plot_dir / f"draft_{draft_length:02d}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)
    return plot_paths


def build_draft_drop_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cutoff_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            if draft_length <= 0:
                continue
            data = results[(batch_size, draft_length)]
            if data.schema_version < 9:
                raise RuntimeError(
                    "Draft-token drop analysis requires raw schema v9 with "
                    "token identities and complete destination-rank routing. "
                    f"{batch_size=}, {draft_length=} uses schema "
                    f"v{data.schema_version}; re-run collect."
                )
            assignments = np.asarray(
                data.global_token_layer_destination_assignment_counts,
                dtype=np.int64,
            )
            offsets = np.asarray(
                data.global_token_barrier_offsets,
                dtype=np.int64,
            )
            expected_shape = (
                data.global_token_position_ids.shape[0],
                data.layers.shape[0],
                data.data_parallel_size,
            )
            if assignments.shape != expected_shape:
                raise RuntimeError(
                    "Token assignment tensor has shape "
                    f"{assignments.shape}; expected {expected_shape} for "
                    f"{batch_size=}, {draft_length=}."
                )
            if offsets.shape != (data.global_barrier_ids.shape[0] + 1,):
                raise RuntimeError(
                    "global_token_barrier_offsets must contain one boundary "
                    "per global barrier."
                )

            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            condition_step_ratios: list[float] = []
            condition_dropped = 0
            condition_draft = 0
            for barrier_idx in np.flatnonzero(target_mask):
                token_start = int(offsets[barrier_idx])
                token_end = int(offsets[barrier_idx + 1])
                positions = np.asarray(
                    data.global_token_position_ids[token_start:token_end],
                    dtype=np.int64,
                )
                source_ranks = np.asarray(
                    data.global_token_source_ranks[token_start:token_end],
                    dtype=np.int64,
                )
                request_ids = np.asarray(
                    data.global_token_request_ids[token_start:token_end],
                    dtype=np.str_,
                )
                barrier_assignments = assignments[token_start:token_end]
                num_draft_tokens = int(np.count_nonzero(positions > 0))
                if num_draft_tokens == 0:
                    continue

                max_position = int(np.max(positions, initial=0))
                position_load = np.zeros(
                    (
                        max_position + 1,
                        data.layers.shape[0],
                        data.data_parallel_size,
                    ),
                    dtype=np.int64,
                )
                for position in range(max_position + 1):
                    position_load[position] = np.sum(
                        barrier_assignments[positions == position],
                        axis=0,
                        dtype=np.int64,
                    )
                layer_rank_load = np.sum(
                    barrier_assignments,
                    axis=0,
                    dtype=np.int64,
                )
                baselines = np.min(layer_rank_load, axis=1)
                cutoffs = np.zeros_like(layer_rank_load, dtype=np.int64)
                for layer_row, layer in enumerate(data.layers):
                    for destination_rank in range(data.data_parallel_size):
                        cumulative = np.cumsum(
                            position_load[:, layer_row, destination_rank]
                        )
                        reached = np.flatnonzero(
                            cumulative >= baselines[layer_row]
                        )
                        cutoff = int(reached[0]) if reached.size else max_position
                        cutoffs[layer_row, destination_rank] = cutoff
                        cutoff_rows.append(
                            {
                                "batch_size": batch_size,
                                "draft_length": draft_length,
                                "global_barrier_id": int(
                                    data.global_barrier_ids[barrier_idx]
                                ),
                                "layer": int(layer),
                                "destination_rank": destination_rank,
                                "baseline_assignments": int(
                                    baselines[layer_row]
                                ),
                                "total_assignments": int(
                                    layer_rank_load[
                                        layer_row, destination_rank
                                    ]
                                ),
                                "cutoff_position": cutoff,
                                "cutoff_position_assignments": int(
                                    position_load[
                                        cutoff, layer_row, destination_rank
                                    ]
                                ),
                                "assignments_through_cutoff": int(
                                    cumulative[cutoff]
                                ),
                                "assignments_after_cutoff": int(
                                    layer_rank_load[
                                        layer_row, destination_rank
                                    ]
                                    - cumulative[cutoff]
                                ),
                            }
                        )

                token_layer_dropped = np.zeros(
                    (positions.shape[0], data.layers.shape[0]),
                    dtype=np.bool_,
                )
                for layer_row, layer in enumerate(data.layers):
                    for destination_rank in range(data.data_parallel_size):
                        token_layer_dropped[:, layer_row] |= (
                            (positions > cutoffs[layer_row, destination_rank])
                            & (positions > 0)
                            & (
                                barrier_assignments[
                                    :, layer_row, destination_rank
                                ]
                                > 0
                            )
                        )
                    layer_dropped = int(
                        np.count_nonzero(token_layer_dropped[:, layer_row])
                    )
                    layer_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "global_barrier_id": int(
                                data.global_barrier_ids[barrier_idx]
                            ),
                            "layer": int(layer),
                            "scheduled_draft_tokens": num_draft_tokens,
                            "unique_dropped_draft_tokens": layer_dropped,
                            "drop_ratio": layer_dropped / num_draft_tokens,
                        }
                    )

                directly_dropped = np.any(token_layer_dropped, axis=1)
                suffix_dropped = np.zeros_like(directly_dropped)
                sequence_keys = sorted(
                    {
                        (int(source_rank), str(request_id))
                        for source_rank, request_id in zip(
                            source_ranks,
                            request_ids,
                        )
                    }
                )
                for source_rank, request_id in sequence_keys:
                    sequence_mask = (
                        (source_ranks == source_rank)
                        & (request_ids == request_id)
                        & (positions > 0)
                    )
                    triggering_positions = positions[
                        sequence_mask & directly_dropped
                    ]
                    if triggering_positions.size:
                        first_drop = int(np.min(triggering_positions))
                        suffix_dropped |= sequence_mask & (
                            positions >= first_drop
                        )
                dropped_count = int(np.count_nonzero(suffix_dropped))
                drop_ratio = dropped_count / num_draft_tokens
                step_rows.append(
                    {
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "global_barrier_id": int(
                            data.global_barrier_ids[barrier_idx]
                        ),
                        "scheduled_draft_tokens": num_draft_tokens,
                        "directly_dropped_unique_draft_tokens": int(
                            np.count_nonzero(directly_dropped)
                        ),
                        "global_suffix_dropped_draft_tokens": dropped_count,
                        "global_suffix_drop_ratio": drop_ratio,
                    }
                )
                condition_step_ratios.append(drop_ratio)
                condition_dropped += dropped_count
                condition_draft += num_draft_tokens

            if not condition_step_ratios:
                raise RuntimeError(
                    "No strict verification-only barriers with draft tokens for "
                    f"{batch_size=}, {draft_length=}."
                )
            condition_rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "num_verification_steps": len(condition_step_ratios),
                    "mean_step_drop_ratio": float(
                        np.mean(condition_step_ratios)
                    ),
                    "weighted_drop_ratio": (
                        condition_dropped / condition_draft
                    ),
                    "total_dropped_draft_tokens": condition_dropped,
                    "total_scheduled_draft_tokens": condition_draft,
                }
            )
    return cutoff_rows, layer_rows, step_rows, condition_rows


def build_sorted_rank_ffn_time_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    sorted_rank_rows: list[dict[str, Any]] = []
    imbalance_rows: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        for draft_length in draft_lengths:
            data = results[(batch_size, draft_length)]
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            sorted_ffn_ms = np.asarray(
                data.global_step_sorted_rank_ffn_ms, dtype=np.float64
            )
            if sorted_ffn_ms.ndim != 2:
                raise ValueError(
                    "global_step_sorted_rank_ffn_ms must be 2-D for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            if sorted_ffn_ms.shape[1] != data.data_parallel_size:
                raise ValueError(
                    "global_step_sorted_rank_ffn_ms rank dimension must match "
                    f"data_parallel_size for batch_size={batch_size}, "
                    f"draft_length={draft_length}: {sorted_ffn_ms.shape[1]} vs "
                    f"{data.data_parallel_size}."
                )
            if sorted_ffn_ms.shape[0] != data.num_global_captured_steps:
                raise ValueError(
                    "global_step_sorted_rank_ffn_ms step dimension must match "
                    f"num_global_captured_steps for batch_size={batch_size}, "
                    f"draft_length={draft_length}: {sorted_ffn_ms.shape[0]} vs "
                    f"{data.num_global_captured_steps}."
                )
            if sorted_ffn_ms.shape[0] == 0:
                raise ValueError(
                    "No globally captured rank-local FFN times for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            if not np.any(target_mask):
                raise ValueError(
                    "No strict target barriers for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )

            ratios = np.asarray(
                data.global_step_ffn_max_mean_ratio, dtype=np.float64
            )
            if ratios.shape != (sorted_ffn_ms.shape[0],):
                raise ValueError(
                    "global_step_ffn_max_mean_ratio must have one value per "
                    f"captured step for batch_size={batch_size}, "
                    f"draft_length={draft_length}: {ratios.shape} vs "
                    f"{(sorted_ffn_ms.shape[0],)}."
                )

            sorted_ffn_ms = sorted_ffn_ms[target_mask]
            ratios = ratios[target_mask]
            num_target_barriers = int(np.count_nonzero(target_mask))
            avg_sorted_ffn_ms = np.mean(sorted_ffn_ms, axis=0)
            heaviest_ffn_ms = float(avg_sorted_ffn_ms[0])
            lightest_ffn_ms = float(avg_sorted_ffn_ms[-1])
            decode_step_scope = (
                "verification_only" if draft_length > 0 else "decode_only"
            )

            for sorted_rank_position, avg_ffn_ms in enumerate(avg_sorted_ffn_ms):
                sorted_rank_rows.append(
                    {
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "decode_step_scope": decode_step_scope,
                        "sorted_rank_position": sorted_rank_position,
                        "avg_local_ffn_ms": float(avg_ffn_ms),
                        "avg_layer_sorted_routed_expert_gpu_ms": float(
                            avg_ffn_ms
                        ),
                        "num_global_captured_steps": num_target_barriers,
                    }
                )

            imbalance_rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "decode_step_scope": decode_step_scope,
                    "num_global_captured_steps": num_target_barriers,
                    "avg_heaviest_local_ffn_ms": heaviest_ffn_ms,
                    "avg_lightest_local_ffn_ms": lightest_ffn_ms,
                    "avg_heaviest_routed_expert_gpu_ms": heaviest_ffn_ms,
                    "avg_lightest_routed_expert_gpu_ms": lightest_ffn_ms,
                    "avg_heaviest_minus_lightest_local_ffn_ms": (
                        heaviest_ffn_ms - lightest_ffn_ms
                    ),
                    "avg_heaviest_over_lightest_local_ffn_ratio": (
                        heaviest_ffn_ms / lightest_ffn_ms
                        if lightest_ffn_ms > 0
                        else float("nan")
                    ),
                    "avg_step_ffn_max_mean_ratio": _finite_mean(ratios),
                    "p50_step_ffn_max_mean_ratio": _finite_percentile(
                        ratios, 50.0
                    ),
                    "p95_step_ffn_max_mean_ratio": _finite_percentile(
                        ratios, 95.0
                    ),
                }
            )

    return sorted_rank_rows, imbalance_rows


def build_sorted_rank_moe_time_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorted_rows: list[dict[str, Any]] = []
    imbalance_rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            values = data.global_step_sorted_rank_moe_gpu_ms[target_mask]
            ratios = data.global_step_moe_max_mean_ratio[target_mask]
            if values.ndim != 2 or values.shape[0] == 0:
                raise ValueError(
                    "No strict sorted-rank MoE CUDA Event data for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            averages = np.mean(values, axis=0)
            for position, value in enumerate(averages):
                sorted_rows.append(
                    {
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "sorted_rank_position": position,
                        "avg_layer_sorted_moe_gpu_ms": float(value),
                        "num_global_barriers": int(values.shape[0]),
                    }
                )
            imbalance_rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "num_global_barriers": int(values.shape[0]),
                    "avg_heaviest_moe_gpu_ms": float(averages[0]),
                    "avg_lightest_moe_gpu_ms": float(averages[-1]),
                    "avg_step_moe_max_mean_ratio": _finite_mean(ratios),
                    "p95_step_moe_max_mean_ratio": _finite_percentile(
                        ratios,
                        95.0,
                    ),
                }
            )
    return sorted_rows, imbalance_rows


def build_sorted_rank_summary_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            validate_barrier_shapes(
                data, batch_size=batch_size, draft_length=draft_length
            )
            if data.global_step_sorted_rank_ffn_ms.shape[0] == 0:
                raise ValueError(
                    "No sorted-rank barrier data for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            if not np.any(target_mask):
                raise ValueError(
                    "No strict target barriers for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            for position in range(data.data_parallel_size):
                rows.append(
                    {
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "sorted_rank_position": position,
                        "avg_ffn_ms": float(
                            np.mean(
                                data.global_step_sorted_rank_ffn_ms[
                                    target_mask, position
                                ]
                            )
                        ),
                        "avg_routed_expert_gpu_ms": float(
                            np.mean(
                                getattr(
                                    data,
                                    "global_step_sorted_rank_routed_expert_gpu_ms",
                                    data.global_step_sorted_rank_ffn_ms,
                                )[
                                    target_mask, position
                                ]
                            )
                        ),
                        "avg_moe_gpu_ms": float(
                            np.mean(
                                getattr(
                                    data,
                                    "global_step_sorted_rank_moe_gpu_ms",
                                    data.global_step_sorted_rank_ffn_ms,
                                )[
                                    target_mask, position
                                ]
                            )
                        ),
                        "avg_local_routed_tokens": float(
                            np.mean(
                                data.global_step_sorted_rank_local_routed_tokens[
                                    target_mask, position
                                ]
                            )
                        ),
                        "avg_local_active_experts": float(
                            np.mean(
                                data.global_step_sorted_rank_local_active_experts[
                                    target_mask, position
                                ]
                            )
                        ),
                        "num_global_barriers": int(np.count_nonzero(target_mask)),
                    }
                )
    return rows


def build_active_expert_ratio_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, Any]]:
    num_experts = int(manifest.get("num_experts", 256))
    rows: list[dict[str, Any]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            target_mask = strict_target_barrier_mask(
                data,
                draft_length=draft_length,
            )
            if not np.any(target_mask):
                raise ValueError(
                    "No strict target barriers for "
                    f"batch_size={batch_size}, draft_length={draft_length}."
                )
            denominator = data.layers.shape[0] * num_experts
            per_barrier_ratio = (
                np.sum(
                    data.rank_layer_local_active_experts[target_mask],
                    axis=(1, 2),
                )
                / denominator
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "num_layers": int(data.layers.shape[0]),
                    "num_experts": num_experts,
                    "num_global_barriers": int(np.count_nonzero(target_mask)),
                    "active_expert_ratio": float(np.mean(per_barrier_ratio)),
                }
            )
    return rows


def build_acceptance_metric_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size in tuple(manifest["batch_sizes"]):
        for draft_length in tuple(manifest["draft_lengths"]):
            data = results[(batch_size, draft_length)]
            elapsed_s = data.vllm_generation_elapsed_ms / 1000.0
            rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "spec_num_drafts": data.spec_num_drafts,
                    "spec_num_draft_tokens": data.spec_num_draft_tokens,
                    "spec_num_accepted_tokens": data.spec_num_accepted_tokens,
                    "acceptance_rate": data.spec_acceptance_rate,
                    "mean_acceptance_length": data.spec_mean_acceptance_length,
                    "drafted_throughput_tok_s": (
                        data.spec_num_draft_tokens / elapsed_s
                        if elapsed_s > 0
                        else 0.0
                    ),
                    "accepted_throughput_tok_s": (
                        data.spec_num_accepted_tokens / elapsed_s
                        if elapsed_s > 0
                        else 0.0
                    ),
                }
            )
    return rows


def plot_speedup_vs_draft_length(
    plot_dir: Path,
    speedup_rows: list[dict[str, float | int]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size in batch_sizes:
        y = [
            next(
                row["vllm_request_tpot_speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for draft_length in draft_lengths
        ]
        ax.plot(draft_lengths, y, marker="o", linewidth=2, label=f"bs={batch_size}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("draft_length")
    ax.set_ylabel("vLLM request TPOT speedup vs draft_length=0")
    ax.set_title("vLLM Request TPOT Speedup vs Draft Length")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "vllm_request_tpot_speedup_vs_draft_length.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_speedup_vs_batch_size(
    plot_dir: Path,
    speedup_rows: list[dict[str, float | int]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for draft_length in draft_lengths:
        y = [
            next(
                row["vllm_request_tpot_speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for batch_size in batch_sizes
        ]
        ax.plot(batch_sizes, y, marker="o", linewidth=2, label=f"d={draft_length}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("vLLM request TPOT speedup vs draft_length=0")
    ax.set_title("vLLM Request TPOT Speedup vs Global Batch Size")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "vllm_request_tpot_speedup_vs_batch_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_decode_throughput_speedup_vs_batch_size(
    plot_dir: Path,
    speedup_rows: list[dict[str, float | int]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for draft_length in draft_lengths:
        y = [
            next(
                row["vllm_generation_throughput_speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for batch_size in batch_sizes
        ]
        ax.plot(batch_sizes, y, marker="o", linewidth=2, label=f"d={draft_length}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("vLLM generation throughput speedup vs draft_length=0")
    ax.set_title("vLLM Generation Throughput Speedup vs Global Batch Size")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "vllm_generation_throughput_speedup_vs_batch_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_acceptance_rate_vs_draft_length(
    plot_dir: Path,
    acceptance_rows: list[dict[str, float | int]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_lengths = tuple(
        draft_length for draft_length in draft_lengths if draft_length > 0
    )
    for batch_size in batch_sizes:
        y = [
            next(
                row["acceptance_rate"]
                for row in acceptance_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for draft_length in plotted_lengths
        ]
        ax.plot(
            plotted_lengths,
            y,
            marker="o",
            linewidth=2,
            label=f"bs={batch_size}",
        )
    ax.set_xlabel("draft_length")
    ax.set_ylabel("accepted draft tokens / drafted tokens")
    ax.set_title("Spec Decode Acceptance Rate vs Draft Length")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "acceptance_rate_vs_draft_length.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_layer_drop_ratio(
    plot_dir: Path,
    layer_rows: list[dict[str, Any]],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    conditions = sorted(
        {
            (int(row["batch_size"]), int(row["draft_length"]))
            for row in layer_rows
        }
    )
    for batch_size, draft_length in conditions:
        selected = [
            row
            for row in layer_rows
            if int(row["batch_size"]) == batch_size
            and int(row["draft_length"]) == draft_length
        ]
        layers = sorted({int(row["layer"]) for row in selected})
        values = [
            float(
                np.mean(
                    [
                        row["drop_ratio"]
                        for row in selected
                        if int(row["layer"]) == layer
                    ]
                )
            )
            for layer in layers
        ]
        ax.plot(
            layers,
            values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"bs={batch_size}, d={draft_length}",
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("mean unique draft-token drop ratio")
    ax.set_title("Layer-local routing-oracle draft-token drop ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = plot_dir / "layer_mean_drop_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_step_drop_distribution(
    plot_dir: Path,
    step_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> Path:
    plt = import_plot_module()
    conditions = sorted(
        {
            (int(row["batch_size"]), int(row["draft_length"]))
            for row in step_rows
        }
    )
    values = [
        [
            float(row["global_suffix_drop_ratio"])
            for row in step_rows
            if int(row["batch_size"]) == batch_size
            and int(row["draft_length"]) == draft_length
        ]
        for batch_size, draft_length in conditions
    ]
    labels = [
        f"bs={batch_size}\nd={draft_length}"
        for batch_size, draft_length in conditions
    ]
    means = [
        next(
            float(row["mean_step_drop_ratio"])
            for row in condition_rows
            if int(row["batch_size"]) == batch_size
            and int(row["draft_length"]) == draft_length
        )
        for batch_size, draft_length in conditions
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.scatter(
        np.arange(1, len(conditions) + 1),
        means,
        color="#E45756",
        marker="D",
        label="arithmetic mean",
        zorder=3,
    )
    ax.set_ylabel("global suffix draft-token drop ratio per step")
    ax.set_title("Per-step routing-oracle draft-token drop distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "step_drop_ratio_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_step_time_breakdown(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    step_rows: list[dict[str, float | int]],
) -> Path:
    plt = import_plot_module()
    rows = [
        next(
            row
            for row in step_rows
            if row["batch_size"] == batch_size and row["draft_length"] == draft_length
        )
        for draft_length in draft_lengths
    ]
    x = np.arange(len(draft_lengths))
    wall = np.asarray(
        [row["normalized_verification_wall"] for row in rows],
        dtype=np.float64,
    )
    other = np.asarray(
        [row["normalized_gpu_other"] for row in rows], dtype=np.float64
    )
    attention = np.asarray(
        [row["normalized_attention_gpu"] for row in rows], dtype=np.float64
    )
    moe = np.asarray(
        [row["normalized_moe_gpu"] for row in rows], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.36
    ax.bar(x - width / 2, wall, width=width, label="verification wall")
    ax.bar(x + width / 2, other, width=width, label="GPU Other")
    ax.bar(
        x + width / 2,
        attention,
        width=width,
        bottom=other,
        label="GPU Attention",
        color="#59a14f",
    )
    ax.bar(
        x + width / 2,
        moe,
        width=width,
        bottom=other + attention,
        label="GPU MoE",
        color="#e15759",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in draft_lengths])
    ax.set_xlabel("draft_length")
    ax.set_ylabel("normalized time (wall and GPU shown separately)")
    ax.set_title(
        f"Critical-rank wall and CUDA Event GPU time (batch_size={batch_size})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = plot_dir / f"batch_size_{batch_size:03d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ffn_vs_draft_length(
    plot_dir: Path,
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
    step_rows: list[dict[str, float | int]],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size in batch_sizes:
        baseline_ffn_ms = float(
            next(
                row["avg_ffn_ms"]
                for row in step_rows
                if row["batch_size"] == batch_size and row["draft_length"] == 0
            )
        )
        y = [
            float(
                next(
                    row["avg_ffn_ms"]
                    for row in step_rows
                    if row["batch_size"] == batch_size
                    and row["draft_length"] == draft_length
                )
            )
            / baseline_ffn_ms
            for draft_length in draft_lengths
        ]
        ax.plot(draft_lengths, y, marker="o", linewidth=2, label=f"bs={batch_size}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("draft_length")
    ax.set_ylabel(
        "avg routed-expert/FFN GPU ms / avg routed-expert/FFN GPU ms(d=0)"
    )
    ax.set_title("Routed Expert/FFN GPU Time vs Draft Length")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "ffn_vs_draft_length.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_condition_grid(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    load_metric_rows: list[dict[str, Any]],
    results: dict[tuple[int, int], LoadedConditionData],
) -> Path:
    plt = import_plot_module()
    sample = results[(batch_size, draft_lengths[0])]
    layers = tuple(sample.layers.tolist())
    fig, axes = plt.subplots(
        len(layers),
        len(draft_lengths),
        figsize=(4.2 * len(draft_lengths), 2.8 * len(layers)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, layer_idx in enumerate(layers):
        for col_idx, draft_length in enumerate(draft_lengths):
            ax = axes[row_idx][col_idx]
            data = results[(batch_size, draft_length)]
            avg_histograms = strict_average_histograms(
                data,
                draft_length=draft_length,
            )
            sorted_counts, sorted_ids = sort_experts_desc(avg_histograms)
            counts = sorted_counts[row_idx]
            expert_ids = sorted_ids[row_idx]
            metrics_row = next(
                row
                for row in load_metric_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
                and row["layer"] == layer_idx
            )
            tick_positions = np.linspace(
                0,
                counts.size - 1,
                num=min(8, counts.size),
                dtype=int,
            )
            ax.bar(np.arange(counts.size), counts, color="#1f77b4", width=1.0)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(expert_ids[pos]) for pos in tick_positions])
            ax.set_title(
                f"layer={layer_idx}, draft={draft_length}\n"
                f"bal={metrics_row['balancedness']:.4f}, "
                f"gini={metrics_row['gini']:.4f}",
                fontsize=10,
            )
            if col_idx == 0:
                ax.set_ylabel("avg routed count")
            if row_idx == len(layers) - 1:
                ax.set_xlabel("expert id (condition-sorted)")
    path = plot_dir / f"batch_size_{batch_size:03d}_grid.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_expert_load(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    results: dict[tuple[int, int], LoadedConditionData],
) -> Path:
    plt = import_plot_module()
    baseline = results[(batch_size, 0)]
    layers = tuple(baseline.layers.tolist())
    baseline_avg = strict_average_histograms(baseline, draft_length=0)
    _, baseline_order = sort_experts_desc(baseline_avg)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 3.0 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    for row_idx, layer_idx in enumerate(layers):
        ax = axes[row_idx]
        tick_positions = np.linspace(
            0,
            baseline_order.shape[1] - 1,
            num=min(8, baseline_order.shape[1]),
            dtype=int,
        )
        tick_labels = [str(baseline_order[row_idx, pos]) for pos in tick_positions]
        for color, draft_length in zip(colors[: len(draft_lengths)], draft_lengths):
            data = results[(batch_size, draft_length)]
            avg_histograms = strict_average_histograms(
                data,
                draft_length=draft_length,
            )
            counts = reorder_histograms_by_expert_order(
                avg_histograms,
                baseline_order,
            )[row_idx]
            ax.plot(
                np.arange(counts.size),
                counts,
                linewidth=1.8,
                color=color,
                label=f"d={draft_length}",
            )
        ax.set_title(f"layer={layer_idx}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("avg routed assignments per global step")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("expert id (baseline-sorted)")
    axes[0].legend(ncol=len(draft_lengths), loc="upper right")
    path = plot_dir / f"batch_size_{batch_size:03d}_expert_load.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_rank_load(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    rank_load_rows: list[dict[str, Any]],
) -> Path:
    plt = import_plot_module()
    layers = sorted(
        {
            int(row["layer"])
            for row in rank_load_rows
            if row["batch_size"] == batch_size
        }
    )
    ep_ranks = sorted(
        {
            int(row["ep_rank"])
            for row in rank_load_rows
            if row["batch_size"] == batch_size
        }
    )
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, axes = plt.subplots(len(layers), 1, figsize=(9, 3.0 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    x = np.arange(len(ep_ranks))
    for row_idx, layer_idx in enumerate(layers):
        ax = axes[row_idx]
        width = 0.8 / max(len(draft_lengths), 1)
        for draft_idx, draft_length in enumerate(draft_lengths):
            values = [
                next(
                    float(row["avg_routed_assignments_per_step"])
                    for row in rank_load_rows
                    if row["batch_size"] == batch_size
                    and row["draft_length"] == draft_length
                    and row["layer"] == layer_idx
                    and row["ep_rank"] == ep_rank
                )
                for ep_rank in ep_ranks
            ]
            offset = (draft_idx - (len(draft_lengths) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[draft_idx % len(colors)],
                label=f"d={draft_length}",
            )
        ax.set_title(f"layer={layer_idx}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(rank) for rank in ep_ranks])
        ax.set_ylabel("avg routed assignments per global step")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("EP rank")
    axes[0].legend(ncol=len(draft_lengths), loc="upper right")
    path = plot_dir / f"batch_size_{batch_size:03d}_rank_load.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sorted_rank_ffn_time(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    sorted_rank_rows: list[dict[str, Any]],
) -> Path:
    plt = import_plot_module()
    positions = sorted(
        {
            int(row["sorted_rank_position"])
            for row in sorted_rank_rows
            if row["batch_size"] == batch_size
        }
    )
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for draft_idx, draft_length in enumerate(draft_lengths):
        values = [
            next(
                float(row["avg_local_ffn_ms"])
                for row in sorted_rank_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
                and row["sorted_rank_position"] == position
            )
            for position in positions
        ]
        ax.plot(
            positions,
            values,
            marker="o",
            color=colors[draft_idx % len(colors)],
            label=f"d={draft_length}",
        )
    ax.set_title(
        f"batch_size={batch_size} sorted rank-local routed-expert/FFN GPU time"
    )
    ax.set_xlabel("sorted rank position (0 = heaviest per barrier)")
    ax.set_ylabel(
        "avg local routed-expert/FFN GPU time after per-barrier sort (ms)"
    )
    ax.set_xticks(positions)
    ax.grid(alpha=0.25)
    ax.legend()
    path = plot_dir / f"sorted_rank_ffn_batch_{batch_size:03d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sorted_rank_metric_by_batch(
    plot_dir: Path,
    *,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    sorted_rank_rows: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_name: str,
) -> Path:
    plt = import_plot_module()
    positions = sorted(
        {
            int(row["sorted_rank_position"])
            for row in sorted_rank_rows
            if row["batch_size"] == batch_size
        }
    )
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for draft_idx, draft_length in enumerate(draft_lengths):
        values = [
            next(
                float(row[metric_key])
                for row in sorted_rank_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
                and row["sorted_rank_position"] == position
            )
            for position in positions
        ]
        ax.plot(
            positions,
            values,
            marker="o",
            color=colors[draft_idx % len(colors)],
            label=f"d={draft_length}",
        )
    ax.set_title(f"batch_size={batch_size} sorted rank {ylabel}")
    ax.set_xlabel("sorted rank position (0 = heaviest per layer)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.grid(alpha=0.25)
    ax.legend()
    path = plot_dir / output_name.format(batch_size=batch_size)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sorted_rank0_ffn_vs_batch_size(
    plot_dir: Path,
    sorted_rank_rows: list[dict[str, Any]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    x = np.arange(len(batch_sizes))
    width = 0.8 / max(len(draft_lengths), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for draft_idx, draft_length in enumerate(draft_lengths):
        values = [
            next(
                float(row["avg_ffn_ms"])
                for row in sorted_rank_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
                and row["sorted_rank_position"] == 0
            )
            for batch_size in batch_sizes
        ]
        offset = (draft_idx - (len(draft_lengths) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=f"d={draft_length}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.set_xlabel("batch_size")
    ax.set_ylabel("avg sorted-rank0 routed-expert/FFN GPU time (ms)")
    ax.set_title("Sorted-rank0 Routed Expert/FFN GPU Time vs Global Batch Size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = plot_dir / "sorted_rank0_ffn_vs_batch_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_routing_time_audit(
    plot_dir: Path,
    condition_rows: list[dict[str, Any]],
    layer_rank_rows: list[dict[str, Any]],
    sorted_rank_rows: list[dict[str, Any]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> list[Path]:
    plt = import_plot_module()
    paths: list[Path] = []
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    for batch_size in batch_sizes:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        positions = sorted(
            {
                int(row["sorted_rank_position"])
                for row in sorted_rank_rows
                if int(row["batch_size"]) == batch_size
            }
        )
        for draft_idx, draft_length in enumerate(draft_lengths):
            selected = [
                row
                for row in sorted_rank_rows
                if int(row["batch_size"]) == batch_size
                and int(row["draft_length"]) == draft_length
            ]
            if not selected:
                continue
            ffn_values = [
                next(
                    float(row["avg_ffn_sorted_routed_expert_gpu_ms"])
                    for row in selected
                    if int(row["sorted_rank_position"]) == position
                )
                for position in positions
            ]
            destination_values = [
                next(
                    float(row["avg_ffn_sorted_v10_destination_tokens"])
                    for row in selected
                    if int(row["sorted_rank_position"]) == position
                )
                for position in positions
            ]
            color = colors[draft_idx % len(colors)]
            axes[0].plot(
                positions,
                ffn_values,
                marker="o",
                color=color,
                label=f"d={draft_length}",
            )
            axes[1].plot(
                positions,
                destination_values,
                marker="o",
                color=color,
                label=f"d={draft_length}",
            )
        axes[0].set_title("FFN-sorted routed-expert GPU time")
        axes[0].set_ylabel("avg routed-expert GPU ms")
        axes[1].set_title("FFN-sorted v10 destination assignments")
        axes[1].set_ylabel("avg destination assignments")
        for axis in axes:
            axis.set_xlabel("sorted rank position")
            axis.set_xticks(positions)
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(f"batch_size={batch_size} routing-time audit")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = plot_dir / f"batch_{batch_size:03d}_ffn_sorted_audit.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    for row in condition_rows:
        batch_size = int(row["batch_size"])
        draft_length = int(row["draft_length"])
        selected = [
            layer_row
            for layer_row in layer_rank_rows
            if int(layer_row["batch_size"]) == batch_size
            and int(layer_row["draft_length"]) == draft_length
        ]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        x = [
            float(layer_row["avg_v10_destination_assignments"])
            for layer_row in selected
        ]
        y = [
            float(layer_row["avg_routed_expert_gpu_ms"])
            for layer_row in selected
        ]
        ranks = [
            int(layer_row["physical_rank"])
            for layer_row in selected
        ]
        scatter = ax.scatter(x, y, c=ranks, cmap="tab10", s=42, alpha=0.85)
        ax.set_xlabel("avg v10 destination assignments")
        ax.set_ylabel("avg routed-expert GPU ms")
        ax.set_title(
            f"batch_size={batch_size}, draft_length={draft_length}: "
            "FFN vs destination load"
        )
        ax.grid(alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="physical rank")
        fig.tight_layout()
        path = (
            plot_dir
            / f"batch_{batch_size:03d}_draft_{draft_length:02d}_scatter.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_active_expert_ratio_vs_batch_size(
    plot_dir: Path,
    active_ratio_rows: list[dict[str, Any]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for draft_length in draft_lengths:
        values = [
            next(
                float(row["active_expert_ratio"])
                for row in active_ratio_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for batch_size in batch_sizes
        ]
        ax.plot(batch_sizes, values, marker="o", linewidth=2, label=f"d={draft_length}")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("active expert ratio")
    ax.set_title("Active Expert Ratio vs Global Batch Size")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    path = plot_dir / "active_expert_ratio_vs_batch_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_routing_time_audit_report(
    manifest: dict[str, Any],
    condition_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Routing Time Audit",
        "",
        "## Scope",
        "",
        (
            "- v10_destination_assignments counts all token/expert assignments "
            "incoming to each destination EP rank."
        ),
        (
            "- v8_emulated_self_assignments replays the old local/self view by "
            "counting only source_rank == destination_rank assignments."
        ),
        (
            "- Strict barriers are verification_only for draft_length > 0 and "
            "decode_only for draft_length = 0, with every rank matching the "
            "same step kind."
        ),
        "",
        "## Condition Verdicts",
        "",
    ]
    for batch_size in tuple(manifest["batch_sizes"]):
        rows = [
            row
            for row in condition_rows
            if int(row["batch_size"]) == batch_size
        ]
        lines.append(f"### batch_size={batch_size}")
        for row in rows:
            lines.append(
                "- draft_length={draft_length}: "
                "ffn_max/min={ffn_ratio:.3f}, "
                "v10_token_max/min={destination_ratio:.3f}, "
                "v8_self_max/min={self_ratio:.3f}, "
                "pearson(ffn,v10)={destination_corr:.3f}, "
                "spearman(ffn,v10)={destination_rank_corr:.3f}, "
                "pearson(ffn,v8)={self_corr:.3f}, "
                "ms_per_destination_max/min={destination_ms_ratio:.3f}, "
                "verdict={verdict}".format(
                    draft_length=int(row["draft_length"]),
                    ffn_ratio=float(row["ffn_max_min_ratio"]),
                    destination_ratio=float(
                        row["v10_destination_token_max_min_ratio"]
                    ),
                    self_ratio=float(
                        row["v8_emulated_self_token_max_min_ratio"]
                    ),
                    destination_corr=float(
                        row["pearson_ffn_v10_destination_tokens"]
                    ),
                    destination_rank_corr=float(
                        row["spearman_ffn_v10_destination_tokens"]
                    ),
                    self_corr=float(
                        row["pearson_ffn_v8_emulated_self_tokens"]
                    ),
                    destination_ms_ratio=float(
                        row[
                            "physical_rank_ms_per_destination_assignment_"
                            "max_min_ratio"
                        ]
                    ),
                    verdict=str(row["interpretation"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Rules",
            "",
            (
                "- If v10 destination token imbalance is near flat, "
                "correlation is low, or ms_per_destination_assignment varies "
                "strongly across physical ranks, current v10 data does not "
                "support explaining FFN time mainly by incoming destination "
                "assignment count."
            ),
            (
                "- If v8 self assignments align better than v10 destination "
                "assignments, the old curve matched the old local/self "
                "accounting view, not the true destination-rank incoming load."
            ),
            (
                "- If v10 destination assignments correlate strongly with FFN "
                "time and per-assignment cost is stable, the current code "
                "supports the load-imbalance hypothesis."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(
    output_dir: Path,
    manifest: dict[str, Any],
    speedup_rows: list[dict[str, float | int]],
    acceptance_rows: list[dict[str, float | int]],
    step_rows: list[dict[str, float | int]],
    load_metric_rows: list[dict[str, Any]],
    rank_ffn_imbalance_rows: list[dict[str, Any]],
) -> str:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    lines = [
        "# Qwen3.6 MTP DP+EP 三子实验报告",
        "",
        "## 实验设置",
        "",
        f"- 模型：`{manifest['model']}`",
        (
            f"- 数据集：`{manifest['dataset']}` / "
            f"`{manifest['dataset_config']}` / `{manifest['dataset_split']}`"
        ),
        f"- global batch_size：`{', '.join(map(str, batch_sizes))}`",
        f"- draft_length：`{', '.join(map(str, draft_lengths))}`",
        f"- data_parallel_size：`{manifest['data_parallel_size']}`",
        f"- num_samples：`{manifest['num_samples']}`",
        f"- batch_size_scope：`{manifest['batch_size_scope']}`",
        f"- mixed_step_policy：`{manifest['mixed_step_policy']}`",
        f"- TPOT 定义：`{manifest.get('tpot_definition', TPOT_DEFINITION)}`",
        "- speedup 口径：vLLM request TPOT histogram 和 generation throughput",
        f"- max_tokens：`{manifest['max_tokens']}`",
        f"- warmup_rounds：`{manifest.get('warmup_rounds', 0)}`",
        (
            "- timing：verification wall-clock 是主性能口径；Attention/MoE/"
            "Other 来自延迟解析的 CUDA Event。"
        ),
        (
            "- NVTX：仅用于 Nsight Systems 区间校验，不作为任何 CSV "
            "数值来源。"
        ),
        "",
        "## vLLM Native Speedup",
        "",
    ]

    for batch_size in batch_sizes:
        candidates = [
            row
            for row in speedup_rows
            if row["batch_size"] == batch_size and row["draft_length"] != 0
        ]
        best = max(
            candidates,
            key=lambda row: float(row["vllm_generation_throughput_speedup"]),
        )
        speedup_summaries = ", ".join(
            f"d={row['draft_length']}:"
            f"tpot={float(row['vllm_request_tpot_speedup']):.3f}x,"
            f"throughput={float(row['vllm_generation_throughput_speedup']):.3f}x"
            for row in candidates
        )
        lines.append(
            f"- batch_size={batch_size}: {speedup_summaries}; "
            f"best_throughput=d={best['draft_length']} "
            f"({float(best['vllm_generation_throughput_speedup']):.3f}x)"
        )

    lines.extend(["", "## Spec Decode Acceptance", ""])
    for batch_size in batch_sizes:
        rows = [
            row
            for row in acceptance_rows
            if row["batch_size"] == batch_size and row["draft_length"] != 0
        ]
        summaries = ", ".join(
            f"d={row['draft_length']}:"
            f"accept={float(row['acceptance_rate']) * 100:.1f}%,"
            f"mean_len={float(row['mean_acceptance_length']):.2f}"
            for row in rows
        )
        lines.append(f"- batch_size={batch_size}: {summaries}")

    lines.extend(["", "## Decode/Verification-only Wall/GPU 时间", ""])
    for batch_size in batch_sizes:
        rows = [
            row for row in step_rows if row["batch_size"] == batch_size
        ]
        lines.append(f"### batch_size={batch_size}")
        for row in rows:
            lines.append(
                f"- draft_length={row['draft_length']}: "
                "verification_wall="
                f"{float(row['avg_verification_wall_ms']):.2f} ms, "
                "moe_gpu="
                f"{float(row['avg_moe_gpu_ms']):.2f} ms, "
                "routed_expert_gpu="
                f"{float(row['avg_routed_expert_gpu_ms']):.2f} ms, "
                f"gpu_other={float(row['avg_gpu_other_ms']):.2f} ms, "
                "moe_gpu_share="
                f"{float(row['moe_gpu_share']) * 100:.1f}%, "
                f"global_captured={int(row['num_global_captured_steps'])}, "
                f"global_candidates={int(row['num_global_candidate_steps'])}, "
                f"prefill_drop={int(row['num_global_prefill_dropped_steps'])}, "
                f"mixed_drop={int(row['num_global_mixed_dropped_steps'])}"
            )

    lines.extend(["", "## Decode/Verification-only Expert Load 分布", ""])
    for batch_size in batch_sizes:
        lines.append(f"### batch_size={batch_size}")
        batch_rows = [
            row for row in load_metric_rows if row["batch_size"] == batch_size
        ]
        layers = sorted({int(row["layer"]) for row in batch_rows})
        for layer in layers:
            candidates = [
                row
                for row in batch_rows
                if row["layer"] == layer and row["draft_length"] != 0
            ]
            worst = min(candidates, key=lambda row: float(row["balancedness_delta"]))
            lines.append(
                f"- layer {layer}: worst balancedness delta at "
                f"d={worst['draft_length']} "
                f"(Δbal={float(worst['balancedness_delta']):+.4f}, "
                f"Δg={float(worst['gini_delta']):+.4f})"
            )

    lines.extend(["", "## Rank-local Routed Expert GPU time 不均衡", ""])
    for batch_size in batch_sizes:
        rows = [
            row
            for row in rank_ffn_imbalance_rows
            if row["batch_size"] == batch_size
        ]
        summaries = ", ".join(
            f"d={row['draft_length']}: "
            f"max/mean={float(row['avg_step_ffn_max_mean_ratio']):.3f}, "
            "routed_expert_gap="
            f"{float(row['avg_heaviest_minus_lightest_local_ffn_ms']):.2f} ms"
            for row in rows
        )
        lines.append(f"- batch_size={batch_size}: {summaries}")
    return "\n".join(lines) + "\n"


def analyze_draft_drop(
    input_dir: Path,
    *,
    skip_plots: bool = False,
) -> None:
    manifest, results = load_all_conditions(input_dir)
    cutoff_rows, layer_rows, step_rows, condition_rows = build_draft_drop_rows(
        manifest,
        results,
    )
    dirs = ensure_analysis_dirs(input_dir)
    save_csv(dirs["tables"] / "draft_drop_cutoffs.csv", cutoff_rows)
    save_csv(dirs["tables"] / "draft_drop_layer_steps.csv", layer_rows)
    save_csv(dirs["tables"] / "draft_drop_step_summary.csv", step_rows)
    save_csv(
        dirs["tables"] / "draft_drop_condition_summary.csv",
        condition_rows,
    )
    if not skip_plots:
        plot_layer_drop_ratio(dirs["draft_drop"], layer_rows)
        plot_step_drop_distribution(
            dirs["draft_drop"],
            step_rows,
            condition_rows,
        )


def analyze_experiment(
    input_dir: Path,
    *,
    skip_plots: bool = False,
    skip_report: bool = False,
) -> None:
    manifest, results = load_all_conditions(input_dir)
    schema_versions = sorted({data.schema_version for data in results.values()})
    if schema_versions != [SCHEMA_VERSION]:
        raise RuntimeError(
            "Unified analysis requires schema v10 CUDA Event raw data; found "
            f"schema versions {schema_versions}. Re-run collect."
        )
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    if 0 not in draft_lengths:
        raise RuntimeError(
            "Unified analysis requires draft_length=0 baseline data. Re-run "
            "collect with --draft-lengths 0 2 4 6."
        )
    dirs = ensure_analysis_dirs(input_dir)

    decode_time_ms_by_condition = {
        condition: data.decode_time_total_ms for condition, data in results.items()
    }
    generation_tokens_by_condition = {
        condition: data.num_generation_tokens_total
        for condition, data in results.items()
    }
    output_tokens_excl_first_by_condition = {
        condition: data.num_output_tokens_excl_first_total
        for condition, data in results.items()
    }
    vllm_request_tpot_ms_by_condition = {
        condition: data.vllm_request_tpot_ms for condition, data in results.items()
    }
    vllm_generation_throughput_tok_s_by_condition = {
        condition: data.vllm_generation_throughput_tok_s
        for condition, data in results.items()
    }
    speedup_rows = build_speedup_rows(
        decode_time_ms_by_condition,
        generation_tokens_by_condition,
        output_tokens_excl_first_by_condition,
        batch_sizes,
        draft_lengths,
        vllm_request_tpot_ms_by_condition=vllm_request_tpot_ms_by_condition,
        vllm_generation_throughput_tok_s_by_condition=(
            vllm_generation_throughput_tok_s_by_condition
        ),
    )
    acceptance_rows = build_acceptance_metric_rows(manifest, results)
    step_rows = build_step_time_rows(manifest, results)
    (
        load_metric_rows,
        condition_sorted_rows,
        baseline_sorted_rows,
        rank_load_rows,
    ) = build_load_distribution_rows(manifest, results)
    (
        sorted_rank_ffn_rows,
        rank_ffn_imbalance_rows,
    ) = build_sorted_rank_ffn_time_rows(manifest, results)
    (
        sorted_rank_moe_rows,
        rank_moe_imbalance_rows,
    ) = build_sorted_rank_moe_time_rows(manifest, results)
    barrier_rank_layer_rows = build_barrier_rank_layer_rows(manifest, results)
    timing_completeness_rows = build_timing_completeness_rows(
        manifest,
        results,
    )
    sorted_rank_summary_rows = build_sorted_rank_summary_rows(manifest, results)
    (
        routing_audit_condition_rows,
        routing_audit_layer_rank_rows,
        routing_audit_sorted_rank_rows,
    ) = build_routing_time_audit_rows(manifest, results)
    active_expert_ratio_rows = build_active_expert_ratio_rows(manifest, results)
    position_rows = build_position_breakdown_rows(manifest, results)
    drop_rows = build_draft_drop_rows(manifest, results)

    save_csv(dirs["tables"] / "speedup_metrics.csv", speedup_rows)
    save_csv(dirs["tables"] / "acceptance_metrics.csv", acceptance_rows)
    save_csv(
        dirs["tables"] / "barrier_rank_layer_metrics.csv",
        barrier_rank_layer_rows,
    )
    save_csv(
        dirs["tables"] / "barrier_rank_layer_cuda_event.csv",
        barrier_rank_layer_rows,
    )
    save_csv(
        dirs["tables"] / "timing_completeness.csv",
        timing_completeness_rows,
    )
    save_csv(dirs["tables"] / "sorted_rank_summary.csv", sorted_rank_summary_rows)
    save_csv(
        dirs["tables"] / "active_expert_ratio.csv",
        active_expert_ratio_rows,
    )
    save_csv(dirs["tables"] / "step_time_breakdown.csv", step_rows)
    save_csv(dirs["tables"] / "load_balance_metrics.csv", load_metric_rows)
    save_csv(
        dirs["tables"] / "averaged_distributions_condition_sorted.csv",
        condition_sorted_rows,
    )
    save_csv(
        dirs["tables"] / "averaged_distributions_baseline_sorted.csv",
        baseline_sorted_rows,
    )
    save_csv(dirs["tables"] / "rank_load_metrics.csv", rank_load_rows)
    save_csv(
        dirs["tables"] / "rank_ffn_time_sorted.csv",
        sorted_rank_ffn_rows,
    )
    save_csv(
        dirs["tables"] / "rank_routed_expert_gpu_time_sorted.csv",
        sorted_rank_ffn_rows,
    )
    save_csv(
        dirs["tables"] / "rank_moe_gpu_time_sorted.csv",
        sorted_rank_moe_rows,
    )
    save_csv(
        dirs["tables"] / "rank_ffn_imbalance_metrics.csv",
        rank_ffn_imbalance_rows,
    )
    save_csv(
        dirs["tables"] / "rank_moe_gpu_imbalance_metrics.csv",
        rank_moe_imbalance_rows,
    )
    save_csv(
        dirs["tables"] / "routing_time_audit_by_condition.csv",
        routing_audit_condition_rows,
    )
    save_csv(
        dirs["tables"] / "routing_time_audit_by_layer_rank.csv",
        routing_audit_layer_rank_rows,
    )
    save_csv(
        dirs["tables"] / "routing_time_sorted_rank_audit.csv",
        routing_audit_sorted_rank_rows,
    )
    save_csv(dirs["tables"] / "position_ffn_breakdown.csv", position_rows)
    cutoff_rows, layer_rows, drop_step_rows, condition_drop_rows = drop_rows
    save_csv(dirs["tables"] / "draft_drop_cutoffs.csv", cutoff_rows)
    save_csv(dirs["tables"] / "draft_drop_layer_steps.csv", layer_rows)
    save_csv(
        dirs["tables"] / "draft_drop_step_summary.csv",
        drop_step_rows,
    )
    save_csv(
        dirs["tables"] / "draft_drop_condition_summary.csv",
        condition_drop_rows,
    )

    rank_trace_rows: list[dict[str, Any]] = []
    for condition in manifest["conditions"]:
        condition_name = Path(condition["raw_path"]).stem
        partial_dir = input_dir / "_dp_partials" / condition_name
        trace_paths = sorted(partial_dir.glob("rank_*.trace.json"))
        if not trace_paths:
            continue
        trace_payloads = [load_rank_trace_payload(path) for path in trace_paths]
        rank_trace_rows.extend(build_rank_trace_rows(condition_name, trace_payloads))
        if not skip_plots:
            plot_rank_trace_timeline(
                dirs["rank_traces"],
                condition_name,
                trace_payloads,
            )

    if rank_trace_rows:
        save_csv(dirs["tables"] / "rank_trace_summary.csv", rank_trace_rows)

    if not skip_report:
        report = build_report(
            input_dir,
            manifest,
            speedup_rows,
            acceptance_rows,
            step_rows,
            load_metric_rows,
            rank_ffn_imbalance_rows,
        )
        with (input_dir / "实验报告.md").open("w", encoding="utf-8") as fp:
            fp.write(report)
        routing_time_report = build_routing_time_audit_report(
            manifest,
            routing_audit_condition_rows,
        )
        with (input_dir / "routing_time_audit_report.md").open(
            "w",
            encoding="utf-8",
        ) as fp:
            fp.write(routing_time_report)

    if not skip_plots:
        plot_speedup_vs_draft_length(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        plot_speedup_vs_batch_size(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        plot_decode_throughput_speedup_vs_batch_size(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        plot_acceptance_rate_vs_draft_length(
            dirs["speedup"], acceptance_rows, batch_sizes, draft_lengths
        )
        plot_ffn_vs_draft_length(
            dirs["time"],
            batch_sizes,
            draft_lengths,
            step_rows,
        )
        for batch_size in batch_sizes:
            plot_step_time_breakdown(
                dirs["time"],
                batch_size,
                draft_lengths,
                step_rows,
            )
            plot_expert_load(
                dirs["expert_load"],
                batch_size,
                draft_lengths,
                results,
            )
            plot_rank_load(
                dirs["rank_load"],
                batch_size,
                draft_lengths,
                rank_load_rows,
            )
            plot_sorted_rank_metric_by_batch(
                dirs["rank_ffn_time_sorted"],
                batch_size=batch_size,
                draft_lengths=draft_lengths,
                sorted_rank_rows=sorted_rank_summary_rows,
                metric_key="avg_ffn_ms",
                ylabel="avg routed-expert/FFN GPU time after per-layer sort (ms)",
                output_name="sorted_rank_ffn_batch_{batch_size:03d}.png",
            )
            plot_sorted_rank_metric_by_batch(
                dirs["rank_ffn_time_sorted"],
                batch_size=batch_size,
                draft_lengths=draft_lengths,
                sorted_rank_rows=sorted_rank_summary_rows,
                metric_key="avg_local_routed_tokens",
                ylabel="avg local routed tokens after per-layer sort",
                output_name="sorted_rank_tokens_batch_{batch_size:03d}.png",
            )
        plot_sorted_rank0_ffn_vs_batch_size(
            dirs["rank_ffn_time_sorted"],
            sorted_rank_summary_rows,
            batch_sizes,
            draft_lengths,
        )
        plot_active_expert_ratio_vs_batch_size(
            dirs["rank_ffn_time_sorted"],
            active_expert_ratio_rows,
            batch_sizes,
            draft_lengths,
        )
        plot_routing_time_audit(
            dirs["routing_time_audit"],
            routing_audit_condition_rows,
            routing_audit_layer_rank_rows,
            routing_audit_sorted_rank_rows,
            batch_sizes,
            draft_lengths,
        )
        plot_position_breakdown(dirs["position_ffn"], position_rows)
        plot_layer_drop_ratio(dirs["draft_drop"], layer_rows)
        plot_step_drop_distribution(
            dirs["draft_drop"],
            drop_step_rows,
            condition_drop_rows,
        )
