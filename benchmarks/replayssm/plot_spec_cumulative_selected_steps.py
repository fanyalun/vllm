# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cumulative speculative expert loads for selected scheduler steps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.replayssm.expert_routing_analysis import (
    PLOT_LAYERS,
    SPEC_STAGES,
    TraceData,
    complete_step_ids,
    expert_counts,
    load_trace,
)

STAGE_LABELS = {
    "target": "target",
    "target_plus_draft_1": "target + draft1",
    "target_plus_draft_1_2": "target + draft1 + draft2",
    "target_plus_draft_1_2_3": "target + draft1 + draft2 + draft3",
}
DRAFT1_DROP_STAGES = (
    ("target", ("spec_target",)),
    (
        "target_plus_unclipped_draft_1",
        ("spec_target", "spec_draft_1"),
    ),
    (
        "target_plus_surviving_draft_1",
        ("spec_target", "spec_draft_1"),
    ),
)
DRAFT1_DROP_STAGE_LABELS = {
    "target": "target",
    "target_plus_unclipped_draft_1": "target + draft1 (unclipped)",
    "target_plus_surviving_draft_1": "target + surviving draft1",
}
PREFIX_CAPACITY_STAGE_LABELS = {
    "target": "target",
    "target_plus_draft_1": "target + retained draft1",
    "target_plus_draft_1_2": "target + retained draft1 + draft2",
    "target_plus_draft_1_2_3": (
        "target + retained draft1 + draft2 + draft3"
    ),
}
DRAFT_ROUTE_KINDS = ("spec_draft_1", "spec_draft_2", "spec_draft_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs=3)
    parser.add_argument("--plot-layers", type=int, nargs="+")
    parser.add_argument("--draft1-drop-layer", type=int)
    parser.add_argument("--all-layer-prefix-capacity", action="store_true")
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--sequence-lengths-output", type=Path)
    parser.add_argument("--layer-lengths-output", type=Path)
    return parser.parse_args()


def representative_steps(complete_steps: np.ndarray) -> tuple[int, int, int]:
    return (
        int(complete_steps[0]),
        int(complete_steps[len(complete_steps) // 2]),
        int(complete_steps[-1]),
    )


def _rank_stage_loads_by_target(
    stage_loads: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    target_order = np.argsort(-stage_loads["target"], axis=1, kind="stable")
    return {
        stage_name: np.take_along_axis(counts, target_order, axis=1)
        for stage_name, counts in stage_loads.items()
    }


def selected_step_loads(
    trace_dir: Path, requested_steps: list[int] | None
) -> tuple[tuple[int, int, int], dict[int, dict[str, np.ndarray]]]:
    trace = load_trace(trace_dir)
    if trace.name != "replayssm_spec_bs128_d3":
        raise ValueError(f"expected Spec BS128 D3 trace, got {trace.name}")
    complete_steps = complete_step_ids(trace)
    if not complete_steps.size:
        raise ValueError("trace has no complete speculative steps")
    steps = (
        tuple(requested_steps)
        if requested_steps is not None
        else representative_steps(complete_steps)
    )
    unknown = sorted(set(steps) - set(complete_steps.tolist()))
    if unknown:
        raise ValueError(f"steps are not complete speculative steps: {unknown}")

    loads_by_step = {}
    for step in steps:
        step_mask = trace.scheduler_steps == step
        stage_loads = {}
        for stage_name, route_kinds in SPEC_STAGES:
            mask = step_mask & np.isin(trace.route_kinds, route_kinds)
            counts = expert_counts(trace.routes, mask)
            expected_rows = 128 * len(route_kinds)
            if int(mask.sum()) != expected_rows:
                raise ValueError(
                    f"step {step} {stage_name} has {mask.sum()} rows, "
                    f"expected {expected_rows}"
                )
            stage_loads[stage_name] = counts
        loads_by_step[int(step)] = _rank_stage_loads_by_target(stage_loads)
    return tuple(int(step) for step in steps), loads_by_step


def _rows_for_kind(trace: TraceData, step: int, route_kind: str) -> np.ndarray:
    mask = (trace.scheduler_steps == step) & (trace.route_kinds == route_kind)
    rows = np.flatnonzero(mask)
    if rows.size != 128:
        raise ValueError(
            f"step {step} {route_kind} has {rows.size} rows, expected 128"
        )
    return rows


def _layer_counts(routes: np.ndarray, num_experts: int) -> np.ndarray:
    return np.bincount(routes.reshape(-1), minlength=num_experts)


def draft1_drop_step_loads(
    trace_dir: Path,
    requested_steps: list[int] | None,
    drop_layer: int,
) -> tuple[
    tuple[int, int, int],
    dict[int, dict[str, np.ndarray]],
    list[dict[str, int]],
]:
    trace = load_trace(trace_dir)
    if trace.name != "replayssm_spec_bs128_d3":
        raise ValueError(f"expected Spec BS128 D3 trace, got {trace.name}")
    num_layers = trace.routes.shape[1]
    num_experts = int(trace.manifest["num_experts"])
    if not 0 <= drop_layer < num_layers:
        raise ValueError(f"drop layer must be in [0, {num_layers - 1}]")
    complete_steps = complete_step_ids(trace)
    if not complete_steps.size:
        raise ValueError("trace has no complete speculative steps")
    steps = (
        tuple(requested_steps)
        if requested_steps is not None
        else representative_steps(complete_steps)
    )
    unknown = sorted(set(steps) - set(complete_steps.tolist()))
    if unknown:
        raise ValueError(f"steps are not complete speculative steps: {unknown}")

    loads_by_step = {}
    stats_rows: list[dict[str, int]] = []
    for step in steps:
        routes_by_kind = {
            route_kind: np.asarray(
                trace.routes[_rows_for_kind(trace, int(step), route_kind)]
            )
            for route_kind in ("spec_target", "spec_draft_1")
        }
        for route_kind, routes in routes_by_kind.items():
            if np.any(np.diff(np.sort(routes, axis=2), axis=2) == 0):
                raise ValueError(
                    f"step {step} {route_kind} has duplicate expert IDs"
                )
        target = routes_by_kind["spec_target"]
        draft1 = routes_by_kind["spec_draft_1"]
        surviving_draft1 = np.ones(128, dtype=np.bool_)
        target_at_drop = _layer_counts(target[:, drop_layer], num_experts)
        capacity = int(target_at_drop.max())
        layer_loads = target_at_drop.copy()
        for row in range(128):
            experts = draft1[row, drop_layer]
            if np.any(layer_loads[experts] >= capacity):
                surviving_draft1[row] = False
            else:
                layer_loads[experts] += 1

        stage_loads = {
            stage_name: np.empty((num_layers, num_experts), dtype=np.int64)
            for stage_name, _ in DRAFT1_DROP_STAGES
        }
        for layer in range(num_layers):
            target_counts = _layer_counts(target[:, layer], num_experts)
            unclipped_draft1_counts = _layer_counts(
                draft1[:, layer], num_experts
            )
            layer_survivors = (
                np.ones(128, dtype=np.bool_)
                if layer < drop_layer
                else surviving_draft1
            )
            draft1_counts = _layer_counts(
                draft1[layer_survivors, layer], num_experts
            )
            stage_loads["target"][layer] = target_counts
            stage_loads["target_plus_unclipped_draft_1"][layer] = (
                target_counts + unclipped_draft1_counts
            )
            stage_loads["target_plus_surviving_draft_1"][layer] = (
                target_counts + draft1_counts
            )
            if layer == drop_layer:
                draft1_peak = stage_loads[
                    "target_plus_surviving_draft_1"
                ][layer].max()
                if draft1_peak > capacity:
                    raise AssertionError(
                        f"step {step} layer {layer} exceeds drop-layer capacity"
                    )
            expected_draft1_assignments = int(layer_survivors.sum()) * 8
            observed_draft1_assignments = int(draft1_counts.sum())
            if observed_draft1_assignments != expected_draft1_assignments:
                raise AssertionError("surviving draft1 assignment count mismatch")
            stats_rows.append(
                {
                    "scheduler_step": int(step),
                    "drop_layer": drop_layer,
                    "max_capacity": capacity,
                    "dropped_draft1_tokens": int(
                        (~surviving_draft1).sum()
                    ),
                    "surviving_draft1_tokens": int(
                        surviving_draft1.sum()
                    ),
                    "layer": layer,
                    "target_peak": int(target_counts.max()),
                    "target_plus_unclipped_draft1_peak": int(
                        stage_loads["target_plus_unclipped_draft_1"][
                            layer
                        ].max()
                    ),
                    "target_plus_surviving_draft1_peak": int(
                        stage_loads["target_plus_surviving_draft_1"][
                            layer
                        ].max()
                    ),
                    "draft1_assignments": observed_draft1_assignments,
                }
            )
        loads_by_step[int(step)] = _rank_stage_loads_by_target(stage_loads)
    return tuple(int(step) for step in steps), loads_by_step, stats_rows


def _aligned_step_routes(
    trace: TraceData, step: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    target_rows = _rows_for_kind(trace, step, "spec_target")
    target_request_ids = trace.request_ids[target_rows]
    routes = [np.asarray(trace.routes[target_rows])]
    for route_kind in DRAFT_ROUTE_KINDS:
        rows = _rows_for_kind(trace, step, route_kind)
        if not np.array_equal(trace.request_ids[rows], target_request_ids):
            raise ValueError(
                f"step {step} {route_kind} request order differs from target"
            )
        routes.append(np.asarray(trace.routes[rows]))
    for route_kind, stage_routes in zip(
        ("spec_target", *DRAFT_ROUTE_KINDS), routes
    ):
        if np.any(np.diff(np.sort(stage_routes, axis=2), axis=2) == 0):
            raise ValueError(f"step {step} {route_kind} has duplicate experts")
    return routes[0], routes[1:], target_rows


def _per_layer_prefix_lengths(
    target: np.ndarray,
    drafts: list[np.ndarray],
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_layers = target.shape[1]
    layer_lengths = np.zeros((num_layers, target.shape[0]), dtype=np.int8)
    capacities = np.empty(num_layers, dtype=np.int64)
    for layer in range(num_layers):
        layer_loads = _layer_counts(target[:, layer], num_experts)
        capacity = int(layer_loads.max())
        capacities[layer] = capacity
        lengths = np.zeros(target.shape[0], dtype=np.int8)
        for draft_index, draft in enumerate(drafts, start=1):
            for sequence in range(target.shape[0]):
                if lengths[sequence] != draft_index - 1:
                    continue
                experts = draft[sequence, layer]
                if np.any(layer_loads[experts] >= capacity):
                    continue
                layer_loads[experts] += 1
                lengths[sequence] = draft_index
        if int(layer_loads.max()) > capacity:
            raise AssertionError(f"layer {layer} exceeds target capacity")
        layer_lengths[layer] = lengths
    return layer_lengths, capacities


def prefix_capacity_step_loads(
    trace_dir: Path,
    requested_steps: list[int] | None,
) -> tuple[
    tuple[int, int, int],
    dict[int, dict[str, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    trace = load_trace(trace_dir)
    if trace.name != "replayssm_spec_bs128_d3":
        raise ValueError(f"expected Spec BS128 D3 trace, got {trace.name}")
    complete_steps = complete_step_ids(trace)
    if not complete_steps.size:
        raise ValueError("trace has no complete speculative steps")
    steps = (
        tuple(requested_steps)
        if requested_steps is not None
        else representative_steps(complete_steps)
    )
    unknown = sorted(set(steps) - set(complete_steps.tolist()))
    if unknown:
        raise ValueError(f"steps are not complete speculative steps: {unknown}")

    num_experts = int(trace.manifest["num_experts"])
    loads_by_step = {}
    summary_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for step in steps:
        target, drafts, target_rows = _aligned_step_routes(trace, int(step))
        layer_lengths, capacities = _per_layer_prefix_lengths(
            target, drafts, num_experts
        )
        final_lengths = layer_lengths.min(axis=0)
        stage_loads = {
            stage_name: np.empty(
                (target.shape[1], num_experts), dtype=np.int64
            )
            for stage_name, _ in SPEC_STAGES
        }
        for layer in range(target.shape[1]):
            cumulative = _layer_counts(target[:, layer], num_experts)
            stage_loads["target"][layer] = cumulative
            for draft_index, draft in enumerate(drafts, start=1):
                retained = final_lengths >= draft_index
                cumulative = cumulative + _layer_counts(
                    draft[retained, layer], num_experts
                )
                stage_name = SPEC_STAGES[draft_index][0]
                stage_loads[stage_name][layer] = cumulative
            if int(cumulative.max()) > int(capacities[layer]):
                raise AssertionError(
                    f"step {step} layer {layer} exceeds target capacity"
                )
            local_histogram = np.bincount(
                layer_lengths[layer], minlength=4
            )
            layer_rows.append(
                {
                    "scheduler_step": int(step),
                    "layer": layer,
                    "max_capacity": int(capacities[layer]),
                    "mean_local_prefix_length": float(
                        layer_lengths[layer].mean()
                    ),
                    "local_length_0_sequences": int(local_histogram[0]),
                    "local_length_1_sequences": int(local_histogram[1]),
                    "local_length_2_sequences": int(local_histogram[2]),
                    "local_length_3_sequences": int(local_histogram[3]),
                    "final_max_expert_load": int(cumulative.max()),
                }
            )
        loads_by_step[int(step)] = _rank_stage_loads_by_target(stage_loads)

        final_histogram = np.bincount(final_lengths, minlength=4)
        summary_rows.append(
            {
                "scheduler_step": int(step),
                "mean_final_draft_length": float(final_lengths.mean()),
                "retained_draft_tokens": int(final_lengths.sum()),
                "retention_rate": float(
                    final_lengths.sum() / (target.shape[0] * len(drafts))
                ),
                "length_0_sequences": int(final_histogram[0]),
                "length_1_sequences": int(final_histogram[1]),
                "length_2_sequences": int(final_histogram[2]),
                "length_3_sequences": int(final_histogram[3]),
            }
        )
        for sequence, final_length in enumerate(final_lengths):
            limiting_layers = np.flatnonzero(
                layer_lengths[:, sequence] == final_length
            )
            row = int(target_rows[sequence])
            sequence_rows.append(
                {
                    "scheduler_step": int(step),
                    "sequence_index": sequence,
                    "request_id": str(trace.request_ids[row]),
                    "question_index": int(trace.question_indices[row]),
                    "target_absolute_position": int(
                        trace.absolute_positions[row]
                    ),
                    "final_draft_length": int(final_length),
                    "limiting_layers": ";".join(
                        str(layer) for layer in limiting_layers
                    ),
                    "all_layer_lengths": ";".join(
                        str(length)
                        for length in layer_lengths[:, sequence]
                    ),
                }
            )
    return (
        tuple(int(step) for step in steps),
        loads_by_step,
        summary_rows,
        sequence_rows,
        layer_rows,
    )


def write_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    steps: tuple[int, int, int],
    loads_by_step: dict[int, dict[str, np.ndarray]],
    output: Path,
    stage_labels: dict[str, str] = STAGE_LABELS,
    stages: tuple[tuple[str, tuple[str, ...]], ...] = SPEC_STAGES,
    plot_layers: tuple[int, ...] = PLOT_LAYERS,
    step_annotations: dict[int, str] | None = None,
    title: str = "Cumulative speculative expert loads for individual complete steps",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    num_layers = loads_by_step[steps[0]]["target"].shape[0]
    if not plot_layers:
        raise ValueError("at least one plot layer is required")
    invalid_layers = [
        layer for layer in plot_layers if not 0 <= layer < num_layers
    ]
    if invalid_layers:
        raise ValueError(f"invalid plot layers: {invalid_layers}")
    figure_width = 16 if len(plot_layers) == 3 else 5.2 * len(plot_layers)
    figure, axes = plt.subplots(
        3,
        len(plot_layers),
        figsize=(figure_width, 13),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    expert_order = np.arange(1, 257)
    for row, step in enumerate(steps):
        for column, layer in enumerate(plot_layers):
            axis = axes[row, column]
            for (stage_name, _), color in zip(stages, colors):
                axis.plot(
                    expert_order,
                    loads_by_step[step][stage_name][layer],
                    label=stage_labels[stage_name],
                    color=color,
                )
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"Layer {layer}")
            if column == 0:
                axis.set_ylabel(f"Step {step}\nrouted assignments")
                if step_annotations is not None:
                    axis.text(
                        0.98,
                        0.96,
                        step_annotations[step],
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8},
                    )
            if row == len(steps) - 1:
                axis.set_xlabel("expert rank fixed by target load")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(stages),
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(title, y=0.965)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.all_layer_prefix_capacity and args.draft1_drop_layer is not None:
        raise ValueError(
            "--all-layer-prefix-capacity and --draft1-drop-layer conflict"
        )
    plot_layers = (
        tuple(args.plot_layers)
        if args.plot_layers is not None
        else PLOT_LAYERS
    )
    if args.all_layer_prefix_capacity:
        (
            steps,
            loads_by_step,
            summary_rows,
            sequence_rows,
            layer_rows,
        ) = prefix_capacity_step_loads(args.trace_dir, args.steps)
        step_annotations = {
            int(row["scheduler_step"]): (
                "final n0/n1/n2/n3="
                f"{row['length_0_sequences']}/"
                f"{row['length_1_sequences']}/"
                f"{row['length_2_sequences']}/"
                f"{row['length_3_sequences']}"
            )
            for row in summary_rows
        }
        plot(
            steps,
            loads_by_step,
            args.output,
            stage_labels=PREFIX_CAPACITY_STAGE_LABELS,
            stages=SPEC_STAGES,
            plot_layers=plot_layers,
            step_annotations=step_annotations,
            title=(
                "All-layer prefix-capacity clipping with per-sequence "
                "minimum lengths"
            ),
        )
        summary_output = args.stats_output or args.output.with_name(
            f"{args.output.stem}_summary.csv"
        )
        sequence_output = args.sequence_lengths_output or args.output.with_name(
            f"{args.output.stem}_sequence_lengths.csv"
        )
        layer_output = args.layer_lengths_output or args.output.with_name(
            f"{args.output.stem}_layer_lengths.csv"
        )
        write_stats(summary_output, summary_rows)
        write_stats(sequence_output, sequence_rows)
        write_stats(layer_output, layer_rows)
        print(f"wrote: {summary_output.resolve()}")
        print(f"wrote: {sequence_output.resolve()}")
        print(f"wrote: {layer_output.resolve()}")
    elif args.draft1_drop_layer is None:
        steps, loads_by_step = selected_step_loads(args.trace_dir, args.steps)
        plot(steps, loads_by_step, args.output, plot_layers=plot_layers)
    else:
        steps, loads_by_step, stats_rows = draft1_drop_step_loads(
            args.trace_dir,
            args.steps,
            args.draft1_drop_layer,
        )
        plot(
            steps,
            loads_by_step,
            args.output,
            stage_labels=DRAFT1_DROP_STAGE_LABELS,
            stages=DRAFT1_DROP_STAGES,
            plot_layers=plot_layers,
            title=(
                "Draft1 token drop at "
                f"Layer {args.draft1_drop_layer}, propagated downstream"
            ),
        )
        stats_output = args.stats_output or args.output.with_suffix(".csv")
        write_stats(stats_output, stats_rows)
        print(f"wrote: {stats_output.resolve()}")
    print(f"selected complete steps: {steps}")
    print(f"wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
