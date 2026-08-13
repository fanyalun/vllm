# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cumulative speculative expert loads for selected scheduler steps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
CASCADE_STAGE_LABELS = {
    "target": "target",
    "target_plus_draft_1": "target + surviving draft1",
    "target_plus_draft_1_2": "target + surviving draft1 + draft2",
    "target_plus_draft_1_2_3": "target + surviving draft1 + draft2 + draft3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs=3)
    parser.add_argument("--draft1-cascade-start-layer", type=int)
    parser.add_argument("--stats-output", type=Path)
    return parser.parse_args()


def representative_steps(complete_steps: np.ndarray) -> tuple[int, int, int]:
    return (
        int(complete_steps[0]),
        int(complete_steps[len(complete_steps) // 2]),
        int(complete_steps[-1]),
    )


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
            stage_loads[stage_name] = np.sort(counts, axis=1)[:, ::-1]
        loads_by_step[int(step)] = stage_loads
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


def draft1_cascade_step_loads(
    trace_dir: Path,
    requested_steps: list[int] | None,
    start_layer: int,
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
    if not 0 <= start_layer < num_layers:
        raise ValueError(f"start layer must be in [0, {num_layers - 1}]")
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
            for route_kind in (
                "spec_target",
                "spec_draft_1",
                "spec_draft_2",
                "spec_draft_3",
            )
        }
        for route_kind, routes in routes_by_kind.items():
            if np.any(np.diff(np.sort(routes, axis=2), axis=2) == 0):
                raise ValueError(
                    f"step {step} {route_kind} has duplicate expert IDs"
                )
        target = routes_by_kind["spec_target"]
        draft1 = routes_by_kind["spec_draft_1"]
        draft2 = routes_by_kind["spec_draft_2"]
        draft3 = routes_by_kind["spec_draft_3"]
        surviving_draft1 = np.ones(128, dtype=np.bool_)
        survivor_masks = np.ones((num_layers, 128), dtype=np.bool_)
        capacities = np.empty(num_layers, dtype=np.int64)

        for layer in range(num_layers):
            target_counts = _layer_counts(target[:, layer], num_experts)
            capacity = int(target_counts.max())
            capacities[layer] = capacity
            entering = int(surviving_draft1.sum())
            if layer >= start_layer:
                layer_loads = target_counts.copy()
                for row in np.flatnonzero(surviving_draft1):
                    experts = draft1[row, layer]
                    if np.any(layer_loads[experts] >= capacity):
                        surviving_draft1[row] = False
                    else:
                        layer_loads[experts] += 1
            survivor_masks[layer] = surviving_draft1
            surviving = int(surviving_draft1.sum())
            stats_rows.append(
                {
                    "scheduler_step": int(step),
                    "layer": layer,
                    "max_capacity": capacity,
                    "entering_draft1_tokens": entering,
                    "dropped_draft1_tokens": entering - surviving,
                    "surviving_draft1_tokens": surviving,
                }
            )

        stage_loads = {
            stage_name: np.empty((num_layers, num_experts), dtype=np.int64)
            for stage_name, _ in SPEC_STAGES
        }
        for layer in range(num_layers):
            target_counts = _layer_counts(target[:, layer], num_experts)
            draft1_counts = _layer_counts(
                draft1[survivor_masks[layer], layer], num_experts
            )
            draft2_counts = _layer_counts(draft2[:, layer], num_experts)
            draft3_counts = _layer_counts(draft3[:, layer], num_experts)
            stage_loads["target"][layer] = target_counts
            stage_loads["target_plus_draft_1"][layer] = (
                target_counts + draft1_counts
            )
            stage_loads["target_plus_draft_1_2"][layer] = (
                target_counts + draft1_counts + draft2_counts
            )
            stage_loads["target_plus_draft_1_2_3"][layer] = (
                target_counts + draft1_counts + draft2_counts + draft3_counts
            )
            if layer >= start_layer:
                draft1_peak = stage_loads["target_plus_draft_1"][layer].max()
                if draft1_peak > capacities[layer]:
                    raise AssertionError(
                        f"step {step} layer {layer} exceeds draft1 capacity"
                    )
            expected_draft1_assignments = int(survivor_masks[layer].sum()) * 8
            observed_draft1_assignments = int(draft1_counts.sum())
            if observed_draft1_assignments != expected_draft1_assignments:
                raise AssertionError("surviving draft1 assignment count mismatch")
        loads_by_step[int(step)] = {
            stage_name: np.sort(counts, axis=1)[:, ::-1]
            for stage_name, counts in stage_loads.items()
        }
    return tuple(int(step) for step in steps), loads_by_step, stats_rows


def write_stats(path: Path, rows: list[dict[str, int]]) -> None:
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
    title: str = "Cumulative speculative expert loads for individual complete steps",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 13),
        sharex=True,
        sharey="col",
    )
    expert_order = np.arange(1, 257)
    for row, step in enumerate(steps):
        for column, layer in enumerate(PLOT_LAYERS):
            axis = axes[row, column]
            for (stage_name, _), color in zip(SPEC_STAGES, colors):
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
                axis.set_ylabel(
                    f"Step {step}\nrouted assignments"
                )
            if row == len(steps) - 1:
                axis.set_xlabel("expert position after within-step load sorting")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(title, y=0.965)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.draft1_cascade_start_layer is None:
        steps, loads_by_step = selected_step_loads(args.trace_dir, args.steps)
        plot(steps, loads_by_step, args.output)
    else:
        steps, loads_by_step, stats_rows = draft1_cascade_step_loads(
            args.trace_dir,
            args.steps,
            args.draft1_cascade_start_layer,
        )
        plot(
            steps,
            loads_by_step,
            args.output,
            stage_labels=CASCADE_STAGE_LABELS,
            title=(
                "Draft1 token-level cascade clipping from "
                f"Layer {args.draft1_cascade_start_layer}"
            ),
        )
        stats_output = args.stats_output or args.output.with_suffix(".csv")
        write_stats(stats_output, stats_rows)
        print(f"wrote: {stats_output.resolve()}")
    print(f"selected complete steps: {steps}")
    print(f"wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
