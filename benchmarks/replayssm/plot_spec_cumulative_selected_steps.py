# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cumulative speculative expert loads for selected scheduler steps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.replayssm.expert_routing_analysis import (
    PLOT_LAYERS,
    SPEC_STAGES,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs=3)
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


def plot(
    steps: tuple[int, int, int],
    loads_by_step: dict[int, dict[str, np.ndarray]],
    output: Path,
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
                    label=STAGE_LABELS[stage_name],
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
    figure.suptitle(
        "Cumulative speculative expert loads for individual complete steps",
        y=0.965,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    steps, loads_by_step = selected_step_loads(args.trace_dir, args.steps)
    plot(steps, loads_by_step, args.output)
    print(f"selected complete steps: {steps}")
    print(f"wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
