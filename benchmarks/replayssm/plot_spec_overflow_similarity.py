# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cross-layer similarity of capacity-clipped speculative tokens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.replayssm.expert_routing_analysis import (
    complete_step_ids,
    load_trace,
)

SPEC_DRAFT_STAGES = (
    ("target_plus_draft_1", "spec_draft_1"),
    ("target_plus_draft_1_2", "spec_draft_2"),
    ("target_plus_draft_1_2_3", "spec_draft_3"),
)
STAGE_LABELS = {
    "target_plus_draft_1": "target + draft1",
    "target_plus_draft_1_2": "target + draft1 + draft2",
    "target_plus_draft_1_2_3": "target + draft1 + draft2 + draft3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def expert_counts(routes: np.ndarray, num_experts: int) -> np.ndarray:
    counts = np.zeros((routes.shape[1], num_experts), dtype=np.int32)
    for layer in range(routes.shape[1]):
        counts[layer] = np.bincount(
            routes[:, layer].reshape(-1), minlength=num_experts
        )
    return counts


def pairwise_jaccard(hits: np.ndarray) -> np.ndarray:
    integer_hits = hits.astype(np.int64)
    intersections = integer_hits.T @ integer_hits
    totals = integer_hits.sum(axis=0)
    unions = totals[:, None] + totals[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.full_like(intersections, np.nan, dtype=np.float64),
        where=unions != 0,
    )


def add_stage_with_clipping(
    routes: np.ndarray,
    cumulative_loads: np.ndarray,
    capacities: np.ndarray,
    cumulative_overflow_hits: np.ndarray,
    row_indices: np.ndarray,
) -> np.ndarray:
    num_layers = routes.shape[1]
    overflow_assignments = np.zeros(num_layers, dtype=np.int32)
    for layer in range(num_layers):
        layer_routes = routes[:, layer]
        for expert in np.unique(layer_routes):
            routed_rows = np.flatnonzero(np.any(layer_routes == expert, axis=1))
            available = max(
                int(capacities[layer] - cumulative_loads[layer, expert]), 0
            )
            overflow_rows = routed_rows[available:]
            if overflow_rows.size:
                cumulative_overflow_hits[row_indices[overflow_rows], layer] = True
                overflow_assignments[layer] += overflow_rows.size
            cumulative_loads[layer, expert] += routed_rows.size
    return overflow_assignments


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    header = ["layer"] + [str(layer) for layer in range(matrix.shape[1])]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        for layer, row in enumerate(matrix):
            writer.writerow([layer, *row])


def plot_similarity(
    matrices: dict[str, np.ndarray],
    stage_summaries: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_matrices = {}
    for stage_name, matrix in matrices.items():
        display_matrix = matrix.copy()
        np.fill_diagonal(display_matrix, np.nan)
        display_matrices[stage_name] = display_matrix

    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("lightgray")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    image = None
    for axis, (stage_name, matrix) in zip(axes, display_matrices.items()):
        summary = stage_summaries[stage_name]
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap=color_map,
            vmin=0,
            vmax=1,
        )
        axis.set_xlabel("layer")
        axis.set_title(
            f"{STAGE_LABELS[stage_name]}\n"
            f"mean off-diagonal Jaccard={summary['mean_jaccard']:.4f}"
        )
    axes[0].set_ylabel("layer")
    figure.subplots_adjust(
        left=0.06, right=0.88, bottom=0.12, top=0.86, wspace=0.2
    )
    colorbar_axis = figure.add_axes((0.9, 0.15, 0.015, 0.65))
    figure.colorbar(
        image,
        cax=colorbar_axis,
        label="mean per-step overflow-token Jaccard",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze(trace_dir: Path, output_root: Path) -> None:
    trace = load_trace(trace_dir)
    if trace.name != "replayssm_spec_bs128_d3":
        raise ValueError(f"expected Spec BS128 D3 trace, got {trace.name}")

    steps = complete_step_ids(trace)
    if not steps.size:
        raise ValueError("trace has no complete speculative steps")
    num_layers = trace.routes.shape[1]
    num_experts = int(trace.manifest["num_experts"])
    matrices_by_stage: dict[str, list[np.ndarray]] = {
        stage_name: [] for stage_name, _ in SPEC_DRAFT_STAGES
    }
    detail_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for step in steps:
        step_indices = np.flatnonzero(trace.scheduler_steps == step)
        target_local = np.flatnonzero(
            trace.route_kinds[step_indices] == "spec_target"
        )
        target_routes = trace.routes[step_indices[target_local]]
        if target_routes.shape[0] != 128:
            raise ValueError(
                f"step {step} spec_target has {target_routes.shape[0]} rows"
            )
        if np.any(np.diff(np.sort(target_routes, axis=2), axis=2) == 0):
            raise ValueError(f"step {step} spec_target has duplicate experts")
        cumulative_loads = expert_counts(target_routes, num_experts)
        capacities = cumulative_loads.max(axis=1)
        cumulative_overflow_hits = np.zeros(
            (step_indices.size, num_layers), dtype=np.bool_
        )
        cumulative_overflow_assignments = np.zeros(num_layers, dtype=np.int32)

        for stage_name, route_kind in SPEC_DRAFT_STAGES:
            stage_local = np.flatnonzero(
                trace.route_kinds[step_indices] == route_kind
            )
            stage_routes = trace.routes[step_indices[stage_local]]
            if stage_routes.shape[0] != 128:
                raise ValueError(
                    f"step {step} {route_kind} has {stage_routes.shape[0]} rows"
                )
            sorted_routes = np.sort(stage_routes, axis=2)
            if np.any(np.diff(sorted_routes, axis=2) == 0):
                raise ValueError(f"step {step} {route_kind} has duplicate experts")
            cumulative_overflow_assignments += add_stage_with_clipping(
                stage_routes,
                cumulative_loads,
                capacities,
                cumulative_overflow_hits,
                stage_local,
            )
            expected_overflow = np.maximum(
                cumulative_loads - capacities[:, None], 0
            ).sum(axis=1)
            if not np.array_equal(
                cumulative_overflow_assignments, expected_overflow
            ):
                raise AssertionError("clipped assignments do not match curve area")
            jaccard = pairwise_jaccard(cumulative_overflow_hits)
            matrices_by_stage[stage_name].append(jaccard)
            integer_hits = cumulative_overflow_hits.astype(np.int64)
            intersections = integer_hits.T @ integer_hits
            totals = integer_hits.sum(axis=0)
            for first_layer in range(num_layers):
                for second_layer in range(first_layer + 1, num_layers):
                    intersection = int(
                        intersections[first_layer, second_layer]
                    )
                    union = int(
                        totals[first_layer]
                        + totals[second_layer]
                        - intersection
                    )
                    pair_rows.append(
                        {
                            "scheduler_step": int(step),
                            "stage": stage_name,
                            "first_layer": first_layer,
                            "second_layer": second_layer,
                            "intersection": intersection,
                            "union": union,
                            "jaccard": (
                                intersection / union if union else ""
                            ),
                        }
                    )
            for layer in range(num_layers):
                detail_rows.append(
                    {
                        "scheduler_step": int(step),
                        "stage": stage_name,
                        "layer": layer,
                        "max_capacity": int(capacities[layer]),
                        "cumulative_assignments": int(
                            cumulative_loads[layer].sum()
                        ),
                        "overflow_assignments": int(
                            cumulative_overflow_assignments[layer]
                        ),
                        "overflow_tokens": int(
                            cumulative_overflow_hits[:, layer].sum()
                        ),
                    }
                )

    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mean_matrices = {
        stage_name: np.nanmean(np.stack(matrices), axis=0)
        for stage_name, matrices in matrices_by_stage.items()
    }
    stage_summaries = {}
    off_diagonal = ~np.eye(num_layers, dtype=np.bool_)
    for stage_name, matrix in mean_matrices.items():
        stage_rows = [row for row in detail_rows if row["stage"] == stage_name]
        stage_summaries[stage_name] = {
            "mean_jaccard": float(np.nanmean(matrix[off_diagonal])),
            "mean_overflow_assignments_per_layer_step": float(
                np.mean([row["overflow_assignments"] for row in stage_rows])
            ),
            "mean_overflow_tokens_per_layer_step": float(
                np.mean([row["overflow_tokens"] for row in stage_rows])
            ),
        }
        write_matrix(
            metrics_dir / f"spec_overflow_jaccard_{stage_name}.csv", matrix
        )

    write_csv(metrics_dir / "spec_overflow_step_layer.csv", detail_rows)
    write_csv(metrics_dir / "spec_overflow_step_pair_jaccard.csv", pair_rows)
    summary = {
        "definition": (
            "Within each complete Spec BS128 D3 step and layer, max_capacity "
            "is the maximum target-only load over all 256 global experts. "
            "Target rows occupy capacity first; draft rows are added in "
            "draft1, draft2, draft3 and trace-row order. Assignments beyond "
            "capacity are clipped. A token is an overflow token for a layer "
            "if at least one of its Top-8 assignments is clipped. Pairwise "
            "layer Jaccard is computed per step and then averaged over steps; "
            "layer pairs with an empty union in a step are excluded."
        ),
        "trace_dir": str(trace_dir.resolve()),
        "complete_steps": int(steps.size),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": int(trace.manifest["top_k"]),
        "stages": stage_summaries,
    }
    (metrics_dir / "spec_overflow_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_similarity(
        mean_matrices,
        stage_summaries,
        figures_dir / "spec_overflow_layer_similarity.png",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    analyze(args.trace_dir, args.output_root)


if __name__ == "__main__":
    main()
