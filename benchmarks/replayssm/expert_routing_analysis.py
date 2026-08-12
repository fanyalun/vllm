# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROUTE_KINDS = (
    "ar_decode",
    "spec_target",
    "spec_draft_1",
    "spec_draft_2",
    "spec_draft_3",
)
SPEC_STAGES = (
    ("target", ("spec_target",)),
    ("target_plus_draft_1", ("spec_target", "spec_draft_1")),
    (
        "target_plus_draft_1_2",
        ("spec_target", "spec_draft_1", "spec_draft_2"),
    ),
    (
        "target_plus_draft_1_2_3",
        ("spec_target", "spec_draft_1", "spec_draft_2", "spec_draft_3"),
    ),
)
INTERNAL_REQUEST_ID = re.compile(r"^(?P<external>.*)-[0-9a-f]{8}$")


@dataclass
class TraceData:
    name: str
    routes: np.memmap
    request_ids: np.ndarray
    scheduler_steps: np.ndarray
    absolute_positions: np.ndarray
    token_ids: np.ndarray
    route_kinds: np.ndarray
    accepted: np.ndarray
    question_indices: np.ndarray
    manifest: dict[str, Any]

    @property
    def num_rows(self) -> int:
        return int(self.routes.shape[0])


def load_trace(trace_dir: Path) -> TraceData:
    manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
    if manifest["state"] != "complete":
        raise ValueError(f"trace is not complete: {trace_dir}")
    shape = tuple(manifest["route_shape"])
    routes = np.memmap(
        trace_dir / manifest["routes_file"],
        dtype=np.dtype(manifest["route_dtype"]),
        mode="r",
        shape=shape,
    )
    num_rows = shape[0]
    request_ids = np.empty(num_rows, dtype=object)
    scheduler_steps = np.empty(num_rows, dtype=np.int32)
    absolute_positions = np.empty(num_rows, dtype=np.int32)
    token_ids = np.empty(num_rows, dtype=np.int32)
    route_kinds = np.empty(num_rows, dtype="U16")
    accepted = np.empty(num_rows, dtype=np.bool_)
    seen_rows = 0
    with (trace_dir / manifest["events_file"]).open(encoding="utf-8") as events:
        for line in events:
            event = json.loads(line)
            start = int(event["binary_row_offset"])
            count = int(event["row_count"])
            end = start + count
            if start != seen_rows:
                raise ValueError(f"non-contiguous trace event at row {start}")
            request_ids[start:end] = event["request_id"]
            scheduler_steps[start:end] = event["scheduler_step"]
            absolute_positions[start:end] = event["absolute_positions"]
            token_ids[start:end] = event["token_ids"]
            route_kinds[start:end] = event["route_kinds"]
            accepted[start:end] = event["accepted"]
            seen_rows = end
    if seen_rows != num_rows:
        raise ValueError(f"events cover {seen_rows} rows, expected {num_rows}")

    output_map: dict[str, int] = {}
    outputs_path = trace_dir / "outputs.jsonl"
    with outputs_path.open(encoding="utf-8") as outputs:
        for line in outputs:
            row = json.loads(line)
            output_map[row["request_id"]] = int(row["question_index"])
    question_indices = np.array(
        [
            output_map[external_request_id(str(request_id), output_map)]
            for request_id in request_ids
        ],
        dtype=np.int32,
    )
    return TraceData(
        name=trace_dir.name,
        routes=routes,
        request_ids=request_ids,
        scheduler_steps=scheduler_steps,
        absolute_positions=absolute_positions,
        token_ids=token_ids,
        route_kinds=route_kinds,
        accepted=accepted,
        question_indices=question_indices,
        manifest=manifest,
    )


def external_request_id(request_id: str, output_map: dict[str, int]) -> str:
    if request_id in output_map:
        return request_id
    match = INTERNAL_REQUEST_ID.fullmatch(request_id)
    if match is not None and match["external"] in output_map:
        return match["external"]
    raise KeyError(f"no output mapping for internal request ID {request_id!r}")


def load_outputs(trace_dir: Path) -> dict[int, list[int]]:
    outputs: dict[int, list[int]] = {}
    with (trace_dir / "outputs.jsonl").open(encoding="utf-8") as output_file:
        for line in output_file:
            row = json.loads(line)
            outputs[int(row["question_index"])] = row["token_ids"]
    return outputs


def compare_outputs(
    first: dict[int, list[int]], second: dict[int, list[int]], limit: int | None = None
) -> dict[str, Any]:
    common = sorted(set(first) & set(second))
    if limit is not None:
        common = [index for index in common if index < limit]
    matching: list[int] = []
    prefix_lengths: dict[int, int] = {}
    differences: list[dict[str, int]] = []
    for question_index in common:
        first_tokens = first[question_index]
        second_tokens = second[question_index]
        prefix = next(
            (
                index
                for index, (first_token, second_token) in enumerate(
                    zip(first_tokens, second_tokens)
                )
                if first_token != second_token
            ),
            min(len(first_tokens), len(second_tokens)),
        )
        prefix_lengths[question_index] = prefix
        if first_tokens == second_tokens:
            matching.append(question_index)
        elif len(differences) < 50:
            differences.append(
                {
                    "question_index": question_index,
                    "first_difference": prefix,
                    "first_token": first_tokens[prefix],
                    "second_token": second_tokens[prefix],
                }
            )
    return {
        "compared_requests": len(common),
        "matching_requests": len(matching),
        "mismatching_requests": len(common) - len(matching),
        "matching_question_indices": matching,
        "matching_prefix_lengths": prefix_lengths,
        "first_differences": differences,
    }


def expert_counts(
    routes: np.ndarray,
    mask: np.ndarray | None = None,
    num_experts: int = 256,
) -> np.ndarray:
    selected = routes if mask is None else routes[mask]
    counts = np.zeros((routes.shape[1], num_experts), dtype=np.int64)
    for layer in range(routes.shape[1]):
        counts[layer] = np.bincount(
            np.asarray(selected[:, layer, :]).reshape(-1), minlength=num_experts
        )
    return counts


def gini(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    total = array.sum()
    if total == 0:
        return 0.0
    array.sort()
    indices = np.arange(1, array.size + 1, dtype=np.float64)
    return float(2 * np.dot(indices, array) / (array.size * total)) - (
        array.size + 1
    ) / array.size


def gini_rows(values: np.ndarray) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    totals = array.sum(axis=1)
    indices = np.arange(1, array.shape[1] + 1, dtype=np.float64)
    result = 2 * (array @ indices) / (array.shape[1] * totals)
    result -= (array.shape[1] + 1) / array.shape[1]
    result[totals == 0] = 0
    return result


def distribution_metrics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(array.sum())
    mean = float(array.mean())
    active = int(np.count_nonzero(array))
    if total == 0:
        return {
            "assignments": 0,
            "active_experts": active,
            "gini": 0.0,
            "cv": 0.0,
            "max_over_mean": 0.0,
            "normalized_entropy": 0.0,
            "hot_10pct_share": 0.0,
        }
    probabilities = array[array > 0] / total
    entropy = -float(np.dot(probabilities, np.log(probabilities)))
    hot_count = math.ceil(array.size * 0.1)
    return {
        "assignments": int(total),
        "active_experts": active,
        "gini": gini(array),
        "cv": float(array.std() / mean) if mean else 0.0,
        "max_over_mean": float(array.max() / mean) if mean else 0.0,
        "normalized_entropy": entropy / math.log(array.size),
        "hot_10pct_share": float(np.partition(array, -hot_count)[-hot_count:].sum())
        / total,
    }


def select_hot_experts(counts: np.ndarray) -> np.ndarray:
    hot_count = math.ceil(counts.shape[1] * 0.1)
    hot = np.empty((counts.shape[0], hot_count), dtype=np.int16)
    expert_ids = np.arange(counts.shape[1])
    for layer, layer_counts in enumerate(counts):
        hot[layer] = np.lexsort((expert_ids, -layer_counts))[:hot_count]
    return hot


def hot_hits(routes: np.ndarray, hot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hits = np.zeros((routes.shape[0], routes.shape[1]), dtype=np.bool_)
    assignment_counts = np.zeros_like(hits, dtype=np.uint8)
    for layer in range(routes.shape[1]):
        routed_hot = np.isin(routes[:, layer, :], hot[layer])
        hits[:, layer] = routed_hot.any(axis=1)
        assignment_counts[:, layer] = routed_hot.sum(axis=1)
    return hits, assignment_counts


def pairwise_hot_metrics(hits: np.ndarray) -> dict[str, np.ndarray]:
    integer_hits = hits.astype(np.int64)
    intersections = integer_hits.T @ integer_hits
    layer_totals = integer_hits.sum(axis=0)
    unions = layer_totals[:, None] + layer_totals[None, :] - intersections
    minimums = np.minimum(layer_totals[:, None], layer_totals[None, :])
    count = max(hits.shape[0], 1)
    probabilities = layer_totals / count
    joint = intersections / count
    independent = probabilities[:, None] * probabilities[None, :]
    denominator = np.sqrt(
        probabilities[:, None]
        * (1 - probabilities[:, None])
        * probabilities[None, :]
        * (1 - probabilities[None, :])
    )
    return {
        "intersection": intersections,
        "jaccard": np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=unions != 0,
        ),
        "overlap": np.divide(
            intersections,
            minimums,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=minimums != 0,
        ),
        "lift": np.divide(
            joint,
            independent,
            out=np.zeros_like(joint, dtype=np.float64),
            where=independent != 0,
        ),
        "phi": np.divide(
            joint - independent,
            denominator,
            out=np.zeros_like(joint, dtype=np.float64),
            where=denominator != 0,
        ),
    }


def pairwise_relation_rows(
    run_name: str,
    slice_name: str,
    matrices: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_layers = next(iter(matrices.values())).shape[0]
    for distance in range(1, num_layers):
        indices = (np.arange(num_layers - distance), np.arange(distance, num_layers))
        for metric_name in ("jaccard", "overlap", "lift", "phi"):
            rows.append(
                {
                    "run": run_name,
                    "slice": slice_name,
                    "relation": "layer_distance",
                    "group": str(distance),
                    "metric": metric_name,
                    "mean": float(matrices[metric_name][indices].mean()),
                    "pairs": num_layers - distance,
                }
            )
    layer_types = np.array(
        [
            "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
            for layer in range(num_layers)
        ]
    )
    pair_groups: dict[str, list[tuple[int, int]]] = {}
    for first in range(num_layers):
        for second in range(first + 1, num_layers):
            group = "+".join(sorted((layer_types[first], layer_types[second])))
            pair_groups.setdefault(group, []).append((first, second))
    for group, pairs in pair_groups.items():
        first_indices, second_indices = zip(*pairs)
        for metric_name in ("jaccard", "overlap", "lift", "phi"):
            values = matrices[metric_name][first_indices, second_indices]
            rows.append(
                {
                    "run": run_name,
                    "slice": slice_name,
                    "relation": "attention_layer_types",
                    "group": group,
                    "metric": metric_name,
                    "mean": float(values.mean()),
                    "pairs": len(pairs),
                }
            )
    return rows


def poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    distribution = np.array([1.0])
    for probability in probabilities:
        distribution = np.convolve(distribution, [1 - probability, probability])
    return distribution


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    rows = []
    for layer, values in enumerate(matrix):
        row: dict[str, Any] = {"layer": layer}
        row.update({f"layer_{index}": value for index, value in enumerate(values)})
        rows.append(row)
    write_csv(path, rows)


def _slice_masks(trace: TraceData) -> dict[str, np.ndarray]:
    masks = {"all_executed": np.ones(trace.num_rows, dtype=np.bool_)}
    masks["accepted_only"] = trace.accepted.copy()
    if np.count_nonzero(~trace.accepted):
        masks["rejected_only"] = ~trace.accepted
    for route_kind in ROUTE_KINDS:
        mask = trace.route_kinds == route_kind
        if mask.any():
            masks[route_kind] = mask
    return masks


def token_id_hot_coverage_rows(
    trace: TraceData, hits: np.ndarray, mask: np.ndarray, slice_name: str
) -> list[dict[str, Any]]:
    selected_indices = np.flatnonzero(mask)
    if not selected_indices.size:
        return []
    order = np.argsort(trace.token_ids[selected_indices], kind="stable")
    sorted_indices = selected_indices[order]
    sorted_token_ids = trace.token_ids[sorted_indices]
    boundaries = np.flatnonzero(np.diff(sorted_token_ids)) + 1
    groups = np.split(sorted_indices, boundaries)
    return [
        {
            "run": trace.name,
            "slice": slice_name,
            "token_id": int(trace.token_ids[group[0]]),
            "occurrences": int(group.size),
            "mean_hot_layers": float(hits[group].sum(axis=1).mean()),
            "strict_all_40_occurrences": int(hits[group].all(axis=1).sum()),
        }
        for group in groups
    ]


def question_mask(trace: TraceData, question_indices: list[int]) -> np.ndarray:
    return np.isin(trace.question_indices, question_indices)


def matching_prefix_mask(
    trace: TraceData, prefix_lengths: dict[int, int]
) -> np.ndarray:
    prompt_starts = {
        int(question): int(
            trace.absolute_positions[trace.question_indices == question].min()
        )
        for question in np.unique(trace.question_indices)
    }
    limits = np.array(
        [
            prompt_starts[int(question)] + prefix_lengths.get(int(question), 0)
            for question in trace.question_indices
        ],
        dtype=np.int32,
    )
    return trace.absolute_positions < limits


def _request_count_matrix(
    trace: TraceData, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    question_indices = np.unique(trace.question_indices[mask])
    matrices = np.zeros(
        (
            question_indices.size,
            trace.routes.shape[1] * trace.manifest["num_experts"],
        ),
        dtype=np.int32,
    )
    for index, question_index in enumerate(question_indices):
        request_mask = mask & (trace.question_indices == question_index)
        matrices[index] = expert_counts(trace.routes, request_mask).reshape(-1)
    return question_indices, matrices


def paired_request_bootstrap(
    first_trace: TraceData,
    first_mask: np.ndarray,
    second_trace: TraceData,
    second_mask: np.ndarray,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    first_questions, first_counts = _request_count_matrix(first_trace, first_mask)
    second_questions, second_counts = _request_count_matrix(second_trace, second_mask)
    common = np.intersect1d(first_questions, second_questions)
    first_lookup = {value: index for index, value in enumerate(first_questions)}
    second_lookup = {value: index for index, value in enumerate(second_questions)}
    first_counts = first_counts[[first_lookup[value] for value in common]]
    second_counts = second_counts[[second_lookup[value] for value in common]]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        common.size, np.full(common.size, 1 / common.size), size=samples
    )
    differences = np.empty(samples, dtype=np.float64)
    chunk_size = 25
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        first_gini = gini_rows(weights[start:end] @ first_counts)
        second_gini = gini_rows(weights[start:end] @ second_counts)
        differences[start:end] = first_gini - second_gini
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": float(
            gini(first_counts.sum(axis=0)) - gini(second_counts.sum(axis=0))
        ),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": samples,
        "paired_requests": int(common.size),
    }


def _full_step_masks(trace: TraceData) -> tuple[np.ndarray, np.ndarray]:
    selected_steps: list[int] = []
    for step in np.unique(trace.scheduler_steps):
        mask = trace.scheduler_steps == step
        if int(mask.sum()) != 512:
            continue
        if "spec" in trace.name:
            counts = {
                kind: int(np.count_nonzero(trace.route_kinds[mask] == kind))
                for kind in ROUTE_KINDS[1:]
            }
            if any(count != 128 for count in counts.values()):
                continue
        selected_steps.append(int(step))
    row_mask = np.isin(trace.scheduler_steps, selected_steps)
    return np.array(selected_steps, dtype=np.int32), row_mask


def block_step_bootstrap(
    first_trace: TraceData,
    second_trace: TraceData,
    samples: int = 1000,
    block_length: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    first_steps, first_mask = _full_step_masks(first_trace)
    second_steps, second_mask = _full_step_masks(second_trace)
    if not first_steps.size or not second_steps.size:
        raise ValueError("equal-token comparison has no complete 512-row steps")

    def step_counts(trace: TraceData, steps: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                expert_counts(trace.routes, trace.scheduler_steps == step).reshape(-1)
                for step in steps
            ]
        )

    first_counts = step_counts(first_trace, first_steps)
    second_counts = step_counts(second_trace, second_steps)
    rng = np.random.default_rng(seed)

    def one_sample(counts: np.ndarray) -> np.ndarray:
        count = counts.shape[0]
        needed_blocks = math.ceil(count / block_length)
        starts = rng.integers(0, count, size=needed_blocks)
        indices = np.concatenate(
            [
                (np.arange(start, start + block_length) % count)
                for start in starts
            ]
        )[:count]
        return counts[indices].sum(axis=0)

    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        differences[index] = gini(one_sample(first_counts)) - gini(
            one_sample(second_counts)
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": gini(expert_counts(first_trace.routes, first_mask))
        - gini(expert_counts(second_trace.routes, second_mask)),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": samples,
        "block_length": block_length,
        "first_full_steps": int(first_steps.size),
        "second_full_steps": int(second_steps.size),
    }


def request_phi_bootstrap(
    trace: TraceData,
    hits: np.ndarray,
    mask: np.ndarray,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    questions = np.unique(trace.question_indices[mask])
    num_layers = hits.shape[1]
    row_counts = np.zeros(questions.size, dtype=np.int64)
    layer_hits = np.zeros((questions.size, num_layers), dtype=np.int64)
    intersections = np.zeros(
        (questions.size, num_layers, num_layers), dtype=np.int64
    )
    for index, question in enumerate(questions):
        request_mask = mask & (trace.question_indices == question)
        request_hits = hits[request_mask].astype(np.int64)
        row_counts[index] = request_hits.shape[0]
        layer_hits[index] = request_hits.sum(axis=0)
        intersections[index] = request_hits.T @ request_hits
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        questions.size, np.full(questions.size, 1 / questions.size), size=samples
    )
    off_diagonal = ~np.eye(num_layers, dtype=np.bool_)
    means = np.empty(samples, dtype=np.float64)
    for index, weight in enumerate(weights):
        count = int(weight @ row_counts)
        totals = weight @ layer_hits
        joint_counts = np.tensordot(weight, intersections, axes=(0, 0))
        probabilities = totals / count
        joint = joint_counts / count
        independent = probabilities[:, None] * probabilities[None, :]
        denominator = np.sqrt(
            probabilities[:, None]
            * (1 - probabilities[:, None])
            * probabilities[None, :]
            * (1 - probabilities[None, :])
        )
        phi = np.divide(
            joint - independent,
            denominator,
            out=np.zeros_like(joint),
            where=denominator != 0,
        )
        means[index] = phi[off_diagonal].mean()
    observed = pairwise_hot_metrics(hits[mask])["phi"][off_diagonal].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_off_diagonal_phi": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": samples,
        "requests": int(questions.size),
    }


def _plot_spec_stages(
    stage_counts: list[tuple[str, np.ndarray]], figures_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(5, 8, figsize=(24, 14), sharex=True)
    expert_ids = np.arange(stage_counts[0][1].shape[1])
    for layer, axis in enumerate(axes.flat):
        for (label, counts), color in zip(stage_counts, colors):
            axis.plot(
                expert_ids,
                counts[layer],
                label=label,
                color=color,
                linewidth=0.8,
            )
        axis.set_title(f"layer {layer}", fontsize=8)
        axis.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    figure.supxlabel("logical expert id")
    figure.supylabel("cumulative routed assignments")
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(figures_dir / f"spec_cumulative_by_layer.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(5, 8, figsize=(24, 14), sharex=True, sharey=True)
    for layer, axis in enumerate(axes.flat):
        for (label, counts), color in zip(stage_counts, colors):
            values = counts[layer] / max(counts[layer].mean(), 1)
            axis.plot(
                np.arange(values.size),
                np.sort(values)[::-1],
                label=label,
                color=color,
                linewidth=0.8,
            )
        axis.set_title(f"layer {layer}", fontsize=8)
        axis.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    figure.supxlabel("expert load rank")
    figure.supylabel("load / layer mean")
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(
            figures_dir / f"spec_cumulative_normalized_by_layer.{suffix}", dpi=180
        )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for (label, counts), color in zip(stage_counts, colors):
        values = counts.reshape(-1) / max(counts.mean(), 1)
        axis.plot(
            np.arange(values.size),
            np.sort(values)[::-1],
            label=label,
            color=color,
        )
    axis.set_xlabel("layer-expert load rank")
    axis.set_ylabel("load / global layer-expert mean")
    axis.legend()
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(figures_dir / f"spec_cumulative_global.{suffix}", dpi=180)
    plt.close(figure)


def _plot_run_comparison(
    counts_by_run: dict[str, np.ndarray], figures_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    for name, counts in counts_by_run.items():
        values = counts.reshape(-1) / max(counts.mean(), 1)
        axis.plot(np.sort(values)[::-1], label=name)
    axis.set_xlabel("layer-expert load rank")
    axis.set_ylabel("load / global layer-expert mean")
    axis.legend()
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(figures_dir / f"global_load_comparison.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for axis, (name, counts) in zip(axes, counts_by_run.items()):
        normalized = counts / np.maximum(counts.mean(axis=1, keepdims=True), 1)
        image = axis.imshow(normalized, aspect="auto", cmap="magma")
        axis.set_ylabel("layer")
        axis.set_title(name)
        figure.colorbar(image, ax=axis, label="load / layer mean")
    axes[-1].set_xlabel("logical expert id")
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(figures_dir / f"expert_load_heatmaps.{suffix}", dpi=180)
    plt.close(figure)


def _plot_layer_stage_metrics(
    stage_counts: list[tuple[str, np.ndarray]], figures_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("gini", "cv", "hot_10pct_share")
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for metric, axis in zip(metrics, axes):
        for label, counts in stage_counts:
            values = [distribution_metrics(row)[metric] for row in counts]
            axis.plot(values, label=label)
        axis.set_ylabel(metric)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("layer")
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(figures_dir / f"spec_cumulative_layer_metrics.{suffix}", dpi=180)
    plt.close(figure)


def _plot_hot_matrix(matrix: np.ndarray, name: str, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, origin="lower", cmap="viridis")
    axis.set_xlabel("layer")
    axis.set_ylabel("layer")
    axis.set_title(name)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(figures_dir / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def analyze_experiment(root: Path, bootstrap_samples: int = 1000) -> None:
    trace_root = root / "trace"
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_names = (
        "replayssm_ar_bs128",
        "replayssm_spec_bs128_d3",
        "replayssm_ar_bs512",
    )
    traces = {name: load_trace(trace_root / name) for name in run_names}
    counts_by_run: dict[str, np.ndarray] = {}
    expert_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    hot_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    hit_layer_histogram_rows: list[dict[str, Any]] = []
    hot_assignment_histogram_rows: list[dict[str, Any]] = []
    layer_relation_rows: list[dict[str, Any]] = []
    vocabulary_rows: list[dict[str, Any]] = []
    global_metrics: dict[str, Any] = {}
    phi_bootstrap: dict[str, dict[str, Any]] = {}

    spec_stage_counts: list[tuple[str, np.ndarray]] = []
    spec_trace = traces["replayssm_spec_bs128_d3"]
    for stage_name, kinds in SPEC_STAGES:
        stage_mask = np.isin(spec_trace.route_kinds, kinds)
        spec_stage_counts.append(
            (stage_name, expert_counts(spec_trace.routes, stage_mask))
        )

    for run_name, trace in traces.items():
        masks = _slice_masks(trace)
        all_counts = expert_counts(trace.routes)
        counts_by_run[run_name] = all_counts
        hot = select_hot_experts(all_counts)
        hits, hot_assignment_counts = hot_hits(trace.routes, hot)
        global_metrics[run_name] = {
            "all_executed": distribution_metrics(all_counts),
            "route_rows": trace.num_rows,
        }

        for layer, experts in enumerate(hot):
            for hot_rank, expert in enumerate(experts, start=1):
                hot_rows.append(
                    {
                        "run": run_name,
                        "layer": layer,
                        "hot_rank": hot_rank,
                        "expert": int(expert),
                        "assignments": int(all_counts[layer, expert]),
                    }
                )
        for layer, counts in enumerate(all_counts):
            total = counts.sum()
            for expert, count in enumerate(counts):
                expert_rows.append(
                    {
                        "run": run_name,
                        "slice": "all_executed",
                        "layer": layer,
                        "expert": expert,
                        "assignments": int(count),
                        "share": float(count / total),
                        "normalized_load": float(count / counts.mean()),
                    }
                )

        for slice_name, mask in masks.items():
            counts = expert_counts(trace.routes, mask)
            global_metrics[run_name][slice_name] = distribution_metrics(counts)
            for layer, layer_counts in enumerate(counts):
                row = {"run": run_name, "slice": slice_name, "layer": layer}
                row.update(distribution_metrics(layer_counts))
                layer_rows.append(row)

            selected_hits = hits[mask]
            selected_assignments = hot_assignment_counts[mask]
            pairwise = pairwise_hot_metrics(selected_hits)
            hit_layers = selected_hits.sum(axis=1)
            probabilities = selected_hits.mean(axis=0)
            null_distribution = poisson_binomial(probabilities)
            observed_distribution = np.bincount(hit_layers, minlength=41)
            profile = {
                "run": run_name,
                "slice": slice_name,
                "tokens": int(mask.sum()),
                "strict_all_40": int(observed_distribution[40]),
                "strict_all_40_share": float(observed_distribution[40] / mask.sum())
                if mask.sum()
                else 0.0,
                "expected_all_40": float(mask.sum() * null_distribution[40]),
                "mean_hot_assignments_per_layer": float(selected_assignments.mean())
                if mask.sum()
                else 0.0,
            }
            for threshold in (10, 20, 30, 36, 40):
                observed = int(observed_distribution[threshold:].sum())
                expected = float(mask.sum() * null_distribution[threshold:].sum())
                profile[f"tokens_k_ge_{threshold}"] = observed
                profile[f"share_k_ge_{threshold}"] = (
                    observed / mask.sum() if mask.sum() else 0
                )
                profile[f"expected_k_ge_{threshold}"] = expected
                profile[f"observed_over_expected_k_ge_{threshold}"] = (
                    observed / expected if expected else 0
                )
            coverage_rows.append(profile)
            for hit_count in range(41):
                hit_layer_histogram_rows.append(
                    {
                        "run": run_name,
                        "slice": slice_name,
                        "hot_hit_layers": hit_count,
                        "tokens": int(observed_distribution[hit_count]),
                        "share": float(observed_distribution[hit_count] / mask.sum()),
                        "independent_layer_expected_tokens": float(
                            mask.sum() * null_distribution[hit_count]
                        ),
                    }
                )
            assignment_histogram = np.bincount(
                selected_assignments.reshape(-1), minlength=9
            )
            for assignment_count in range(9):
                hot_assignment_histogram_rows.append(
                    {
                        "run": run_name,
                        "slice": slice_name,
                        "hot_assignments_in_top8": assignment_count,
                        "layer_token_occurrences": int(
                            assignment_histogram[assignment_count]
                        ),
                        "share": float(
                            assignment_histogram[assignment_count]
                            / assignment_histogram.sum()
                        ),
                    }
                )

            file_stem = f"{run_name}_{slice_name}"
            for metric_name, matrix in pairwise.items():
                write_matrix(
                    metrics_dir / f"pairwise_{metric_name}_{file_stem}.csv",
                    matrix,
                )
            layer_relation_rows.extend(
                pairwise_relation_rows(run_name, slice_name, pairwise)
            )
            phi_bootstrap.setdefault(run_name, {})[slice_name] = (
                request_phi_bootstrap(
                    trace, hits, mask, samples=bootstrap_samples
                )
            )
            if slice_name == "all_executed":
                _plot_hot_matrix(pairwise["phi"], f"hot_phi_{run_name}", figures_dir)

            vocabulary_rows.extend(
                token_id_hot_coverage_rows(trace, hits, mask, slice_name)
            )

        for step in np.unique(trace.scheduler_steps):
            step_mask = trace.scheduler_steps == step
            for layer, counts in enumerate(expert_counts(trace.routes, step_mask)):
                row = {
                    "run": run_name,
                    "scheduler_step": int(step),
                    "route_rows": int(step_mask.sum()),
                    "layer": layer,
                }
                row.update(distribution_metrics(counts))
                step_rows.append(row)
        for window_start in range(0, trace.num_rows, 512):
            window_end = min(window_start + 512, trace.num_rows)
            if window_end - window_start != 512:
                continue
            mask = np.zeros(trace.num_rows, dtype=np.bool_)
            mask[window_start:window_end] = True
            for layer, counts in enumerate(expert_counts(trace.routes, mask)):
                row = {
                    "run": run_name,
                    "window_start": window_start,
                    "route_rows": 512,
                    "layer": layer,
                }
                row.update(distribution_metrics(counts))
                window_rows.append(row)

    stage_rows: list[dict[str, Any]] = []
    stage_hot_sets: dict[str, np.ndarray] = {}
    for stage_name, counts in spec_stage_counts:
        stage_hot_sets[stage_name] = select_hot_experts(counts)
        for layer, layer_counts in enumerate(counts):
            row = {"stage": stage_name, "layer": layer}
            row.update(distribution_metrics(layer_counts))
            stage_rows.append(row)
        for layer, layer_counts in enumerate(counts):
            for expert, value in enumerate(layer_counts):
                expert_rows.append(
                    {
                        "run": "replayssm_spec_bs128_d3",
                        "slice": stage_name,
                        "layer": layer,
                        "expert": expert,
                        "assignments": int(value),
                        "share": float(value / layer_counts.sum()),
                        "normalized_load": float(value / layer_counts.mean()),
                    }
                )
    stage_jaccard_rows: list[dict[str, Any]] = []
    final_stage = SPEC_STAGES[-1][0]
    for stage_name, hot in stage_hot_sets.items():
        for layer in range(hot.shape[0]):
            intersection = len(
                set(hot[layer]) & set(stage_hot_sets[final_stage][layer])
            )
            union = len(set(hot[layer]) | set(stage_hot_sets[final_stage][layer]))
            stage_jaccard_rows.append(
                {
                    "stage": stage_name,
                    "compared_with": final_stage,
                    "layer": layer,
                    "jaccard": intersection / union,
                }
            )

    ar128 = traces["replayssm_ar_bs128"]
    spec128 = traces["replayssm_spec_bs128_d3"]
    ar512 = traces["replayssm_ar_bs512"]
    outputs = {
        name: load_outputs(trace_root / name)
        for name in run_names
    }
    ar_spec_consistency = compare_outputs(
        outputs["replayssm_ar_bs128"],
        outputs["replayssm_spec_bs128_d3"],
    )
    ar_batch_consistency = compare_outputs(
        outputs["replayssm_ar_bs128"],
        outputs["replayssm_ar_bs512"],
        limit=128,
    )
    output_consistency = {
        "ar128_vs_spec128": ar_spec_consistency,
        "ar128_vs_first128_of_ar512": ar_batch_consistency,
    }
    (metrics_dir / "output_consistency.json").write_text(
        json.dumps(output_consistency, indent=2, sort_keys=True) + "\n"
    )
    h1_same_batch = paired_request_bootstrap(
        spec128,
        np.ones(spec128.num_rows, dtype=np.bool_),
        ar128,
        np.ones(ar128.num_rows, dtype=np.bool_),
        samples=bootstrap_samples,
    )
    h1_equal_token = block_step_bootstrap(
        spec128, ar512, samples=bootstrap_samples
    )
    matching_requests = ar_spec_consistency["matching_question_indices"]
    h1_matching_requests = None
    if matching_requests:
        h1_matching_requests = paired_request_bootstrap(
            spec128,
            question_mask(spec128, matching_requests),
            ar128,
            question_mask(ar128, matching_requests),
            samples=bootstrap_samples,
        )
    prefix_lengths = {
        int(question): int(length)
        for question, length in ar_spec_consistency[
            "matching_prefix_lengths"
        ].items()
    }
    spec_prefix_mask = matching_prefix_mask(spec128, prefix_lengths)
    ar_prefix_mask = matching_prefix_mask(ar128, prefix_lengths)
    h1_matching_prefix = paired_request_bootstrap(
        spec128,
        spec_prefix_mask,
        ar128,
        ar_prefix_mask,
        samples=bootstrap_samples,
    )

    layer_direction_summary: dict[str, Any] = {}
    spec_layer_counts = counts_by_run["replayssm_spec_bs128_d3"]
    ar_layer_counts = counts_by_run["replayssm_ar_bs128"]
    for metric_name in ("gini", "cv", "max_over_mean", "hot_10pct_share"):
        spec_values = np.array(
            [distribution_metrics(row)[metric_name] for row in spec_layer_counts]
        )
        ar_values = np.array(
            [distribution_metrics(row)[metric_name] for row in ar_layer_counts]
        )
        layer_direction_summary[metric_name] = {
            "layers_increased": int(np.count_nonzero(spec_values > ar_values)),
            "layers_decreased": int(np.count_nonzero(spec_values < ar_values)),
            "layers_equal": int(np.count_nonzero(spec_values == ar_values)),
            "mean_difference": float((spec_values - ar_values).mean()),
        }
    bootstrap_summary = {
        "h1_same_batch_spec128_minus_ar128_global_gini": h1_same_batch,
        "h1_equal_step_tokens_spec128_minus_ar512_global_gini": h1_equal_token,
        "h1_matching_requests_spec128_minus_ar128_global_gini": (
            h1_matching_requests
        ),
        "h1_matching_prefix_spec128_minus_ar128_global_gini": h1_matching_prefix,
        "layer_direction_spec128_vs_ar128": layer_direction_summary,
        "h2_mean_off_diagonal_phi": phi_bootstrap,
    }

    rank_rows: list[dict[str, Any]] = []
    for run_name, counts in counts_by_run.items():
        for rank, expert_slice in ((0, slice(0, 128)), (1, slice(128, 256))):
            rank_rows.append(
                {
                    "run": run_name,
                    "ep_rank": rank,
                    "expert_range": f"{expert_slice.start}-{expert_slice.stop - 1}",
                    "assignments": int(counts[:, expert_slice].sum()),
                }
            )

    write_csv(metrics_dir / "expert_load.csv", expert_rows)
    write_csv(metrics_dir / "layer_imbalance.csv", layer_rows)
    write_csv(metrics_dir / "step_imbalance.csv", step_rows)
    write_csv(metrics_dir / "window_512_imbalance.csv", window_rows)
    write_csv(metrics_dir / "hot_experts.csv", hot_rows)
    write_csv(metrics_dir / "hot_token_coverage.csv", coverage_rows)
    write_csv(
        metrics_dir / "hot_hit_layer_histogram.csv", hit_layer_histogram_rows
    )
    write_csv(
        metrics_dir / "hot_assignment_histogram.csv",
        hot_assignment_histogram_rows,
    )
    write_csv(metrics_dir / "hot_layer_relations.csv", layer_relation_rows)
    write_csv(metrics_dir / "token_id_hot_coverage.csv", vocabulary_rows)
    write_csv(metrics_dir / "spec_cumulative_layer_metrics.csv", stage_rows)
    write_csv(metrics_dir / "spec_cumulative_hot_jaccard.csv", stage_jaccard_rows)
    write_csv(metrics_dir / "ep_rank_load.csv", rank_rows)
    (metrics_dir / "global_imbalance.json").write_text(
        json.dumps(global_metrics, indent=2, sort_keys=True) + "\n"
    )
    (metrics_dir / "bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, indent=2, sort_keys=True) + "\n"
    )

    _plot_spec_stages(spec_stage_counts, figures_dir)
    _plot_layer_stage_metrics(spec_stage_counts, figures_dir)
    _plot_run_comparison(counts_by_run, figures_dir)

    same_support = h1_same_batch["ci_low"] > 0
    equal_support = h1_equal_token["ci_low"] > 0
    spec_phi = phi_bootstrap["replayssm_spec_bs128_d3"]["all_executed"]
    h2_support = spec_phi["ci_low"] > 0
    mismatch_count = ar_spec_consistency["mismatching_requests"]
    matching_count = ar_spec_consistency["matching_requests"]
    output_caveat = (
        "AR128 与 Spec128 输出完全一致，可以直接解释同轨迹路由差异。"
        if mismatch_count == 0
        else (
            f"AR128 与 Spec128 有 {mismatch_count}/"
            f"{ar_spec_consistency['compared_requests']} 个请求输出不完全一致；"
            "负载差异同时包含投机执行与后续生成轨迹变化，因果判断必须结合"
            "匹配请求和匹配前缀敏感性结果。"
        )
    )
    matching_sensitivity = (
        "无完整匹配请求。"
        if h1_matching_requests is None
        else (
            f"{matching_count} 个完整匹配请求上的 Gini 差为 "
            f"{h1_matching_requests['difference']:.6f}，95% CI "
            f"[{h1_matching_requests['ci_low']:.6f}, "
            f"{h1_matching_requests['ci_high']:.6f}]。"
        )
    )
    same_label = "支持" if same_support else "证据不确定或反向"
    equal_label = "支持" if equal_support else "证据不确定或反向"
    h2_label = "支持" if h2_support else "证据不确定或反向"
    report_lines = [
        "# ReplaySSM 专家路由负载实验报告",
        "",
        "## 实验结论",
        "",
        (
            f"- 猜想 1 同 BS 主对照：{same_label}。Spec128 - AR128 的全局 "
            f"Gini 差为 {h1_same_batch['difference']:.6f}，95% CI "
            f"[{h1_same_batch['ci_low']:.6f}, "
            f"{h1_same_batch['ci_high']:.6f}]。"
        ),
        (
            f"- 猜想 1 等 step 执行 token 对照：{equal_label}。Spec128 - "
            f"AR512 的全局 Gini 差为 {h1_equal_token['difference']:.6f}，"
            f"95% CI [{h1_equal_token['ci_low']:.6f}, "
            f"{h1_equal_token['ci_high']:.6f}]。"
        ),
        (
            f"- 猜想 2：{h2_label}。Spec128 全执行量的跨层平均非对角 "
            f"phi 为 {spec_phi['mean_off_diagonal_phi']:.6f}，95% CI "
            f"[{spec_phi['ci_low']:.6f}, {spec_phi['ci_high']:.6f}]。"
        ),
        "",
        "## 输出一致性与敏感性",
        "",
        f"- {output_caveat}",
        f"- {matching_sensitivity}",
        (
            "- 逐请求匹配前缀上的 Gini 差为 "
            f"{h1_matching_prefix['difference']:.6f}，95% CI "
            f"[{h1_matching_prefix['ci_low']:.6f}, "
            f"{h1_matching_prefix['ci_high']:.6f}]。"
        ),
        (
            "- AR128 与 AR512 前 128 个请求中有 "
            f"{ar_batch_consistency['matching_requests']}/"
            f"{ar_batch_consistency['compared_requests']} 个输出完全一致。"
        ),
        "",
        "## 指标语义",
        "",
        (
            "- Global expert distribution 将 40 层 × 256 个专家视为 "
            "10,240 个独立 layer-expert 实例。"
        ),
        (
            "- Spec 主结果包含 target model 实际执行的 target 和三个 "
            "draft verify positions，包括 rejected draft rows。"
        ),
        "- MTP drafter 自身的 MoE 路由未绑定到 capturer，不计入任何指标。",
        (
            "- Hot expert 为每层负载最高的 26 个专家；token 在一层 "
            "top-8 中命中至少一个 hot expert 即计为该层 hot hit。"
        ),
        (
            "- BS128 Spec 与 BS512 AR 的等 token 结论只使用每步严格 "
            "512 个 target-model route rows 的稳态 steps。"
        ),
        "- 路由追踪启用了 eager 执行，因此本文不报告吞吐结论。",
        "",
        "## 可复核产物",
        "",
        "- `metrics/global_imbalance.json`：全局和分 slice 负载指标。",
        "- `metrics/layer_imbalance.csv`：逐层指标。",
        (
            "- `metrics/step_imbalance.csv` 与 "
            "`window_512_imbalance.csv`：自然 step 和固定 512-row 窗口指标。"
        ),
        (
            "- `metrics/hot_token_coverage.csv` 与 `pairwise_*.csv`："
            "严格交集、K 分布和跨层矩阵。"
        ),
        "- `metrics/output_consistency.json`：输出差异位置及匹配前缀。",
        (
            "- `figures/spec_cumulative_by_layer.*`：target 到 draft_3 的"
            "逐层累计负载曲线。"
        ),
    ]
    report = "\n".join(report_lines) + "\n"
    (root / "report_zh.md").write_text(report, encoding="utf-8")
