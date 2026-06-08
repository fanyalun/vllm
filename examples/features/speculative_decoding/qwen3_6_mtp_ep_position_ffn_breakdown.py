# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from mtp_ep_load_balance_utils import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_LAYERS,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_EXPERTS,
    DEFAULT_NUM_SAMPLES,
)

DEFAULT_LOCAL_MODEL = (
    "/home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"
)
DEFAULT_BATCH_SIZES = (8, 16)
DEFAULT_DRAFT_LENGTHS = (2, 4, 6)
DEFAULT_DATA_PARALLEL_SIZE = 4
DEFAULT_MAX_MODEL_LEN = 768


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path("results") / f"qwen3_6_mtp_position_ffn_{timestamp}"


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--draft-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_DRAFT_LENGTHS),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=DEFAULT_DATA_PARALLEL_SIZE,
    )
    parser.add_argument("--local-gpu-ids", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--trace-steps-per-rank", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)


def add_condition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--draft-length", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=DEFAULT_DATA_PARALLEL_SIZE,
    )
    parser.add_argument("--local-gpu-ids", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-cache-path", type=Path, default=None)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--trace-steps-per-rank", type=int, default=0)


def add_rank_args(parser: argparse.ArgumentParser) -> None:
    add_condition_args(parser)
    parser.add_argument("--dp-rank", type=int, required=True)
    parser.add_argument("--dp-local-rank", type=int, required=True)
    parser.add_argument("--dp-master-ip", required=True)
    parser.add_argument("--dp-master-port", type=int, required=True)
    parser.add_argument("--rank-output-path", type=Path, required=True)


def _read_manifest(input_dir: Path) -> dict:
    manifest_path = input_dir / "collect_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _raw_path(input_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return input_dir / path


def _strict_verification_mask(
    global_step_kinds: np.ndarray,
    rank_step_kinds: np.ndarray,
    *,
    num_steps: int,
) -> np.ndarray:
    if global_step_kinds.shape != (num_steps,):
        raise RuntimeError(
            "global_step_kinds must align with position-level data: "
            f"{global_step_kinds.shape} vs {(num_steps,)}."
        )
    if rank_step_kinds.ndim != 2 or rank_step_kinds.shape[0] != num_steps:
        raise RuntimeError(
            "rank_step_kinds must align with position-level data: "
            f"{rank_step_kinds.shape} vs ({num_steps}, num_ranks)."
        )
    return (global_step_kinds == "verification_only") & np.all(
        rank_step_kinds == "verification_only",
        axis=1,
    )


def build_position_rows(input_dir: Path) -> list[dict[str, int | float]]:
    manifest = _read_manifest(input_dir)
    rows: list[dict[str, int | float]] = []
    for condition in manifest["conditions"]:
        batch_size = int(condition["batch_size"])
        draft_length = int(condition["draft_length"])
        if draft_length <= 0:
            continue
        path = _raw_path(input_dir, str(condition["raw_path"]))
        with np.load(path, allow_pickle=False) as data:
            required_keys = (
                "global_step_position_sorted_rank_ffn_ms",
                "global_step_position_sorted_rank_local_routed_tokens",
                "global_step_kinds",
                "rank_step_kinds",
            )
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise RuntimeError(
                    f"{path} has no position-level data for "
                    f"{', '.join(missing_keys)}. Re-run collect with "
                    "qwen3_6_mtp_ep_position_ffn_breakdown.py."
                )
            ffn = np.asarray(
                data["global_step_position_sorted_rank_ffn_ms"],
                dtype=np.float64,
            )
            tokens = np.asarray(
                data["global_step_position_sorted_rank_local_routed_tokens"],
                dtype=np.float64,
            )
            target_mask = _strict_verification_mask(
                np.asarray(data["global_step_kinds"], dtype=np.str_),
                np.asarray(data["rank_step_kinds"], dtype=np.str_),
                num_steps=int(ffn.shape[0]),
            )
        position_count = draft_length + 1
        if (
            ffn.ndim != 3
            or ffn.shape[1] < position_count
            or tokens.shape != ffn.shape
        ):
            raise RuntimeError(
                f"{path} has FFN shape {ffn.shape} and token shape "
                f"{tokens.shape}; expected matching 3D arrays with at least "
                f"{position_count} verification positions."
            )
        if not np.any(target_mask):
            raise RuntimeError(
                f"{path} has no strict global verification-only barriers."
            )
        ffn = ffn[target_mask]
        tokens = tokens[target_mask]
        avg_ffn = ffn[:, :position_count, :].mean(axis=0)
        avg_tokens = tokens[:, :position_count, :].mean(axis=0)
        num_steps = int(ffn.shape[0])
        for position in range(position_count):
            for sorted_rank in range(avg_ffn.shape[1]):
                rows.append(
                    {
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "verification_position": position,
                        "sorted_rank_position": sorted_rank,
                        "num_global_steps": num_steps,
                        "avg_attributed_ffn_ms": float(
                            avg_ffn[position, sorted_rank]
                        ),
                        "avg_local_routed_tokens": float(
                            avg_tokens[position, sorted_rank]
                        ),
                    }
                )
    return rows


def write_position_csv(rows: list[dict[str, int | float]], output_dir: Path) -> Path:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "position_ffn_breakdown.csv"
    fieldnames = [
        "batch_size",
        "draft_length",
        "verification_position",
        "sorted_rank_position",
        "num_global_steps",
        "avg_attributed_ffn_ms",
        "avg_local_routed_tokens",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _build_position_metric_matrix(
    selected: list[dict[str, int | float]],
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
    rows: list[dict[str, int | float]],
    output_dir: Path,
) -> list[Path]:
    matplotlib_cache_dir = Path("/tmp/matplotlib")
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "position_ffn_breakdown"
    plot_dir.mkdir(parents=True, exist_ok=True)
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
            token_axis = axes[1][column]
            selected = [
                row
                for row in rows
                if int(row["batch_size"]) == batch_size
                and int(row["draft_length"]) == draft_length
            ]
            if not selected:
                ffn_axis.set_axis_off()
                token_axis.set_axis_off()
                continue
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
            token_values = _build_position_metric_matrix(
                selected,
                verification_positions,
                rank_positions,
                "avg_local_routed_tokens",
            )
            labels = [
                f"pos {position}" for position in verification_positions
            ]
            ffn_axis.stackplot(
                x,
                ffn_values,
                labels=labels,
                colors=colors[: len(verification_positions)],
                alpha=0.85,
            )
            ffn_axis.plot(
                x,
                ffn_values.sum(axis=0),
                color="black",
                linewidth=1.5,
                marker="o",
                label="total",
            )
            token_axis.stackplot(
                x,
                token_values,
                labels=labels,
                colors=colors[: len(verification_positions)],
                alpha=0.85,
            )
            token_axis.plot(
                x,
                token_values.sum(axis=0),
                color="black",
                linewidth=1.5,
                marker="o",
                label="total",
            )
            ffn_axis.set_title(f"batch_size={batch_size}")
            token_axis.set_xlabel(
                "sorted rank position (0 = heaviest per layer)"
            )
            for axis in (ffn_axis, token_axis):
                axis.grid(True, alpha=0.25)
                axis.set_xticks(rank_positions)
        axes[0][0].set_ylabel(
            "avg attributed FFN time after per-layer sort (ms)"
        )
        axes[1][0].set_ylabel(
            "avg local routed tokens after per-layer sort"
        )
        handles, labels = axes[0][-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(
            f"draft_length={draft_length} verification-position FFN and "
            "routed-token breakdown"
        )
        fig.tight_layout(rect=(0, 0, 0.9, 0.92))
        plot_path = plot_dir / f"draft_{draft_length:02d}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)
    return plot_paths


def analyze_position_breakdown(input_dir: Path) -> None:
    rows = build_position_rows(input_dir)
    if not rows:
        raise RuntimeError(f"No speculative conditions found under {input_dir}.")
    csv_path = write_position_csv(rows, input_dir)
    plot_paths = plot_position_breakdown(rows, input_dir)
    print(f"[position-analysis] wrote {csv_path}", flush=True)
    for plot_path in plot_paths:
        print(f"[position-analysis] wrote {plot_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Qwen3.6 MTP EP verification-position FFN attribution "
            "and draw stacked sorted-rank breakdown plots."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect the 8/16 x draft 2/4/6 matrix and draw plots.",
    )
    add_runtime_args(collect_parser)

    collect_one_parser = subparsers.add_parser("collect-one", help=argparse.SUPPRESS)
    add_condition_args(collect_one_parser)

    collect_one_rank_parser = subparsers.add_parser(
        "collect-one-rank",
        help=argparse.SUPPRESS,
    )
    add_rank_args(collect_one_rank_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Draw position-breakdown plots from an existing collected run.",
    )
    analyze_parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entrypoint = Path(__file__).resolve()
    if hasattr(args, "batch_sizes"):
        args.batch_sizes = tuple(args.batch_sizes)
    if hasattr(args, "draft_lengths"):
        args.draft_lengths = tuple(args.draft_lengths)
    if hasattr(args, "layers"):
        args.layers = tuple(args.layers)

    if args.command == "collect":
        from mtp_ep_experiment_runtime import collect_experiment

        output_dir = args.output_dir or default_output_dir()
        collect_experiment(args, output_dir, entrypoint)
        analyze_position_breakdown(output_dir)
        return

    if args.command == "collect-one":
        from mtp_ep_experiment_runtime import collect_one_condition

        collect_one_condition(args, args.output_dir, entrypoint)
        return

    if args.command == "collect-one-rank":
        from mtp_ep_experiment_runtime import collect_one_rank

        collect_one_rank(args)
        return

    analyze_position_breakdown(args.input_dir)


if __name__ == "__main__":
    main()
