# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path

from mtp_ep_experiment_analysis import analyze_experiment
from mtp_ep_experiment_runtime import (
    add_hybrid_prediction_trace_args,
    collect_experiment,
    collect_one_condition,
    collect_one_rank,
    default_output_dir,
)
from mtp_ep_load_balance_utils import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_DATA_PARALLEL_SIZE,
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_DRAFT_LENGTHS,
    DEFAULT_LAYERS,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NUM_EXPERTS,
    DEFAULT_NUM_SAMPLES,
)


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--hybrid-spec-state-offload-mode",
        choices=("disabled", "predict_last"),
        default="disabled",
    )
    parser.add_argument(
        "--hybrid-spec-state-ewma-alpha",
        type=float,
        default=0.5,
    )
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
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--trace-steps-per-rank", type=int, default=0)
    parser.add_argument("--enable-nvtx-ranges", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_hybrid_prediction_trace_args(parser)

    # Collection subprocesses use the same public `collect` mode.
    parser.add_argument(
        "--internal-stage",
        choices=("experiment", "condition", "rank"),
        default="experiment",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--draft-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--prompt-cache-path",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dp-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dp-local-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dp-master-ip", help=argparse.SUPPRESS)
    parser.add_argument("--dp-master-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rank-output-path",
        type=Path,
        help=argparse.SUPPRESS,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect or analyze the unified Qwen3.6 MTP DP+EP experiment. "
            "Analysis produces performance, load, position, and draft-drop "
            "outputs using complete destination-rank routing."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect schema v10 CUDA Event raw data for all conditions.",
    )
    add_collect_args(collect_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Generate all CSV tables, plots, and the experiment report.",
    )
    analyze_parser.add_argument("--input-dir", type=Path, required=True)
    analyze_parser.add_argument("--skip-plots", action="store_true")
    analyze_parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args(argv)


def _require_internal_args(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        options = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(
            f"Internal collect stage {args.internal_stage!r} requires {options}."
        )


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        analyze_experiment(
            args.input_dir,
            skip_plots=args.skip_plots,
            skip_report=args.skip_report,
        )
        return

    args.layers = tuple(args.layers)
    entrypoint = Path(__file__).resolve()
    if args.internal_stage == "rank":
        _require_internal_args(
            args,
            (
                "batch_size",
                "draft_length",
                "output_dir",
                "prompt_cache_path",
                "dp_rank",
                "dp_local_rank",
                "dp_master_ip",
                "dp_master_port",
                "rank_output_path",
            ),
        )
        collect_one_rank(args)
        return
    if args.internal_stage == "condition":
        _require_internal_args(
            args,
            (
                "batch_size",
                "draft_length",
                "output_dir",
                "prompt_cache_path",
            ),
        )
        collect_one_condition(args, args.output_dir, entrypoint)
        return

    args.batch_sizes = tuple(args.batch_sizes)
    args.draft_lengths = tuple(args.draft_lengths)
    if 0 not in args.draft_lengths:
        raise ValueError(
            "Unified collection requires draft_length=0 as the performance "
            "and load-balance baseline."
        )
    output_dir = args.output_dir or default_output_dir()
    collect_experiment(args, output_dir, entrypoint)


if __name__ == "__main__":
    main()
