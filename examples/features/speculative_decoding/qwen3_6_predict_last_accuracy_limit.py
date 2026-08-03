# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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

EXPERIMENT_ENTRYPOINT = Path(__file__).resolve().with_name(
    "qwen3_6_mtp_ep_load_balance_experiment.py"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_ACCURACIES = (0.6, 0.8, 1.0)


def _normalize_draft_lengths(draft_lengths: list[int]) -> list[int]:
    ordered = list(dict.fromkeys(draft_lengths))
    if 0 not in ordered:
        ordered = [0, *ordered]
    return ordered


def _accuracy_tag(accuracy: float) -> str:
    return f"{int(round(float(accuracy) * 100)):03d}"


def _simulation_dir(root: Path, accuracy: float) -> Path:
    return root / f"sim_acc_{_accuracy_tag(accuracy)}"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = REPO_ROOT / ".venv" / "bin"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    return env


def _resolve_data_parallel_size(args: argparse.Namespace) -> int:
    explicit = getattr(args, "data_parallel_size", None)
    if explicit is not None:
        return int(explicit)
    local_gpu_ids = str(getattr(args, "local_gpu_ids", "") or "")
    gpu_ids = [gpu_id.strip() for gpu_id in local_gpu_ids.split(",") if gpu_id.strip()]
    if gpu_ids:
        return len(gpu_ids)
    return DEFAULT_DATA_PARALLEL_SIZE


def _run_command(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        cwd=REPO_ROOT,
        env=_subprocess_env(),
    )


def _add_common_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[128],
    )
    parser.add_argument(
        "--draft-lengths",
        nargs="+",
        type=int,
        default=[0, 4, 6],
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
        default=None,
    )
    parser.add_argument("--local-gpu-ids", default="0,1")
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
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-report", action="store_true")


def _build_collect_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    trace_mode: str,
    oracle_trace_root: Path | None,
    target_accuracy: float,
) -> list[str]:
    command = [
        sys.executable,
        str(EXPERIMENT_ENTRYPOINT),
        "collect",
        "--model",
        args.model,
        "--hybrid-spec-state-offload-mode",
        "predict_last",
        "--hybrid-spec-state-ewma-alpha",
        "0.5",
        "--dataset",
        args.dataset,
        "--dataset-split",
        args.dataset_split,
        "--num-samples",
        str(args.num_samples),
        "--batch-sizes",
        *(str(batch_size) for batch_size in args.batch_sizes),
        "--draft-lengths",
        *(str(draft_length) for draft_length in _normalize_draft_lengths(args.draft_lengths)),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(_resolve_data_parallel_size(args)),
        "--local-gpu-ids",
        str(args.local_gpu_ids),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--layers",
        *(str(layer) for layer in args.layers),
        "--num-experts",
        str(args.num_experts),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--trace-steps-per-rank",
        str(args.trace_steps_per_rank),
        "--output-dir",
        str(output_dir),
        "--hybrid-prediction-trace-mode",
        trace_mode,
        "--hybrid-prediction-target-accuracy",
        str(target_accuracy),
        "--hybrid-prediction-sim-mode",
        "exact_upper_bound",
        "--hybrid-prediction-sim-seed",
        "0",
    ]
    if args.dataset_config is not None:
        command.extend(["--dataset-config", str(args.dataset_config)])
    if oracle_trace_root is not None:
        command.extend(
            [
                "--hybrid-prediction-oracle-trace-root",
                str(oracle_trace_root),
            ]
        )
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    return command


def _build_analyze_command(
    result_dir: Path,
    *,
    skip_plots: bool,
    skip_report: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(EXPERIMENT_ENTRYPOINT),
        "analyze",
        "--input-dir",
        str(result_dir),
    ]
    if skip_plots:
        command.append("--skip-plots")
    if skip_report:
        command.append("--skip-report")
    return command


def _run_collect_and_analyze(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    trace_mode: str,
    oracle_trace_root: Path | None,
    target_accuracy: float,
) -> None:
    _run_command(
        _build_collect_command(
            args,
            output_dir=output_dir,
            trace_mode=trace_mode,
            oracle_trace_root=oracle_trace_root,
            target_accuracy=target_accuracy,
        )
    )
    _run_command(
        _build_analyze_command(
            output_dir,
            skip_plots=args.skip_plots,
            skip_report=args.skip_report,
        )
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_table_rows_by_condition(
    result_dir: Path,
    table_name: str,
) -> dict[tuple[int, int], dict[str, str]]:
    rows = _load_csv_rows(result_dir / "tables" / table_name)
    return {
        (int(row["batch_size"]), int(row["draft_length"])): row
        for row in rows
    }


def _load_manifest_by_condition(
    result_dir: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    with (result_dir / "collect_manifest.json").open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return {
        (int(row["batch_size"]), int(row["draft_length"])): row
        for row in payload["conditions"]
    }


def _float_or_nan(value: str) -> float:
    lower = value.lower()
    if lower == "nan":
        return math.nan
    return float(value)


def _load_condition_metrics(result_dir: Path) -> dict[tuple[int, int], dict[str, float]]:
    speedup_rows = _load_table_rows_by_condition(result_dir, "speedup_metrics.csv")
    prediction_rows = _load_table_rows_by_condition(
        result_dir,
        "hybrid_prediction_metrics.csv",
    )
    step_rows = _load_table_rows_by_condition(result_dir, "step_time_breakdown.csv")
    manifest_rows = _load_manifest_by_condition(result_dir)

    metrics: dict[tuple[int, int], dict[str, float]] = {}
    for key, speedup_row in speedup_rows.items():
        prediction_row = prediction_rows.get(key, {})
        step_row = step_rows[key]
        manifest_row = manifest_rows[key]
        total_predictions = float(prediction_row.get("hybrid_prediction_total", 0.0))
        exact_match_count = float(
            prediction_row.get("hybrid_prediction_exact_match", 0.0)
        )
        exact_match_rate = (
            exact_match_count / total_predictions
            if total_predictions > 0
            else math.nan
        )
        metrics[key] = {
            "tpot_ms": float(speedup_row["tpot_ms"]),
            "vllm_generation_throughput_tok_s": float(
                speedup_row["vllm_generation_throughput_tok_s"]
            ),
            "avg_verification_wall_ms": float(step_row["avg_verification_wall_ms"]),
            "achieved_exact_match_rate": exact_match_rate,
            "hybrid_replay_prepare_copy_ms": float(
                manifest_row.get("hybrid_replay_prepare_copy_ms", 0.0)
            ),
            "hybrid_replay_repair_compute_ms": float(
                manifest_row.get("hybrid_replay_repair_compute_ms", 0.0)
            ),
            "hybrid_replay_verify_attention_ms": float(
                manifest_row.get("hybrid_replay_verify_attention_ms", 0.0)
            ),
            "hybrid_prediction_total": total_predictions,
        }
    return metrics


def _build_compare_rows(
    *,
    disabled_metrics: dict[tuple[int, int], dict[str, float]],
    simulation_metrics: dict[tuple[int, int], dict[str, float]],
    target_accuracy: float,
    batch_sizes: list[int],
    draft_lengths: list[int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size in batch_sizes:
        for draft_length in draft_lengths:
            key = (batch_size, draft_length)
            disabled_row = disabled_metrics[key]
            simulation_row = simulation_metrics[key]
            tpot_ms = float(simulation_row["tpot_ms"])
            disabled_tpot_ms = float(disabled_row["tpot_ms"])
            throughput = float(simulation_row["vllm_generation_throughput_tok_s"])
            disabled_throughput = float(
                disabled_row["vllm_generation_throughput_tok_s"]
            )
            verification_wall_ms = float(simulation_row["avg_verification_wall_ms"])
            disabled_verification_wall_ms = float(
                disabled_row["avg_verification_wall_ms"]
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "target_accuracy": target_accuracy,
                    "achieved_exact_match_rate": float(
                        simulation_row["achieved_exact_match_rate"]
                    ),
                    "hybrid_prediction_total": int(
                        simulation_row["hybrid_prediction_total"]
                    ),
                    "tpot_ms": tpot_ms,
                    "disabled_tpot_ms": disabled_tpot_ms,
                    "vs_disabled_tpot_delta_ms": tpot_ms - disabled_tpot_ms,
                    "vs_disabled_tpot_delta_pct": (
                        (tpot_ms / disabled_tpot_ms - 1.0) * 100.0
                    ),
                    "vllm_generation_throughput_tok_s": throughput,
                    "disabled_vllm_generation_throughput_tok_s": (
                        disabled_throughput
                    ),
                    "vs_disabled_throughput_delta_tok_s": (
                        throughput - disabled_throughput
                    ),
                    "vs_disabled_throughput_delta_pct": (
                        (throughput / disabled_throughput - 1.0) * 100.0
                    ),
                    "avg_verification_wall_ms": verification_wall_ms,
                    "disabled_avg_verification_wall_ms": (
                        disabled_verification_wall_ms
                    ),
                    "vs_disabled_verification_wall_delta_ms": (
                        verification_wall_ms - disabled_verification_wall_ms
                    ),
                    "hybrid_replay_prepare_copy_ms": float(
                        simulation_row["hybrid_replay_prepare_copy_ms"]
                    ),
                    "hybrid_replay_repair_compute_ms": float(
                        simulation_row["hybrid_replay_repair_compute_ms"]
                    ),
                    "hybrid_replay_verify_attention_ms": float(
                        simulation_row["hybrid_replay_verify_attention_ms"]
                    ),
                }
            )
    return rows


def _oracle_beats_disabled(compare_rows: list[dict[str, float | int]]) -> bool:
    for row in compare_rows:
        if float(row["target_accuracy"]) < 0.999:
            continue
        if int(row["draft_length"]) == 0:
            continue
        if (
            float(row["vs_disabled_tpot_delta_ms"]) < 0.0
            or float(row["vs_disabled_throughput_delta_tok_s"]) > 0.0
        ):
            return True
    return False


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    oracle_dir: Path,
    disabled_baseline_dir: Path,
    compare_rows: list[dict[str, float | int]],
) -> None:
    lines = [
        "# Qwen3.6 predict_last Oracle Accuracy Limit Report",
        "",
        f"- oracle trace source: `{oracle_dir}`",
        f"- disabled baseline source: `{disabled_baseline_dir}`",
        "",
    ]
    oracle_beats_disabled = _oracle_beats_disabled(compare_rows)
    if oracle_beats_disabled:
        lines.extend(
            [
                "## Verdict",
                "",
                "100% oracle prediction at least matches or beats the disabled "
                "baseline on a primary metric for one speculative condition. "
                "The mechanism is not ruled out by this oracle-ceiling test.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Verdict",
                "",
                "Even with 100% oracle prediction accuracy, the current "
                "`predict_last` runtime path does not beat the disabled "
                "baseline on TPOT or generation throughput for the tested "
                "speculative conditions. Under this runtime shape, the memory "
                "optimization mechanism does not stand up.",
                "",
            ]
        )

    lines.extend(
        [
            "## Key Rows",
            "",
            "| batch | draft | target_acc | achieved_exact | tpot_ms | disabled_tpot_ms | throughput | disabled_throughput |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compare_rows:
        lines.append(
            "| "
            f"{int(row['batch_size'])} | "
            f"{int(row['draft_length'])} | "
            f"{float(row['target_accuracy']):.2f} | "
            f"{float(row['achieved_exact_match_rate']):.4f} | "
            f"{float(row['tpot_ms']):.4f} | "
            f"{float(row['disabled_tpot_ms']):.4f} | "
            f"{float(row['vllm_generation_throughput_tok_s']):.4f} | "
            f"{float(row['disabled_vllm_generation_throughput_tok_s']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_accuracy_limit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disabled_metrics = _load_condition_metrics(args.disabled_baseline_dir)
    oracle_metrics = _load_condition_metrics(args.oracle_dir)
    del oracle_metrics

    compare_rows: list[dict[str, float | int]] = []
    batch_sizes = list(args.batch_sizes)
    draft_lengths = _normalize_draft_lengths(list(args.draft_lengths))
    for target_accuracy in args.target_accuracies:
        simulation_dir = _simulation_dir(args.simulation_root, target_accuracy)
        simulation_metrics = _load_condition_metrics(simulation_dir)
        compare_rows.extend(
            _build_compare_rows(
                disabled_metrics=disabled_metrics,
                simulation_metrics=simulation_metrics,
                target_accuracy=target_accuracy,
                batch_sizes=batch_sizes,
                draft_lengths=draft_lengths,
            )
        )

    _write_csv(output_dir / "simulated_accuracy_limit_metrics.csv", compare_rows)
    _write_report(
        output_dir / "oracle_limit_report.md",
        oracle_dir=args.oracle_dir,
        disabled_baseline_dir=args.disabled_baseline_dir,
        compare_rows=compare_rows,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect oracle acceptance traces, simulate target prediction "
            "accuracies, and compare predict_last against a disabled baseline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect-oracle")
    _add_common_collect_args(collect_parser)
    collect_parser.add_argument("--output-dir", type=Path, required=True)

    simulate_parser = subparsers.add_parser("simulate")
    _add_common_collect_args(simulate_parser)
    simulate_parser.add_argument("--output-dir", type=Path, required=True)
    simulate_parser.add_argument(
        "--oracle-trace-root",
        type=Path,
        required=True,
    )
    simulate_parser.add_argument(
        "--target-accuracies",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_ACCURACIES),
    )

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument(
        "--disabled-baseline-dir",
        type=Path,
        required=True,
    )
    analyze_parser.add_argument("--oracle-dir", type=Path, required=True)
    analyze_parser.add_argument("--simulation-root", type=Path, required=True)
    analyze_parser.add_argument(
        "--target-accuracies",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_ACCURACIES),
    )
    analyze_parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[128],
    )
    analyze_parser.add_argument(
        "--draft-lengths",
        nargs="+",
        type=int,
        default=[0, 4, 6],
    )

    run_all_parser = subparsers.add_parser("run-all")
    _add_common_collect_args(run_all_parser)
    run_all_parser.add_argument("--output-dir", type=Path, required=True)
    run_all_parser.add_argument(
        "--disabled-baseline-dir",
        type=Path,
        required=True,
    )
    run_all_parser.add_argument(
        "--target-accuracies",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_ACCURACIES),
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "collect-oracle":
        _run_collect_and_analyze(
            args,
            output_dir=args.output_dir,
            trace_mode="record",
            oracle_trace_root=None,
            target_accuracy=1.0,
        )
        return
    if args.command == "simulate":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for target_accuracy in args.target_accuracies:
            _run_collect_and_analyze(
                args,
                output_dir=_simulation_dir(args.output_dir, target_accuracy),
                trace_mode="replay",
                oracle_trace_root=args.oracle_trace_root,
                target_accuracy=target_accuracy,
            )
        return
    if args.command == "analyze":
        analyze_accuracy_limit(args)
        return

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    oracle_dir = root / "oracle"
    simulation_root = root
    analysis_dir = root / "analysis"
    _run_collect_and_analyze(
        args,
        output_dir=oracle_dir,
        trace_mode="record",
        oracle_trace_root=None,
        target_accuracy=1.0,
    )
    for target_accuracy in args.target_accuracies:
        _run_collect_and_analyze(
            args,
            output_dir=_simulation_dir(simulation_root, target_accuracy),
            trace_mode="replay",
            oracle_trace_root=oracle_dir,
            target_accuracy=target_accuracy,
        )
    analyze_accuracy_limit(
        argparse.Namespace(
            output_dir=analysis_dir,
            disabled_baseline_dir=args.disabled_baseline_dir,
            oracle_dir=oracle_dir,
            simulation_root=simulation_root,
            target_accuracies=args.target_accuracies,
            batch_sizes=args.batch_sizes,
            draft_lengths=args.draft_lengths,
        )
    )


if __name__ == "__main__":
    main()
