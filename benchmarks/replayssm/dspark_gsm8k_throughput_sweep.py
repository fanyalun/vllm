# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a resumable single-GPU Qwen3.6 DSpark throughput sweep on GSM8K."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BATCH_SIZES = (32, 64, 128, 256)
DEFAULT_DRAFT_LENGTHS = (2, 3, 4, 5)


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_gpu_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative GPU indices"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spec-model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument(
        "--draft-lengths",
        type=parse_int_list,
        default=DEFAULT_DRAFT_LENGTHS,
    )
    parser.add_argument("--gpus", type=parse_gpu_list, default=(0, 1))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--draft-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cell-dir", help=argparse.SUPPRESS)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_questions(dataset_path: str, count: int) -> list[list[dict[str, str]]]:
    questions = []
    with open(dataset_path, encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                questions.append(json.loads(line)["question"])
    if len(questions) < count:
        raise ValueError(
            f"GSM8K has {len(questions)} rows, fewer than requested {count}"
        )
    return [
        [{"role": "user", "content": question}]
        for question in questions[:count]
    ]


def sum_counter(metrics: list[object], name: str) -> float:
    return sum(
        metric.value
        for metric in metrics
        if getattr(metric, "name", None) == name and hasattr(metric, "value")
    )


def capture_sizes(batch_size: int, verify_width: int) -> list[int]:
    request_sizes = (1, 2, 4, 8, 16, 24, 32, 48, 64)
    return sorted(
        {
            min(request_size, batch_size) * verify_width
            for request_size in request_sizes
        }
    )


def run_worker(args: argparse.Namespace) -> None:
    import torch

    from vllm import LLM, SamplingParams

    if args.batch_size is None or args.draft_length is None:
        raise ValueError("worker requires --batch-size and --draft-length")
    if args.physical_gpu is None or args.cell_dir is None:
        raise ValueError("worker requires --physical-gpu and --cell-dir")

    cell_dir = Path(args.cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    verify_width = args.draft_length + 1
    max_num_batched_tokens = 8192 + args.batch_size * args.draft_length
    capture = capture_sizes(args.batch_size, verify_width)
    started_at = utc_now()
    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        additional_config={"gdn_prefill_backend": "triton"},
        compilation_config={
            "cudagraph_capture_sizes": capture,
            "max_cudagraph_capture_size": max(capture),
        },
        kernel_config={"enable_flashinfer_autotune": False},
        speculative_config={
            "method": "dspark",
            "model": args.spec_model,
            "num_speculative_tokens": args.draft_length,
            "moe_backend": "triton",
        },
    )
    init_elapsed = time.perf_counter() - init_start

    messages = load_questions(args.dataset_path, args.batch_size)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=0,
    )
    chat_kwargs = {"enable_thinking": False}

    torch.cuda.synchronize()
    warmup_start = time.perf_counter()
    warmup_outputs = llm.chat(
        messages[:1],
        sampling_params,
        chat_template_kwargs=chat_kwargs,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    warmup_elapsed = time.perf_counter() - warmup_start
    warmup_tokens = sum(
        len(output.outputs[0].token_ids) for output in warmup_outputs
    )
    if len(warmup_outputs) != 1 or warmup_tokens != args.max_tokens:
        raise AssertionError(
            "single-sample warmup did not produce the requested token count"
        )

    metrics_before = llm.get_metrics()
    accepted_before = sum_counter(
        metrics_before, "vllm:spec_decode_num_accepted_tokens"
    )
    drafts_before = sum_counter(metrics_before, "vllm:spec_decode_num_drafts")

    torch.cuda.synchronize()
    formal_start = time.perf_counter()
    outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs=chat_kwargs,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    formal_elapsed = time.perf_counter() - formal_start

    output_lengths = [len(output.outputs[0].token_ids) for output in outputs]
    expected_tokens = args.batch_size * args.max_tokens
    produced_tokens = sum(output_lengths)
    if len(outputs) != args.batch_size:
        raise AssertionError(f"expected {args.batch_size} outputs, got {len(outputs)}")
    if produced_tokens != expected_tokens or set(output_lengths) != {args.max_tokens}:
        raise AssertionError(
            f"expected {expected_tokens} completion tokens, got {produced_tokens}"
        )

    metrics_after = llm.get_metrics()
    accepted = (
        sum_counter(metrics_after, "vllm:spec_decode_num_accepted_tokens")
        - accepted_before
    )
    drafts = (
        sum_counter(metrics_after, "vllm:spec_decode_num_drafts")
        - drafts_before
    )
    result = {
        "status": "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "model": args.model,
        "spec_model": args.spec_model,
        "spec_method": "dspark",
        "physical_gpu": args.physical_gpu,
        "visible_cuda_device": 0,
        "tensor_parallel_size": 1,
        "expert_parallel": False,
        "batch_size": args.batch_size,
        "formal_sample_count": len(outputs),
        "warmup_sample_count": len(warmup_outputs),
        "draft_length": args.draft_length,
        "verify_width": verify_width,
        "max_tokens_per_sample": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "target_token_scheduler_budget": 8192,
        "formal_completion_tokens": produced_tokens,
        "warmup_completion_tokens": warmup_tokens,
        "model_init_elapsed_s": init_elapsed,
        "warmup_elapsed_s": warmup_elapsed,
        "formal_elapsed_s": formal_elapsed,
        "completion_throughput_tok_s": produced_tokens / formal_elapsed,
        "request_throughput_req_s": args.batch_size / formal_elapsed,
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafts,
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else None,
        "cuda_graph_capture_sizes": capture,
        "dataset_path": args.dataset_path,
        "dataset_sha256": hashlib.sha256(
            Path(args.dataset_path).read_bytes()
        ).hexdigest(),
    }
    write_json(cell_dir / "result.json", result)
    (cell_dir / "cell_complete").touch()
    print("CELL_RESULT " + json.dumps(result), flush=True)


def cell_name(batch_size: int, draft_length: int) -> str:
    return f"bs{batch_size}_d{draft_length}"


def result_is_complete(
    cell_dir: Path, batch_size: int, draft_length: int, max_tokens: int
) -> bool:
    result_path = cell_dir / "result.json"
    marker_path = cell_dir / "cell_complete"
    if not result_path.is_file() or not marker_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    return (
        result.get("status") == "complete"
        and result.get("batch_size") == batch_size
        and result.get("formal_sample_count") == batch_size
        and result.get("warmup_sample_count") == 1
        and result.get("draft_length") == draft_length
        and result.get("formal_completion_tokens") == batch_size * max_tokens
    )


def worker_command(
    args: argparse.Namespace,
    batch_size: int,
    draft_length: int,
    gpu: int,
    cell_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        args.model,
        "--spec-model",
        args.spec_model,
        "--dataset-path",
        args.dataset_path,
        "--output-root",
        args.output_root,
        "--batch-size",
        str(batch_size),
        "--draft-length",
        str(draft_length),
        "--physical-gpu",
        str(gpu),
        "--cell-dir",
        str(cell_dir),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]


def launch_cell(
    args: argparse.Namespace,
    batch_size: int,
    draft_length: int,
    gpu: int,
    cell_dir: Path,
) -> tuple[subprocess.Popen[str], object]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_file = (cell_dir / "run.log").open("a", encoding="utf-8")
    command = worker_command(args, batch_size, draft_length, gpu, cell_dir)
    (cell_dir / "command.txt").write_text(" ".join(command) + "\n")
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["HF_DATASETS_OFFLINE"] = "1"
    child_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    python_bin = str(Path(sys.executable).resolve().parent)
    child_env["PATH"] = os.pathsep.join(
        value for value in (python_bin, child_env.get("PATH")) if value
    )
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )
    print(
        f"LAUNCHED {cell_name(batch_size, draft_length)} "
        f"gpu={gpu} pid={process.pid} log={cell_dir / 'run.log'}",
        flush=True,
    )
    return process, log_file


def collect_results(
    output_root: Path,
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
    max_tokens: int,
) -> list[dict[str, object]]:
    results = []
    for batch_size in batch_sizes:
        for draft_length in draft_lengths:
            cell_dir = output_root / "cells" / cell_name(batch_size, draft_length)
            if not result_is_complete(
                cell_dir, batch_size, draft_length, max_tokens
            ):
                raise RuntimeError(f"incomplete or invalid cell: {cell_dir}")
            results.append(json.loads((cell_dir / "result.json").read_text()))
    return results


def write_summary(output_root: Path, results: list[dict[str, object]]) -> None:
    fieldnames = [
        "batch_size",
        "draft_length",
        "completion_throughput_tok_s",
        "formal_elapsed_s",
        "formal_sample_count",
        "formal_completion_tokens",
        "warmup_sample_count",
        "warmup_completion_tokens",
        "mean_acceptance_length",
        "physical_gpu",
    ]
    with (output_root / "summary.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result[name] for name in fieldnames})
    write_json(output_root / "summary.json", results)


def plot_results(output_root: Path, results: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6.25), constrained_layout=True)
    batches = sorted({int(result["batch_size"]) for result in results})
    for batch_size in batches:
        batch_results = sorted(
            (
                result
                for result in results
                if int(result["batch_size"]) == batch_size
            ),
            key=lambda result: int(result["draft_length"]),
        )
        axis.plot(
            [int(result["draft_length"]) for result in batch_results],
            [
                float(result["completion_throughput_tok_s"])
                for result in batch_results
            ],
            marker="o",
            linewidth=2,
            label=f"Batch {batch_size}",
        )
    axis.set_xlabel("Draft length")
    axis.set_ylabel("Completion throughput (tokens/s)")
    axis.set_title("Qwen3.6 + DSpark on GSM8K (single GPU per run)")
    axis.set_xticks(sorted({int(result["draft_length"]) for result in results}))
    axis.grid(True, alpha=0.25)
    axis.legend(title="Formal sample count")
    figure.savefig(output_root / "throughput_by_draft_length.png", dpi=240)
    figure.savefig(output_root / "throughput_by_draft_length.pdf")
    plt.close(figure)


def run_sweep(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cells = [
        (batch_size, draft_length)
        for batch_size in args.batch_sizes
        for draft_length in args.draft_lengths
    ]
    manifest = {
        "status": "running",
        "created_at": utc_now(),
        "repo": str(Path.cwd()),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(Path(args.model).resolve()),
        "spec_model": str(Path(args.spec_model).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "batch_sizes": args.batch_sizes,
        "draft_lengths": args.draft_lengths,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "target_token_scheduler_budget_per_cell": 8192,
        "gpus": args.gpus,
        "expected_cells": len(cells),
        "warmup_sample_count_per_cell": 1,
        "formal_sample_count_rule": "equals batch_size",
        "throughput_definition": (
            "formal completion tokens / formal llm.chat wall time"
        ),
    }
    write_json(output_root / "run_manifest.json", manifest)

    pending = [
        cell
        for cell in cells
        if not result_is_complete(
            output_root / "cells" / cell_name(*cell),
            cell[0],
            cell[1],
            args.max_tokens,
        )
    ]
    active: dict[int, tuple[subprocess.Popen[str], object, tuple[int, int]]] = {}
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            batch_size, draft_length = pending.pop(0)
            cell_dir = output_root / "cells" / cell_name(
                batch_size, draft_length
            )
            process, log_file = launch_cell(
                args, batch_size, draft_length, gpu, cell_dir
            )
            active[gpu] = (process, log_file, (batch_size, draft_length))

        time.sleep(2)
        for gpu, (process, log_file, cell) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            del active[gpu]
            batch_size, draft_length = cell
            cell_dir = output_root / "cells" / cell_name(*cell)
            if return_code != 0 or not result_is_complete(
                cell_dir, batch_size, draft_length, args.max_tokens
            ):
                manifest["status"] = "failed"
                manifest["failed_cell"] = cell_name(*cell)
                manifest["failed_return_code"] = return_code
                manifest["updated_at"] = utc_now()
                write_json(output_root / "run_manifest.json", manifest)
                for running, running_log, _ in active.values():
                    running.terminate()
                    running_log.close()
                raise RuntimeError(
                    f"cell {cell_name(*cell)} failed; inspect {cell_dir / 'run.log'}"
                )
            print(
                f"COMPLETED {cell_name(*cell)} gpu={gpu} "
                f"result={cell_dir / 'result.json'}",
                flush=True,
            )

    results = collect_results(
        output_root, args.batch_sizes, args.draft_lengths, args.max_tokens
    )
    write_summary(output_root, results)
    plot_results(output_root, results)
    matrix = {
        "status": "complete",
        "completed_at": utc_now(),
        "expected_cells": len(cells),
        "completed_cells": len(results),
        "batch_sizes": args.batch_sizes,
        "draft_lengths": args.draft_lengths,
        "all_warmups_are_single_sample": all(
            result["warmup_sample_count"] == 1 for result in results
        ),
        "all_formal_sample_counts_match_batch": all(
            result["formal_sample_count"] == result["batch_size"]
            for result in results
        ),
        "figure_png": str(output_root / "throughput_by_draft_length.png"),
        "summary_csv": str(output_root / "summary.csv"),
    }
    write_json(output_root / "matrix_complete.json", matrix)
    manifest["status"] = "complete"
    manifest["updated_at"] = utc_now()
    write_json(output_root / "run_manifest.json", manifest)
    print("MATRIX_COMPLETE " + json.dumps(matrix), flush=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
