# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and audit the Llama-3.1-8B EAGLE3 asynchronous SSD phase-A matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

MODES = ("ar", "sync", "async_jit", "async_cache")
ENGINES = ("eager", "graph")
BATCH_SIZES = (1, 4, 16)
PERF_ORDER = (
    ("sync", "async_cache"),
    ("async_cache", "sync"),
    ("sync", "async_cache"),
)
SPEC_METRIC_PREFIX = "vllm:spec_decode_"
ASYNC_METRIC_PREFIX = "vllm:async_draft_"


@dataclass(frozen=True)
class Cell:
    suite: str
    mode: str
    engine: str
    batch_size: int
    repeat: int = 0
    order: int = 0
    variant: str = "fixed"

    @property
    def name(self) -> str:
        suffix = ""
        if self.repeat:
            suffix += f"_r{self.repeat}_o{self.order}"
        if self.variant != "fixed":
            suffix += f"_{self.variant}"
        return f"{self.suite}_{self.mode}_{self.engine}_b{self.batch_size}{suffix}"


class GpuSampler:
    def __init__(self, output_path: Path, interval: float = 0.25):
        self.output_path = output_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        with self.output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                (
                    "timestamp",
                    "gpu_index",
                    "gpu_uuid",
                    "utilization_gpu_percent",
                    "memory_used_mib",
                    "power_draw_w",
                )
            )
            while not self._stop.is_set():
                command = [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ]
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                timestamp = time.time()
                for line in result.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 5:
                        writer.writerow((timestamp, *fields))
                output.flush()
                self._stop.wait(self.interval)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_template_token_ids(rendered: object) -> list[int]:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if (
        isinstance(rendered, list)
        and len(rendered) == 1
        and isinstance(rendered[0], list)
    ):
        rendered = rendered[0]
    if not isinstance(rendered, list) or not all(
        isinstance(token_id, int) for token_id in rendered
    ):
        raise TypeError("chat template did not return a flat integer token list")
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-device", type=int, default=0)
    parser.add_argument("--draft-device", type=int, default=1)
    parser.add_argument("--num-prompts-per-dataset", type=int, default=32)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=512)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument(
        "--tie-logprob-tolerance",
        type=float,
        default=0.1,
        help="Maximum two-sided Target top-1 tie gap for FP16 audit",
    )
    parser.add_argument("--draft-tie-logit-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--acceptance-length-relative-tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "cell",
            "correctness",
            "lifecycle",
            "performance",
            "audit",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--cell", help="Run one matrix cell with --phase cell")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port-base", type=int, default=43100)
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkpoint_manifest(model_root: Path) -> dict[str, object]:
    files = sorted(
        path
        for pattern in (
            "config.json",
            "generation_config.json",
            "tokenizer*.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "*.bin",
            "*.bin.index.json",
        )
        for path in model_root.glob(pattern)
        if path.is_file()
    )
    return {
        "path": str(model_root.resolve()),
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def prepare_prompts(
    args: argparse.Namespace,
    output_root: Path,
) -> list[dict[str, Any]]:
    prompt_path = output_root / "prompts.json"
    if prompt_path.is_file():
        prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
        if len(prompts) == 4 * args.num_prompts_per_dataset:
            return prompts

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    datasets = (
        ("humaneval", "humaneval/humaneval_data_10000.jsonl"),
        ("alpaca", "alpaca/alpaca_data_10000.jsonl"),
        ("gsm8k", "gsm8k/gsm8k_data_10000.jsonl"),
        ("ultrafeedback", "ultrafeedback/ultrafeedback_data_10000.jsonl"),
    )
    prompts: list[dict[str, Any]] = []
    for dataset_name, relative_path in datasets:
        source_path = Path(args.dataset_root) / relative_path
        rows = []
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) == args.num_prompts_per_dataset:
                    break
        if len(rows) != args.num_prompts_per_dataset:
            raise ValueError(
                f"{source_path} has {len(rows)} usable rows, expected "
                f"{args.num_prompts_per_dataset}"
            )
        for dataset_index, row in enumerate(rows):
            text = row["text"]
            token_ids = normalize_template_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            token_ids = token_ids[: args.input_length]
            prompts.append(
                {
                    "prompt_index": len(prompts),
                    "dataset": dataset_name,
                    "dataset_index": dataset_index,
                    "token_ids": token_ids,
                    "token_sha256": sha256_json(token_ids),
                }
            )
    write_json(prompt_path, prompts)
    return prompts


def prepare_manifest(
    args: argparse,
    output_root: Path,
    prompts: list[dict[str, Any]],
) -> None:
    import vllm

    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--", ".", ":(exclude)TDO"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            ".",
            ":(exclude)TDO",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_files = sorted(filter(None, untracked_output.splitlines()))
    worktree_hasher = hashlib.sha256()
    worktree_hasher.update(tracked_diff)
    for relative_path in untracked_files:
        worktree_hasher.update(relative_path.encode())
        worktree_hasher.update(b"\0")
        worktree_hasher.update(Path(relative_path).read_bytes())
        worktree_hasher.update(b"\0")
    topology = command_output(["nvidia-smi", "topo", "-m"])
    gpu_query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    manifest = {
        "artifact_kind": "async_ssd_eagle3_phase_a",
        "created_at_utc": utc_now(),
        "git": {
            "branch": command_output(["git", "branch", "--show-current"]),
            "head": command_output(["git", "rev-parse", "HEAD"]),
            "working_tree_diff_sha256": worktree_hasher.hexdigest(),
            "untracked_files_in_fingerprint": untracked_files,
        },
        "source": {
            "python": sys.executable,
            "vllm": vllm.__file__,
        },
        "models": {
            "target": checkpoint_manifest(Path(args.target)),
            "draft": checkpoint_manifest(Path(args.draft)),
            "dtype": "float16",
            "num_speculative_tokens": args.num_speculative_tokens,
            "fan_out": 3,
            "draft_sample_method": "greedy",
            "rejection_sample_method": "standard",
        },
        "workload": {
            "prompt_count": len(prompts),
            "prompt_suite_sha256": sha256_json(prompts),
            "input_length_cap": args.input_length,
            "output_length": args.output_length,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "topology": {
            "target_device": args.target_device,
            "draft_device": args.draft_device,
            "gpus": gpu_query,
            "nvidia_smi_topo": topology,
        },
        "correctness_contract": {
            "exact_output_tokens_by_default": True,
            "allow_first_divergence_only_when_both_modes_show_a_top1_tie": True,
            "tie_logprob_tolerance": args.tie_logprob_tolerance,
            "tie_tolerance_scope": "first Target token divergence only",
            "draft_tie_logit_tolerance": args.draft_tie_logit_tolerance,
            "preserve_ssd_approximate_branch_semantics": True,
            "sync_vs_forced_jit_must_match_before_first_draft_tie": True,
            "draft_tie_can_explain_later_outcome_cascade": True,
            "target_tie_can_explain_outcome_cascade_only_at_same_offset": True,
            "acceptance_length_relative_tolerance": (
                args.acceptance_length_relative_tolerance
            ),
            "acceptance_length_includes_target_recovery_token": True,
            "accepted_prefix_divergence_before_first_draft_tie_is_forbidden": True,
            "draft_tie_requires_each_mode_to_have_a_small_top1_top2_gap": True,
            "batch_shape_numerical_exception_requires": [
                "direct Sync-vs-JIT output exact",
                "same prompt passes every smaller batch",
                "same unordered Draft top-2 token set",
                "no accepted-prefix mismatch",
                "cell acceptance-length relative difference within tolerance",
            ],
            "cache_hit_tokens_are_provisional_and_may_differ_from_jit": True,
        },
    }
    write_json(output_root / "manifest.json", manifest)


def correctness_cells() -> list[Cell]:
    return [
        Cell("correctness", mode, engine, batch_size)
        for engine in ENGINES
        for batch_size in BATCH_SIZES
        for mode in MODES
    ]


def lifecycle_cells() -> list[Cell]:
    return [
        Cell("lifecycle", "async_cache", "eager", 16, variant="mixed_abort"),
        Cell("lifecycle", "async_cache", "eager", 4, variant="preemption"),
        Cell("lifecycle", "async_cache", "eager", 1, variant="chunked_prefill"),
    ]


def performance_cells() -> list[Cell]:
    cells = []
    for batch_size in BATCH_SIZES:
        for repeat, pair in enumerate(PERF_ORDER, start=1):
            for order, mode in enumerate(pair, start=1):
                cells.append(
                    Cell(
                        "performance",
                        mode,
                        "graph",
                        batch_size,
                        repeat=repeat,
                        order=order,
                    )
                )
    return cells


def prepare_matrix(output_root: Path) -> None:
    cells = correctness_cells() + lifecycle_cells() + performance_cells()
    write_json(
        output_root / "matrix.json",
        {
            "status": "expected",
            "created_at_utc": utc_now(),
            "cells": [asdict(cell) | {"name": cell.name} for cell in cells],
            "correctness_cell_count": len(correctness_cells()),
            "lifecycle_cell_count": len(lifecycle_cells()),
            "performance_cell_count": len(performance_cells()),
        },
    )


def spec_config(args: argparse.Namespace, mode: str) -> dict[str, object] | None:
    if mode == "ar":
        return None
    config: dict[str, object] = {
        "model": str(Path(args.draft).resolve()),
        "method": "eagle3",
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_tensor_parallel_size": 1,
        "draft_sample_method": "greedy",
        "rejection_sample_method": "standard",
    }
    if mode in ("async_jit", "async_cache"):
        config["async_draft_device"] = args.draft_device
    return config


def server_command(
    args: argparse,
    cell: Cell,
    port: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(Path(args.target).resolve()),
        "--served-model-name",
        "async-ssd-eagle3",
        "--dtype",
        "float16",
        "--device-ids",
        str(args.target_device),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(cell.batch_size),
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--no-enable-log-requests",
        "--port",
        str(port),
    ]
    if cell.engine == "eager":
        command.append("--enforce-eager")
    if cell.variant == "chunked_prefill":
        index = command.index("4096")
        command[index] = "64"
    if cell.variant == "preemption":
        max_model_len_index = command.index("--max-model-len") + 1
        command[max_model_len_index] = "256"
        command.extend(("--num-gpu-blocks-override", "32"))
    config = spec_config(args, cell.mode)
    if config is not None:
        command.extend(("--speculative-config", json.dumps(config)))
    return command


def wait_for_server(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"server did not become ready within {timeout}s")


def stop_server(process: subprocess.Popen[str]) -> dict[str, object]:
    started = time.monotonic()
    forced = False
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            forced = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    return {
        "exit_code": process.returncode,
        "forced_kill": forced,
        "shutdown_seconds": time.monotonic() - started,
    }


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s]+)\s+([-+0-9.eE]+)$", line)
        if match is None:
            continue
        name, raw_value = match.groups()
        is_spec_metric = name.startswith((SPEC_METRIC_PREFIX, ASYNC_METRIC_PREFIX))
        if is_spec_metric or name.startswith("vllm:num_preemptions"):
            values[name] = float(raw_value)
    return values


def scrape_metrics(port: int) -> tuple[str, dict[str, float]]:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=30)
    response.raise_for_status()
    return response.text, parse_prometheus(response.text)


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {
        name: value - before.get(name, 0.0)
        for name, value in after.items()
        if name.startswith((SPEC_METRIC_PREFIX, ASYNC_METRIC_PREFIX))
        or name.startswith("vllm:num_preemptions")
    }


def stream_request(
    *,
    port: int,
    request_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    logprobs: int | None,
    abort_after_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "async-ssd-eagle3",
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "request_id": request_id,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
        payload["return_tokens_as_token_ids"] = True

    token_ids: list[int] = []
    top_logprobs: list[dict[str, float] | None] = []
    start = time.perf_counter()
    first_token_at: float | None = None
    usage = None
    aborted = False
    with requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json=payload,
        stream=True,
        timeout=(30, 900),
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices") or ():
                delta_ids = choice.get("token_ids") or []
                if delta_ids and first_token_at is None:
                    first_token_at = time.perf_counter()
                token_ids.extend(delta_ids)
                choice_logprobs = choice.get("logprobs")
                if choice_logprobs:
                    top_logprobs.extend(choice_logprobs.get("top_logprobs") or [])
            if abort_after_tokens is not None and len(token_ids) >= abort_after_tokens:
                aborted = True
                response.close()
                break
    end = time.perf_counter()
    ttft = None if first_token_at is None else first_token_at - start
    tpot = None
    if first_token_at is not None and len(token_ids) > 1:
        tpot = (end - first_token_at) / (len(token_ids) - 1)
    return {
        "request_id": request_id,
        "token_ids": token_ids,
        "token_sha256": sha256_json(token_ids),
        "top_logprobs": top_logprobs,
        "started_at": start,
        "first_token_at": first_token_at,
        "finished_at": end,
        "latency_seconds": end - start,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "usage": usage,
        "aborted": aborted,
    }


def run_requests(
    *,
    port: int,
    cell: Cell,
    prompts: list[dict[str, Any]],
    output_length: int,
    logprobs: int | None,
    prefix: str,
    mixed_lengths: bool = False,
    abort_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    abort_indices = abort_indices or set()
    start = time.perf_counter()
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cell.batch_size) as executor:
        for prompt in prompts:
            prompt_index = int(prompt["prompt_index"])
            max_tokens = output_length
            if mixed_lengths:
                max_tokens = (32, 64, 96, 128)[prompt_index % 4]
            futures.append(
                executor.submit(
                    stream_request,
                    port=port,
                    request_id=(f"{prefix}-{cell.name}-formal-{prompt_index:03d}"),
                    prompt_token_ids=prompt["token_ids"],
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    abort_after_tokens=8 if prompt_index in abort_indices else None,
                )
            )
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    results.sort(key=lambda result: result["request_id"])
    return results, elapsed


def warmup_server(
    args: argparse,
    port: int,
    cell: Cell,
    prompt: dict[str, Any],
) -> dict[str, object]:
    started = time.perf_counter()
    requests_completed = 0
    tokens = 0
    while time.perf_counter() - started < args.warmup_seconds:
        result = stream_request(
            port=port,
            request_id=f"warmup-{cell.name}-{requests_completed}",
            prompt_token_ids=prompt["token_ids"],
            max_tokens=min(64, args.output_length),
            logprobs=None,
        )
        requests_completed += 1
        tokens += len(result["token_ids"])
    return {
        "seconds": time.perf_counter() - started,
        "requests": requests_completed,
        "completion_tokens": tokens,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = max(math.ceil(quantile * len(values)) - 1, 0)
    return values[index]


def summarize_requests(
    results: list[dict[str, Any]],
    elapsed: float,
    gpu_count: int,
) -> dict[str, object]:
    completed = [result for result in results if not result["aborted"]]
    completion_tokens = sum(len(result["token_ids"]) for result in completed)
    ttfts = [result["ttft_seconds"] for result in completed if result["ttft_seconds"]]
    tpots = [result["tpot_seconds"] for result in completed if result["tpot_seconds"]]
    latencies = [result["latency_seconds"] for result in completed]
    return {
        "request_count": len(results),
        "completed_request_count": len(completed),
        "aborted_request_count": len(results) - len(completed),
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "completion_throughput_tok_s": completion_tokens / elapsed,
        "tokens_per_gpu_second": completion_tokens / elapsed / gpu_count,
        "ttft_p50_seconds": percentile(ttfts, 0.5),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "tpot_p50_seconds": percentile(tpots, 0.5),
        "tpot_p95_seconds": percentile(tpots, 0.95),
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
    }


def cell_is_complete(cell_dir: Path) -> bool:
    marker = cell_dir / "cell_complete.json"
    result = cell_dir / "result.json"
    if not marker.is_file() or not result.is_file():
        return False
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    result_value = json.loads(result.read_text(encoding="utf-8"))
    return marker_value.get("status") == result_value.get("status") == "complete"


def run_cell(
    args: argparse,
    output_root: Path,
    prompts: list[dict[str, Any]],
    cell: Cell,
    port: int,
) -> None:
    cell_dir = output_root / "cells" / cell.name
    if args.resume and cell_is_complete(cell_dir):
        print(f"SKIP complete cell {cell.name}", flush=True)
        return
    cell_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "cell_complete.json",
        "failure_marker.json",
        "proposals.jsonl",
    ):
        (cell_dir / stale_name).unlink(missing_ok=True)
    started_at_utc = utc_now()
    command = server_command(args, cell, port)
    (cell_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}"
            f"{environment.get('PATH', '')}",
        }
    )
    if cell.mode == "async_jit":
        environment["ASYNC_DRAFT_FORCE_JIT"] = "1"
    if cell.suite in ("correctness", "lifecycle") and cell.mode != "ar":
        environment["REPLAYSSM_SPEC_DECODE_TRACE_PATH"] = str(
            cell_dir / "proposals.jsonl"
        )
    if cell.suite == "correctness" and cell.mode != "ar":
        environment["REPLAYSSM_SPEC_DECODE_TRACE_LOGITS"] = "1"
    recorded_environment = {
        name: environment[name]
        for name in (
            "ASYNC_DRAFT_FORCE_JIT",
            "REPLAYSSM_SPEC_DECODE_TRACE_PATH",
            "REPLAYSSM_SPEC_DECODE_TRACE_LOGITS",
            "VLLM_WORKER_MULTIPROC_METHOD",
        )
        if name in environment
    }
    write_json(cell_dir / "environment.json", recorded_environment)
    log_file = (cell_dir / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    sampler = GpuSampler(cell_dir / "gpu_samples.csv")
    shutdown: dict[str, object] = {}
    try:
        wait_for_server(port, process, args.startup_timeout)
        warmup = warmup_server(args, port, cell, prompts[0])
        before_text, before = scrape_metrics(port)
        (cell_dir / "metrics_before.prom").write_text(before_text, encoding="utf-8")
        sampler.start()

        cell_prompts = prompts
        mixed = False
        abort_indices: set[int] = set()
        if cell.suite == "lifecycle":
            cell_prompts = prompts[: max(cell.batch_size * 2, 8)]
            if cell.variant == "mixed_abort":
                mixed = True
                abort_indices = {1, 5}
            elif cell.variant == "preemption":
                cell_prompts = prompts[:8]
            elif cell.variant == "chunked_prefill":
                cell_prompts = prompts[:4]

        logprobs = 20 if cell.suite == "correctness" else None
        output_length = args.output_length
        if cell.suite == "lifecycle":
            output_length = min(output_length, 128)
        results, elapsed = run_requests(
            port=port,
            cell=cell,
            prompts=cell_prompts,
            output_length=output_length,
            logprobs=logprobs,
            prefix="ssd",
            mixed_lengths=mixed,
            abort_indices=abort_indices,
        )
        sampler.stop()
        after_text, after = scrape_metrics(port)
        (cell_dir / "metrics_after.prom").write_text(after_text, encoding="utf-8")
        deltas = metric_delta(before, after)
        write_json(cell_dir / "metrics_delta.json", deltas)
        write_json(cell_dir / "requests.json", results)

        gpu_count = 2 if cell.mode.startswith("async") else 1
        summary = summarize_requests(results, elapsed, gpu_count)
        expected_tokens = None
        if cell.suite in ("correctness", "performance"):
            expected_tokens = len(cell_prompts) * args.output_length
            if summary["completion_tokens"] != expected_tokens:
                raise AssertionError(
                    f"{cell.name} produced {summary['completion_tokens']} tokens; "
                    f"expected {expected_tokens}"
                )
        if cell.variant == "mixed_abort" and summary["aborted_request_count"] != 2:
            raise AssertionError("mixed_abort did not abort exactly two requests")
        if cell.variant == "preemption":
            preemptions = sum(
                value
                for name, value in deltas.items()
                if name.startswith("vllm:num_preemptions")
            )
            if preemptions <= 0:
                raise AssertionError("preemption cell did not trigger preemption")

        result = {
            "status": "complete",
            "cell": asdict(cell) | {"name": cell.name},
            "started_at_utc": started_at_utc,
            "warmup": warmup,
            "summary": summary,
            "expected_completion_tokens": expected_tokens,
            "metrics_delta": deltas,
            "runtime_environment": recorded_environment,
        }
        write_json(cell_dir / "result.json", result)
    except BaseException as error:
        sampler.stop()
        write_json(
            cell_dir / "failure_marker.json",
            {
                "status": "failed",
                "cell": asdict(cell) | {"name": cell.name},
                "error": f"{type(error).__name__}: {error}",
                "timestamp_utc": utc_now(),
            },
        )
        raise
    finally:
        shutdown = stop_server(process)
        log_file.close()
        write_json(cell_dir / "shutdown.json", shutdown)
    if shutdown.get("forced_kill"):
        raise RuntimeError(f"{cell.name} required a forced server kill")
    write_json(
        cell_dir / "cell_complete.json",
        {
            "status": "complete",
            "cell": cell.name,
            "completed_at_utc": utc_now(),
        },
    )
    print(f"COMPLETE {cell.name}", flush=True)


def load_requests(output_root: Path, cell: Cell) -> list[dict[str, Any]]:
    path = output_root / "cells" / cell.name / "requests.json"
    return json.loads(path.read_text(encoding="utf-8"))


def request_map(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    mapped = {}
    for result in results:
        match = re.search(r"formal-(\d+)$", result["request_id"])
        if match:
            mapped[int(match.group(1))] = result
    return mapped


def top_logprob(top: dict[str, float] | None, token_id: int) -> float | None:
    if top is None:
        return None
    return top.get(f"token_id:{token_id}")


def audit_request_pair(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    baseline_tokens = baseline["token_ids"]
    candidate_tokens = candidate["token_ids"]
    if baseline_tokens == candidate_tokens:
        return {"status": "exact"}
    mismatch = next(
        (
            index
            for index, pair in enumerate(zip(baseline_tokens, candidate_tokens))
            if pair[0] != pair[1]
        ),
        min(len(baseline_tokens), len(candidate_tokens)),
    )
    if mismatch >= len(baseline_tokens) or mismatch >= len(candidate_tokens):
        return {"status": "failed", "reason": "length_mismatch", "offset": mismatch}
    baseline_token = baseline_tokens[mismatch]
    candidate_token = candidate_tokens[mismatch]
    baseline_top = baseline["top_logprobs"][mismatch]
    candidate_top = candidate["top_logprobs"][mismatch]
    baseline_own = top_logprob(baseline_top, baseline_token)
    baseline_other = top_logprob(baseline_top, candidate_token)
    candidate_own = top_logprob(candidate_top, candidate_token)
    candidate_other = top_logprob(candidate_top, baseline_token)
    if None in (baseline_own, baseline_other, candidate_own, candidate_other):
        return {
            "status": "failed",
            "reason": "divergent_token_missing_from_top_logprobs",
            "offset": mismatch,
            "baseline_token": baseline_token,
            "candidate_token": candidate_token,
        }
    baseline_gap = abs(float(baseline_own) - float(baseline_other))
    candidate_gap = abs(float(candidate_own) - float(candidate_other))
    if baseline_gap <= tolerance and candidate_gap <= tolerance:
        return {
            "status": "target_top1_tie_equivalent",
            "offset": mismatch,
            "baseline_token": baseline_token,
            "candidate_token": candidate_token,
            "baseline_logprob_gap": baseline_gap,
            "candidate_logprob_gap": candidate_gap,
        }
    return {
        "status": "failed",
        "reason": "non_tie_token_divergence",
        "offset": mismatch,
        "baseline_token": baseline_token,
        "candidate_token": candidate_token,
        "baseline_logprob_gap": baseline_gap,
        "candidate_logprob_gap": candidate_gap,
    }


def normalized_trace(path: Path) -> dict[int, list[dict[str, Any]]]:
    requests: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            match = re.search(r"formal-(\d+)", record["request_id"])
            if match is None:
                continue
            prompt_index = int(match.group(1))
            normalized = {
                field: record[field]
                for field in (
                    "accepted_draft_count",
                    "num_rejected",
                    "recovery_token",
                    "accepted_draft_tokens",
                    "draft_tokens",
                )
            }
            if "draft_top2" in record:
                normalized["draft_top2"] = record["draft_top2"]
            requests.setdefault(prompt_index, []).append(normalized)
    return requests


def _trace_outcome(record: dict[str, Any]) -> tuple[object, ...]:
    return tuple(
        record[field]
        for field in (
            "accepted_draft_count",
            "num_rejected",
            "recovery_token",
            "accepted_draft_tokens",
        )
    )


def _draft_tie_at_divergence(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    position: int,
    tolerance: float,
) -> dict[str, Any]:
    baseline_token = baseline["draft_tokens"][position]
    candidate_token = candidate["draft_tokens"][position]
    result = {
        "position": position,
        "baseline_token": baseline_token,
        "candidate_token": candidate_token,
    }
    try:
        baseline_top2 = baseline["draft_top2"][position]
        candidate_top2 = candidate["draft_top2"][position]
        baseline_ids = baseline_top2["token_ids"]
        candidate_ids = candidate_top2["token_ids"]
        baseline_logits = baseline_top2["logits"]
        candidate_logits = candidate_top2["logits"]
        if baseline_token not in baseline_ids or candidate_token not in candidate_ids:
            raise ValueError("selected token is not in top-2")
        baseline_gap = abs(baseline_logits[0] - baseline_logits[1])
        candidate_gap = abs(candidate_logits[0] - candidate_logits[1])
    except (KeyError, IndexError, TypeError):
        return result | {"status": "failed", "reason": "missing_draft_top2"}
    except ValueError:
        return result | {"status": "failed", "reason": "invalid_draft_top2"}
    return result | {
        "status": (
            "draft_top1_tie_equivalent"
            if baseline_gap <= tolerance and candidate_gap <= tolerance
            else "failed"
        ),
        "reason": (
            None
            if baseline_gap <= tolerance and candidate_gap <= tolerance
            else "non_tie_draft_divergence"
        ),
        "baseline_logit_gap": baseline_gap,
        "candidate_logit_gap": candidate_gap,
        "baseline_runner_up_token": baseline_ids[1],
        "candidate_runner_up_token": candidate_ids[1],
    }


def audit_draft_trace_pair(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
    target_output_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    common = min(len(baseline), len(candidate))
    if common == 0:
        return {
            "status": "failed",
            "reason": "empty_trace",
            "baseline_rounds": len(baseline),
            "candidate_rounds": len(candidate),
        }

    for round_index in range(common):
        if _trace_outcome(baseline[round_index]) != _trace_outcome(
            candidate[round_index]
        ):
            baseline_chunk = [
                *baseline[round_index]["accepted_draft_tokens"],
                baseline[round_index]["recovery_token"],
            ]
            candidate_chunk = [
                *candidate[round_index]["accepted_draft_tokens"],
                candidate[round_index]["recovery_token"],
            ]
            local_offset = next(
                (
                    index
                    for index, pair in enumerate(zip(baseline_chunk, candidate_chunk))
                    if pair[0] != pair[1]
                ),
                min(len(baseline_chunk), len(candidate_chunk)),
            )
            baseline_prefix = sum(
                record["accepted_draft_count"] + 1 for record in baseline[:round_index]
            )
            candidate_prefix = sum(
                record["accepted_draft_count"] + 1 for record in candidate[:round_index]
            )
            output_offset = baseline_prefix + local_offset
            if (
                baseline_prefix == candidate_prefix
                and target_output_audit is not None
                and target_output_audit.get("status") == "target_top1_tie_equivalent"
                and target_output_audit.get("offset") == output_offset
            ):
                return {
                    "status": "target_top1_tie_cascade_equivalent",
                    "baseline_rounds": len(baseline),
                    "candidate_rounds": len(candidate),
                    "strict_prefix_rounds": round_index,
                    "first_outcome_divergence_round": round_index,
                    "first_outcome_divergence_output_offset": output_offset,
                    "target_tie": target_output_audit,
                }
            return {
                "status": "failed",
                "reason": "outcome_mismatch_before_first_draft_tie",
                "round": round_index,
                "candidate_prefix_tokens": candidate_prefix,
                "baseline_prefix_tokens": baseline_prefix,
                "outcome_divergence_output_offset": output_offset,
                "target_output_audit": target_output_audit,
            }
        baseline_tokens = baseline[round_index]["draft_tokens"]
        candidate_tokens = candidate[round_index]["draft_tokens"]
        if baseline_tokens == candidate_tokens:
            continue
        mismatch = next(
            index
            for index, pair in enumerate(zip(baseline_tokens, candidate_tokens))
            if pair[0] != pair[1]
        )
        if round_index + 1 < common:
            baseline_accepted = baseline[round_index + 1]["accepted_draft_count"]
            candidate_accepted = candidate[round_index + 1]["accepted_draft_count"]
            if mismatch < min(baseline_accepted, candidate_accepted):
                return {
                    "status": "failed",
                    "reason": "accepted_draft_prefix_mismatch",
                    "round": round_index,
                    "position": mismatch,
                    "baseline_accepted_draft_count": baseline_accepted,
                    "candidate_accepted_draft_count": candidate_accepted,
                }
        tie = _draft_tie_at_divergence(
            baseline[round_index],
            candidate[round_index],
            mismatch,
            tolerance,
        )
        tie["round"] = round_index
        if tie["status"] == "failed":
            return tie
        return {
            "status": "draft_top1_tie_cascade_equivalent",
            "baseline_rounds": len(baseline),
            "candidate_rounds": len(candidate),
            "strict_prefix_rounds": round_index + 1,
            "first_tie_divergence": tie,
        }

    if abs(len(baseline) - len(candidate)) > 1:
        return {
            "status": "failed",
            "reason": "trace_length_mismatch_without_draft_tie",
            "baseline_rounds": len(baseline),
            "candidate_rounds": len(candidate),
        }
    return {
        "status": "exact",
        "baseline_rounds": len(baseline),
        "candidate_rounds": len(candidate),
        "unverified_terminal_round_delta": len(baseline) - len(candidate),
    }


def draft_trace_statistics(traces: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    records = [record for trace in traces.values() for record in trace]
    accepted_tokens = sum(record["accepted_draft_count"] for record in records)
    return {
        "num_drafts": len(records),
        "num_draft_tokens": sum(len(record["draft_tokens"]) for record in records),
        "num_accepted_tokens": accepted_tokens,
        "accepted_counts_per_position": [
            sum(record["accepted_draft_count"] > position for record in records)
            for position in range(7)
        ],
        "mean_accepted_draft_length": (
            accepted_tokens / len(records) if records else 0.0
        ),
        "mean_acceptance_length": (
            1.0 + accepted_tokens / len(records) if records else 0.0
        ),
    }


def is_batch_shape_numerical_exception(
    trace_audit: dict[str, Any],
    output_audit: dict[str, Any],
    batch_size: int,
    smaller_batch_audits: list[dict[str, Any]],
    acceptance_relative_difference: float,
    acceptance_relative_tolerance: float,
) -> bool:
    if (
        trace_audit.get("reason") != "non_tie_draft_divergence"
        or output_audit.get("status") != "exact"
        or batch_size <= min(BATCH_SIZES)
        or acceptance_relative_difference > acceptance_relative_tolerance
    ):
        return False
    expected_smaller_batches = {
        candidate for candidate in BATCH_SIZES if candidate < batch_size
    }
    observed_smaller_batches = {
        audit["batch_size"]
        for audit in smaller_batch_audits
        if audit.get("status") != "failed"
    }
    if observed_smaller_batches != expected_smaller_batches:
        return False
    baseline_top2 = {
        trace_audit.get("baseline_token"),
        trace_audit.get("baseline_runner_up_token"),
    }
    candidate_top2 = {
        trace_audit.get("candidate_token"),
        trace_audit.get("candidate_runner_up_token"),
    }
    return None not in baseline_top2 and baseline_top2 == candidate_top2


def audit_correctness(args: argparse, output_root: Path) -> bool:
    cells = correctness_cells()
    missing = [
        cell.name
        for cell in cells
        if not cell_is_complete(output_root / "cells" / cell.name)
    ]
    if missing:
        write_json(
            output_root / "correctness_audit.json",
            {"status": "incomplete", "missing_cells": missing},
        )
        return False

    comparisons = []
    trace_comparisons = []
    trace_summaries = []
    failed = []
    ties = []
    for engine in ENGINES:
        for batch_size in BATCH_SIZES:
            baseline_cell = Cell("correctness", "ar", engine, batch_size)
            baseline = request_map(load_requests(output_root, baseline_cell))
            mode_requests = {}
            for mode in ("sync", "async_jit", "async_cache"):
                cell = Cell("correctness", mode, engine, batch_size)
                cell_result = json.loads(
                    (output_root / "cells" / cell.name / "result.json").read_text(
                        encoding="utf-8"
                    )
                )
                candidate = request_map(load_requests(output_root, cell))
                mode_requests[mode] = candidate
                for prompt_index in sorted(baseline):
                    audit = audit_request_pair(
                        baseline[prompt_index],
                        candidate[prompt_index],
                        args.tie_logprob_tolerance,
                    )
                    record = {
                        "cell": cell.name,
                        "baseline_cell": baseline_cell.name,
                        "prompt_index": prompt_index,
                        **audit,
                    }
                    comparisons.append(record)
                    if audit["status"] == "failed":
                        failed.append(record)
                    elif audit["status"] == "target_top1_tie_equivalent":
                        ties.append(record)

                if mode.startswith("async"):
                    metrics = cell_result["metrics_delta"]
                    jit_fallbacks = sum(
                        value
                        for name, value in metrics.items()
                        if name.startswith("vllm:async_draft_jit_fallbacks_total")
                    )
                    if jit_fallbacks <= 0:
                        failed.append(
                            {
                                "status": "failed",
                                "reason": "async_path_has_no_jit_fallbacks",
                                "cell": cell.name,
                            }
                        )
                    if mode == "async_cache":
                        cache_hits = sum(
                            value
                            for name, value in metrics.items()
                            if name.startswith("vllm:async_draft_cache_hits_total")
                        )
                        if cache_hits <= 0:
                            failed.append(
                                {
                                    "status": "failed",
                                    "reason": "async_cache_path_has_no_hits",
                                    "cell": cell.name,
                                }
                            )

            sync_trace = normalized_trace(
                output_root
                / "cells"
                / Cell("correctness", "sync", engine, batch_size).name
                / "proposals.jsonl"
            )
            jit_trace = normalized_trace(
                output_root
                / "cells"
                / Cell("correctness", "async_jit", engine, batch_size).name
                / "proposals.jsonl"
            )
            if set(sync_trace) != set(jit_trace):
                failed.append(
                    {
                        "status": "failed",
                        "reason": "sync_forced_jit_request_set_mismatch",
                        "engine": engine,
                        "batch_size": batch_size,
                    }
                )
                continue
            sync_stats = draft_trace_statistics(sync_trace)
            jit_stats = draft_trace_statistics(jit_trace)
            sync_mean = sync_stats["mean_acceptance_length"]
            jit_mean = jit_stats["mean_acceptance_length"]
            acceptance_relative_difference = (
                abs(sync_mean - jit_mean) / sync_mean
                if sync_mean
                else (0.0 if jit_mean == 0 else math.inf)
            )
            trace_summary = {
                "engine": engine,
                "batch_size": batch_size,
                "sync": sync_stats,
                "async_jit": jit_stats,
                "acceptance_length_relative_difference": (
                    acceptance_relative_difference
                ),
                "acceptance_length_relative_tolerance": (
                    args.acceptance_length_relative_tolerance
                ),
            }
            trace_summaries.append(trace_summary)
            if (
                acceptance_relative_difference
                > args.acceptance_length_relative_tolerance
            ):
                failed.append(
                    {
                        "status": "failed",
                        "reason": "acceptance_length_relative_difference_exceeded",
                        **trace_summary,
                    }
                )
            for prompt_index in sorted(sync_trace):
                sync_jit_output_audit = audit_request_pair(
                    mode_requests["sync"][prompt_index],
                    mode_requests["async_jit"][prompt_index],
                    args.tie_logprob_tolerance,
                )
                trace_audit = audit_draft_trace_pair(
                    sync_trace[prompt_index],
                    jit_trace[prompt_index],
                    args.draft_tie_logit_tolerance,
                    sync_jit_output_audit,
                )
                smaller_batch_audits = [
                    record
                    for record in trace_comparisons
                    if record["engine"] == engine
                    and record["prompt_index"] == prompt_index
                    and record["batch_size"] < batch_size
                ]
                if is_batch_shape_numerical_exception(
                    trace_audit,
                    sync_jit_output_audit,
                    batch_size,
                    smaller_batch_audits,
                    acceptance_relative_difference,
                    args.acceptance_length_relative_tolerance,
                ):
                    trace_audit = {
                        **trace_audit,
                        "status": "batch_shape_numerical_equivalent",
                        "original_status": "failed",
                        "waiver": {
                            "reason": "batch_shape_numerical_nondeterminism",
                            "smaller_batch_evidence": [
                                {
                                    "batch_size": record["batch_size"],
                                    "status": record["status"],
                                }
                                for record in smaller_batch_audits
                            ],
                            "acceptance_length_relative_difference": (
                                acceptance_relative_difference
                            ),
                        },
                    }
                trace_record = {
                    "engine": engine,
                    "batch_size": batch_size,
                    "prompt_index": prompt_index,
                    "sync_jit_output_audit": sync_jit_output_audit,
                    **trace_audit,
                }
                trace_comparisons.append(trace_record)
                if trace_audit["status"] == "failed":
                    failed.append(trace_record)

    status = "passed" if not failed else "failed"
    audit = {
        "status": status,
        "completed_at_utc": utc_now(),
        "comparison_count": len(comparisons),
        "exact_count": sum(record["status"] == "exact" for record in comparisons),
        "tie_equivalent_count": len(ties),
        "failed_count": len(failed),
        "tie_logprob_tolerance": args.tie_logprob_tolerance,
        "draft_tie_logit_tolerance": args.draft_tie_logit_tolerance,
        "acceptance_length_relative_tolerance": (
            args.acceptance_length_relative_tolerance
        ),
        "tie_equivalent_divergences": ties,
        "failures": failed,
        "comparisons": comparisons,
        "draft_trace_comparisons": trace_comparisons,
        "draft_trace_summaries": trace_summaries,
    }
    write_json(output_root / "correctness_audit.json", audit)
    return status == "passed"


def metric_total(metrics: Mapping[str, float], metric_name: str) -> float:
    return sum(
        value
        for name, value in metrics.items()
        if name == metric_name or name.startswith(f"{metric_name}{{")
    )


def gpu_sample_summary(path: Path) -> dict[str, dict[str, float | int]]:
    samples: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            gpu = row["gpu_index"]
            values = samples.setdefault(gpu, {"utilization": [], "power": []})
            values["utilization"].append(float(row["utilization_gpu_percent"]))
            values["power"].append(float(row["power_draw_w"]))
    return {
        gpu: {
            "sample_count": len(values["utilization"]),
            "mean_utilization_gpu_percent": statistics.mean(values["utilization"]),
            "mean_power_draw_w": statistics.mean(values["power"]),
        }
        for gpu, values in sorted(samples.items())
        if values["utilization"]
    }


def _load_cell_json(cell_dir: Path, name: str) -> Any:
    return json.loads((cell_dir / name).read_text(encoding="utf-8"))


def _common_cell_failures(cell: Cell, cell_dir: Path) -> list[str]:
    failures = []
    if not cell_is_complete(cell_dir):
        failures.append("missing or inconsistent completion marker")
        return failures
    if (cell_dir / "failure_marker.json").exists():
        failures.append("failure_marker.json is present")
    if not (cell_dir / "shutdown.json").is_file():
        failures.append("shutdown.json is missing")
        return failures
    shutdown = _load_cell_json(cell_dir, "shutdown.json")
    if shutdown.get("exit_code") != 0:
        failures.append(f"server exit_code is {shutdown.get('exit_code')!r}")
    if shutdown.get("forced_kill") is not False:
        failures.append("server required a forced kill")
    result = _load_cell_json(cell_dir, "result.json")
    if result.get("status") != "complete":
        failures.append(f"result status is {result.get('status')!r}")
    if result.get("cell", {}).get("name") != cell.name:
        failures.append("result cell identity does not match directory")
    server_log = cell_dir / "server.log"
    if not server_log.is_file():
        failures.append("server.log is missing")
    else:
        log_text = server_log.read_text(encoding="utf-8", errors="replace")
        child_shutdown_errors = (
            "did not exit; terminating it.",
            "Failed to stop async draft",
            "Failed to shut down async draft runner",
        )
        if any(marker in log_text for marker in child_shutdown_errors):
            failures.append("async Draft child did not shut down cleanly")
    return failures


def audit_lifecycle(output_root: Path) -> bool:
    expectations = {
        "mixed_abort": (32, 30, 2),
        "preemption": (8, 8, 0),
        "chunked_prefill": (4, 4, 0),
    }
    records = []
    failures = []
    for cell in lifecycle_cells():
        cell_dir = output_root / "cells" / cell.name
        cell_failures = _common_cell_failures(cell, cell_dir)
        if not cell_failures:
            result = _load_cell_json(cell_dir, "result.json")
            summary = result["summary"]
            metrics = result["metrics_delta"]
            expected = expectations[cell.variant]
            observed = (
                summary.get("request_count"),
                summary.get("completed_request_count"),
                summary.get("aborted_request_count"),
            )
            if observed != expected:
                cell_failures.append(
                    f"request lifecycle counts are {observed}, expected {expected}"
                )
            hits = metric_total(metrics, "vllm:async_draft_cache_hits_total")
            misses = metric_total(metrics, "vllm:async_draft_cache_misses_total")
            jit = metric_total(metrics, "vllm:async_draft_jit_fallbacks_total")
            preemptions = metric_total(metrics, "vllm:num_preemptions_total")
            if hits <= 0 or misses <= 0:
                cell_failures.append("cache hit and miss paths were not both exercised")
            if jit != misses:
                cell_failures.append(f"JIT fallbacks {jit} != cache misses {misses}")
            if cell.variant == "preemption" and preemptions <= 0:
                cell_failures.append("preemption path was not exercised")
            trace_path = cell_dir / "proposals.jsonl"
            trace_records = (
                sum(1 for line in trace_path.open(encoding="utf-8") if line.strip())
                if trace_path.is_file()
                else 0
            )
            if trace_records <= 0:
                cell_failures.append("proposal trace is empty or missing")
        else:
            hits = misses = jit = preemptions = 0.0
            trace_records = 0
            observed = None
        record = {
            "cell": cell.name,
            "variant": cell.variant,
            "status": "passed" if not cell_failures else "failed",
            "expected_request_counts": expectations[cell.variant],
            "observed_request_counts": observed,
            "cache_hits": hits,
            "cache_misses": misses,
            "jit_fallbacks": jit,
            "preemptions": preemptions,
            "proposal_trace_records": trace_records,
            "failures": cell_failures,
        }
        records.append(record)
        failures.extend(
            {"cell": cell.name, "reason": reason} for reason in cell_failures
        )
    status = "passed" if not failures else "failed"
    write_json(
        output_root / "lifecycle_audit.json",
        {
            "status": status,
            "completed_at_utc": utc_now(),
            "expected_cells": [cell.name for cell in lifecycle_cells()],
            "records": records,
            "failures": failures,
        },
    )
    return not failures


def audit_performance(output_root: Path) -> bool:
    cells = performance_cells()
    missing = [
        cell.name
        for cell in cells
        if not cell_is_complete(output_root / "cells" / cell.name)
    ]
    if missing:
        write_json(
            output_root / "performance_summary.json",
            {"status": "incomplete", "missing_cells": missing},
        )
        return False

    manifest = _load_cell_json(output_root, "manifest.json")
    workload = manifest["workload"]
    expected_requests = int(workload["prompt_count"])
    expected_tokens = expected_requests * int(workload["output_length"])
    rows = []
    semantic_failures = []
    for cell in cells:
        cell_dir = output_root / "cells" / cell.name
        cell_failures = _common_cell_failures(cell, cell_dir)
        result = _load_cell_json(cell_dir, "result.json")
        summary = result["summary"]
        warmup = result.get("warmup", {})
        metrics = result["metrics_delta"]
        command_path = cell_dir / "command.json"
        command = (
            _load_cell_json(cell_dir, "command.json") if command_path.is_file() else []
        )
        if not command:
            cell_failures.append("command.json is empty or missing")
        observed_requests = summary.get("request_count")
        observed_completed = summary.get("completed_request_count")
        observed_aborted = summary.get("aborted_request_count")
        completion_tokens = summary.get("completion_tokens")
        if (observed_requests, observed_completed, observed_aborted) != (
            expected_requests,
            expected_requests,
            0,
        ):
            cell_failures.append(
                "request counts are "
                f"{(observed_requests, observed_completed, observed_aborted)}, "
                f"expected {(expected_requests, expected_requests, 0)}"
            )
        if completion_tokens != expected_tokens:
            cell_failures.append(
                f"completion tokens are {completion_tokens}, expected {expected_tokens}"
            )
        if result.get("expected_completion_tokens") != expected_tokens:
            cell_failures.append("recorded expected completion-token count is wrong")
        if float(warmup.get("seconds", 0.0)) < 30.0:
            cell_failures.append("warmup was shorter than 30 seconds")
        elapsed = float(summary["elapsed_seconds"])
        measured_throughput = float(summary["completion_throughput_tok_s"])
        recomputed_throughput = float(completion_tokens) / elapsed
        if not math.isclose(measured_throughput, recomputed_throughput, rel_tol=1e-9):
            cell_failures.append("completion throughput does not recompute")
        if "--no-async-scheduling" not in command:
            cell_failures.append("--no-async-scheduling is missing")
        if "--no-enable-prefix-caching" not in command:
            cell_failures.append("prefix caching was not explicitly disabled")
        hits = metric_total(metrics, "vllm:async_draft_cache_hits_total")
        misses = metric_total(metrics, "vllm:async_draft_cache_misses_total")
        jit = metric_total(metrics, "vllm:async_draft_jit_fallbacks_total")
        evictions = metric_total(metrics, "vllm:async_draft_cache_evictions_total")
        ipc_bytes = metric_total(metrics, "vllm:async_draft_ipc_bytes_total")
        wait_seconds = metric_total(metrics, "vllm:async_draft_wait_seconds_total")
        branch_seconds = metric_total(
            metrics, "vllm:async_draft_branch_build_seconds_total"
        )
        overlap_name = "vllm:async_draft_overlap_seconds_total"
        overlap_available = any(
            name == overlap_name or name.startswith(f"{overlap_name}{{")
            for name in metrics
        )
        overlap_seconds = metric_total(metrics, overlap_name)
        if cell.mode == "async_cache":
            if hits <= 0 or misses <= 0 or ipc_bytes <= 0 or branch_seconds <= 0:
                cell_failures.append(
                    "async cache/JIT/IPC/branch paths were not exercised"
                )
            if jit != misses:
                cell_failures.append(f"JIT fallbacks {jit} != cache misses {misses}")
        gpu_path = cell_dir / "gpu_samples.csv"
        gpu_samples = gpu_sample_summary(gpu_path) if gpu_path.is_file() else {}
        if not gpu_samples:
            cell_failures.append("GPU utilization samples are empty or missing")
        if str(manifest["topology"]["target_device"]) not in gpu_samples:
            cell_failures.append("Target GPU has no utilization samples")
        if (
            cell.mode == "async_cache"
            and str(manifest["topology"]["draft_device"]) not in gpu_samples
        ):
            cell_failures.append("Draft GPU has no utilization samples")
        semantic_failures.extend(
            {"cell": cell.name, "reason": reason} for reason in cell_failures
        )
        rows.append(
            {
                "cell": cell.name,
                "mode": cell.mode,
                "batch_size": cell.batch_size,
                "repeat": cell.repeat,
                "order": cell.order,
                **summary,
                "warmup_seconds": warmup.get("seconds"),
                "cache_hits": hits,
                "cache_misses": misses,
                "jit_fallbacks": jit,
                "cache_evictions": evictions,
                "ipc_bytes": ipc_bytes,
                "wait_seconds": wait_seconds,
                "branch_build_seconds": branch_seconds,
                "overlap_seconds": overlap_seconds if overlap_available else None,
                "gpu0_mean_utilization_percent": gpu_samples.get("0", {}).get(
                    "mean_utilization_gpu_percent"
                ),
                "gpu1_mean_utilization_percent": gpu_samples.get("1", {}).get(
                    "mean_utilization_gpu_percent"
                ),
                "semantic_status": "passed" if not cell_failures else "failed",
            }
        )
    by_batch: dict[int, dict[str, object]] = {}
    for batch_size in BATCH_SIZES:
        values = {}
        for mode in ("sync", "async_cache"):
            mode_rows = [
                row
                for row in rows
                if row["batch_size"] == batch_size and row["mode"] == mode
            ]
            throughputs = [row["completion_throughput_tok_s"] for row in mode_rows]
            values[mode] = {
                "median_completion_throughput_tok_s": statistics.median(throughputs),
                "mean_completion_throughput_tok_s": statistics.mean(throughputs),
                "stdev_completion_throughput_tok_s": statistics.stdev(throughputs),
                "raw_completion_throughput_tok_s": throughputs,
            }
        sync_median = values["sync"]["median_completion_throughput_tok_s"]
        async_median = values["async_cache"]["median_completion_throughput_tok_s"]
        by_batch[batch_size] = {
            **values,
            "speedup": async_median / sync_median,
            "async_faster": async_median > sync_median,
        }
    semantic_passed = not semantic_failures
    primary_gate_passed = bool(by_batch[1]["async_faster"])
    passed = semantic_passed and primary_gate_passed
    write_json(
        output_root / "performance_summary.json",
        {
            "status": "passed" if passed else "failed",
            "semantic_status": "passed" if semantic_passed else "failed",
            "semantic_failures": semantic_failures,
            "primary_gate": "B=1 async median completion tok/s > sync median",
            "primary_gate_passed": primary_gate_passed,
            "overlap_metric_available": any(
                row["overlap_seconds"] is not None for row in rows
            ),
            "overlap_metric_note": (
                "Formal cells predate the overlap observability metric; "
                "separate post-run probes measure the available overlap window."
            ),
            "by_batch_size": by_batch,
            "raw_rows": rows,
        },
    )
    with (output_root / "performance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_performance_plot(output_root, by_batch)
    return passed


def render_performance_plot(
    output_root: Path,
    by_batch: dict[int, dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    batches = list(BATCH_SIZES)
    sync = [
        by_batch[batch]["sync"]["median_completion_throughput_tok_s"]  # type: ignore[index]
        for batch in batches
    ]
    async_values = [
        by_batch[batch]["async_cache"][  # type: ignore[index]
            "median_completion_throughput_tok_s"
        ]
        for batch in batches
    ]
    x = range(len(batches))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar([value - width / 2 for value in x], sync, width, label="Sync")
    axis.bar(
        [value + width / 2 for value in x],
        async_values,
        width,
        label="Async SSD F=3",
    )
    axis.set_xticks(list(x), [str(batch) for batch in batches])
    axis.set_xlabel("Continuous batch size")
    axis.set_ylabel("Median completion tokens/s")
    axis.set_title("Llama-3.1-8B + EAGLE3 D=7 FP16")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "performance.png", dpi=160)
    plt.close(figure)


def finalize_matrix(args: argparse, output_root: Path) -> bool:
    correctness = audit_correctness(args, output_root)
    lifecycle = audit_lifecycle(output_root)
    performance = audit_performance(output_root)
    complete = correctness and lifecycle and performance
    marker = {
        "status": "complete" if complete else "incomplete",
        "completed_at_utc": utc_now(),
        "correctness": correctness,
        "lifecycle": lifecycle,
        "performance": performance,
    }
    if complete:
        write_json(output_root / "matrix_complete.json", marker)
    else:
        write_json(output_root / "matrix_incomplete.json", marker)
    return complete


def run_cells(
    args: argparse,
    output_root: Path,
    prompts: list[dict[str, Any]],
    cells: list[Cell],
) -> None:
    for index, cell in enumerate(cells):
        run_cell(args, output_root, prompts, cell, args.port_base + index)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prompts = prepare_prompts(args, output_root)
    if args.phase in ("prepare", "all"):
        prepare_manifest(args, output_root, prompts)
        prepare_matrix(output_root)
        if args.phase == "prepare":
            return 0
    if args.phase == "cell":
        if not args.cell:
            raise ValueError("--phase cell requires --cell")
        cells = correctness_cells() + lifecycle_cells() + performance_cells()
        matches = [cell for cell in cells if cell.name == args.cell]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous matrix cell: {args.cell!r}")
        run_cells(args, output_root, prompts, matches)
        return 0
    if args.phase in ("correctness", "all"):
        run_cells(args, output_root, prompts, correctness_cells())
        if not audit_correctness(args, output_root):
            return 2
    if args.phase in ("lifecycle", "all"):
        run_cells(args, output_root, prompts, lifecycle_cells())
        if not audit_lifecycle(output_root):
            return 3
    if args.phase in ("performance", "all"):
        if not audit_correctness(args, output_root):
            raise RuntimeError("Performance is gated on a complete correctness audit")
        run_cells(args, output_root, prompts, performance_cells())
        if not audit_performance(output_root):
            return 4
    if args.phase in ("audit", "all"):
        return 0 if finalize_matrix(args, output_root) else 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
