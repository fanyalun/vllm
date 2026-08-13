# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect and analyze ReplaySSM expert-routing traces on a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "/data1/fanya/Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl"
DEFAULT_OUTPUT_ROOT = (
    "/home/fanya/replayssm_build_artifacts/"
    "expert_routing_qwen36_ep2_20260812_target_only"
)
RUN_SPECS = (
    ("replayssm_ar_bs128", "ar", 128),
    ("replayssm_spec_bs128_d3", "spec", 128),
    ("replayssm_ar_bs512", "ar", 512),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_questions(dataset_path: Path, count: int) -> list[str]:
    questions = []
    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                questions.append(json.loads(line)["question"])
            if len(questions) == count:
                break
    if len(questions) != count:
        raise ValueError(
            f"dataset has only {len(questions)} questions, expected {count}"
        )
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "analyze"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--num-spec", type=int, default=3)
    parser.add_argument("--buffer-len", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--run-name", choices=tuple(spec[0] for spec in RUN_SPECS))
    parser.add_argument("--mode", choices=("ar", "spec"))
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    dataset = Path(args.dataset)
    output_root = Path(args.output_root)
    if not model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model}")
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    load_questions(dataset, 512)
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    disk = os.statvfs(output_root.parent)
    free_bytes = disk.f_bavail * disk.f_frsize
    if free_bytes < 10 * 1024**3:
        raise OSError(f"less than 10 GiB free under {output_root.parent}")
    return {
        "timestamp": utc_now(),
        "model": str(model.resolve()),
        "dataset": str(dataset.resolve()),
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "gpus": gpu_result.stdout.strip().splitlines(),
        "topology": topology.stdout,
        "free_bytes": free_bytes,
        "contract": {
            "tensor_parallel_size": 2,
            "expert_parallel_size": 2,
            "data_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "all2all_backend": "allgather_reducescatter",
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "num_speculative_tokens": args.num_spec,
            "replayssm_buffer_len": args.buffer_len,
            "dtype": "bfloat16",
            "mamba_ssm_cache_dtype": "float32",
            "sampling": "greedy, ignore_eos, seed=0",
            "route_scope": "target model decode/verify; excludes MTP drafter",
        },
    }


def worker_command(
    args: argparse.Namespace, run_name: str, mode: str, batch_size: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--output-root",
        args.output_root,
        "--run-name",
        run_name,
        "--mode",
        mode,
        "--batch-size",
        str(batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--num-spec",
        str(args.num_spec),
        "--buffer-len",
        str(args.buffer_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]


def run_worker_subprocess(
    args: argparse.Namespace, run_name: str, mode: str, batch_size: int
) -> None:
    output_root = Path(args.output_root)
    log_path = output_root / "logs" / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = worker_command(args, run_name, mode, batch_size)
    with log_path.open("x", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{run_name} worker exited with code {return_code}")


def validate_worker_output(
    output_root: Path,
    run_name: str,
    mode: str,
    batch_size: int,
    max_tokens: int,
) -> dict[str, Any]:
    trace_dir = output_root / "trace" / run_name
    manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
    if manifest["state"] != "complete":
        raise AssertionError(f"trace is not complete for {run_name}")
    if manifest["route_shape"][1:] != [40, 8]:
        raise AssertionError(
            f"unexpected route shape for {run_name}: {manifest['route_shape']}"
        )
    if manifest["route_shape"][0] <= 0:
        raise AssertionError(f"trace has no decode rows for {run_name}")
    output_rows = [
        json.loads(line)
        for line in (trace_dir / "outputs.jsonl").read_text().splitlines()
    ]
    if len(output_rows) != batch_size:
        raise AssertionError(
            f"expected {batch_size} outputs for {run_name}, got {len(output_rows)}"
        )
    if any(len(row["token_ids"]) != max_tokens for row in output_rows):
        raise AssertionError(
            f"not all {run_name} requests produced {max_tokens} tokens"
        )
    route_kinds: set[str] = set()
    all_accepted = True
    with (trace_dir / "events.jsonl").open(encoding="utf-8") as event_file:
        for line in event_file:
            event = json.loads(line)
            route_kinds.update(event["route_kinds"])
            all_accepted &= all(event["accepted"])
    expected_kinds = (
        {"ar_decode"}
        if mode == "ar"
        else {
            "spec_target",
            "spec_draft_1",
            "spec_draft_2",
            "spec_draft_3",
        }
    )
    if route_kinds != expected_kinds:
        raise AssertionError(
            f"unexpected route kinds for {run_name}: {sorted(route_kinds)}"
        )
    if mode == "ar" and not all_accepted:
        raise AssertionError(f"AR trace contains rejected rows for {run_name}")
    return {
        "route_rows": manifest["route_shape"][0],
        "events": manifest["num_events"],
        "outputs": len(output_rows),
        "route_kinds": sorted(route_kinds),
    }


def run_all(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = preflight(args)
    manifest.update(
        {
            "state": "running",
            "run_specs": [
                {"run_name": name, "mode": mode, "batch_size": batch_size}
                for name, mode, batch_size in RUN_SPECS
            ],
        }
    )
    write_json(output_root / "run_manifest.json", manifest)
    status = {"state": "running", "started_at": utc_now(), "runs": {}}
    write_json(output_root / "status.json", status)
    try:
        for run_name, mode, batch_size in RUN_SPECS:
            status["runs"][run_name] = {
                "state": "running",
                "started_at": utc_now(),
            }
            write_json(output_root / "status.json", status)
            run_worker_subprocess(args, run_name, mode, batch_size)
            validation = validate_worker_output(
                output_root,
                run_name,
                mode,
                batch_size,
                args.max_tokens,
            )
            status["runs"][run_name].update(
                {
                    "state": "complete",
                    "completed_at": utc_now(),
                    "validation": validation,
                }
            )
            write_json(output_root / "status.json", status)
        analyze(args)
        status.update({"state": "complete", "completed_at": utc_now()})
        manifest.update({"state": "complete", "completed_at": utc_now()})
        write_json(output_root / "status.json", status)
        write_json(output_root / "run_manifest.json", manifest)
    except BaseException as error:
        failure = {
            "state": "failed",
            "timestamp": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        status.update({"state": "failed", "completed_at": utc_now()})
        manifest.update({"state": "failed", "completed_at": utc_now()})
        write_json(output_root / "failure.json", failure)
        write_json(output_root / "status.json", status)
        write_json(output_root / "run_manifest.json", manifest)
        raise


def run_worker(args: argparse.Namespace) -> None:
    if args.run_name is None or args.mode is None or args.batch_size is None:
        raise ValueError("worker requires --run-name, --mode, and --batch-size")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    venv_bin = str(Path(sys.executable).absolute().parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    import matplotlib
    import numpy
    import torch

    from vllm import LLM, SamplingParams

    trace_dir = Path(args.output_root).resolve() / "trace" / args.run_name
    questions = load_questions(Path(args.dataset), args.batch_size)
    messages = [[{"role": "user", "content": question}] for question in questions]
    trace_config = {
        "output_dir": str(trace_dir),
        "run_name": args.run_name,
        "decode_only": True,
        "completion_marker": "worker_complete",
    }
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "all2all_backend": "allgather_reducescatter",
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "float32",
        "language_model_only": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "enforce_eager": True,
        "enable_return_routed_experts": True,
        "disable_log_stats": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 0,
        "additional_config": {
            "gdn_prefill_backend": "triton",
            "routed_experts_trace": trace_config,
        },
        "kernel_config": {"enable_flashinfer_autotune": False},
        "replayssm_buffer_len": args.buffer_len,
    }
    if args.mode == "ar":
        llm_kwargs["use_replayssm"] = True
    else:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_spec,
        }
        llm_kwargs["use_replayssm_spec"] = True

    llm = None
    try:
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            ignore_eos=True,
            seed=0,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if len(outputs) != args.batch_size:
            raise AssertionError(
                f"expected {args.batch_size} outputs, got {len(outputs)}"
            )
        output_path = trace_dir / "outputs.jsonl"
        with output_path.open("x", encoding="utf-8") as output_file:
            for question_index, output in enumerate(outputs):
                completion = output.outputs[0]
                if len(completion.token_ids) != args.max_tokens:
                    raise AssertionError(
                        f"request {question_index} produced "
                        f"{len(completion.token_ids)} tokens"
                    )
                output_file.write(
                    json.dumps(
                        {
                            "question_index": question_index,
                            "request_id": output.request_id,
                            "token_ids": list(completion.token_ids),
                            "text": completion.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        worker_summary = {
            "run_name": args.run_name,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "elapsed_seconds_diagnostic_only": elapsed,
            "numpy_version": numpy.__version__,
            "matplotlib_version": matplotlib.__version__,
            "torch_version": torch.__version__,
            "completed_at": utc_now(),
        }
        write_json(trace_dir / "worker_summary.json", worker_summary)
        (trace_dir / "worker_complete").touch(exist_ok=False)
        print("WORKER_SUMMARY " + json.dumps(worker_summary), flush=True)
    except BaseException as error:
        failure = {
            "state": "failed",
            "timestamp": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if trace_dir.exists():
            write_json(trace_dir / "worker_failure.json", failure)
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def analyze(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from expert_routing_analysis import analyze_experiment

    analyze_experiment(
        Path(args.output_root).resolve(),
        bootstrap_samples=args.bootstrap_samples,
    )
    output_root = Path(args.output_root).resolve()
    completed_at = utc_now()
    for filename in ("status.json", "run_manifest.json"):
        path = output_root / filename
        payload = json.loads(path.read_text())
        payload.update({"state": "complete", "completed_at": completed_at})
        write_json(path, payload)
    failure_path = output_root / "failure.json"
    if failure_path.exists():
        recovered_path = output_root / "logs" / "recovered_analysis_failure.json"
        os.replace(failure_path, recovered_path)


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_all(args)
    elif args.command == "worker":
        run_worker(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
