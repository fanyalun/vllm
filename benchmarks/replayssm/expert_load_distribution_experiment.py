# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare target-model expert loads for equal-token Spec16+d3 and AR64."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_QWEN_MODEL = "/data1/fanya/Qwen/Qwen3.6-35B-A3B"
DEFAULT_GEMMA_MODEL = (
    "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it"
)
DEFAULT_GEMMA_DRAFTER = (
    "/home/fanya/data1/fanya/models/"
    "gemma-4-26B-A4B-it-speculator.eagle3"
)
DEFAULT_DATASET = "/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl"
DEFAULT_OUTPUT_PARENT = "/home/fanya/replayssm_build_artifacts"
SAMPLE_COUNT = 64
DP_SIZE = 2
TOP_K = 8
MAX_TOKENS = 64
MAX_MODEL_LEN = 512
MAX_NUM_BATCHED_TOKENS = 4096
SPEC_STAGES = (
    "spec_target",
    "spec_draft_1",
    "spec_draft_2",
    "spec_draft_3",
)
INTERNAL_REQUEST_ID = re.compile(r"^(?P<external>.*)-[0-9a-f]{8}$")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_path: str
    drafter_path: str | None
    draft_method: str
    num_layers: int
    num_experts: int
    selected_layers: tuple[int, int, int]


@dataclass(frozen=True)
class TraceRows:
    manifest: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    routes: np.ndarray


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any("question" not in row for row in rows):
        raise ValueError(f"dataset does not contain JSONL question rows: {path}")
    return rows


def build_sample_manifest(
    dataset_path: Path,
    *,
    seed: int = 0,
    count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    rows = load_dataset(dataset_path)
    if len(rows) < count:
        raise ValueError(f"dataset has {len(rows)} rows, expected at least {count}")
    selected = np.random.default_rng(seed).choice(
        len(rows), size=count, replace=False
    )
    samples = []
    for sample_id, source_index_value in enumerate(selected.tolist()):
        source_index = int(source_index_value)
        question = str(rows[source_index]["question"])
        samples.append(
            {
                "sample_id": sample_id,
                "sample_order": sample_id,
                "source_line_number": source_index + 1,
                "source_index": source_index,
                "question_sha256": sha256_text(question),
                "question": question,
                "data_parallel_rank": sample_id % DP_SIZE,
                "rank_order": sample_id // DP_SIZE,
            }
        )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset": str(dataset_path.resolve()),
        "dataset_rows": len(rows),
        "sampling": {
            "library": "numpy.random.default_rng",
            "seed": seed,
            "replace": False,
            "count": count,
        },
        "rank_assignment": "even sample_order to rank 0; odd to rank 1",
        "samples": samples,
    }


def validate_sample_manifest(payload: dict[str, Any]) -> None:
    samples = payload.get("samples", [])
    if len(samples) != SAMPLE_COUNT:
        raise AssertionError(
            f"sample manifest has {len(samples)} samples, expected {SAMPLE_COUNT}"
        )
    ids = [int(sample["sample_id"]) for sample in samples]
    source_indices = [int(sample["source_index"]) for sample in samples]
    if ids != list(range(SAMPLE_COUNT)):
        raise AssertionError("sample IDs are not the stable order 0..63")
    if len(set(source_indices)) != SAMPLE_COUNT:
        raise AssertionError("sample manifest contains duplicate source rows")
    for sample in samples:
        sample_id = int(sample["sample_id"])
        if int(sample["data_parallel_rank"]) != sample_id % DP_SIZE:
            raise AssertionError("sample manifest DP rank assignment is unstable")
        if sha256_text(str(sample["question"])) != sample["question_sha256"]:
            raise AssertionError("sample question hash mismatch")


def model_specs(args: argparse.Namespace) -> dict[str, ModelSpec]:
    return {
        "qwen36": ModelSpec(
            key="qwen36",
            model_path=args.qwen_model,
            drafter_path=None,
            draft_method="mtp_replayssm",
            num_layers=40,
            num_experts=256,
            selected_layers=(4, 20, 36),
        ),
        "gemma4": ModelSpec(
            key="gemma4",
            model_path=args.gemma_model,
            drafter_path=args.gemma_drafter,
            draft_method="eagle3",
            num_layers=30,
            num_experts=128,
            selected_layers=(3, 15, 27),
        ),
    }


def model_spec_from_args(args: argparse.Namespace) -> ModelSpec:
    return model_specs(args)[args.model_family]


def cell_names() -> tuple[tuple[str, str], ...]:
    return (
        ("qwen36", "ar"),
        ("qwen36", "spec"),
        ("gemma4", "ar"),
        ("gemma4", "spec"),
    )


def attempt_number(path: Path) -> int:
    return int(path.name.removeprefix("attempt_"))


def attempts_for(cell_dir: Path) -> list[Path]:
    return sorted(
        (path for path in cell_dir.glob("attempt_[0-9][0-9][0-9]") if path.is_dir()),
        key=attempt_number,
    )


def next_attempt_dir(cell_dir: Path) -> Path:
    attempts = attempts_for(cell_dir)
    next_number = attempt_number(attempts[-1]) + 1 if attempts else 1
    return cell_dir / f"attempt_{next_number:03d}"


def rank_dir(attempt_dir: Path, rank: int) -> Path:
    return attempt_dir / f"rank_{rank}"


def load_outputs(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_trace(path: Path) -> TraceRows:
    manifest = read_json(path / "trace_manifest.json")
    dtype = np.dtype(manifest["route_dtype"])
    shape = tuple(int(value) for value in manifest["route_shape"])
    routes = np.fromfile(path / manifest["routes_file"], dtype=dtype)
    expected_size = math.prod(shape)
    if routes.size != expected_size:
        raise AssertionError(
            f"{path}: route file has {routes.size} values, expected {expected_size}"
        )
    routes = routes.reshape(shape)
    events = tuple(
        json.loads(line)
        for line in (path / manifest["events_file"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    return TraceRows(manifest=manifest, events=events, routes=routes)


def validate_rank_artifacts(
    path: Path,
    *,
    rank: int,
    model: ModelSpec,
    mode: str,
) -> dict[str, Any]:
    required = (
        path / "worker_complete",
        path / "worker_summary.json",
        path / "outputs.jsonl",
        path / "trace" / "trace_manifest.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise AssertionError(f"rank {rank} missing artifacts: {missing}")
    trace = load_trace(path / "trace")
    manifest = trace.manifest
    expected_method = model.draft_method if mode == "spec" else "none"
    expected = {
        "state": "complete",
        "data_parallel_rank": rank,
        "model_family": model.key,
        "draft_method": expected_method,
        "num_layers": model.num_layers,
        "num_experts": model.num_experts,
        "top_k": TOP_K,
        "expected_target_moe_layers": model.num_layers,
        "tensor_parallel_size": 1,
        "data_parallel_size": DP_SIZE,
        "expert_parallel_size": DP_SIZE,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"rank {rank} trace manifest mismatch: {mismatches}")
    if tuple(manifest["route_shape"][1:]) != (model.num_layers, TOP_K):
        raise AssertionError(f"rank {rank} has invalid route shape")
    outputs = load_outputs(path / "outputs.jsonl")
    if len(outputs) != SAMPLE_COUNT // DP_SIZE:
        raise AssertionError(f"rank {rank} has {len(outputs)} outputs, expected 32")
    if any(len(output["token_ids"]) != MAX_TOKENS for output in outputs):
        raise AssertionError(f"rank {rank} does not have exactly 64 output tokens")
    expected_samples = set(range(rank, SAMPLE_COUNT, DP_SIZE))
    actual_samples = {int(output["sample_id"]) for output in outputs}
    if actual_samples != expected_samples:
        raise AssertionError(f"rank {rank} output sample IDs are incomplete")
    return {
        "rank": rank,
        "outputs": len(outputs),
        "route_rows": int(manifest["route_shape"][0]),
        "events": int(manifest["num_events"]),
    }


def validate_attempt(
    attempt_dir: Path,
    model: ModelSpec,
    mode: str,
    sample_manifest_path: Path,
) -> dict[str, Any]:
    attempt_manifest = read_json(attempt_dir / "attempt_manifest.json")
    if Path(attempt_manifest["sample_manifest"]).resolve() != (
        sample_manifest_path.resolve()
    ):
        raise AssertionError("attempt references a different sample manifest")
    ranks = [
        validate_rank_artifacts(
            rank_dir(attempt_dir, rank), rank=rank, model=model, mode=mode
        )
        for rank in range(DP_SIZE)
    ]
    return {"attempt": attempt_dir.name, "ranks": ranks}


def find_complete_attempt(
    cell_dir: Path,
    model: ModelSpec,
    mode: str,
    sample_manifest_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    for attempt_dir in reversed(attempts_for(cell_dir)):
        try:
            validation = validate_attempt(
                attempt_dir, model, mode, sample_manifest_path
            )
        except (AssertionError, FileNotFoundError, KeyError, ValueError):
            continue
        return attempt_dir, validation
    return None


def command_for_attempt(
    args: argparse.Namespace,
    model: ModelSpec,
    mode: str,
    attempt_dir: Path,
    sample_manifest_path: Path,
) -> list[str]:
    command = [
        str(Path(sys.executable).absolute()),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "worker",
        "--model-family",
        model.key,
        "--mode",
        mode,
        "--attempt-dir",
        str(attempt_dir),
        "--sample-manifest",
        str(sample_manifest_path),
        "--qwen-model",
        args.qwen_model,
        "--gemma-model",
        args.gemma_model,
        "--gemma-drafter",
        args.gemma_drafter,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    return command


def run_subprocess_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    if return_code:
        raise RuntimeError(f"subprocess exited with code {return_code}")


def run_command(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def preflight(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    specs = model_specs(args)
    paths = {
        "qwen_model": Path(args.qwen_model),
        "gemma_model": Path(args.gemma_model),
        "gemma_drafter": Path(args.gemma_drafter),
        "dataset": Path(args.dataset),
        "venv_python": Path(sys.executable),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"preflight paths do not exist: {missing}")
    rows = load_dataset(paths["dataset"])
    if len(rows) != 1319:
        raise AssertionError(f"expected 1319 GSM8K rows, found {len(rows)}")
    gpu_lines = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,memory.total,memory.used,"
            "utilization.gpu",
            "--format=csv,noheader",
        ]
    ).splitlines()
    if len(gpu_lines) != DP_SIZE:
        raise AssertionError(f"expected two GPUs, found {len(gpu_lines)}")
    topology = run_command(["nvidia-smi", "topo", "-m"])
    disk = shutil.disk_usage(output_root.parent)
    if disk.free < 10 * 1024**3:
        raise OSError(f"less than 10 GiB free under {output_root.parent}")
    import vllm

    dry_validations = dry_validate_configs(args, specs)

    return {
        "timestamp": utc_now(),
        "git_branch": run_command(["git", "branch", "--show-current"]),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_command(["git", "status", "--short"]),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "vllm_import_path": str(Path(vllm.__file__).resolve()),
        "platform": platform.platform(),
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "dataset_rows": len(rows),
        "gpus": gpu_lines,
        "topology": topology,
        "disk_free_bytes": disk.free,
        "models": {key: asdict(value) for key, value in specs.items()},
        "engine_config_dry_validation": dry_validations,
        "contract": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 2,
            "expert_parallel_size": 2,
            "spec_global_batch_size": 16,
            "spec_max_num_seqs_per_rank": 8,
            "ar_global_batch_size": 64,
            "ar_max_num_seqs_per_rank": 32,
            "max_tokens": MAX_TOKENS,
            "min_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens_per_rank": MAX_NUM_BATCHED_TOKENS,
            "all2all_backend": "allgather_reducescatter",
            "enable_eplb": False,
            "enable_dbo": False,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "enforce_eager": True,
        },
    }


def dry_validate_configs(
    args: argparse.Namespace, specs: dict[str, ModelSpec]
) -> list[dict[str, Any]]:
    from vllm.engine.arg_utils import EngineArgs

    env_defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29599",
    }
    original = {key: os.environ.get(key) for key in env_defaults}
    os.environ.update(env_defaults)
    validations = []
    try:
        for family, mode in cell_names():
            worker_args = argparse.Namespace(**vars(args))
            worker_args.model_family = family
            worker_args.mode = mode
            kwargs = worker_llm_kwargs(
                worker_args, specs[family], Path("/dry-run"), rank=0
            )
            config = EngineArgs(**kwargs).create_engine_config()
            validations.append(
                {
                    "model_family": family,
                    "mode": mode,
                    "tensor_parallel_size": (
                        config.parallel_config.tensor_parallel_size
                    ),
                    "data_parallel_size": (
                        config.parallel_config.data_parallel_size
                    ),
                    "expert_parallel_size": (
                        config.parallel_config.world_size
                        if config.parallel_config.enable_expert_parallel
                        else 1
                    ),
                    "num_speculative_tokens": config.num_speculative_tokens,
                    "num_hidden_layers": int(
                        config.model_config.hf_text_config.num_hidden_layers
                    ),
                    "num_experts": specs[family].num_experts,
                    "top_k": TOP_K,
                    "state": "valid",
                }
            )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return validations


def run_all(args: argparse.Namespace) -> None:
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = (
            Path(DEFAULT_OUTPUT_PARENT)
            / f"expert_load_distribution_{timestamp_slug()}"
        )
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"output root already exists; pass --resume to reuse: {output_root}"
        )
    if args.resume and (output_root / "experiment_complete").is_file():
        status = read_json(output_root / "status.json")
        manifest = read_json(output_root / "run_manifest.json")
        if status.get("state") != "complete" or manifest.get("state") != "complete":
            raise AssertionError(
                "completion marker exists but manifests are incomplete"
            )
        print(f"EXPERIMENT_ALREADY_COMPLETE {output_root}", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    run_manifest_path = output_root / "run_manifest.json"
    sample_manifest_path = output_root / "sample_manifest.json"
    try:
        preflight_payload = preflight(args, output_root)
        if sample_manifest_path.exists():
            samples = read_json(sample_manifest_path)
            validate_sample_manifest(samples)
            if Path(samples["dataset"]).resolve() != Path(args.dataset).resolve():
                raise AssertionError("resume dataset differs from sample manifest")
        else:
            samples = build_sample_manifest(Path(args.dataset).resolve())
            validate_sample_manifest(samples)
            write_json(sample_manifest_path, samples)
        manifest = {
            **preflight_payload,
            "state": "running",
            "output_root": str(output_root),
            "sample_manifest": str(sample_manifest_path),
            "cells": [
                {"model_family": family, "mode": mode}
                for family, mode in cell_names()
            ],
        }
        write_json(run_manifest_path, manifest)
        status: dict[str, Any] = {
            "state": "running",
            "started_at": utc_now(),
            "cells": {},
        }
        write_json(status_path, status)
        specs = model_specs(args)
        selected_attempts: dict[str, str] = {}
        for family, mode in cell_names():
            model = specs[family]
            cell_key = f"{family}/{mode}"
            cell_dir = output_root / "cells" / family / mode
            complete = find_complete_attempt(
                cell_dir, model, mode, sample_manifest_path
            )
            if complete is not None and args.resume:
                complete_dir, validation = complete
                status["cells"][cell_key] = {
                    "state": "complete",
                    "resumed": True,
                    "attempt": complete_dir.name,
                    "validation": validation,
                }
                selected_attempts[cell_key] = str(complete_dir)
                write_json(status_path, status)
                continue
            attempt_dir = next_attempt_dir(cell_dir)
            attempt_dir.mkdir(parents=True)
            attempt_manifest = {
                "state": "running",
                "started_at": utc_now(),
                "model_family": family,
                "mode": mode,
                "model": asdict(model),
                "sample_manifest": str(sample_manifest_path),
            }
            write_json(attempt_dir / "attempt_manifest.json", attempt_manifest)
            status["cells"][cell_key] = {
                "state": "running",
                "attempt": attempt_dir.name,
                "started_at": utc_now(),
            }
            write_json(status_path, status)
            command = command_for_attempt(
                args, model, mode, attempt_dir, sample_manifest_path
            )
            write_json(attempt_dir / "launch_command.json", {"argv": command})
            try:
                run_subprocess_logged(command, attempt_dir / "torchrun.log")
                validation = validate_attempt(
                    attempt_dir, model, mode, sample_manifest_path
                )
                (attempt_dir / "cell_complete").touch(exist_ok=False)
                attempt_manifest.update(
                    {
                        "state": "complete",
                        "completed_at": utc_now(),
                        "validation": validation,
                    }
                )
                write_json(
                    attempt_dir / "attempt_manifest.json", attempt_manifest
                )
                status["cells"][cell_key].update(
                    {
                        "state": "complete",
                        "completed_at": utc_now(),
                        "validation": validation,
                    }
                )
                selected_attempts[cell_key] = str(attempt_dir)
                write_json(status_path, status)
            except BaseException as error:
                failure = {
                    "state": "failed",
                    "timestamp": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                write_json(attempt_dir / "failure.json", failure)
                attempt_manifest.update(failure)
                write_json(
                    attempt_dir / "attempt_manifest.json", attempt_manifest
                )
                status["cells"][cell_key].update(failure)
                write_json(status_path, status)
                raise
        write_json(output_root / "selected_attempts.json", selected_attempts)
        analyze_experiment(output_root, specs, selected_attempts)
        status.update({"state": "complete", "completed_at": utc_now()})
        manifest.update({"state": "complete", "completed_at": utc_now()})
        write_json(status_path, status)
        write_json(run_manifest_path, manifest)
        (output_root / "experiment_complete").touch(exist_ok=False)
        print(f"EXPERIMENT_COMPLETE {output_root}", flush=True)
    except BaseException as error:
        failure = {
            "state": "failed",
            "timestamp": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(output_root / "failure.json", failure)
        if status_path.exists():
            status = read_json(status_path)
            status.update(failure)
            write_json(status_path, status)
        if run_manifest_path.exists():
            manifest = read_json(run_manifest_path)
            manifest.update(failure)
            write_json(run_manifest_path, manifest)
        raise


def worker_llm_kwargs(
    args: argparse.Namespace,
    model: ModelSpec,
    trace_dir: Path,
    rank: int,
) -> dict[str, Any]:
    trace_config = {
        "output_dir": str(trace_dir),
        "run_name": f"{model.key}_{args.mode}_rank{rank}",
        "decode_only": True,
        "completion_marker": "../worker_complete",
        "data_parallel_rank": rank,
        "model_family": model.key,
        "draft_method": model.draft_method if args.mode == "spec" else "none",
        "expected_target_moe_layers": model.num_layers,
    }
    kwargs: dict[str, Any] = {
        "model": model.model_path,
        "tensor_parallel_size": 1,
        "data_parallel_size": DP_SIZE,
        "distributed_executor_backend": "external_launcher",
        "enable_expert_parallel": True,
        "all2all_backend": "allgather_reducescatter",
        "enable_eplb": False,
        "enable_dbo": False,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "language_model_only": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": 8 if args.mode == "spec" else 32,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "enforce_eager": True,
        "enable_return_routed_experts": True,
        "disable_log_stats": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 0,
        "additional_config": {"routed_experts_trace": trace_config},
    }
    if model.key == "qwen36":
        kwargs.update(
            {
                "mamba_ssm_cache_dtype": "float32",
                "replayssm_buffer_len": 16,
            }
        )
        if args.mode == "ar":
            kwargs["use_replayssm"] = True
        else:
            kwargs["use_replayssm_spec"] = True
            kwargs["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": 3,
            }
    elif args.mode == "spec":
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": model.drafter_path,
            "num_speculative_tokens": 3,
        }
    return kwargs


def run_worker(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank not in (0, 1) or local_rank not in (0, 1) or world_size != DP_SIZE:
        raise RuntimeError(
            f"worker requires two-rank torchrun, got rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )
    attempt_dir = Path(args.attempt_dir).resolve()
    output_dir = rank_dir(attempt_dir, rank)
    output_dir.mkdir(parents=True, exist_ok=False)
    worker_log = output_dir / "worker.log"
    append_jsonl(
        worker_log,
        {"event": "worker_start", "timestamp": utc_now(), "rank": rank},
    )
    model = model_spec_from_args(args)
    sample_manifest = read_json(Path(args.sample_manifest))
    validate_sample_manifest(sample_manifest)
    samples = [
        sample
        for sample in sample_manifest["samples"]
        if int(sample["data_parallel_rank"]) == rank
    ]
    samples.sort(key=lambda sample: int(sample["rank_order"]))
    if len(samples) != SAMPLE_COUNT // DP_SIZE:
        raise AssertionError(f"rank {rank} did not receive 32 samples")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    venv_bin = str(Path(sys.executable).absolute().parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM, SamplingParams

    llm = None
    trace_dir = output_dir / "trace"
    try:
        llm_kwargs = worker_llm_kwargs(args, model, trace_dir, rank)
        write_json(
            output_dir / "worker_config.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "model_family": model.key,
                "mode": args.mode,
                "sample_ids": [int(sample["sample_id"]) for sample in samples],
                "llm_kwargs": llm_kwargs,
            },
        )
        llm = LLM(**llm_kwargs)
        configured_rank = (
            llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
        )
        if configured_rank != rank:
            raise AssertionError(
                f"vLLM configured DP rank {configured_rank}, torch rank is {rank}"
            )
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            min_tokens=MAX_TOKENS,
            ignore_eos=True,
            seed=0,
        )
        messages = [
            [{"role": "user", "content": str(sample["question"])}]
            for sample in samples
        ]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
        if len(outputs) != len(samples):
            raise AssertionError(
                f"rank {rank} produced {len(outputs)} outputs, expected 32"
            )
        with (output_dir / "outputs.jsonl").open("x", encoding="utf-8") as file:
            for sample, output in zip(samples, outputs, strict=True):
                completion = output.outputs[0]
                if len(completion.token_ids) != MAX_TOKENS:
                    raise AssertionError(
                        f"sample {sample['sample_id']} produced "
                        f"{len(completion.token_ids)} tokens"
                    )
                prompt_token_ids = list(output.prompt_token_ids or [])
                file.write(
                    json.dumps(
                        {
                            "sample_id": int(sample["sample_id"]),
                            "source_line_number": int(
                                sample["source_line_number"]
                            ),
                            "question_sha256": sample["question_sha256"],
                            "request_id": output.request_id,
                            "prompt_token_count": len(prompt_token_ids),
                            "prompt_token_ids": prompt_token_ids,
                            "token_ids": list(completion.token_ids),
                            "text": completion.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary = {
            "state": "complete",
            "completed_at": utc_now(),
            "model_family": model.key,
            "mode": args.mode,
            "data_parallel_rank": rank,
            "outputs": len(outputs),
            "max_tokens": MAX_TOKENS,
        }
        write_json(output_dir / "worker_summary.json", summary)
        (output_dir / "worker_complete").touch(exist_ok=False)
        append_jsonl(worker_log, {"event": "worker_complete", **summary})
        print("WORKER_COMPLETE " + json.dumps(summary), flush=True)
    except BaseException as error:
        failure = {
            "state": "failed",
            "timestamp": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "rank": rank,
        }
        write_json(output_dir / "worker_failure.json", failure)
        append_jsonl(worker_log, {"event": "worker_failed", **failure})
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def event_rows(
    trace: TraceRows,
    outputs: list[dict[str, Any]],
    rank: int,
) -> list[dict[str, Any]]:
    output_by_request = {str(row["request_id"]): row for row in outputs}
    rows: list[dict[str, Any]] = []
    for event in trace.events:
        request_id = str(event["request_id"])
        output_request_id = request_id
        if output_request_id not in output_by_request:
            match = INTERNAL_REQUEST_ID.fullmatch(request_id)
            if match is not None:
                output_request_id = match["external"]
        if output_request_id not in output_by_request:
            raise AssertionError(f"trace request is absent from outputs: {request_id}")
        output = output_by_request[output_request_id]
        count = int(event["row_count"])
        offset = int(event["binary_row_offset"])
        arrays = (
            event["absolute_positions"],
            event["token_ids"],
            event["row_in_request"],
            event["route_kinds"],
            event["accepted"],
        )
        if any(len(array) != count for array in arrays):
            raise AssertionError("trace event metadata lengths do not match")
        for row_index in range(count):
            absolute_position = int(event["absolute_positions"][row_index])
            rows.append(
                {
                    "rank": rank,
                    "scheduler_step": int(event["scheduler_step"]),
                    "request_id": request_id,
                    "sample_id": int(output["sample_id"]),
                    "prompt_token_count": int(output["prompt_token_count"]),
                    "absolute_position": absolute_position,
                    "generation_position": absolute_position
                    - int(output["prompt_token_count"]),
                    "token_id": int(event["token_ids"][row_index]),
                    "row_in_request": int(event["row_in_request"][row_index]),
                    "route_kind": str(event["route_kinds"][row_index]),
                    "accepted": bool(event["accepted"][row_index]),
                    "routes": trace.routes[offset + row_index],
                }
            )
    if len(rows) != int(trace.manifest["route_shape"][0]):
        raise AssertionError("events do not cover every binary route row")
    return rows


def group_by_step(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["scheduler_step"])].append(row)
    return dict(grouped)


def _position_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    positions = np.asarray(
        [int(row["generation_position"]) for row in rows], dtype=np.int64
    )
    return {
        "min": int(positions.min()),
        "median": float(np.median(positions)),
        "max": int(positions.max()),
    }


def validate_step(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    model: ModelSpec,
) -> dict[str, Any]:
    reasons: list[str] = []
    rank_counts = Counter(int(row["rank"]) for row in rows)
    kinds = Counter(str(row["route_kind"]) for row in rows)
    requests = {(int(row["rank"]), str(row["request_id"])) for row in rows}
    identities = [
        (int(row["rank"]), str(row["request_id"]), str(row["route_kind"]))
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        reasons.append("duplicate request/stage row")
    if set(rank_counts) != {0, 1}:
        reasons.append("missing DP rank")
    if mode == "spec":
        if rank_counts != Counter({0: 32, 1: 32}):
            reasons.append(f"rank row counts are {dict(rank_counts)}, expected 32 each")
        if kinds != Counter({stage: 16 for stage in SPEC_STAGES}):
            reasons.append(f"stage counts are {dict(kinds)}, expected 16 each")
        if len(requests) != 16:
            reasons.append(f"active request count is {len(requests)}, expected 16")
        for request in requests:
            request_kinds = {
                str(row["route_kind"])
                for row in rows
                if (int(row["rank"]), str(row["request_id"])) == request
            }
            if request_kinds != set(SPEC_STAGES):
                reasons.append(f"request {request} does not contain all spec stages")
                break
        position_rows = [
            row for row in rows if row["route_kind"] == "spec_target"
        ]
    else:
        if rank_counts != Counter({0: 32, 1: 32}):
            reasons.append(f"rank row counts are {dict(rank_counts)}, expected 32 each")
        if kinds != Counter({"ar_decode": 64}):
            reasons.append(f"route kinds are {dict(kinds)}, expected 64 AR rows")
        if len(requests) != SAMPLE_COUNT:
            reasons.append(f"active request count is {len(requests)}, expected 64")
        position_rows = rows
    if rows:
        for layer in model.selected_layers:
            assignments = sum(int(row["routes"][layer].size) for row in rows)
            if assignments != SAMPLE_COUNT * TOP_K:
                reasons.append(
                    f"layer {layer} has {assignments} assignments, expected 512"
                )
            if any(
                len(set(int(value) for value in row["routes"][layer])) != TOP_K
                for row in rows
            ):
                reasons.append(f"layer {layer} contains duplicate expert IDs")
            if any(
                int(row["routes"][layer].min()) < 0
                or int(row["routes"][layer].max()) >= model.num_experts
                for row in rows
            ):
                reasons.append(f"layer {layer} contains out-of-range expert IDs")
    summary: dict[str, Any] = {
        "valid": not reasons,
        "reasons": reasons,
        "rows": len(rows),
        "rank_rows": dict(sorted(rank_counts.items())),
        "stage_rows": dict(sorted(kinds.items())),
        "active_requests": len(requests),
        "sample_ids": sorted({int(row["sample_id"]) for row in rows}),
    }
    if position_rows:
        summary["position"] = _position_summary(position_rows)
    if mode == "spec":
        draft_rows = [row for row in rows if row["route_kind"] != "spec_target"]
        accepted = sum(bool(row["accepted"]) for row in draft_rows)
        summary["draft_acceptance"] = {
            "accepted_rows": accepted,
            "draft_rows": len(draft_rows),
            "acceptance_rate": accepted / len(draft_rows) if draft_rows else 0.0,
            "by_stage": {
                stage: {
                    "accepted": sum(
                        bool(row["accepted"])
                        for row in draft_rows
                        if row["route_kind"] == stage
                    ),
                    "total": sum(
                        row["route_kind"] == stage for row in draft_rows
                    ),
                }
                for stage in SPEC_STAGES[1:]
            },
        }
    return summary


def choose_distinct_steps(
    candidates: list[dict[str, Any]], targets: tuple[float, float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if len(valid) < 2:
        raise AssertionError(f"only {len(valid)} complete candidate steps")
    selected = []
    used: set[int] = set()
    for target in targets:
        choices = [
            candidate
            for candidate in valid
            if int(candidate["scheduler_step"]) not in used
        ]
        chosen = min(
            choices,
            key=lambda candidate: (
                abs(float(candidate["position"]["median"]) - target),
                int(candidate["scheduler_step"]),
            ),
        )
        chosen = dict(chosen)
        chosen["selection_target"] = target
        chosen["selection_distance"] = abs(
            float(chosen["position"]["median"]) - target
        )
        selected.append(chosen)
        used.add(int(chosen["scheduler_step"]))
    return selected[0], selected[1]


def ranked_loads(
    rows: list[dict[str, Any]], layer: int, num_experts: int
) -> list[tuple[int, int, int]]:
    counts = np.zeros(num_experts, dtype=np.int64)
    for row in rows:
        np.add.at(counts, np.asarray(row["routes"][layer], dtype=np.int64), 1)
    order = sorted(range(num_experts), key=lambda expert: (-counts[expert], expert))
    return [
        (load_rank, expert, int(counts[expert]))
        for load_rank, expert in enumerate(order, start=1)
    ]


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    return float(
        (2 * np.sum(indices * sorted_values) / np.sum(sorted_values)
         - (values.size + 1))
        / values.size
    )


def imbalance_metrics(counts: np.ndarray) -> dict[str, float | int]:
    counts = np.asarray(counts, dtype=np.float64)
    mean = float(counts.mean())
    top_count = max(1, math.ceil(0.1 * counts.size))
    return {
        "active_experts": int(np.count_nonzero(counts)),
        "gini": gini(counts),
        "cv": float(counts.std() / mean) if mean else 0.0,
        "max_over_mean": float(counts.max() / mean) if mean else 0.0,
        "zero_load_share": float(np.count_nonzero(counts == 0) / counts.size),
        "top_10pct_assignment_share": float(
            np.sort(counts)[-top_count:].sum() / counts.sum()
        )
        if counts.sum()
        else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_outputs(
    ar_outputs: list[dict[str, Any]], spec_outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    ar_by_sample = {int(row["sample_id"]): row for row in ar_outputs}
    spec_by_sample = {int(row["sample_id"]): row for row in spec_outputs}
    if set(ar_by_sample) != set(range(SAMPLE_COUNT)) or set(spec_by_sample) != set(
        range(SAMPLE_COUNT)
    ):
        raise AssertionError("output consistency requires exactly sample IDs 0..63")
    comparisons = []
    for sample_id in range(SAMPLE_COUNT):
        ar_tokens = [int(value) for value in ar_by_sample[sample_id]["token_ids"]]
        spec_tokens = [
            int(value) for value in spec_by_sample[sample_id]["token_ids"]
        ]
        prefix = 0
        while (
            prefix < len(ar_tokens)
            and prefix < len(spec_tokens)
            and ar_tokens[prefix] == spec_tokens[prefix]
        ):
            prefix += 1
        comparisons.append(
            {
                "sample_id": sample_id,
                "exact_match": ar_tokens == spec_tokens,
                "matching_prefix_length": prefix,
                "first_difference": None
                if prefix == len(ar_tokens) == len(spec_tokens)
                else prefix,
                "ar_token_at_first_difference": ar_tokens[prefix]
                if prefix < len(ar_tokens)
                else None,
                "spec_token_at_first_difference": spec_tokens[prefix]
                if prefix < len(spec_tokens)
                else None,
            }
        )
    return {
        "samples": SAMPLE_COUNT,
        "exact_matches": sum(row["exact_match"] for row in comparisons),
        "mismatches": sum(not row["exact_match"] for row in comparisons),
        "comparisons": comparisons,
        "interpretation": (
            "Mismatches are retained: load curves describe the realized natural "
            "workload and can include batch-numerical or trajectory differences."
        ),
    }


def plot_model(
    model: ModelSpec,
    selected_rows: dict[tuple[str, str], list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    colors = {"spec": "#d55e00", "ar": "#0072b2"}
    labels = {"spec": "Spec16+d3", "ar": "AR64"}
    for row_index, label in enumerate(("early", "late")):
        for column_index, layer in enumerate(model.selected_layers):
            axis = axes[row_index, column_index]
            for mode in ("spec", "ar"):
                ranked = ranked_loads(
                    selected_rows[(label, mode)], layer, model.num_experts
                )
                axis.plot(
                    [row[0] for row in ranked],
                    [row[2] for row in ranked],
                    color=colors[mode],
                    label=labels[mode],
                    linewidth=1.6,
                )
            axis.set_title(f"{label.capitalize()} · Layer {layer}")
            axis.set_xlim(1, model.num_experts)
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Independent load rank")
            if column_index == 0:
                axis.set_ylabel("Routed assignment count")
            if row_index == 0 and column_index == 0:
                axis.legend()
    figure.suptitle(
        f"{model.key}: equal 64 target-router tokens per physical step\n"
        "Each curve is sorted independently; x does not identify the same expert",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_dir / f"{model.key}_expert_load_composite.png", dpi=180)
    figure.savefig(output_dir / f"{model.key}_expert_load_composite.pdf")
    plt.close(figure)


def load_cell(
    attempt_dir: Path, model: ModelSpec, mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    all_outputs = []
    for rank in range(DP_SIZE):
        path = rank_dir(attempt_dir, rank)
        validate_rank_artifacts(path, rank=rank, model=model, mode=mode)
        outputs = load_outputs(path / "outputs.jsonl")
        trace = load_trace(path / "trace")
        all_outputs.extend(outputs)
        all_rows.extend(event_rows(trace, outputs, rank))
    return all_rows, all_outputs


def analyze_model(
    output_root: Path,
    model: ModelSpec,
    selected_attempts: dict[str, str],
) -> dict[str, Any]:
    analysis_dir = output_root / "analysis" / model.key
    analysis_dir.mkdir(parents=True, exist_ok=False)
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    outputs_by_mode: dict[str, list[dict[str, Any]]] = {}
    candidates_by_mode: dict[str, list[dict[str, Any]]] = {}
    grouped_by_mode: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for mode in ("ar", "spec"):
        attempt = Path(selected_attempts[f"{model.key}/{mode}"])
        rows, outputs = load_cell(attempt, model, mode)
        rows_by_mode[mode] = rows
        outputs_by_mode[mode] = outputs
        grouped = group_by_step(rows)
        grouped_by_mode[mode] = grouped
        candidates = []
        for step, step_rows in sorted(grouped.items()):
            summary = validate_step(step_rows, mode=mode, model=model)
            summary["scheduler_step"] = step
            candidates.append(summary)
        candidates_by_mode[mode] = candidates
    spec_early, spec_late = choose_distinct_steps(
        candidates_by_mode["spec"], (16.0, 48.0)
    )
    ar_targets = (
        float(spec_early["position"]["median"]),
        float(spec_late["position"]["median"]),
    )
    ar_early, ar_late = choose_distinct_steps(
        candidates_by_mode["ar"], ar_targets
    )
    selected = {
        "early": {"spec": spec_early, "ar": ar_early},
        "late": {"spec": spec_late, "ar": ar_late},
    }
    selected_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label in ("early", "late"):
        for mode in ("spec", "ar"):
            step = int(selected[label][mode]["scheduler_step"])
            selected_rows[(label, mode)] = grouped_by_mode[mode][step]
    selected_payload = {
        "model_family": model.key,
        "selection_rules": {
            "spec": "valid complete step nearest offsets 16 and 48; tie by step",
            "ar": "valid complete step nearest selected Spec medians; tie by step",
            "distinct_steps_required": True,
        },
        "selected": selected,
        "candidates": candidates_by_mode,
    }
    write_json(analysis_dir / "selected_steps.json", selected_payload)
    load_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    metric_json: list[dict[str, Any]] = []
    owners = DP_SIZE
    experts_per_owner = model.num_experts // owners
    for label in ("early", "late"):
        for mode in ("spec", "ar"):
            step_summary = selected[label][mode]
            step_rows = selected_rows[(label, mode)]
            for layer in model.selected_layers:
                ranked = ranked_loads(step_rows, layer, model.num_experts)
                if sum(row[2] for row in ranked) != SAMPLE_COUNT * TOP_K:
                    raise AssertionError("selected panel assignment total is not 512")
                counts = np.zeros(model.num_experts, dtype=np.int64)
                for load_rank, expert, count in ranked:
                    counts[expert] = count
                    load_rows.append(
                        {
                            "model": model.key,
                            "representative_step": label,
                            "mode": mode,
                            "scheduler_step": int(
                                step_summary["scheduler_step"]
                            ),
                            "position_min": step_summary["position"]["min"],
                            "position_median": step_summary["position"]["median"],
                            "position_max": step_summary["position"]["max"],
                            "layer": layer,
                            "load_rank": load_rank,
                            "logical_expert_id": expert,
                            "assignment_count": count,
                            "ep_owner_rank": expert // experts_per_owner,
                        }
                    )
                metrics = imbalance_metrics(counts)
                owner_totals = [
                    int(
                        counts[
                            owner * experts_per_owner : (owner + 1)
                            * experts_per_owner
                        ].sum()
                    )
                    for owner in range(owners)
                ]
                metric = {
                    "model": model.key,
                    "representative_step": label,
                    "mode": mode,
                    "scheduler_step": int(step_summary["scheduler_step"]),
                    "layer": layer,
                    **metrics,
                    "ep_owner_0_assignments": owner_totals[0],
                    "ep_owner_1_assignments": owner_totals[1],
                }
                metric_rows.append(metric)
                metric_json.append(metric)
    write_csv(analysis_dir / "selected_step_loads.csv", load_rows)
    write_csv(analysis_dir / "imbalance_summary.csv", metric_rows)
    write_json(analysis_dir / "imbalance_summary.json", metric_json)
    consistency = compare_outputs(outputs_by_mode["ar"], outputs_by_mode["spec"])
    write_json(analysis_dir / "output_consistency.json", consistency)
    plot_model(model, selected_rows, analysis_dir)
    return {
        "model_family": model.key,
        "selected_steps": selected,
        "valid_candidate_steps": {
            mode: sum(candidate["valid"] for candidate in candidates)
            for mode, candidates in candidates_by_mode.items()
        },
        "output_exact_matches": consistency["exact_matches"],
        "output_mismatches": consistency["mismatches"],
        "artifacts": {
            "png": str(analysis_dir / f"{model.key}_expert_load_composite.png"),
            "pdf": str(analysis_dir / f"{model.key}_expert_load_composite.pdf"),
            "selected_step_loads": str(
                analysis_dir / "selected_step_loads.csv"
            ),
            "selected_steps": str(analysis_dir / "selected_steps.json"),
            "imbalance_summary_csv": str(
                analysis_dir / "imbalance_summary.csv"
            ),
            "imbalance_summary_json": str(
                analysis_dir / "imbalance_summary.json"
            ),
            "output_consistency": str(
                analysis_dir / "output_consistency.json"
            ),
        },
    }


def analyze_experiment(
    output_root: Path,
    specs: dict[str, ModelSpec],
    selected_attempts: dict[str, str],
) -> None:
    analysis_root = output_root / "analysis"
    if analysis_root.exists():
        raise FileExistsError(f"analysis output already exists: {analysis_root}")
    summaries = [
        analyze_model(output_root, specs[family], selected_attempts)
        for family in ("qwen36", "gemma4")
    ]
    write_json(
        analysis_root / "analysis_manifest.json",
        {
            "state": "complete",
            "completed_at": utc_now(),
            "summaries": summaries,
        },
    )
    (analysis_root / "analysis_complete").touch(exist_ok=False)


def analyze_command(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    selected_attempts = read_json(output_root / "selected_attempts.json")
    analyze_experiment(output_root, model_specs(args), selected_attempts)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--gemma-drafter", default=DEFAULT_GEMMA_DRAFTER)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    run_parser.add_argument("--output-root")
    run_parser.add_argument("--resume", action="store_true")
    worker_parser = subparsers.add_parser("worker")
    add_shared_arguments(worker_parser)
    worker_parser.add_argument(
        "--model-family", choices=("qwen36", "gemma4"), required=True
    )
    worker_parser.add_argument("--mode", choices=("ar", "spec"), required=True)
    worker_parser.add_argument("--attempt-dir", required=True)
    worker_parser.add_argument("--sample-manifest", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    add_shared_arguments(analyze_parser)
    analyze_parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_all(args)
    elif args.command == "worker":
        run_worker(args)
    else:
        analyze_command(args)


if __name__ == "__main__":
    main()
