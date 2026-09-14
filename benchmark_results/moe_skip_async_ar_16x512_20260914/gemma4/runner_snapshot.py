# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fixed expert budgets on the same 16 prompts and 512 output tokens."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen36": (
        "/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        "moe_skip_qwen36_multicategory_128x512_20260906",
    ),
    "gemma4": (
        "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it",
        "moe_skip_gemma4_multicategory_128x512_d64_20260906",
    ),
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_metrics(metrics, d):
    accepted = metrics["per_step_accepted"]
    drafted = metrics["per_step_drafted"]
    steps = metrics["num_spec_steps"]
    if not (
        metrics["num_spec_tokens"] == d
        and len(accepted) == len(drafted) == steps
        and steps > 0
        and sum(accepted) == metrics["num_accepted_draft_tokens"]
        and sum(drafted) == metrics["num_draft_tokens"]
        and all(0 <= a <= n <= d for a, n in zip(accepted, drafted, strict=True))
        and metrics["acceptance_histogram"] == [accepted.count(i) for i in range(d + 1)]
    ):
        raise RuntimeError("Inconsistent per-request acceptance counters")


def worker(config_path):
    from vllm import LLM, SamplingParams

    config = json.loads(config_path.read_text())
    dataset = Path(config["dataset"])
    if sha256(dataset) != config["dataset_sha256"]:
        raise RuntimeError("Dataset fingerprint changed")
    samples = [json.loads(line) for line in dataset.read_text().splitlines()]
    spec = None
    if config["h"]:
        spec = {
            "method": "moe_skip",
            "moe_skip_top_h": config["h"],
            "num_speculative_tokens": config["d"],
        }
    started = time.perf_counter()
    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        async_scheduling=config.get("async_scheduling", False),
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed" if spec else "none",
        disable_log_stats=True,
        seed=0,
    )
    init_seconds = time.perf_counter() - started
    sampling = SamplingParams(temperature=0, max_tokens=512, ignore_eos=True)
    warmups = []
    for index in range(2):
        started = time.perf_counter()
        llm.generate([samples[0]["prompt"]], sampling, use_tqdm=False)
        warmups.append(time.perf_counter() - started)
        print(f"WARMUP {index + 1}/2 {warmups[-1]:.3f}s", flush=True)
    print("WARMUP_COMPLETE", flush=True)
    outputs = []
    for sample in samples:
        started = time.perf_counter()
        result = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        output = result.outputs[0]
        if len(output.token_ids) != 512:
            raise RuntimeError("Expected exactly 512 output tokens")
        if len(result.prompt_token_ids) != sample["prompt_token_count"]:
            raise RuntimeError("Prompt tokenization changed")
        metrics = None
        if spec:
            metrics = output.spec_decode_metrics.to_dict()
            validate_metrics(metrics, config["d"])
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "category": sample["category"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_tokens": len(result.prompt_token_ids),
                "token_ids": list(output.token_ids),
                "finish_reason": output.finish_reason,
                "e2e_seconds": elapsed,
                "spec_decode_metrics": metrics,
            }
        )
        write_json(
            config_path.parent / "progress.json",
            {
                "completed_samples": len(outputs),
                "expected_samples": 16,
                "outputs": outputs,
            },
        )
        print(f"SAMPLE_COMPLETE {len(outputs)}/16 {elapsed:.3f}s", flush=True)
    total = sum(o["e2e_seconds"] for o in outputs)
    write_json(
        config_path.parent / "result.json",
        {
            **config,
            "init_seconds": init_seconds,
            "warmup_seconds": warmups,
            "e2e_seconds": total,
            "output_tokens": 8192,
            "output_tokens_per_second": 8192 / total,
            "outputs": outputs,
        },
    )
    (config_path.parent / "CELL_COMPLETE").write_text("16 x 512 completed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--gpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--async-ar-only", action="store_true")
    args = parser.parse_args()
    if args.cell:
        worker(args.cell)
        return
    if args.model is None or args.gpu is None or args.run_dir is None:
        parser.error("--model, --gpu and --run-dir are required")
    run_dir = args.run_dir.resolve() / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    model, source_name = MODELS[args.model]
    source = ROOT / "benchmark_results" / source_name / "dataset"
    manifest = args.dataset or source / "multicategory_128.jsonl"
    samples = [json.loads(s) for s in manifest.read_text().splitlines()][:16]
    if len(samples) != 16 or sorted(
        Counter(s["category"] for s in samples).values()
    ) != [4, 4, 4, 4]:
        raise RuntimeError("Expected four samples per category")
    for sample in samples:
        if (
            hashlib.sha256(sample["prompt"].encode()).hexdigest()
            != sample["prompt_sha256"]
        ):
            raise RuntimeError("Source prompt hash mismatch")
    dataset = run_dir / "dataset.jsonl"
    payload = "".join(json.dumps(s) + "\n" for s in samples)
    pairs = [(h, d) for h in (2, 4, 6, 8) for d in (4, 8, 16, 32)]
    random.Random(20260914).shuffle(pairs)
    cells = [{"name": "ar_start", "h": 0, "d": 0}]
    cells += [{"name": f"h{h}_d{d}", "h": h, "d": d} for h, d in pairs]
    cells += [{"name": "ar_end", "h": 0, "d": 0}]
    if args.async_ar_only:
        cells = [cells[0], cells[-1]]
        for cell in cells:
            cell["async_scheduling"] = True
    contract_path = run_dir / "contract.json"
    if contract_path.exists():
        if not args.resume:
            raise RuntimeError("Use a fresh directory or --resume")
        contract = json.loads(contract_path.read_text())
        if (
            dataset.read_text() != payload
            or contract["gpu"] != args.gpu
            or contract["script_sha256"] != sha256(Path(__file__))
            or contract["cells"] != cells
        ):
            raise RuntimeError("Resume inputs changed")
    else:
        dataset.write_text(payload)
        contract = {
            "model": args.model,
            "model_path": model,
            "gpu": args.gpu,
            "cells": cells,
            "dataset": str(dataset),
            "dataset_source": str(manifest.resolve()),
            "dataset_sha256": sha256(dataset),
            "samples": 16,
            "output_tokens_per_sample": 512,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "batch_size": 1,
            "tensor_parallel_size": 1,
            "cuda_graph": True,
            "prefix_caching": False,
            "async_scheduling": args.async_ar_only,
            "warmup_requests": 2,
            "max_model_len": 1024,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.95,
            "spec_metrics": "detailed; included in measured wall time",
            "timer": "sum of llm.generate wall times; prefill + decode + API",
            "script_sha256": sha256(Path(__file__)),
            "model_config_sha256": sha256(Path(model) / "config.json"),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
        write_json(contract_path, contract)
        (run_dir / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
        for name, command in {
            "source_diff.patch": ["git", "diff", "HEAD"],
            "git_status.txt": ["git", "status", "--short"],
            "gpu_start.txt": ["nvidia-smi"],
        }.items():
            (run_dir / name).write_bytes(subprocess.check_output(command, cwd=ROOT))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_MOE_SKIP_TRACE", "VLLM_DRAFT_TOPK_TRACE")):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    )
    write_json(
        run_dir / "environment.json",
        {
            k: v
            for k, v in env.items()
            if k.startswith(("VLLM_", "CUDA_", "HF_", "OMP_", "MKL_"))
            and "TOKEN" not in k
        },
    )
    for cell in cells:
        directory = run_dir / cell["name"]
        if (directory / "CELL_COMPLETE").exists():
            result = json.loads((directory / "result.json").read_text())
            if len(result["outputs"]) != 16 or any(
                len(o["token_ids"]) != 512 for o in result["outputs"]
            ):
                raise RuntimeError(f"Invalid completed cell: {directory}")
            continue
        directory.mkdir(exist_ok=True)
        config_path = directory / "config.json"
        write_json(
            config_path,
            {
                **cell,
                "model": args.model,
                "model_path": model,
                "dataset": str(dataset),
                "dataset_sha256": sha256(dataset),
            },
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            str(config_path),
        ]
        write_json(directory / "command.json", command)
        print(f"START {args.model} {cell['name']}", flush=True)
        with (directory / "run.log").open("a") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            write_json(directory / "failed.json", {"returncode": result.returncode})
            raise RuntimeError(f"Failed cell: {directory}")
        print(f"COMPLETE {args.model} {cell['name']}", flush=True)
    (run_dir / "MEASUREMENTS_COMPLETE").write_text(
        f"{len(cells)} cells; audit pending\n"
    )


if __name__ == "__main__":
    main()
