# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small shared-weight top-p versus static top-h experiment."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_static_budget import MODELS, ROOT, sha256, validate_metrics, write_json


def worker(path):
    from vllm import LLM, SamplingParams

    config = json.loads(path.read_text())
    dataset = Path(config["dataset"])
    assert sha256(dataset) == config["dataset_sha256"]
    samples = [json.loads(line) for line in dataset.read_text().splitlines()]
    spec = None
    extra = {}
    if config["h"]:
        spec = {
            "method": "moe_skip",
            "moe_skip_top_h": config["h"],
            "num_speculative_tokens": config["d"],
        }
    if config["p"] is not None:
        extra["worker_extension_cls"] = "benchmarks.moe_skip.top_p_worker.TopPWorker"
    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        async_scheduling=spec is None,
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed" if spec else "none",
        disable_log_stats=True,
        seed=0,
        **extra,
    )
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    for sample in samples:
        llm.generate([sample["prompt"]], sampling, use_tqdm=False)
    if extra:
        llm.collective_rpc("reset_budget_counts")
    print("WARMUP_COMPLETE", flush=True)
    outputs = []
    for sample in samples:
        started = time.perf_counter()
        result = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        output = result.outputs[0]
        assert len(output.token_ids) == 128
        assert len(result.prompt_token_ids) == sample["prompt_token_count"]
        metrics = output.spec_decode_metrics.to_dict() if spec else None
        if spec:
            validate_metrics(metrics, config["d"])
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "category": sample["category"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_tokens": len(result.prompt_token_ids),
                "token_ids": list(output.token_ids),
                "e2e_seconds": elapsed,
                "spec_decode_metrics": metrics,
            }
        )
        write_json(path.parent / "progress.json", outputs)
        print(f"SAMPLE_COMPLETE {len(outputs)}/4 {elapsed:.3f}s", flush=True)
    budgets = llm.collective_rpc("collect_budget_counts") if extra else None
    if extra and not any(sum(v) for worker in budgets for v in worker.values()):
        raise RuntimeError("Top-p did not record any expert routing")
    write_json(
        path.parent / "result.json",
        {
            **config,
            "outputs": outputs,
            "budget_histograms": budgets,
            "e2e_seconds": sum(o["e2e_seconds"] for o in outputs),
            "output_tokens": 512,
        },
    )
    (path.parent / "CELL_COMPLETE").write_text("4 x 128 completed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--gpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cell:
        worker(args.cell)
        return
    if None in (args.model, args.gpu, args.run_dir):
        parser.error("--model, --gpu and --run-dir are required")
    folder = args.run_dir.resolve() / args.model
    folder.mkdir(parents=True, exist_ok=True)
    model, _ = MODELS[args.model]
    source = ROOT / "benchmark_results/moe_skip_static_budget_16x512_20260914"
    samples = [
        json.loads(s)
        for s in (source / args.model / "dataset.jsonl").read_text().splitlines()
    ]
    selected = {}
    for sample in samples:
        selected.setdefault(sample["category"], sample)
    assert len(selected) == 4
    dataset = folder / "dataset.jsonl"
    payload = "".join(json.dumps(s) + "\n" for s in selected.values())
    d = 8 if args.model == "qwen36" else 4
    cells = [{"name": "ar_start", "h": 0, "p": None, "d": 0}]
    cells += [{"name": "h8", "h": 8, "p": None, "d": d}]
    cells += [{"name": "p1_control", "h": 8, "p": 1.0, "d": d}]
    cells += [
        {"name": "p08", "h": 8, "p": 0.8, "d": d},
        {"name": "h4", "h": 4, "p": None, "d": d},
        {"name": "p07", "h": 8, "p": 0.7, "d": d},
        {"name": "h2", "h": 2, "p": None, "d": d},
        {"name": "p09", "h": 8, "p": 0.9, "d": d},
        {"name": "h6", "h": 6, "p": None, "d": d},
        {"name": "ar_end", "h": 0, "p": None, "d": 0},
    ]
    fingerprints = {
        name: sha256(Path(__file__).parent / name)
        for name in ("run_top_p_probe.py", "top_p_worker.py")
    }
    contract = {
        "model": args.model,
        "model_path": model,
        "gpu": args.gpu,
        "cells": cells,
        "samples": 4,
        "output_length": 128,
        "dataset_payload": payload,
        "source_sha256": fingerprints,
        "model_config_sha256": sha256(Path(model) / "config.json"),
        "routing": "native top-8 mass; retained weights renormalized; h=1..8",
        "timer": "llm.generate wall time including prefill and detailed metrics",
        "warmup": "one full request per measured prompt, before counter reset",
    }
    contract_path = folder / "contract.json"
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("Resume contract mismatch or missing --resume")
        if dataset.read_text() != payload:
            raise RuntimeError("Resume dataset mismatch")
    else:
        write_json(contract_path, contract)
        dataset.write_text(payload)
        for name in fingerprints:
            (folder / name).write_bytes((Path(__file__).parent / name).read_bytes())
        (folder / "git_head.txt").write_bytes(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        )
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_MOE_SKIP_TRACE", "VLLM_DRAFT_TOPK_TRACE")):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    )
    for cell in cells:
        directory = folder / cell["name"]
        config = {
            **cell,
            "model": args.model,
            "model_path": model,
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "source_sha256": fingerprints,
        }
        if (directory / "CELL_COMPLETE").exists():
            previous = json.loads((directory / "result.json").read_text())
            if any(previous.get(k) != v for k, v in config.items()):
                raise RuntimeError("Completed cell identity mismatch")
            assert len(previous["outputs"]) == 4
            assert all(len(o["token_ids"]) == 128 for o in previous["outputs"])
            continue
        directory.mkdir(exist_ok=True)
        write_json(directory / "config.json", config)
        env.pop("MOE_SKIP_BENCH_TOP_P", None)
        if cell["p"] is not None:
            env["MOE_SKIP_BENCH_TOP_P"] = str(cell["p"])
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            str(directory / "config.json"),
        ]
        write_json(directory / "command.json", command)
        print(f"START {args.model} {cell['name']}", flush=True)
        with (directory / f"run_{time.time_ns()}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"Cell failed: {directory}")
        print(f"COMPLETE {args.model} {cell['name']}", flush=True)
    (folder / "MEASUREMENTS_COMPLETE").write_text("10 cells; audit pending\n")


if __name__ == "__main__":
    main()
