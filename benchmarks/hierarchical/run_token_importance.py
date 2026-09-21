# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma same-block token importance versus static Pre-Verify expert budgets."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it"
ASSISTANT = MODEL + "-assistant"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker(path):
    config = json.loads(path.read_text())
    os.environ["PREVERIFY_EXPERT_POOL"] = config["pool"]
    from vllm import LLM, SamplingParams

    dataset = Path(config["dataset"])
    assert digest(dataset) == config["dataset_sha256"]
    samples = [json.loads(s) for s in dataset.read_text().splitlines()]
    spec = (
        None
        if config["h"] == 0
        else {
            "method": "hierarchical",
            "model": ASSISTANT,
            "inner_method": "mtp",
            "inner_num_speculative_tokens": 4,
            "inner_num_rounds": 4,
            "num_speculative_tokens": 20,
            "moe_skip_top_h": config["h"],
            "draft_sample_method": "greedy",
        }
    )
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=False,
        async_scheduling=spec is None,
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed" if spec else "none",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="token_importance_worker.TokenImportanceWorker",
    )
    params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, seed=0)
    for sample in samples:
        llm.generate([sample["prompt"]], params, use_tqdm=False)
    llm.collective_rpc("begin_pool_measurement")
    print("WARMUP_COMPLETE", flush=True)
    outputs = []
    for sample in samples:
        started = time.perf_counter()
        result = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        assert len(result.outputs[0].token_ids) == 128
        assert len(result.prompt_token_ids) == sample["prompt_token_count"]
        metrics = result.outputs[0].spec_decode_metrics.to_dict() if spec else None
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "category": sample["category"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_tokens": len(result.prompt_token_ids),
                "token_ids": list(result.outputs[0].token_ids),
                "e2e_seconds": elapsed,
                "spec_decode_metrics": metrics,
            }
        )
        write_json(path.parent / "progress.json", outputs)
        print(f"SAMPLE_COMPLETE {len(outputs)}/4 {elapsed:.3f}s", flush=True)
    measurement = llm.collective_rpc("collect_pool_measurement")
    if config["pool"] != "none":
        assert (
            len([v for w in measurement for v in w["layers"].values() if v[-1] > 0])
            == 30
        )
    write_json(
        path.parent / "result.json",
        {
            **config,
            "outputs": outputs,
            "measurement": measurement,
            "e2e_seconds": sum(o["e2e_seconds"] for o in outputs),
            "output_tokens": 512,
        },
    )
    (path.parent / "CELL_COMPLETE").write_text("4 x 128 completed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell", type=Path)
    args = parser.parse_args()
    if args.cell:
        worker(args.cell)
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    folder = args.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source = (
        ROOT
        / "benchmark_results/.sources/moe_skip_top_p_4x128_20260915_run2"
        / "gemma4/dataset.jsonl"
    )
    dataset = folder / "dataset.jsonl"
    cells = [
        {"name": "attention60", "pool": "attention", "h": 8},
        {"name": "ar_start", "pool": "none", "h": 0},
        {"name": "routing60", "pool": "routing", "h": 8},
        {"name": "h4", "pool": "none", "h": 4},
        {"name": "h6", "pool": "none", "h": 6},
        {"name": "h8", "pool": "none", "h": 8},
        {"name": "ar_end", "pool": "none", "h": 0},
    ]
    fingerprints = {
        name: digest(Path(__file__).parent / name)
        for name in ("run_token_importance.py", "token_importance_worker.py")
    }
    contract = {
        "model_path": MODEL,
        "assistant_path": ASSISTANT,
        "assistant_config_sha256": digest(Path(ASSISTANT) / "config.json"),
        "model_config_sha256": digest(Path(MODEL) / "config.json"),
        "gpu": args.gpu,
        "cells": cells,
        "source_sha256": fingerprints,
        "dataset_sha256": digest(source),
        "samples": 4,
        "output_length": 128,
        "B": 1,
        "inner_method": "mtp",
        "inner_D": 4,
        "rounds": 4,
        "outer_D": 20,
        "budget": "ceil(0.6 * union of native top8 experts on draft rows 1..D)",
        "score": "mean-head full-context attention from last draft to earlier "
        "draft rows times normalized native top8 gate probability",
        "score_rows": "exclude anchor row 0 and last row D; routing ablation "
        "uses the same rows with unit importance",
        "mask_rows": "all Pre-Verify rows, including anchor and last draft",
        "zero_expert_rows": "routed contribution zero; no budget-expanding fallback",
        "weights": "renormalize retained gate probabilities, preserve per-expert scale",
        "timer": "llm.generate wall time includes scoring, selection, MTP, "
        "Pre-Verify, Target, API and metrics",
    }
    contract_path = folder / "contract.json"
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("Resume contract mismatch")
        assert digest(dataset) == contract["dataset_sha256"]
    else:
        write_json(contract_path, contract)
        dataset.write_bytes(source.read_bytes())
        for name in fingerprints:
            (folder / name).write_bytes((Path(__file__).parent / name).read_bytes())
        (folder / "git_head.txt").write_bytes(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        )
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            (
                "VLLM_HIERARCHICAL",
                "VLLM_MOE_SKIP_TRACE",
                "VLLM_DRAFT_TOPK_TRACE",
                "MOE_SKIP_BENCH",
            )
        ):
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
            "model_path": MODEL,
            "dataset": str(dataset),
            "dataset_sha256": digest(dataset),
            "source_sha256": fingerprints,
        }
        if (directory / "CELL_COMPLETE").exists():
            previous = json.loads((directory / "result.json").read_text())
            assert all(previous[k] == v for k, v in config.items())
            assert len(previous["outputs"]) == 4
            assert all(len(o["token_ids"]) == 128 for o in previous["outputs"])
            continue
        directory.mkdir(exist_ok=True)
        write_json(directory / "config.json", config)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            str(directory / "config.json"),
        ]
        write_json(directory / "command.json", command)
        print(f"START {cell['name']}", flush=True)
        with (directory / f"run_{time.time_ns()}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"Failed: {directory}")
        print(f"COMPLETE {cell['name']}", flush=True)
    (folder / "MEASUREMENTS_COMPLETE").write_text("7 cells; audit pending\n")


if __name__ == "__main__":
    main()
