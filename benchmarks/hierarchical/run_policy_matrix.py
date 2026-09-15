# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched AR, MTP D4 and three h4 stopping policies at B1/4/8/16."""

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


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker(path):
    from vllm import LLM, SamplingParams

    config = json.loads(path.read_text())
    samples = [json.loads(s) for s in Path(config["dataset"]).read_text().splitlines()]
    assert len(samples) == 16
    mode, batch_size = config["mode"], config["batch_size"]
    spec = None
    if mode == "mtp":
        spec = {
            "method": "mtp",
            "model": ASSISTANT,
            "num_speculative_tokens": 4,
            "draft_sample_method": "greedy",
        }
    elif mode != "ar":
        spec = {
            "method": "hierarchical",
            "model": ASSISTANT,
            "inner_method": "mtp",
            "inner_num_speculative_tokens": 4,
            "inner_num_rounds": 4,
            "num_speculative_tokens": 20,
            "moe_skip_top_h": 4,
            "draft_sample_method": "greedy",
            "hierarchical_stop_policy": mode,
        }
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=batch_size,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=False,
        async_scheduling=False,
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed" if spec else "none",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="benchmarks.hierarchical.policy_worker.PolicyWorker",
    )

    def generate(length=None):
        length = length or config["output_length"]
        params = SamplingParams(
            temperature=0, max_tokens=length, ignore_eos=True, seed=0
        )
        outputs, batches = [], []
        for start in range(0, 16, batch_size):
            group = samples[start : start + batch_size]
            begin = time.perf_counter()
            results = llm.generate([r["prompt"] for r in group], params, use_tqdm=False)
            elapsed = time.perf_counter() - begin
            batches.append({"start": start, "size": len(group), "seconds": elapsed})
            for sample, result in zip(group, results, strict=True):
                tokens = list(result.outputs[0].token_ids)
                assert len(tokens) == length
                assert len(result.prompt_token_ids) == sample["prompt_token_count"]
                outputs.append(
                    {
                        "prompt_sha256": sample["prompt_sha256"],
                        "category": sample["category"],
                        "token_ids": tokens,
                        "spec_decode_metrics": result.outputs[
                            0
                        ].spec_decode_metrics.to_dict()
                        if spec
                        else None,
                    }
                )
            print(f"BATCH_COMPLETE {start + len(group)}/16 {elapsed:.3f}s", flush=True)
        return outputs, batches

    generate(min(64, config["output_length"]))
    print("WARMUP_COMPLETE", flush=True)
    for attempt in range(3):
        before = llm.collective_rpc("policy_counters", kwargs={"reset": True})[0]
        outputs, batches = generate()
        counters = llm.collective_rpc("policy_counters")[0]
        result = {
            **config,
            "outputs": outputs,
            "batches": batches,
            "counters": counters,
            "warmup_graphs": before.get("preverify_graphs", 0),
            "seconds": sum(r["seconds"] for r in batches),
            "output_tokens": 16 * config["output_length"],
            "attempt": attempt,
        }
        write_json(path.parent / f"attempt_{attempt}.json", result)
        if counters.get("preverify_graphs", 0) == before.get("preverify_graphs", 0):
            break
        print("NEW_GRAPH_DURING_MEASUREMENT_RETRY", flush=True)
    else:
        raise RuntimeError("Preverify graph coverage did not stabilize")
    write_json(path.parent / "result.json", result)
    (path.parent / "CELL_COMPLETE").write_text(
        "16 requests, exact output lengths; warm graphs\n"
    )


def idle_gpu():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        index, used = map(int, line.split(","))
        if used < 1000:
            return index
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if args.cell:
        worker(args.cell)
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        write_json(
            root / "status.json",
            {
                "state": "waiting_for_existing_job",
                "wait_pid": args.wait_pid,
                "pid": os.getpid(),
                "completed_cells": 0,
                "expected_cells": 20,
            },
        )
        while Path(f"/proc/{args.wait_pid}").exists():
            print(f"WAITING_FOR_EXISTING_JOB {args.wait_pid}", flush=True)
            time.sleep(30)
    source = (
        ROOT
        / "benchmark_results/gemma_h4_confidence_late_16x128_20260915/dataset.jsonl"
    )
    if not (root / "dataset.jsonl").exists():
        (root / "dataset.jsonl").write_bytes(source.read_bytes())
    cells = [
        (b, mode)
        for b in (1, 4, 8, 16)
        for mode in ("ar", "mtp", "low_error", "balanced", "aggressive")
    ]
    if args.smoke:
        cells = [(4, "low_error"), (16, "low_error")]
    paths = [
        Path(__file__),
        Path(__file__).with_name("policy_worker.py"),
        Path(__file__).with_name("finish_policy_matrix.py"),
        Path(__file__).with_name("summarize_policy_matrix.py"),
        Path(__file__).with_name("plot_policy_matrix.py"),
        ROOT / "vllm/config/speculative.py",
    ] + list((ROOT / "vllm/v1/worker/gpu/spec_decode/hierarchical").glob("*.py"))
    fingerprints = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    for path in paths:
        snapshot = root / "source" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    (root / "git_head.txt").write_bytes(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
    )
    (root / "gpu.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))
    write_json(
        root / "contract.json",
        {
            "cells": cells,
            "samples": 16,
            "output_length": 32 if args.smoke else 512,
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_sha256": fingerprints,
            "warmup_output_length": 32 if args.smoke else 64,
            "compiler_cache": "isolated by experiment and mode",
            "model": MODEL,
            "assistant": ASSISTANT,
            "model_config_sha256": hashlib.sha256(
                (Path(MODEL) / "config.json").read_bytes()
            ).hexdigest(),
            "assistant_config_sha256": hashlib.sha256(
                (Path(ASSISTANT) / "config.json").read_bytes()
            ).hexdigest(),
            "protocol": "fixed groups, TP1, synchronous scheduling, "
            "raw prompt, greedy, ignore_eos",
        },
    )
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            (
                "VLLM_HIERARCHICAL",
                "VLLM_MOE_SKIP_TRACE",
                "VLLM_DRAFT_TOPK_TRACE",
                "PREVERIFY_EXPERT",
            )
        ):
            env.pop(key)
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    )
    for batch_size, mode in cells:
        folder = root / f"b{batch_size}_{mode}"
        if (folder / "CELL_COMPLETE").exists():
            continue
        folder.mkdir(exist_ok=True)
        write_json(
            root / "status.json",
            {
                "state": "waiting_for_idle_gpu",
                "cell": folder.name,
                "pid": os.getpid(),
                "completed_cells": sum(
                    (root / f"b{b}_{m}" / "CELL_COMPLETE").exists() for b, m in cells
                ),
                "expected_cells": len(cells),
            },
        )
        while True:
            gpu = idle_gpu()
            time.sleep(30)
            if gpu is not None and gpu == idle_gpu():
                break
            print("WAITING_FOR_STABLE_IDLE_GPU", flush=True)
        config = {
            "mode": mode,
            "batch_size": batch_size,
            "gpu": gpu,
            "dataset": str(root / "dataset.jsonl"),
            "output_length": 32 if args.smoke else 512,
            "source_sha256": fingerprints,
        }
        write_json(folder / "config.json", config)
        command = [
            sys.executable,
            "-m",
            "benchmarks.hierarchical.run_policy_matrix",
            "--cell",
            str(folder / "config.json"),
        ]
        write_json(folder / "command.json", command)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["VLLM_CACHE_ROOT"] = str(root / "compiler_cache" / mode)
        print(f"START {folder.name} GPU{gpu}", flush=True)
        write_json(
            root / "status.json",
            {
                "state": "running",
                "cell": folder.name,
                "gpu": gpu,
                "pid": os.getpid(),
                "completed_cells": sum(
                    (root / f"b{b}_{m}" / "CELL_COMPLETE").exists() for b, m in cells
                ),
                "expected_cells": len(cells),
            },
        )
        with (folder / "run.log").open("w") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            write_json(
                root / "status.json",
                {
                    "state": "failed",
                    "cell": folder.name,
                    "returncode": process.returncode,
                },
            )
            process.check_returncode()
        print(f"COMPLETE {folder.name}", flush=True)
    (root / "MEASUREMENTS_COMPLETE").write_text(
        f"{len(cells)} cells completed; analysis pending\n"
    )
    if not args.smoke:
        from benchmarks.hierarchical.finish_policy_matrix import finish

        finish(root)
    write_json(
        root / "status.json",
        {
            "state": "complete",
            "completed_cells": len(cells),
            "visual_review": "pending" if not args.smoke else "not_applicable",
        },
    )


if __name__ == "__main__":
    main()
