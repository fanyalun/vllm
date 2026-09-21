# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce full-budget speedups, then measure complete execution paths."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_static_budget import MODELS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmark_results/.sources/moe_skip_static_budget_16x512_20260914"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(config_path):
    from vllm import LLM, SamplingParams

    config = json.loads(config_path.read_text())
    output_dir = config_path.parent
    spec = (
        {"method": "moe_skip", "moe_skip_top_h": 8, "num_speculative_tokens": 8}
        if config["mode"] == "h8_d8"
        else None
    )
    llm = LLM(
        model=config["model"],
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        async_scheduling=config["mode"] == "ar_async",
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed" if spec else "none",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="performance_path_worker.PerformancePathWorker",
    )
    samples = [
        json.loads(line) for line in Path(config["dataset"]).read_text().splitlines()
    ][:2]
    sampling = SamplingParams(temperature=0, max_tokens=512, ignore_eos=True)
    for sample in samples:
        llm.generate([sample["prompt"]], sampling, use_tqdm=False)
    print("WARMUP_COMPLETE", flush=True)
    results = []
    for repeat in range(3):
        for sample in samples:
            start = time.perf_counter()
            output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
            elapsed = time.perf_counter() - start
            assert len(output.outputs[0].token_ids) == 512
            results.append(
                dict(
                    repeat=repeat,
                    sample_index=sample["sample_index"],
                    seconds=elapsed,
                    token_ids=list(output.outputs[0].token_ids),
                )
            )
            write(output_dir / "e2e.json", results)
            print(f"E2E_COMPLETE {len(results)}/6 {elapsed:.4f}s", flush=True)
    diagnostics = []
    for mode, length in [("events", 512), ("profile", 64)]:
        llm.collective_rpc("begin_path_measurement", args=(mode,))
        start = time.perf_counter()
        output = llm.generate(
            [samples[0]["prompt"]],
            SamplingParams(temperature=0, max_tokens=length, ignore_eos=True),
            use_tqdm=False,
        )[0]
        elapsed = time.perf_counter() - start
        rows = llm.collective_rpc(
            "collect_path_measurement",
            args=(str(output_dir / "trace.json.gz"),),
        )[0]
        diagnostics.append(
            dict(
                **rows,
                seconds=elapsed,
                token_ids=list(output.outputs[0].token_ids),
            )
        )
        write(output_dir / "diagnostics.json", diagnostics)
        print(f"DIAGNOSTIC_COMPLETE {mode}", flush=True)
    write(output_dir / "complete.json", {"e2e_requests": 6, "diagnostics": 2})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--wait-pids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.output is None:
        parser.error("--output is required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    script_dir = Path(__file__).resolve().parent
    write(
        root / "contract.json",
        dict(
            models=list(MODELS),
            modes=["ar_sync", "h8_d8", "ar_async"],
            samples=2,
            output_tokens=512,
            repeats=3,
            gpu=args.gpu,
            wait_pids=args.wait_pids,
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            scope=(
                "Diagnostic only; no replacement for the 16x512 matrix. "
                "Profile timing is not E2E throughput."
            ),
        ),
    )
    for name in ("diagnose_performance_paths.py", "performance_path_worker.py"):
        (root / name).write_bytes((script_dir / name).read_bytes())
    while any(Path(f"/proc/{pid}").exists() for pid in args.wait_pids):
        write(
            root / "status.json",
            {"state": "waiting_for_existing_matrix", "pids": args.wait_pids},
        )
        time.sleep(10)
    while True:
        memory = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    args.gpu,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if memory < 1024:
            break
        write(
            root / "status.json", {"state": "waiting_for_free_gpu", "used_mib": memory}
        )
        time.sleep(10)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("VLLM_MOE_SKIP_TRACE_DIR", "VLLM_DRAFT_TOPK_TRACE_DIR"):
        env.pop(key, None)
    for model, model_path in MODELS.items():
        dataset = SOURCE / model / "dataset.jsonl"
        for mode in ("ar_sync", "h8_d8", "ar_async"):
            cell_dir = root / model / mode
            cell_dir.mkdir(parents=True)
            config_path = cell_dir / "config.json"
            write(
                config_path,
                dict(
                    model=model_path[0],
                    mode=mode,
                    dataset=str(dataset),
                    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                ),
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(config_path),
            ]
            write(cell_dir / "command.json", command)
            write(
                root / "status.json", {"state": "running", "model": model, "mode": mode}
            )
            print(f"START {model} {mode}", flush=True)
            with (cell_dir / "run.log").open("w") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                write(root / "status.json", {"state": "failed", "cell": str(cell_dir)})
                raise RuntimeError(f"Diagnostic failed: {cell_dir}")
            print(f"DONE {model} {mode}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "analyze_performance_paths.py"),
            "--output",
            str(root),
        ],
        cwd=ROOT,
        check=True,
    )
    write(root / "status.json", {"state": "complete"})


if __name__ == "__main__":
    main()
