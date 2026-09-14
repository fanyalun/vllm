# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure sequential B=1 request latency without draft-quality instrumentation."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from benchmark_integrity import prepare_performance_retry

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen36": {
        "model": "/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        "mtp": None,
        "dspark": "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
        "source": "moe_skip_qwen36_multicategory_128x512_20260906",
    },
    "gemma4": {
        "model": "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it",
        "mtp": "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it-assistant",
        "dspark": "/home/fanya/data1/fanya/models/gemma4-26b-a4b-dspark",
        "source": "moe_skip_gemma4_multicategory_128x512_d64_20260906",
    },
}
METHODS = {
    "ar": [0],
    "moe_skip": [4, 8, 16, 32],
    "mtp": [4, 8, 16, 32],
    "dspark": [4, 8],
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_cell(path):
    config = json.loads(path.read_text())
    for key in ("VLLM_MOE_SKIP_TRACE_DIR", "VLLM_DRAFT_TOPK_TRACE_DIR"):
        if os.environ.get(key):
            raise RuntimeError(f"Performance timing requires {key} to be unset")
    from vllm import LLM, SamplingParams

    spec = None
    method = config["method"]
    if method != "ar":
        spec = {"method": method, "num_speculative_tokens": config["d"]}
        if method == "moe_skip":
            spec["moe_skip_top_h"] = 4
        elif config["spec_model"]:
            spec["model"] = config["spec_model"]
    start = time.perf_counter()
    llm = LLM(
        model=config["model"],
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        speculative_config=spec,
        per_request_spec_decode_metrics="none",
        disable_log_stats=True,
        seed=0,
    )
    init_seconds = time.perf_counter() - start
    sampling = SamplingParams(temperature=0, max_tokens=512, ignore_eos=True)
    samples = [
        json.loads(line) for line in Path(config["dataset"]).read_text().splitlines()
    ]
    start = time.perf_counter()
    llm.generate([samples[0]["prompt"]], sampling, use_tqdm=False)
    warmup_seconds = time.perf_counter() - start
    print(f"WARMUP_COMPLETE {warmup_seconds:.3f}s", flush=True)
    outputs = []
    for sample in samples:
        start = time.perf_counter()
        output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        completion = output.outputs[0]
        if len(completion.token_ids) != 512:
            raise RuntimeError("Expected exactly 512 generated tokens")
        if len(output.prompt_token_ids) != sample["prompt_token_count"]:
            raise RuntimeError("Prompt token count differs from source manifest")
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "category": sample["category"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_tokens": len(output.prompt_token_ids),
                "token_ids": list(completion.token_ids),
                "e2e_seconds": elapsed,
            }
        )
        write_json(
            path.parent / "progress.json",
            {"completed_samples": len(outputs), "expected_samples": 16},
        )
        print(f"SAMPLE_COMPLETE {len(outputs)}/16 {elapsed:.3f}s", flush=True)
    total = sum(output["e2e_seconds"] for output in outputs)
    write_json(
        path.parent / "result.json",
        {
            **config,
            "init_seconds": init_seconds,
            "warmup_seconds": warmup_seconds,
            "e2e_seconds": total,
            "mean_request_seconds": total / 16,
            "output_tokens": 8192,
            "output_tokens_per_second": 8192 / total,
            "outputs": outputs,
        },
    )
    (path.parent / "CELL_COMPLETE").write_text("16 requests x 512 tokens\n")


def summarize(run_dir, contract):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for name in contract["models"]:
        baseline = json.loads((run_dir / name / "ar_d0/result.json").read_text())
        for method, lengths in METHODS.items():
            for d in lengths:
                directory = run_dir / name / f"{method}_d{d}"
                if not (directory / "CELL_COMPLETE").exists():
                    raise RuntimeError(f"Incomplete cell: {directory}")
                result = json.loads((directory / "result.json").read_text())
                outputs = result["outputs"]
                if len(outputs) != 16 or any(
                    len(o["token_ids"]) != 512 for o in outputs
                ):
                    raise RuntimeError(f"Incomplete outputs: {directory}")
                if [o["prompt_sha256"] for o in outputs] != [
                    o["prompt_sha256"] for o in baseline["outputs"]
                ]:
                    raise RuntimeError("Ordered prompt identities differ")
                rows.append(
                    {
                        "model": name,
                        "method": method,
                        "d": d,
                        "e2e_seconds": result["e2e_seconds"],
                        "mean_request_seconds": result["mean_request_seconds"],
                        "output_tokens_per_second": result["output_tokens_per_second"],
                        "speedup_vs_ar": baseline["e2e_seconds"]
                        / result["e2e_seconds"],
                        "exact_ar_requests": sum(
                            a["token_ids"] == b["token_ids"]
                            for a, b in zip(outputs, baseline["outputs"], strict=True)
                        ),
                    }
                )
    with (run_dir / "performance.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(
        len(contract["models"]),
        3,
        figsize=(16, 5 * len(contract["models"])),
        squeeze=False,
        constrained_layout=True,
    )
    labels = {"moe_skip": "MoE-Skip (top-4)", "mtp": "MTP", "dspark": "DSpark"}
    metrics = [
        ("mean_request_seconds", "Mean request E2E (s; lower is better)"),
        ("output_tokens_per_second", "Output tokens/s (higher is better)"),
        ("speedup_vs_ar", "E2E speedup over AR"),
    ]
    for index, name in enumerate(contract["models"]):
        display_name = {
            "qwen36": "Qwen3.6-35B-A3B",
            "gemma4": "Gemma4-26B-A4B",
        }[name]
        model_rows = [row for row in rows if row["model"] == name]
        for axis, (metric, title) in zip(axes[index], metrics, strict=True):
            for method, label in labels.items():
                values = [row for row in model_rows if row["method"] == method]
                axis.plot(
                    [r["d"] for r in values],
                    [r[metric] for r in values],
                    marker="o",
                    label=label,
                )
            ar = next(r[metric] for r in model_rows if r["method"] == "ar")
            axis.axhline(ar, color="0.4", linestyle="--", label="AR")
            axis.set(
                xlabel="Draft length D",
                ylabel=title,
                title=display_name,
                xticks=[4, 8, 16, 32],
            )
            axis.set_xscale("log", base=2)
            axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle(
        "16 fixed prompts | 512 output tokens | B=1 | TP=1 | CUDA Graph\n"
        "Prefill + decode + offline API; one warmup excluded; no traces"
    )
    for extension in ("png", "pdf"):
        fig.savefig(run_dir / f"e2e_performance.{extension}", dpi=180)
    plt.close(fig)
    write_json(
        run_dir / "audit.json",
        {
            "status": "passed",
            "cells": len(rows),
            "requests": len(rows) * 16,
            "output_tokens": len(rows) * 8192,
            "contract": contract,
            "ar_parity": [
                {k: r[k] for k in ("model", "method", "d", "exact_ar_requests")}
                for r in rows
            ],
        },
    )
    (run_dir / "RUN_COMPLETE").write_text("All cells audited and plotted\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cell:
        run_cell(args.cell)
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    run_dir = args.run_dir.resolve()
    if args.summarize:
        summarize(run_dir, json.loads((run_dir / "contract.json").read_text()))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "contract.json").exists() and not args.resume:
        raise RuntimeError("Use a fresh run directory")
    contract = {
        "models": args.models,
        "methods": METHODS,
        "num_samples": 16,
        "max_tokens": 512,
        "batch_size": 1,
        "top_h": 4,
        "max_model_len": 1024,
        "max_num_batched_tokens": 4096,
        "temperature": 0,
        "ignore_eos": True,
        "prefix_caching": False,
        "warmup_requests": 1,
        "cuda_device": args.cuda_device,
        "timer": "sum of individual llm.generate wall times, prefill plus decode",
        "datasets": {},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    if args.resume:
        contract = json.loads((run_dir / "contract.json").read_text())
        if args.models != contract["models"]:
            raise RuntimeError("Resume must use the original model list")
        if args.cuda_device != contract["cuda_device"]:
            raise RuntimeError("Resume must use the original GPU")
    else:
        (run_dir / "source_diff.patch").write_bytes(
            subprocess.check_output(["git", "diff"], cwd=ROOT)
        )
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    for key in ("VLLM_MOE_SKIP_TRACE_DIR", "VLLM_DRAFT_TOPK_TRACE_DIR"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=args.cuda_device,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
    )
    for name in args.models:
        model = MODELS[name]
        source = ROOT / "benchmark_results" / model["source"] / "dataset"
        samples = [
            json.loads(line)
            for line in (source / "multicategory_128.jsonl").read_text().splitlines()
        ][:16]
        if set(Counter(r["category"] for r in samples).values()) != {4}:
            raise RuntimeError("Expected four samples from each category")
        dataset = run_dir / f"{name}_16.jsonl"
        payload = "".join(json.dumps(row) + "\n" for row in samples)
        if args.resume and dataset.read_text() != payload:
            raise RuntimeError("Resume dataset changed")
        dataset.write_text(payload)
        contract["datasets"][name] = {
            "source": str(source),
            "path": str(dataset),
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        }
    write_json(run_dir / "contract.json", contract)
    for name in args.models:
        model = MODELS[name]
        for method, lengths in METHODS.items():
            for d in lengths:
                directory = run_dir / name / f"{method}_d{d}"
                if args.resume and (directory / "CELL_COMPLETE").exists():
                    result = json.loads((directory / "result.json").read_text())
                    if len(result["outputs"]) != 16 or any(
                        len(o["token_ids"]) != 512 for o in result["outputs"]
                    ):
                        raise RuntimeError(f"Invalid completed cell: {directory}")
                    print(f"SKIP {name} {method} D={d}", flush=True)
                    continue
                prepare_performance_retry(directory)
                config = directory / "config.json"
                write_json(
                    config,
                    {
                        "model": model["model"],
                        "method": method,
                        "d": d,
                        "spec_model": model.get(method),
                        "dataset": contract["datasets"][name]["path"],
                    },
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--cell",
                    str(config),
                ]
                write_json(directory / "command.json", command)
                print(f"START {name} {method} D={d}", flush=True)
                with (directory / "run.log").open("w") as log:
                    result = subprocess.run(
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    write_json(
                        run_dir / "failed.json",
                        {"cell": str(directory), "returncode": result.returncode},
                    )
                    raise RuntimeError(f"Cell failed: {directory}")
                print(f"COMPLETE {name} {method} D={d}", flush=True)
    summarize(run_dir, contract)


if __name__ == "__main__":
    main()
