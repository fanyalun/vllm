# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure complete draft proposals on real sequential B=1 generation requests."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from run_performance import METHODS, MODELS, ROOT, write_json


def cell(config_path):
    from vllm import LLM, SamplingParams

    config = json.loads(config_path.read_text())
    method, d = config["method"], config["d"]
    spec = {"method": method, "num_speculative_tokens": d}
    if method == "moe_skip":
        spec["moe_skip_top_h"] = 4
    elif config["spec_model"]:
        spec["model"] = config["spec_model"]
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
        worker_extension_cls="draft_timing_worker.DraftTimingWorker",
    )
    samples = [
        json.loads(line) for line in Path(config["dataset"]).read_text().splitlines()
    ]
    sampling = SamplingParams(temperature=0, max_tokens=512, ignore_eos=True)
    llm.generate([samples[0]["prompt"]], sampling, use_tqdm=False)
    print("WARMUP_COMPLETE", flush=True)
    rows, outputs = [], []
    for sample in samples:
        llm.collective_rpc("begin_draft_timing")
        output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        timing = llm.collective_rpc("collect_draft_timing")
        assert len(timing) == 1 and timing[0]
        assert len(output.outputs[0].token_ids) == 512
        assert len(output.prompt_token_ids) == sample["prompt_token_count"]
        for row in timing[0]:
            assert row["num_reqs"] == 1 and row["draft_width"] == d
            assert row["stream_elapsed_ms"] > 0
            rows.append({"sample_index": sample["sample_index"], **row})
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "prompt_sha256": sample["prompt_sha256"],
                "token_ids": list(output.outputs[0].token_ids),
            }
        )
        write_json(
            config_path.parent / "progress.json",
            {"completed_samples": len(outputs), "proposal_calls": len(rows)},
        )
        print(f"SAMPLE_COMPLETE {len(outputs)}/16 calls={len(timing[0])}", flush=True)
    write_json(
        config_path.parent / "result.json",
        {**config, "outputs": outputs, "proposals": rows},
    )
    (config_path.parent / "CELL_COMPLETE").write_text("16 x 512; all proposals saved\n")


def summarize(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    contract = json.loads((root / "contract.json").read_text())
    rows, raw = [], []
    for name in contract["models"]:
        identity = None
        for method, lengths in METHODS.items():
            if method == "ar":
                continue
            for d in lengths:
                directory = root / name / f"{method}_d{d}"
                assert (directory / "CELL_COMPLETE").exists(), directory
                result = json.loads((directory / "result.json").read_text())
                assert len(result["outputs"]) == 16
                prompts = [o["prompt_sha256"] for o in result["outputs"]]
                if identity is None:
                    identity = prompts
                assert identity == prompts
                assert all(len(o["token_ids"]) == 512 for o in result["outputs"])
                for row in result["proposals"]:
                    raw.append({"model": name, "method": method, "d": d, **row})
                for prefill in (False, True):
                    calls = [
                        r for r in result["proposals"] if r["has_prefill"] == prefill
                    ]
                    assert calls
                    times = np.array([r["stream_elapsed_ms"] for r in calls])
                    rows.append(
                        {
                            "model": name,
                            "method": method,
                            "d": d,
                            "phase": "prefill_proposal"
                            if prefill
                            else "decode_proposal",
                            "calls": len(calls),
                            "mean_block_ms": float(times.mean()),
                            "median_block_ms": float(np.median(times)),
                            "p90_block_ms": float(np.percentile(times, 90)),
                            "amortized_ms_per_draft_token": float(times.mean() / d),
                            "mean_cpu_submit_ms": float(
                                np.mean([r["cpu_submit_ms"] for r in calls])
                            ),
                        }
                    )
    for filename, values in (("draft_timing.csv", rows), ("proposal_timings.csv", raw)):
        with (root / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    fig, axes = plt.subplots(
        len(contract["models"]),
        2,
        squeeze=False,
        figsize=(12, 4.8 * len(contract["models"])),
        constrained_layout=True,
    )
    labels = {"moe_skip": "MoE-Skip (top-4)", "mtp": "MTP", "dspark": "DSpark"}
    for index, name in enumerate(contract["models"]):
        for axis, metric, label in zip(
            axes[index],
            ["mean_block_ms", "amortized_ms_per_draft_token"],
            ["Full D-token proposal (ms)", "Amortized cost / draft token (ms)"],
            strict=True,
        ):
            for method, display in labels.items():
                values = [
                    r
                    for r in rows
                    if r["model"] == name
                    and r["method"] == method
                    and r["phase"] == "decode_proposal"
                ]
                axis.plot(
                    [r["d"] for r in values],
                    [r[metric] for r in values],
                    marker="o",
                    label=display,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
            axis.set(
                xlabel="Draft length D",
                ylabel=label,
                title={"qwen36": "Qwen3.6-35B-A3B", "gemma4": "Gemma4-26B-A4B"}[name],
            )
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle(
        "Draft construction: CUDA-event interval around propose()\n"
        "B=1 | CUDA Graph | 16 x 512 | decode proposals only | lower is better"
    )
    for ext in ("png", "pdf"):
        fig.savefig(root / f"draft_construction.{ext}", dpi=180)
    for axis in axes.flat:
        axis.set_yscale("log")
        axis.set_ylabel(axis.get_ylabel() + "; log scale")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"draft_construction_log.{ext}", dpi=180)
    plt.close(fig)
    write_json(
        root / "audit.json",
        {
            "status": "passed",
            "cells": len(rows) // 2,
            "requests": 16 * len(rows) // 2,
            "proposal_calls": len(raw),
            "boundary": "propose inputs/state preparation + draft forward + sampling",
            "metric": "CUDA stream elapsed interval, including CPU-induced idle gaps",
            "excluded": "target verify; initialization; warmup; RPC collection",
            "amortized_note": "block time / D, not measured single-forward latency",
            "aggregation": "proposal-call-weighted mean; prefill proposals separate",
        },
    )
    (root / "RUN_COMPLETE").write_text("All cells audited and plotted\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.cell:
        cell(args.cell)
        return
    root = args.run_dir.resolve()
    if args.summarize:
        summarize(root)
        return
    assert args.model
    root.mkdir(parents=True, exist_ok=False)
    model = MODELS[args.model]
    source = (
        ROOT / "benchmark_results" / model["source"] / "dataset/multicategory_128.jsonl"
    )
    samples = [json.loads(line) for line in source.read_text().splitlines()][:16]
    dataset = root / "samples.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in samples))
    write_json(
        root / "contract.json",
        {
            "models": [args.model],
            "cuda_device": args.cuda_device,
            "num_samples": 16,
            "output_length": 512,
            "batch_size": 1,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "source_dataset": str(source),
            "warmup_requests": 1,
            "methods": {m: ds for m, ds in METHODS.items() if m != "ar"},
        },
    )
    env = os.environ.copy()
    for key in ("VLLM_MOE_SKIP_TRACE_DIR", "VLLM_DRAFT_TOPK_TRACE_DIR"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=args.cuda_device,
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        MPLCONFIGDIR="/tmp/moe_timing_mpl",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve()) + os.pathsep + str(ROOT)
    for method, lengths in METHODS.items():
        if method == "ar":
            continue
        for d in lengths:
            directory = root / args.model / f"{method}_d{d}"
            directory.mkdir(parents=True)
            config = directory / "config.json"
            write_json(
                config,
                {
                    "model": model["model"],
                    "method": method,
                    "d": d,
                    "spec_model": model.get(method),
                    "dataset": str(dataset),
                },
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cell",
                str(config),
            ]
            write_json(directory / "command.json", command)
            print(f"START {args.model} {method} D={d}", flush=True)
            with (directory / "run.log").open("w") as log:
                result = subprocess.run(
                    command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                raise RuntimeError(f"Failed cell: {directory}")
            print(f"COMPLETE {args.model} {method} D={d}", flush=True)
    summarize(root)


if __name__ == "__main__":
    main()
