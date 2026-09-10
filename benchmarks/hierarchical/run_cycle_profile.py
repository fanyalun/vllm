# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused, paired uninstrumented/profiled verification-cycle experiment."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["ar", "mtp", "dspark", "moe_skip"], required=True
    )
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument(
        "--gdn-mode", choices=["none", "ssm_mean", "input_mean"], default="none"
    )
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--legacy-mm-inputs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--warmup-all", action="store_true")
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--draft-model")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--worker-extension", default="cycle_worker.CycleWorker")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["e2e", "profile", "e2e_after"],
        default=["e2e", "profile", "e2e_after"],
    )
    args = parser.parse_args()
    if args.gdn_mode != "none" and not args.rounds:
        parser.error("--gdn-mode requires hierarchical --rounds")
    if args.method == "ar" and (args.rounds or "profile" in args.phases):
        parser.error("AR control requires no --rounds and uninstrumented phases")
    from vllm import LLM, SamplingParams

    root = Path(__file__).resolve().parents[2]
    dataset = args.dataset or (
        root / "benchmarks/hierarchical/previous_config_20260909/samples_16.jsonl"
    )
    samples = [json.loads(line) for line in dataset.read_text().splitlines()][
        : args.samples
    ]
    spec = {
        "method": args.method,
        "num_speculative_tokens": args.draft_length,
        "draft_sample_method": "greedy",
    }
    if args.rounds:
        assert args.draft_length == 4
        spec = {
            "method": "hierarchical",
            "inner_method": args.method,
            "inner_num_speculative_tokens": 4,
            "inner_num_rounds": args.rounds,
            "moe_skip_top_h": 4,
            "draft_sample_method": "greedy",
            "preverify_gdn_mode": args.gdn_mode,
        }
    if args.method == "dspark":
        spec["model"] = "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark"
    if args.draft_model:
        spec["model"] = args.draft_model
    if args.method == "moe_skip":
        spec["moe_skip_top_h"] = 4
    config = dict(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        async_scheduling=args.async_scheduling,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config=None if args.method == "ar" else spec,
        disable_log_stats=True,
        seed=0,
        worker_extension_cls=args.worker_extension,
    )
    if args.legacy_mm_inputs:
        assert not args.rounds
        del config["limit_mm_per_prompt"]
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    save(
        "config.json",
        {
            "llm": config,
            "samples": samples,
            "max_tokens": args.max_tokens,
            "warmup_all": args.warmup_all,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD", "--", "vllm"])
            ).hexdigest(),
            "mean_kernel_sha256": (
                hashlib.sha256(
                    (
                        root / "vllm/model_executor/layers/mamba/gdn/mean_update.py"
                    ).read_bytes()
                ).hexdigest()
                if args.gdn_mode != "none"
                else None
            ),
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        },
    )
    llm = LLM(**config)
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_tokens, ignore_eos=True
    )
    for sample in samples if args.warmup_all else samples[:1]:
        llm.generate([sample["prompt"]], sampling, use_tqdm=False)
    print("WARMUP_COMPLETE", flush=True)
    results = []
    for phase in args.phases:
        for sample in samples:
            assert (
                hashlib.sha256(sample["prompt"].encode()).hexdigest()
                == sample["prompt_sha256"]
            )
            if phase == "profile":
                llm.collective_rpc("begin_cycle_measurement")
            start = time.perf_counter()
            output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
            elapsed = time.perf_counter() - start
            tokens = list(output.outputs[0].token_ids)
            assert len(tokens) == args.max_tokens
            assert len(output.prompt_token_ids) == sample["prompt_token_count"]
            measured = (
                llm.collective_rpc("collect_cycle_measurement")[0]
                if phase == "profile"
                else {}
            )
            results.append(
                {
                    "phase": phase,
                    "sample_index": sample["sample_index"],
                    "e2e_seconds": elapsed,
                    "token_ids": tokens,
                    **measured,
                }
            )
            save("result.json", results)
            print(
                f"COMPLETE {phase} {sample['sample_index']} {elapsed:.3f}s", flush=True
            )
    (args.output / "MEASUREMENT_COMPLETE").write_text(
        "Measured requested passes; not a correctness gate.\n"
    )


if __name__ == "__main__":
    main()
