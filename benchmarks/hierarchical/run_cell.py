# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("ar", "mtp", "dspark", "moe_skip", "hierarchical"),
        required=True,
    )
    parser.add_argument("--inner-method", choices=("mtp", "dspark"), default="mtp")
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--dspark-model",
        default="/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
    )
    parser.add_argument(
        "--dataset",
        default=(
            "benchmark_results/sampling_acceptance_t1_p095_4x256_d16_d32_"
            "20260908/datasets/qwen36_first4.jsonl"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-h", type=int, default=4)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--log")
    parser.add_argument("--launch-blocking", action="store_true")
    parser.add_argument("--check-preverify", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--trace-dir")
    parser.add_argument("--ssm-dtype", choices=("auto", "float32"), default="float32")
    args = parser.parse_args()
    if args.log:
        with open(args.log, "w") as log_file:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.check_preverify:
        os.environ["VLLM_HIERARCHICAL_CHECK_PREVERIFY"] = "1"
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if args.trace_dir:
        os.environ["VLLM_HIERARCHICAL_TRACE_DIR"] = args.trace_dir
    venv_bin = Path(__file__).resolve().parents[2] / ".venv/bin"
    os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ["PATH"]
    from vllm import LLM, SamplingParams

    config = None
    if args.method != "ar":
        config = {
            "method": args.method,
            "num_speculative_tokens": (
                20 if args.method in ("moe_skip", "hierarchical") else 4
            ),
            "draft_sample_method": "greedy",
        }
        if args.method in ("moe_skip", "hierarchical"):
            config["moe_skip_top_h"] = args.top_h
        if args.method == "hierarchical":
            config.update(
                inner_method=args.inner_method,
                inner_num_rounds=4,
                inner_num_speculative_tokens=4,
            )
        if args.method == "dspark" or (
            args.method == "hierarchical" and args.inner_method == "dspark"
        ):
            config["model"] = args.dspark_model
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        json.loads(line) for line in Path(args.dataset).read_text().splitlines()
    ][: args.num_samples]
    if len(samples) != args.num_samples:
        raise ValueError("Dataset contains fewer samples than requested")
    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=args.eager,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype=args.ssm_dtype,
        limit_mm_per_prompt={"image": 0, "video": 0},
        async_scheduling=False,
        speculative_config=config,
        per_request_spec_decode_metrics="detailed" if config else "none",
        disable_log_stats=True,
        seed=20260908,
    )
    result = {
        "args": vars(args),
        "speculative_config": config,
        "init_seconds": time.perf_counter() - started,
        "outputs": [],
    }
    params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95 if args.temperature else 1.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=20260908,
    )
    llm.generate([samples[0]["prompt"]], params, use_tqdm=False)
    print("WARMUP_COMPLETE", flush=True)
    for repeat in range(args.repeats):
        for index, sample in enumerate(samples):
            digest = hashlib.sha256(sample["prompt"].encode()).hexdigest()
            assert digest == sample["prompt_sha256"]
            params.seed = 20260908 + index
            started = time.perf_counter()
            request = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
            elapsed = time.perf_counter() - started
            tokens = list(request.outputs[0].token_ids)
            assert len(tokens) == args.max_tokens
            spec_metrics = getattr(request.outputs[0], "spec_decode_metrics", None)
            if dataclasses.is_dataclass(spec_metrics):
                spec_metrics = dataclasses.asdict(spec_metrics)
            timing = request.metrics
            ttft = decode_seconds = None
            if timing is not None and timing.first_token_ts > 0:
                ttft = timing.first_token_latency
                decode_seconds = timing.last_token_ts - timing.first_token_ts
            result["outputs"].append(
                {
                    "repeat": repeat,
                    "sample_index": index,
                    "prompt_sha256": digest,
                    "token_ids": tokens,
                    "e2e_seconds": elapsed,
                    "ttft_seconds": ttft,
                    "decode_seconds": decode_seconds,
                    "seed": params.seed,
                    "spec_decode_metrics": spec_metrics,
                }
            )
            output.write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(
                f"SAMPLE_COMPLETE repeat={repeat} index={index} seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
