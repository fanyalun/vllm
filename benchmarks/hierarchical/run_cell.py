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
    parser.add_argument(
        "--gdn-mode",
        choices=("none", "ssm_mean", "input_mean", "replay_tail"),
        default="none",
    )
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--dspark-model",
        default="/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
    )
    parser.add_argument(
        "--dataset",
        default=(
            "benchmark_results/.sources/sampling_acceptance_t1_p095_4x256_d16_d32_"
            "20260908/datasets/qwen36_first4.jsonl"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-h", type=int, default=4)
    parser.add_argument(
        "--batch-policy",
        choices=(
            "batch_top_half",
            "batch_max_gap",
            "batch_top_half_top1",
            "batch_max_gap_top1",
        ),
    )
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 32), default=1)
    parser.add_argument("--draft-tokens", type=int)
    parser.add_argument("--inner-rounds", type=int, default=4)
    parser.add_argument("--ar-reference", type=Path)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--log")
    parser.add_argument("--launch-blocking", action="store_true")
    parser.add_argument("--check-preverify", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--trace-dir")
    parser.add_argument(
        "--ssm-dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--routing-counts", action="store_true")
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--capture-sizes", type=int, nargs="+")
    args = parser.parse_args()
    if args.gdn_mode != "none" and args.method != "hierarchical":
        parser.error("--gdn-mode requires --method hierarchical")
    if args.batch_policy and args.method not in ("moe_skip", "hierarchical"):
        parser.error("--batch-policy requires moe_skip or hierarchical")
    if args.num_samples % args.batch_size:
        parser.error("--num-samples must be divisible by --batch-size")
    if args.routing_counts and (args.method != "hierarchical" or not args.batch_policy):
        parser.error("--routing-counts requires hierarchical with a batch policy")
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
                args.draft_tokens
                if args.draft_tokens is not None
                else (
                    args.inner_rounds * 5
                    if args.method == "hierarchical"
                    else 20
                    if args.method == "moe_skip"
                    else 4
                )
            ),
            "draft_sample_method": "greedy",
        }
        if args.method in ("moe_skip", "hierarchical"):
            if args.batch_policy:
                config["moe_skip_batch_policy"] = args.batch_policy
            else:
                config["moe_skip_top_h"] = args.top_h
        if args.method == "hierarchical":
            config.update(
                inner_method=args.inner_method,
                inner_num_rounds=args.inner_rounds,
                inner_num_speculative_tokens=4,
                preverify_gdn_mode=args.gdn_mode,
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
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        compilation_config=(
            {"cudagraph_capture_sizes": args.capture_sizes}
            if args.capture_sizes
            else None
        ),
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype=args.ssm_dtype,
        limit_mm_per_prompt={"image": 0, "video": 0},
        async_scheduling=False,
        speculative_config=config,
        per_request_spec_decode_metrics="detailed" if config else "none",
        disable_log_stats=True,
        seed=20260908,
        kernel_config={"moe_backend": "triton"},
        worker_extension_cls=(
            "benchmarks.hierarchical.routing_count_worker.RoutingCountWorker"
            if args.routing_counts
            else ""
        ),
    )
    result = {
        "args": vars(args),
        "speculative_config": config,
        "init_seconds": time.perf_counter() - started,
        "outputs": [],
        "batches": [],
    }
    params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95 if args.temperature else 1.0,
        max_tokens=args.warmup_tokens or args.max_tokens,
        ignore_eos=True,
        seed=20260908,
    )
    llm.generate(
        [s["prompt"] for s in samples[: args.batch_size]], params, use_tqdm=False
    )
    print("WARMUP_COMPLETE", flush=True)
    for repeat in range(args.repeats):
        for start in range(0, len(samples), args.batch_size):
            group = samples[start : start + args.batch_size]
            group_params = [
                SamplingParams(
                    temperature=args.temperature,
                    top_p=0.95 if args.temperature else 1.0,
                    max_tokens=args.max_tokens,
                    ignore_eos=True,
                    seed=20260908 + start + i,
                )
                for i in range(len(group))
            ]
            started = time.perf_counter()
            requests = llm.generate(
                [s["prompt"] for s in group], group_params, use_tqdm=False
            )
            elapsed = time.perf_counter() - started
            result["batches"].append(
                dict(
                    repeat=repeat,
                    start=start,
                    elapsed_seconds=elapsed,
                    returned_tokens=sum(len(r.outputs[0].token_ids) for r in requests),
                )
            )
            for offset, (sample, request) in enumerate(zip(group, requests)):
                index = start + offset
                digest = hashlib.sha256(sample["prompt"].encode()).hexdigest()
                assert digest == sample["prompt_sha256"]
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
                        "e2e_seconds": elapsed if args.batch_size == 1 else None,
                        "batch_start": start,
                        "ttft_seconds": ttft,
                        "decode_seconds": decode_seconds,
                        "seed": 20260908 + index,
                        "spec_decode_metrics": spec_metrics,
                    }
                )
            output.write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(
                f"BATCH_COMPLETE repeat={repeat} start={start} seconds={elapsed:.3f}",
                flush=True,
            )
    result["returned_token_throughput"] = sum(
        b["returned_tokens"] for b in result["batches"]
    ) / sum(b["elapsed_seconds"] for b in result["batches"])
    metrics = [
        r["spec_decode_metrics"]
        for r in result["outputs"]
        if r["spec_decode_metrics"] is not None
    ]
    if metrics:
        drafted = sum(m["num_draft_tokens"] for m in metrics)
        accepted = sum(
            sum(i * n for i, n in enumerate(m["histogram"])) for m in metrics
        )
        steps = sum(sum(m["histogram"]) for m in metrics)
        result["acceptance"] = dict(
            drafted=drafted,
            accepted=accepted,
            steps=steps,
            acceptance_rate=accepted / drafted if drafted else None,
            mean_acceptance_length=1 + accepted / steps if steps else None,
            mean_outer_accepted=accepted / steps if steps else None,
            mean_outer_submitted=drafted / steps if steps else None,
        )
    if args.ar_reference:
        ar = json.loads(args.ar_reference.read_text())
        assert ar.get("complete") and ar["speculative_config"] is None
        assert ar["args"]["batch_size"] == args.batch_size
        reference = {(r["repeat"], r["sample_index"]): r for r in ar["outputs"]}
        parity = []
        for row in result["outputs"]:
            ref = reference[row["repeat"], row["sample_index"]]
            assert row["prompt_sha256"] == ref["prompt_sha256"]
            assert row["seed"] == ref["seed"]
            parity.append(row["token_ids"] == ref["token_ids"])
        result["ar_parity"] = dict(matched=sum(parity), total=len(parity))
    if args.routing_counts:
        llm.collective_rpc("setup_routing_counts")
        llm.generate(
            [s["prompt"] for s in samples[: args.batch_size]], params, use_tqdm=False
        )
        llm.collective_rpc("begin_routing_counts")
        observed = []
        for repeat in range(args.repeats):
            for start in range(0, len(samples), args.batch_size):
                group = samples[start : start + args.batch_size]
                group_params = [
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=0.95 if args.temperature else 1.0,
                        max_tokens=args.max_tokens,
                        ignore_eos=True,
                        seed=20260908 + start + i,
                    )
                    for i in range(len(group))
                ]
                requests = llm.generate(
                    [s["prompt"] for s in group], group_params, use_tqdm=False
                )
                for offset, request in enumerate(requests):
                    out = request.outputs[0]
                    observed.append(
                        dict(
                            repeat=repeat,
                            sample_index=start + offset,
                            token_ids=list(out.token_ids),
                            spec_decode_metrics=dataclasses.asdict(
                                out.spec_decode_metrics
                            ),
                        )
                    )
        result["routing_counts"] = llm.collective_rpc("collect_routing_counts")[0]
        assert any(
            row["active_requests"] == args.batch_size
            for row in result["routing_counts"]["preverify_calls"]
        ), "Requested batch size was never reached"
        result["instrumented_outputs"] = observed
        result["instrumentation_parity"] = dict(
            tokens=all(
                a["token_ids"] == b["token_ids"]
                for a, b in zip(result["outputs"], observed, strict=True)
            ),
            acceptance=all(
                a["spec_decode_metrics"] == b["spec_decode_metrics"]
                for a, b in zip(result["outputs"], observed, strict=True)
            ),
        )
        output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        assert all(result["instrumentation_parity"].values()), (
            "Instrumentation parity failed"
        )
    result["complete"] = True
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
