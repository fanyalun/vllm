# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One fresh-process end-to-end comparison cell, with excluded warmup."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def generate_cohort(llm, prompts, params):
    if len(prompts) == 1:
        return llm.generate(prompts, params, use_tqdm=False)
    core = llm.llm_engine.engine_core
    core.call_utility("pause_scheduler", "keep", False)
    try:
        llm.enqueue(prompts, params, use_tqdm=False)
    finally:
        core.call_utility("resume_scheduler")
    return llm.wait_for_completion(use_tqdm=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument(
        "--method", choices=["ar", "sd", "replayssm", "dual"], required=True
    )
    parser.add_argument(
        "--policy", choices=["ar", "d4", "d8", "p08", "p06"], required=True
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--kv-gib", type=float, default=10)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["DSPARK_MATRIX_STATS_PATH"] = str(root / "request_stats.jsonl")
    import torch

    from vllm import LLM, SamplingParams

    source = json.loads(Path(args.prompts).read_text())
    rows = source["samples"][: args.samples]
    prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row in rows]
    draft = 0 if args.method == "ar" else (4 if args.policy == "d4" else 8)
    width = draft + 1
    captures = sorted(
        {args.batch * width}
        | {n for n in (1, 2, 4, 8, 16, 32, 64, 128) if n <= args.batch * width}
    )
    config = dict(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        dtype="bfloat16",
        tensor_parallel_size=1,
        language_model_only=True,
        mamba_ssm_cache_dtype="float32",
        max_model_len=1024,
        max_num_seqs=args.batch,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enforce_eager=False,
        gpu_memory_utilization=0.95,
        kv_cache_memory_bytes=int(args.kv_gib * 1024**3),
        seed=20260915,
        async_scheduling=False,
        scheduler_cls="benchmarks.replayssm.dspark_matrix_stats.MatrixScheduler",
        worker_extension_cls="benchmarks.replayssm.qwen36_a100_matrix.JitMonitorExtension",
        additional_config={"gdn_prefill_backend": "triton"},
        compilation_config={
            "cudagraph_capture_sizes": captures,
            "max_cudagraph_capture_size": max(captures),
        },
        kernel_config={"enable_flashinfer_autotune": False},
        disable_log_stats=False,
    )
    if draft:
        spec = dict(
            method="dspark",
            model="/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
            num_speculative_tokens=draft,
        )
        if args.policy.startswith("p"):
            spec["dspark_confidence_threshold"] = {"p08": 0.8, "p06": 0.6}[args.policy]
        config["speculative_config"] = spec
    if args.method in ("replayssm", "dual"):
        config.update(
            use_replayssm_spec=True,
            replayssm_buffer_len=16,
            replayssm_spec_dual_checkpoint=args.method == "dual",
        )
    write_json(root / "config.json", config)
    llm = LLM(**config)
    llm.collective_rpc("benchmark_install_jit_counter")
    params = SamplingParams(
        temperature=0, max_tokens=args.tokens, ignore_eos=True, seed=20260915
    )
    warmup_params = SamplingParams(
        temperature=0, max_tokens=32, ignore_eos=True, seed=20260915
    )
    prefill_warmup_params = SamplingParams(
        temperature=0, max_tokens=1, ignore_eos=True, seed=20260915
    )

    def generate():
        outputs = []
        for start in range(0, len(prompts), args.batch):
            outputs.extend(
                generate_cohort(llm, prompts[start : start + args.batch], params)
            )
        return outputs

    warmups = []
    for warm in range(1):
        before = len(llm.collective_rpc("benchmark_read_jit_counter")[0])
        out = generate_cohort(llm, prompts[: args.batch], warmup_params)
        for start in range(args.batch, len(prompts), args.batch):
            out.extend(
                generate_cohort(
                    llm, prompts[start : start + args.batch], prefill_warmup_params
                )
            )
        after = len(llm.collective_rpc("benchmark_read_jit_counter")[0])
        warmups.append(
            dict(
                index=warm,
                jit_events=after - before,
                token_ids=[list(o.outputs[0].token_ids) for o in out],
            )
        )
    write_json(root / "warmup.json", warmups)
    result = dict(
        method=args.method,
        policy=args.policy,
        batch=args.batch,
        samples=len(prompts),
        output_tokens_per_request=args.tokens,
        prompts_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        repeats=[],
        complete=False,
    )
    for repeat in range(args.repeats):
        before = len(llm.collective_rpc("benchmark_read_jit_counter")[0])
        torch.accelerator.synchronize()
        start = time.perf_counter()
        outputs = generate()
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        events = llm.collective_rpc("benchmark_read_jit_counter")[0][before:]
        records = [
            dict(
                req_id=o.request_id,
                sample_id=rows[i]["id"],
                token_ids=list(o.outputs[0].token_ids),
                prompt_token_ids=list(o.prompt_token_ids),
            )
            for i, o in enumerate(outputs)
        ]
        assert all(len(r["token_ids"]) == args.tokens for r in records)
        assert [r["prompt_token_ids"] for r in records] == [
            r["prompt_token_ids"] for r in rows
        ]
        raw = [
            json.loads(line)
            for line in (root / "request_stats.jsonl").read_text().splitlines()
        ]
        # LLM's numeric external IDs gain an eight-character suffix in the core.
        by_id = {r["req_id"].rsplit("-", 1)[0]: r for r in raw}
        histogram = {}
        for r in records:
            assert r["req_id"] in by_id, r["req_id"]
            r["internal_req_id"] = by_id[r["req_id"]]["req_id"]
            r["initial_admission_width"] = by_id[r["req_id"]]["initial_admission_width"]
            r["acceptance_histogram"] = by_id[r["req_id"]]["histogram"]
            for key, count in r["acceptance_histogram"].items():
                histogram[key] = histogram.get(key, 0) + count
        result["repeats"].append(
            dict(
                repeat=repeat,
                elapsed_s=elapsed,
                output_tokens=len(records) * args.tokens,
                throughput_tps=len(records) * args.tokens / elapsed,
                jit_events=events,
                requests=records,
                histogram=histogram,
            )
        )
        write_json(root / "result.json", result)
        print(
            json.dumps(
                dict(
                    repeat=repeat,
                    throughput_tps=result["repeats"][-1]["throughput_tps"],
                    jit_count=len(events),
                )
            ),
            flush=True,
        )
    result["complete"] = True
    result["cohort_admission_exact"] = all(
        q["initial_admission_width"] == args.batch
        for r in result["repeats"]
        for q in r["requests"]
    )
    result["jit_clean"] = all(not r["jit_events"] for r in result["repeats"])
    result["repeat_tokens_equal"] = (
        None
        if args.repeats == 1
        else all(
            [q["token_ids"] for q in r["requests"]]
            == [q["token_ids"] for q in result["repeats"][0]["requests"]]
            for r in result["repeats"]
        )
    )
    write_json(root / "result.json", result)


if __name__ == "__main__":
    main()
