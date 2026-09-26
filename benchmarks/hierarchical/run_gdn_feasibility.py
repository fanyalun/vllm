# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure GDN-only fixed drafting with native exact outer verification."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl"),
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument(
        "--variant", choices=("ar", "mtp", "v2", "full", "native_draft"), default="v2"
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--kv-gib", type=float)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--capture-state-on-cpu", action="store_true")
    parser.add_argument("--draft-block-graph", action="store_true")
    parser.add_argument("--batch-sharded-sampling", action="store_true")
    parser.add_argument("--release-mtp-bootstrap", action="store_true")
    parser.add_argument("--target-chunk-verify", action="store_true")
    parser.add_argument("--target-eager", action="store_true")
    parser.add_argument(
        "--target-chunk-precision", choices=("bf16", "fp32"), default="bf16"
    )
    args = parser.parse_args()
    if args.draft_block_graph and (args.eager or args.variant in ("ar", "mtp")):
        parser.error("--draft-block-graph requires graph-mode full-model drafting")
    if args.release_mtp_bootstrap and args.variant in ("ar", "mtp"):
        parser.error("--release-mtp-bootstrap requires full-model drafting")
    if args.target_chunk_verify and (
        not (args.eager or args.target_eager)
        or args.variant in ("ar", "mtp")
        or not 8 <= args.length <= 63
    ):
        parser.error("--target-chunk-verify requires eager full-model draft K=8..63")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    from vllm import LLM, SamplingParams

    def save(name, value):
        (args.output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )

    samples = [json.loads(line) for line in args.dataset.read_text().splitlines()][
        : args.batch
    ]
    prompts = [sample.get("prompt", sample.get("question")) for sample in samples]
    assert len(prompts) == args.batch and len(set(prompts)) == args.batch
    spec = None
    if args.variant == "mtp":
        spec = dict(method="mtp", num_speculative_tokens=4)
    elif args.variant != "ar":
        spec = dict(
            method="mtp",
            num_speculative_tokens=args.length,
            draft_sample_method="greedy",
        )
    config = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch,
        max_num_batched_tokens=args.token_budget,
        gpu_memory_utilization=0.95,
        cpu_offload_gb=args.cpu_offload_gb,
        enable_prefix_caching=False,
        async_scheduling=False,
        mamba_cache_mode="none",
        mamba_ssm_cache_dtype="float32",
        kernel_config={"moe_backend": "triton"},
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=args.eager,
        seed=42,
        disable_log_stats=True,
        speculative_config=spec,
        enable_batch_sharded_sampling=args.batch_sharded_sampling,
        worker_extension_cls="benchmarks.hierarchical.gdn_feasibility_worker.GdnFeasibilityWorker",
    )
    if args.kv_gib:
        config["kv_cache_memory_bytes"] = int(args.kv_gib * 2**30)
    if not args.eager:
        config["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": sorted(
                {
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    args.batch * (args.length + 1),
                }
            ),
        }
    save(
        "manifest.json",
        {
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "config": config,
            "prompts": prompts,
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                json.dumps(prompts, ensure_ascii=False).encode()
            ).hexdigest(),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD", "--", "vllm"])
            ).hexdigest(),
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("gdn_feasibility_worker.py"),
                    *(
                        [
                            Path(__file__).with_name("gdn_chunk_target.py"),
                            Path(__file__).parents[1]
                            / "kernels/benchmark_gdn_chunk_verify.py",
                            Path(__file__).parents[1] / "kernels/gdn_parallel_fp32.py",
                        ]
                        if args.target_chunk_verify
                        else []
                    ),
                )
            },
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                    "--format=csv",
                ],
                text=True,
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "diagnostic_environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_LAUNCH_BLOCKING",
                    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
                )
            },
            "scope": "Native autoregressive baseline"
            if args.variant == "ar"
            else "Native MTP four-candidate baseline"
            if args.variant == "mtp"
            else (
                "Unmodified FFN/MoE; fixed autoregressive V2 draft; full-model "
                "verification; TP extension is benchmark-only local-head projection "
                "adapter; no inner MTP proposals"
            ),
        },
    )
    for source in (
        Path(__file__),
        Path(__file__).with_name("gdn_feasibility_worker.py"),
        *(
            [
                Path(__file__).with_name("gdn_chunk_target.py"),
                Path(__file__).parents[1] / "kernels/benchmark_gdn_chunk_verify.py",
                Path(__file__).parents[1] / "kernels/gdn_parallel_fp32.py",
            ]
            if args.target_chunk_verify
            else []
        ),
    ):
        (args.output / source.name).write_bytes(source.read_bytes())
    llm = LLM(**config)
    info = llm.collective_rpc(
        "setup_feasibility",
        args=(
            args.variant,
            args.length,
            not args.eager,
            args.capture_state_on_cpu,
            args.draft_block_graph,
            args.release_mtp_bootstrap,
            args.target_chunk_verify,
            args.target_eager,
            args.target_chunk_precision,
        ),
    )
    save("initialization.json", info)
    params = SamplingParams(
        temperature=0, max_tokens=args.tokens, ignore_eos=True, seed=42
    )

    def generate():
        llm.collective_rpc("synchronize_feasibility")
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        llm.collective_rpc("synchronize_feasibility")
        elapsed = time.perf_counter() - start
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        assert all(len(row) == args.tokens for row in tokens)
        return dict(
            seconds=elapsed,
            token_ids=tokens,
            returned_tokens=sum(map(len, tokens)),
            tokens_per_second=sum(map(len, tokens)) / elapsed,
        )

    warmup = generate()
    save("warmup.json", warmup)
    timings = []
    for repeat in range(args.repeats):
        llm.collective_rpc("feasibility_measure")
        row = generate()
        row["repeat"] = repeat
        row["warmup_equal"] = row["token_ids"] == warmup["token_ids"]
        timings.append(row)
        save("timings.json", timings)
        print(
            json.dumps({k: v for k, v in row.items() if k != "token_ids"}), flush=True
        )
    llm.collective_rpc(
        "feasibility_measure",
        kwargs={"audit": True, "probe": args.probe, "quality": args.quality},
    )
    audited = generate()
    audits = llm.collective_rpc("feasibility_collect")
    save(
        "audit.json",
        {
            "output": audited,
            "workers": audits,
            "timing_equal": audited["token_ids"] == timings[-1]["token_ids"],
        },
    )
    save(
        "complete.json",
        {
            "completed": True,
            "repeat_equal": all(
                row["token_ids"] == timings[0]["token_ids"] for row in timings
            ),
            "warmup_equal": all(row["warmup_equal"] for row in timings),
            "audit_equal": audited["token_ids"] == timings[-1]["token_ids"],
            "actual_peak_batch": max(
                len(row["request_ids"]) for row in audits[0]["steps"]
            ),
            "tokens_per_second": sum(row["returned_tokens"] for row in timings)
            / sum(row["seconds"] for row in timings),
        },
    )


if __name__ == "__main__":
    main()
