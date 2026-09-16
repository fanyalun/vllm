# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired greedy acceptance test for routing weight scaling."""

import argparse
import json
import os
from pathlib import Path

from run_static_budget import MODELS, ROOT, sha256, validate_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("renormalize", "preserve"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--draft-length", type=int, default=4)
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--top-h", type=int, choices=range(1, 9))
    routing.add_argument("--expert-top-p", type=float)
    routing.add_argument("--expert-min-weight", type=float)
    parser.add_argument("--track-expert-counts", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.draft_length < 1:
        parser.error("--samples and --draft-length must be positive")
    if args.expert_top_p is not None and not 0 < args.expert_top_p <= 1:
        parser.error("--expert-top-p must be in (0, 1]")
    if args.expert_min_weight is not None and not 0 < args.expert_min_weight <= 1:
        parser.error("--expert-min-weight must be in (0, 1]")
    adaptive = args.expert_top_p is not None or args.expert_min_weight is not None
    top_h = args.top_h if args.top_h is not None else 4
    count_experts = adaptive or args.track_expert_counts
    os.environ["MOE_SKIP_WEIGHT_MODE"] = args.mode
    for key in (
        "MOE_SKIP_BENCH_TOP_P",
        "MOE_SKIP_BENCH_MIN_WEIGHT",
        "MOE_SKIP_BENCH_COUNT_ONLY",
    ):
        os.environ.pop(key, None)
    if args.expert_top_p is not None:
        os.environ["MOE_SKIP_BENCH_TOP_P"] = str(args.expert_top_p)
    elif args.expert_min_weight is not None:
        os.environ["MOE_SKIP_BENCH_MIN_WEIGHT"] = str(args.expert_min_weight)
    elif count_experts:
        os.environ["MOE_SKIP_BENCH_COUNT_ONLY"] = "1"
    from vllm import LLM, SamplingParams

    dataset = args.dataset or (
        ROOT
        / "benchmark_results/moe_skip_top_p_4x128_20260915"
        / args.model
        / "dataset.jsonl"
    )
    samples = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(samples) == args.samples
    assert len({sample["prompt_sha256"] for sample in samples}) == args.samples
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    config = {
        "model": args.model,
        "mode": args.mode,
        "h": None if adaptive else top_h,
        "expert_top_p": args.expert_top_p,
        "expert_min_weight": args.expert_min_weight,
        "threshold_fallback": "none",
        "histogram_first_bin": 1 if args.expert_top_p is not None else 0,
        "d": args.draft_length,
        "samples": args.samples,
        "max_tokens": 128,
        "temperature": 0,
        "seed": 0,
        "enforce_eager": not args.cuda_graphs,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "model_path": MODELS[args.model][0],
    }
    write_json(args.output / "config.json", config)
    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=1,
        enforce_eager=config["enforce_eager"],
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={
            "method": "moe_skip",
            "moe_skip_top_h": 8 if adaptive else top_h,
            "moe_skip_weight_mode": args.mode,
            "num_speculative_tokens": args.draft_length,
        },
        per_request_spec_decode_metrics="detailed",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls=("top_p_worker.TopPWorker" if count_experts else ""),
    )
    if count_experts:
        llm.collective_rpc("reset_budget_counts")
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    outputs = []
    for sample in samples:
        result = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        output = result.outputs[0]
        assert len(output.token_ids) == 128
        assert len(result.prompt_token_ids) == sample["prompt_token_count"]
        metrics = output.spec_decode_metrics.to_dict()
        validate_metrics(metrics, args.draft_length)
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "category": sample["category"],
                "prompt_sha256": sample["prompt_sha256"],
                "token_ids": list(output.token_ids),
                "metrics": metrics,
            }
        )
        write_json(args.output / "progress.json", outputs)
        print(f"SAMPLE_COMPLETE {len(outputs)}/{args.samples}", flush=True)
    accepted = sum(o["metrics"]["num_accepted_draft_tokens"] for o in outputs)
    drafted = sum(o["metrics"]["num_draft_tokens"] for o in outputs)
    steps = sum(o["metrics"]["num_spec_steps"] for o in outputs)
    budgets = None
    if count_experts:
        budgets = llm.collective_rpc("collect_budget_counts")
        if not any(sum(v) for worker in budgets for v in worker.values()):
            raise RuntimeError("Expert-count instrumentation did not record routing")
    write_json(
        args.output / "result.json",
        {
            **config,
            "outputs": outputs,
            "accepted": accepted,
            "drafted": drafted,
            "steps": steps,
            "acceptance_rate": accepted / drafted,
            "mean_acceptance_length": 1 + accepted / steps,
            "budget_histograms": budgets,
        },
    )
    (args.output / "CELL_COMPLETE").write_text(f"{args.samples} x 128 completed\n")


if __name__ == "__main__":
    main()
