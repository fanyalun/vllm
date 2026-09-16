# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired 4 x 128 greedy acceptance smoke test for routing weight scaling."""

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
    args = parser.parse_args()
    os.environ["MOE_SKIP_WEIGHT_MODE"] = args.mode
    from vllm import LLM, SamplingParams

    dataset = (
        ROOT
        / "benchmark_results/moe_skip_top_p_4x128_20260915"
        / args.model
        / "dataset.jsonl"
    )
    samples = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(samples) == 4
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "mode": args.mode,
        "h": 4,
        "d": 4,
        "samples": 4,
        "max_tokens": 128,
        "temperature": 0,
        "seed": 0,
        "enforce_eager": True,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "model_path": MODELS[args.model][0],
    }
    write_json(args.output / "config.json", config)
    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={
            "method": "moe_skip",
            "moe_skip_top_h": 4,
            "num_speculative_tokens": 4,
        },
        per_request_spec_decode_metrics="detailed",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="weight_ablation_worker.WeightAblationWorker",
    )
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    outputs = []
    for sample in samples:
        result = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        output = result.outputs[0]
        assert len(output.token_ids) == 128
        assert len(result.prompt_token_ids) == sample["prompt_token_count"]
        metrics = output.spec_decode_metrics.to_dict()
        validate_metrics(metrics, 4)
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
        print(f"SAMPLE_COMPLETE {len(outputs)}/4", flush=True)
    accepted = sum(o["metrics"]["num_accepted_draft_tokens"] for o in outputs)
    drafted = sum(o["metrics"]["num_draft_tokens"] for o in outputs)
    steps = sum(o["metrics"]["num_spec_steps"] for o in outputs)
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
        },
    )
    (args.output / "CELL_COMPLETE").write_text("4 x 128 completed\n")


if __name__ == "__main__":
    main()
