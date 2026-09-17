# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure h8/h4/p0.125 with V0/V2/V3 on identical actual MTP windows."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--boundaries", nargs="+", type=int, default=[32, 96])
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    samples = [json.loads(x) for x in args.dataset.read_text().splitlines()][
        : args.samples
    ]
    ar = {
        r["sample"]: r["token_ids"]
        for r in json.loads(args.ar.read_text())
        if r["repeat"] == 0
    }
    assert len(samples) == args.samples
    assert all(len(ar[i]) >= max(args.boundaries) for i in range(args.samples))
    args.output.mkdir(parents=True, exist_ok=False)
    from vllm import LLM, SamplingParams

    config = dict(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="float32",
        async_scheduling=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
        seed=42,
        speculative_config=dict(
            method="hierarchical",
            inner_method="mtp",
            inner_num_speculative_tokens=4,
            inner_num_rounds=4,
            moe_skip_top_h=8,
            moe_skip_min_weight=None,
            preverify_gdn_mode="replay_tail",
            hierarchical_stop_policy="balanced",
            draft_sample_method="greedy",
        ),
        worker_extension_cls="moe_gdn_ablation_worker.MoeGdnAblationWorker",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                config=config,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                samples=samples,
                boundaries=args.boundaries,
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                ar_sha256=hashlib.sha256(args.ar.read_bytes()).hexdigest(),
                dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                prefix=(
                    "canonical prompt plus saved AR tokens; capture actual MTP anchor+4"
                ),
            ),
            indent=2,
        )
    )
    llm = LLM(**config)
    llm.collective_rpc("initialize_ablation")
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
    for index, sample in enumerate(samples):
        for j, boundary in enumerate(args.boundaries):
            path = args.output / f"sample_{index}_boundary_{boundary}.json"
            llm.collective_rpc(
                "prepare_ablation", args=(str(path.resolve()), bool((index + j) % 2))
            )
            llm.generate(
                [
                    dict(
                        prompt_token_ids=sample["prompt_token_ids"]
                        + ar[index][:boundary]
                    )
                ],
                params,
                use_tqdm=False,
            )
            assert path.exists(), "Fixed-window audit did not execute"
            print(index, boundary, "complete", flush=True)
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                windows=len(samples) * len(args.boundaries),
                cells=9,
                tokens_per_request=5,
                warmup=50,
                timing_samples=200,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
