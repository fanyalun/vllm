# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the frozen recommendation through normal LLM configuration."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inner-method", choices=("mtp", "dspark"), required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PATH"] = (
        str(Path(__file__).resolve().parents[2] / ".venv/bin")
        + os.pathsep
        + os.environ["PATH"]
    )
    from vllm import LLM, SamplingParams

    freeze = json.loads((args.root / "final/freeze.json").read_text())
    case = freeze["selected"][args.inner_method]
    _, tail, stop = case.split(":")
    sample = json.loads((args.root / "final_prompts.jsonl").read_text().splitlines()[0])
    config = dict(
        method="hierarchical",
        inner_method=args.inner_method,
        inner_num_speculative_tokens=4,
        inner_num_rounds=4,
        moe_skip_top_h=4,
        draft_sample_method="greedy",
        hierarchical_stop_policy=stop,
        preverify_gdn_mode="replay_tail",
        preverify_gdn_update_policy="three_level_p50",
        preverify_gdn_tail_policy=tail,
    )
    if args.inner_method == "dspark":
        config["model"] = "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark"
    llm = LLM(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="float32",
        async_scheduling=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
        seed=42,
        speculative_config=config,
    )
    params = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True, seed=42)
    outputs = [
        llm.generate([sample["prompt"]], params, use_tqdm=False)[0].outputs[0]
        for _ in range(2)
    ]
    assert list(outputs[0].token_ids) == list(outputs[1].token_ids)
    assert len(outputs[1].token_ids) == 256
    output = args.root / "direct_api"
    output.mkdir(exist_ok=True)
    (output / f"{args.inner_method}.json").write_text(
        json.dumps(
            dict(
                config=config,
                sample=0,
                prompt_sha256=sample["prompt_sha256"],
                token_ids=list(outputs[1].token_ids),
                text=outputs[1].text,
                repeatable=True,
                worker_case_switching=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
