# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the public windowed configuration without benchmark case switching."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PATH"] = (
        str(Path(__file__).resolve().parents[2] / ".venv/bin")
        + os.pathsep
        + os.environ["PATH"]
    )
    from vllm import LLM, SamplingParams

    config = dict(
        method="hierarchical",
        inner_method="mtp",
        inner_num_speculative_tokens=4,
        inner_num_rounds=4,
        moe_skip_top_h=4,
        hierarchical_stop_policy="balanced",
        draft_sample_method="greedy",
        preverify_gdn_mode="replay_tail",
        preverify_gdn_group_mode="none",
        preverify_gdn_update_policy="windowed_three_level",
        preverify_gdn_mode_window_size=5,
        preverify_gdn_tau_alpha=0.95,
        preverify_gdn_tau_beta=0.36328125,
        preverify_gdn_optimization="combined",
        preverify_gdn_tail_policy="carry",
    )
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
    sample = json.loads(args.dataset.read_text().splitlines()[0])
    try:
        llm.generate(
            [sample["prompt"]],
            SamplingParams(temperature=0.5, max_tokens=1),
            use_tqdm=False,
        )
    except ValueError as exc:
        assert "temperature=0" in str(exc)
    else:
        raise AssertionError("Stochastic request was not rejected")
    params = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True, seed=42)
    tokens = [
        llm.generate([sample["prompt"]], params, use_tqdm=False)[0].outputs[0].token_ids
        for _ in range(2)
    ]
    assert len(tokens[0]) == 256 and tokens[0] == tokens[1]
    args.output.write_text(
        json.dumps(
            dict(
                config=config,
                token_ids=list(tokens[0]),
                repeated_outputs_equal=True,
                stochastic_request_rejected=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
