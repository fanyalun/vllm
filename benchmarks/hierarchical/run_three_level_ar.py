# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy AR token control for the frozen Three-Level evaluation prompts."""

import argparse
import hashlib
import json
import os
import time
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

    samples = [json.loads(x) for x in args.dataset.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
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
    )
    params = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True, seed=42)
    llm.generate([samples[0]["prompt"]], params, use_tqdm=False)
    rows = []
    for repeat in range(2):
        for index, sample in enumerate(samples):
            assert (
                hashlib.sha256(sample["prompt"].encode()).hexdigest()
                == sample["prompt_sha256"]
            )
            start = time.perf_counter()
            result = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
            elapsed = time.perf_counter() - start
            tokens = list(result.outputs[0].token_ids)
            assert len(tokens) == 256
            assert list(result.prompt_token_ids) == sample["prompt_token_ids"]
            if repeat:
                assert tokens == rows[index]["token_ids"]
            rows.append(
                dict(
                    sample=index,
                    repeat=repeat,
                    seconds=elapsed,
                    token_ids=tokens,
                    text=result.outputs[0].text,
                    prompt_sha256=sample["prompt_sha256"],
                )
            )
            (args.output / "results.json").write_text(json.dumps(rows, indent=2))
            print("COMPLETE", repeat, index, elapsed, flush=True)
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                samples=len(samples),
                repeatable=True,
                seed=42,
                dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
