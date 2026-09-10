# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure full/top-4 forwards with identical candidate inputs and state."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen36", "gemma4"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled-totals", action="store_true")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    model = {
        "qwen36": "/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        "gemma4": "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it",
    }[args.model]
    spec = dict(
        method="hierarchical",
        inner_method="mtp",
        inner_num_speculative_tokens=4,
        inner_num_rounds=1,
        moe_skip_top_h=4,
        draft_sample_method="greedy",
    )
    if args.model == "gemma4":
        spec["model"] = "/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it-assistant"
    config = dict(
        model=model,
        tensor_parallel_size=1,
        enforce_eager=not args.compiled_totals,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        async_scheduling=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config=spec,
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="forward_stage_worker.ForwardStageWorker",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["FORWARD_STAGE_OUTPUT"] = str(args.output.resolve())
    os.environ["FORWARD_STAGE_COMPILED"] = "1" if args.compiled_totals else "0"

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    dataset = Path(__file__).parent / "previous_config_20260909/samples_16.jsonl"
    samples = [json.loads(line) for line in dataset.read_text().splitlines()][:3]
    save(
        "config.json",
        {
            "llm": config,
            "samples": samples,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "environment": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("VLLM_", "CUDA_", "HF_"))
            },
            "contract": (
                "Same model instance, fixed candidate IDs and private GDN prefix "
                "state; widths 4 and 5; 20 paired graph replays after 3 excluded "
                "replays. Total includes backbone, logits and argmax; excludes "
                "metadata/state restore and proposal generation."
            ),
        },
    )
    llm = LLM(**config)
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    llm.generate([samples[0]["prompt"]], sampling, use_tqdm=False)
    print("WARMUP_COMPLETE", flush=True)
    results = []
    for sample in samples:
        baseline = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        llm.collective_rpc("begin_forward_stages")
        output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        rows = llm.collective_rpc("collect_forward_stages")[0]
        assert len(rows) == (80 if args.compiled_totals else 160), len(rows)
        assert list(baseline.outputs[0].token_ids) == list(output.outputs[0].token_ids)
        results.append(
            {
                "sample_index": sample["sample_index"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_tokens": len(output.prompt_token_ids),
                "token_ids": list(output.outputs[0].token_ids),
                "control_token_ids": list(baseline.outputs[0].token_ids),
                "rows": rows,
            }
        )
        save("result.json", results)
        print(f"SAMPLE_COMPLETE {sample['sample_index']}", flush=True)
    (args.output / "MEASUREMENT_COMPLETE").write_text(
        "3 fixed-prefix paired cases; output controls passed\n"
    )


if __name__ == "__main__":
    main()
