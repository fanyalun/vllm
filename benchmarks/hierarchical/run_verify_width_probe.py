# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe fixed-prefix Gemma full/h4 forwards across verification widths."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams

    root = Path("benchmark_results/.sources/gemma_h4_policy_16x512_20260915_run3")
    dataset = root / "dataset.jsonl"
    samples = [json.loads(line) for line in dataset.read_text().splitlines()]
    ar = json.loads((root / "b16_ar/result.json").read_text())["outputs"]
    config = dict(
        model="/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        async_scheduling=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
        seed=0,
        speculative_config=dict(
            method="hierarchical",
            model="/home/fanya/data1/fanya/models/gemma-4-26B-A4B-it-assistant",
            inner_method="mtp",
            inner_num_speculative_tokens=4,
            inner_num_rounds=6,
            moe_skip_top_h=4,
            draft_sample_method="greedy",
        ),
        worker_extension_cls=(
            "benchmarks.hierarchical.verify_width_worker.VerifyWidthWorker"
        ),
    )

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    try:
        cupti_version = importlib.metadata.version("cupti-python")
    except importlib.metadata.PackageNotFoundError:
        cupti_version = "unavailable; FlashInfer CUDA event fallback"
    save(
        "manifest.json",
        dict(
            config=config,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True),
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            cupti_python=cupti_version,
            environment={
                k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "CUDA_"))
            },
            contract="Fixed prompt plus 64 AR tokens. Nested candidate prefixes. "
            "Private verification forward includes backbone, logits, top2 and argmax; "
            "excludes metadata, sampling, scheduler and draft generation. "
            "Manual CUDA graph replay; FlashInfer CUPTI cold L2; 20 repeats. "
            "Full h8 is a Target-forward proxy, not execute_model timing.",
        ),
    )
    llm = LLM(**config)
    tokenizer = llm.get_tokenizer()
    prompts = [
        dict(prompt_token_ids=tokenizer.encode(s["prompt"]) + a["token_ids"][:64])
        for s, a in zip(samples, ar)
    ]
    candidates = [a["token_ids"][64:95] for a in ar]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    results = []
    for batch in (1, 4, 8, 16):
        baseline = llm.generate(prompts[:batch], sampling, use_tqdm=False)
        llm.collective_rpc("begin_width_probe", args=(candidates[:batch],))
        actual = llm.generate(prompts[:batch], sampling, use_tqdm=False)
        rows = llm.collective_rpc("collect_width_probe")[0]
        control = [list(x.outputs[0].token_ids) for x in baseline]
        observed = [list(x.outputs[0].token_ids) for x in actual]
        assert control == observed, "probe changed generated output"
        assert len(rows) == 5, len(rows)
        results.append(dict(batch=batch, control=control, observed=observed, rows=rows))
        save("result.json", results)
        print(f"BATCH_COMPLETE {batch}", flush=True)
    (args.output / "MEASUREMENT_COMPLETE").write_text("All four controls passed\n")


if __name__ == "__main__":
    main()
