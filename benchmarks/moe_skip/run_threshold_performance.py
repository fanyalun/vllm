# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired threshold smoke with warmup and per-request wall time."""

import argparse
import json
import os
import time
from pathlib import Path

from run_static_budget import MODELS, sha256, validate_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--path", choices=("default", "legacy", "h4"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for key in (
        "MOE_SKIP_BENCH_MIN_WEIGHT",
        "MOE_SKIP_BENCH_TOP_P",
        "MOE_SKIP_BENCH_COUNT_ONLY",
    ):
        os.environ.pop(key, None)
    spec = {"method": "moe_skip", "num_speculative_tokens": 16}
    if args.path == "h4":
        spec["moe_skip_top_h"] = 4
    if args.path == "legacy":
        spec["moe_skip_top_h"] = 8
        os.environ["MOE_SKIP_BENCH_MIN_WEIGHT"] = "0.125"
        os.environ["MOE_SKIP_WEIGHT_MODE"] = "preserve"
    from vllm import LLM, SamplingParams

    samples = [json.loads(s) for s in args.dataset.read_text().splitlines()]
    config = dict(
        model=args.model,
        path=args.path,
        spec=spec,
        d=16,
        dataset_sha256=sha256(args.dataset),
        cuda_graphs=args.cuda_graphs,
    )
    write_json(args.output / "config.json", config)
    llm = LLM(
        model=MODELS[args.model][0],
        tensor_parallel_size=1,
        enforce_eager=not args.cuda_graphs,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="top_p_worker.TopPWorker" if args.path == "legacy" else "",
    )
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    warmup = []
    for sample in samples:
        result = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        warmup.append(list(result.outputs[0].token_ids))
    print("WARMUP_COMPLETE", flush=True)
    outputs = []
    for sample, expected in zip(samples, warmup, strict=True):
        start = time.perf_counter()
        result = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        completion = result.outputs[0]
        metrics = completion.spec_decode_metrics.to_dict()
        validate_metrics(metrics, 16)
        assert len(completion.token_ids) == 128
        assert (
            result.prompt_token_ids
            and len(result.prompt_token_ids) == sample["prompt_token_count"]
        )
        outputs.append(
            dict(
                prompt_sha256=sample["prompt_sha256"],
                token_ids=list(completion.token_ids),
                metrics=metrics,
                seconds=elapsed,
                warmup_equal=list(completion.token_ids) == expected,
            )
        )
        write_json(args.output / "progress.json", outputs)
        print(f"SAMPLE_COMPLETE {len(outputs)}/{len(samples)}", flush=True)
    write_json(
        args.output / "result.json",
        dict(
            **config,
            outputs=outputs,
            seconds=sum(x["seconds"] for x in outputs),
            tokens_per_second=128 * len(samples) / sum(x["seconds"] for x in outputs),
        ),
    )
    (args.output / "CELL_COMPLETE").write_text("completed\n")


if __name__ == "__main__":
    main()
