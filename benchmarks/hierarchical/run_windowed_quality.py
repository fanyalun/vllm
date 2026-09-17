# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit L5/L16 and 4xD5 on canonical AR prefixes using the actual preverify model."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--rejection-audit", action="store_true")
    parser.add_argument("--boundaries", nargs="+", type=int, default=[32, 96, 160, 240])
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PATH"] = (
        str(Path(__file__).resolve().parents[2] / ".venv/bin")
        + os.pathsep
        + os.environ["PATH"]
    )
    from vllm import LLM, SamplingParams

    samples = [json.loads(line) for line in args.dataset.read_text().splitlines()][
        : args.samples
    ]
    assert len(samples) == args.samples
    ar = {
        r["sample"]: r["token_ids"]
        for r in json.loads(args.ar.read_text())
        if r["repeat"] == 0
    }
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [
        "benchmarks/hierarchical/run_windowed_quality.py",
        "benchmarks/hierarchical/windowed_quality_worker.py",
        "benchmarks/hierarchical/windowed_cost.py",
        "benchmarks/hierarchical/windowed_rejections.py",
        "vllm/model_executor/layers/mamba/gdn/replay_tail_update.py",
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "vllm/v1/worker/gpu/spec_decode/hierarchical/state.py",
        "vllm/v1/worker/gpu/spec_decode/hierarchical/speculator.py",
    ]
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                command=sys.argv,
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                source_sha256={
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in sources
                },
                ar_sha256=hashlib.sha256(args.ar.read_bytes()).hexdigest(),
                dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                samples=args.samples,
                boundaries=args.boundaries,
            ),
            indent=2,
        )
    )
    for name in sources:
        target = args.output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    llm = LLM(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="float32",
        limit_mm_per_prompt={"image": 0, "video": 0},
        async_scheduling=False,
        disable_log_stats=True,
        seed=42,
        speculative_config=dict(
            method="hierarchical",
            inner_method="mtp",
            inner_num_speculative_tokens=4,
            inner_num_rounds=4,
            moe_skip_top_h=4,
            draft_sample_method="greedy",
            hierarchical_stop_policy="balanced",
            preverify_gdn_mode="replay_tail",
        ),
        worker_extension_cls="windowed_quality_worker.WindowedQualityWorker",
    )
    llm.collective_rpc("set_replay_case", args=("none:carry:balanced",))
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, seed=42)
    for index, sample in enumerate(samples):
        for boundary in args.boundaries:
            length = 20 if boundary == 96 else 16
            tokens = ar[index]
            path = args.output / f"sample_{index}_boundary_{boundary}.json"
            llm.collective_rpc(
                "prepare_window_quality",
                args=(
                    tokens[boundary - 1 : boundary - 1 + length],
                    tokens[boundary : boundary + length],
                    str(path.resolve()),
                    args.rejection_audit,
                ),
            )
            llm.generate(
                [
                    {
                        "prompt_token_ids": sample["prompt_token_ids"]
                        + tokens[: boundary - 1]
                    }
                ],
                params,
                use_tqdm=False,
            )
            assert path.exists(), "Preverify audit did not execute"
            print(index, boundary, "complete", flush=True)
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                samples=len(samples),
                boundaries=args.boundaries,
                expected_windows=len(samples) * len(args.boundaries),
                variants=7,
                prefix="prompt + first boundary-1 AR tokens; then consume anchor",
                reference="same-input native GDN/top-h4 and greedy AR next tokens",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
