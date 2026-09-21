# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run D32 fixed two-level or original balanced three-level quality probes.

This B1 benchmark uses a capacity-40 hierarchical container to reserve buffers,
then installs a benchmark-only proposal controller. The fixed mode never calls
MTP during real proposals. Timing includes this container's allocation choices;
it must not be described as a tuned two-level performance baseline.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("two_level_fixed", "three_level_balanced"), required=True
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.tokens, args.repeats) <= 0:
        parser.error("samples, tokens, and repeats must be positive")
    samples = [json.loads(x) for x in args.dataset.read_text().splitlines()][
        : args.samples
    ]
    assert len(samples) == args.samples
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams

    config = dict(
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
        enforce_eager=args.eager,
        speculative_config=dict(
            method="hierarchical",
            inner_method="mtp",
            inner_num_speculative_tokens=4,
            inner_num_rounds=8,
            moe_skip_top_h=8,
            moe_skip_min_weight=0.125,
            moe_skip_weight_mode="preserve",
            preverify_gdn_mode="replay_tail",
            hierarchical_stop_policy="balanced",
            draft_sample_method="greedy",
        ),
        worker_extension_cls="long_draft_worker.LongDraftWorker",
    )

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    save(
        "manifest.json",
        dict(
            config=config,
            probe_mode=args.mode,
            actual_gdn_policy="windowed_three_level",
            actual_gdn_window=1,
            actual_max_rounds=32,
            delivery_limit=32,
            allocated_capacity=40,
            samples=samples,
            tokens=args.tokens,
            repeats=args.repeats,
            gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("long_draft_worker.py"),
                )
            },
        ),
    )
    llm = LLM(**config)
    llm.collective_rpc("setup_long_draft", args=(args.mode,))
    params = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
    prompts = [sample["prompt"] for sample in samples]

    def generate():
        return [
            list(llm.generate([prompt], params, use_tqdm=False)[0].outputs[0].token_ids)
            for prompt in prompts
        ]

    expected = generate()
    save("warmup_tokens.json", expected)
    rows = []
    for repeat in range(args.repeats):
        llm.collective_rpc("begin_batch_measurement")
        start = time.perf_counter()
        tokens = generate()
        elapsed = time.perf_counter() - start
        rows.append(dict(repeat=repeat, elapsed_s=elapsed, tokens=tokens))
        save("timings.json", rows)
        assert tokens == expected, "Repeat output differs"
    llm.collective_rpc("begin_batch_measurement", kwargs={"audit": True})
    audited = generate()
    audit = llm.collective_rpc("end_batch_measurement")[0]
    audit["instrumentation_equal"] = audited == expected
    save("audit.json", audit)
    assert audited == expected, "Instrumentation changed output"
    assert all(len(tokens) == args.tokens for tokens in expected)
    windows = []
    for row in audit["outer"]:
        for request, proposed, sampled, cycle in zip(
            row["request_ids"],
            row["scheduled"],
            row["sampled"],
            row["proposal_cycles"],
            strict=True,
        ):
            if cycle is None or proposed == 0:
                continue
            accepted = sampled - 1
            assert 0 <= accepted <= proposed <= 32
            windows.append(
                dict(request=request, cycle=cycle, proposed=proposed, accepted=accepted)
            )
    assert windows, "No matched verification windows"

    def distribution(rows):
        if not rows:
            return dict(windows=0)
        lengths = Counter(row["proposed"] for row in rows)
        accepted = Counter(row["accepted"] for row in rows)
        n = len(rows)
        total_a = sum(row["accepted"] for row in rows)
        total_l = sum(row["proposed"] for row in rows)
        return dict(
            windows=n,
            accepted_tokens=total_a,
            proposed_tokens=total_l,
            weighted_acceptance_rate=total_a / total_l,
            mean_accepted=total_a / n,
            mean_proposed=total_l / n,
            proposed_histogram=[lengths[i] for i in range(33)],
            accepted_histogram=[accepted[i] for i in range(33)],
            survival=[sum(row["accepted"] >= i for row in rows) / n for i in range(33)],
            zero_acceptance_rate=accepted[0] / n,
            full_acceptance_rate=sum(row["accepted"] == row["proposed"] for row in rows)
            / n,
        )

    save(
        "acceptance.json",
        dict(
            all_windows=distribution(windows),
            full_d32_windows=distribution(
                [row for row in windows if row["proposed"] == 32]
            ),
            windows=windows,
            note=(
                "Counts exclude Target correction/bonus; "
                "final output truncation may censor windows"
            ),
        ),
    )
    save("complete.json", dict(completed=True, instrumentation_equal=True))


if __name__ == "__main__":
    main()
