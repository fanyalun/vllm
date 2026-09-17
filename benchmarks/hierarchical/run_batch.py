# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real-request batches, with a separate instrumented output-parity pass."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--case",
        choices=["ar", "native", "v1", "v2", "v3", "v4d", "v4q", "v4dq"],
        default="v3",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmark_results/three_level_p50_20260916/final_prompts.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--batched-tokens", type=int, default=256)
    parser.add_argument("--kv-gib", type=float)
    parser.add_argument(
        "--cases", nargs="+", choices=["v1", "v2", "v3", "v4d", "v4q", "v4dq"]
    )
    parser.add_argument("--batches", nargs="+", type=int)
    args = parser.parse_args()
    if args.cases and args.case in ("ar", "native"):
        parser.error("--cases requires a windowed initial --case")
    if args.batches and (args.case != "ar" or max(args.batches) > args.batch):
        parser.error("--batches requires AR and batch capacity >= every measured batch")
    args.output.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(x) for x in args.dataset.read_text().splitlines()][
        : args.samples
    ]
    assert len(samples) == args.samples
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams

    spec = dict(
        method="hierarchical",
        inner_method="mtp",
        inner_num_speculative_tokens=4,
        inner_num_rounds=4,
        moe_skip_top_h=4,
        hierarchical_stop_policy="balanced",
        draft_sample_method="greedy",
    )
    if args.case not in ("ar", "native"):
        spec.update(
            preverify_gdn_mode="replay_tail",
            preverify_gdn_update_policy="windowed_three_level",
            preverify_gdn_tail_policy="carry",
            preverify_gdn_mode_window_size=1 if args.case == "v2" else 5,
            preverify_gdn_tau_beta=0.0 if args.case == "v1" else 0.36328125,
            preverify_gdn_tau_alpha=0.95,
            preverify_gdn_optimization={
                "v4d": "cumulative_decay",
                "v4q": "multi_query",
                "v4dq": "combined",
            }.get(args.case, "none"),
        )
    config = dict(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=args.batch,
        max_num_batched_tokens=args.batched_tokens
        if args.case == "ar"
        else max(args.batched_tokens, 21 * args.batch),
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="float32",
        async_scheduling=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config=None if args.case == "ar" else spec,
        disable_log_stats=True,
        seed=42,
        enforce_eager=args.eager,
        worker_extension_cls="batch_worker.BatchWorker",
    )
    if args.kv_gib is not None:
        config["kv_cache_memory_bytes"] = int(args.kv_gib * 2**30)

    def save(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")

    save(
        "manifest.json",
        dict(
            config=config,
            args={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            samples=samples,
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        ),
    )
    llm = LLM(**config)
    llm.collective_rpc("setup_batch_measurement")
    params = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
    prompts = [sample["prompt"] for sample in samples]
    groups = [prompts[i : i + args.batch] for i in range(0, len(prompts), args.batch)]
    if args.cases or args.batches:
        run_cases(llm, args, groups, params, samples)
        return
    # Warm each prompt batch; save output identity for repeatability checks.
    warm = [llm.generate(group, params, use_tqdm=False) for group in groups]
    expected = [[list(x.outputs[0].token_ids) for x in group] for group in warm]
    save("warmup_tokens.json", expected)
    rows = []
    for repeat in range(args.repeats):
        llm.collective_rpc("begin_batch_measurement")
        started = time.perf_counter()
        results = [llm.generate(group, params, use_tqdm=False) for group in groups]
        elapsed = time.perf_counter() - started
        audit = llm.collective_rpc("end_batch_measurement")[0]
        tokens = [[list(x.outputs[0].token_ids) for x in group] for group in results]
        row = dict(
            repeat=repeat,
            elapsed_s=elapsed,
            output_tokens=sum(len(x) for group in tokens for x in group),
            repeat_equal=tokens == expected,
            tokens=tokens,
            audit=audit,
        )
        rows.append(row)
        save("timings.json", rows)
    llm.collective_rpc(
        "begin_batch_measurement", kwargs={"audit": True, "profile": args.profile}
    )
    measured = [llm.generate(group, params, use_tqdm=False) for group in groups[:1]]
    profile_path = (
        str((args.output / "profile.json").resolve()) if args.profile else None
    )
    audit = llm.collective_rpc(
        "end_batch_measurement", kwargs={"profile_path": profile_path}
    )[0]
    audit["instrumentation_equal"] = [
        [list(x.outputs[0].token_ids) for x in group] for group in measured
    ] == expected[:1]
    audit["requests"] = [
        dict(
            request_id=x.request_id,
            sample_id=sample["sample_id"],
            prompt_sha256=sample["prompt_sha256"],
        )
        for x, sample in zip(measured[0], samples[: args.batch], strict=True)
    ]
    save("audit.json", audit)
    if args.quality and args.case != "ar":
        llm.collective_rpc("begin_batch_measurement", kwargs={"quality": True})
        quality = llm.generate(groups[0], params, use_tqdm=False)
        checks = llm.collective_rpc("end_batch_measurement")[0]
        checks["output_equal"] = [
            list(x.outputs[0].token_ids) for x in quality
        ] == expected[0]
        save("quality.json", checks)
    save(
        "complete.json",
        completion(args.batch, rows, audit),
    )


def run_cases(llm, args, groups, params, samples):
    cases = args.cases or [f"b{n}" for n in args.batches]
    batch_sizes = {
        case: int(case[1:]) if args.batches else args.batch for case in cases
    }
    prompts = [x for group in groups for x in group]
    case_groups = {
        case: [prompts[i : i + n] for i in range(0, len(prompts), n)]
        for case, n in batch_sizes.items()
    }

    def select(case):
        if not args.batches:
            llm.collective_rpc("set_batch_case", args=(case,))

    expected = {}
    results = {case: [] for case in cases}
    for case in cases:
        select(case)
        expected[case] = [
            [
                list(x.outputs[0].token_ids)
                for x in llm.generate(group, params, use_tqdm=False)
            ]
            for group in case_groups[case]
        ]
        directory = args.output / case
        directory.mkdir(exist_ok=True)
        (directory / "warmup_tokens.json").write_text(json.dumps(expected[case]))
    for repeat in range(args.repeats):
        for case in cases if repeat % 2 == 0 else list(reversed(cases)):
            select(case)
            llm.collective_rpc("begin_batch_measurement")
            started = time.perf_counter()
            outputs = [
                llm.generate(group, params, use_tqdm=False)
                for group in case_groups[case]
            ]
            elapsed = time.perf_counter() - started
            audit = llm.collective_rpc("end_batch_measurement")[0]
            tokens = [
                [list(x.outputs[0].token_ids) for x in group] for group in outputs
            ]
            results[case].append(
                dict(
                    repeat=repeat,
                    elapsed_s=elapsed,
                    output_tokens=sum(len(x) for group in tokens for x in group),
                    repeat_equal=tokens == expected[case],
                    tokens=tokens,
                    audit=audit,
                )
            )
            directory = args.output / case
            directory.mkdir(exist_ok=True)
            (directory / "timings.json").write_text(json.dumps(results[case], indent=2))
    for case in cases:
        select(case)
        llm.collective_rpc(
            "begin_batch_measurement", kwargs={"audit": True, "profile": args.profile}
        )
        outputs = llm.generate(case_groups[case][0], params, use_tqdm=False)
        directory = args.output / case
        path = str((directory / "profile.json").resolve()) if args.profile else None
        audit = llm.collective_rpc(
            "end_batch_measurement", kwargs={"profile_path": path}
        )[0]
        audit["instrumentation_equal"] = [
            list(x.outputs[0].token_ids) for x in outputs
        ] == expected[case][0]
        audit["requests"] = [
            dict(
                request_id=x.request_id,
                sample_id=sample["sample_id"],
                prompt_sha256=sample["prompt_sha256"],
            )
            for x, sample in zip(outputs, samples[: batch_sizes[case]], strict=True)
        ]
        (directory / "audit.json").write_text(json.dumps(audit, indent=2))
        if args.quality and args.case != "ar":
            llm.collective_rpc("begin_batch_measurement", kwargs={"quality": True})
            quality = llm.generate(groups[0], params, use_tqdm=False)
            checks = llm.collective_rpc("end_batch_measurement")[0]
            checks["output_equal"] = [
                list(x.outputs[0].token_ids) for x in quality
            ] == expected[case][0]
            (directory / "quality.json").write_text(json.dumps(checks, indent=2))
        (directory / "complete.json").write_text(
            json.dumps(
                completion(batch_sizes[case], results[case], audit),
                indent=2,
            )
        )


def completion(requested_batch, timings, audit):
    observed = max(
        [x.get("active_batch", 1) for x in audit["inner"]]
        or [
            x["num_reqs"]
            for x in audit.get("target_batches", [])
            if not x["has_prefill"]
        ]
        or [0]
    )
    return dict(
        completed=True,
        requested_batch=requested_batch,
        observed_max_active_batch=observed,
        full_batch_observed=observed == requested_batch,
        repeat_equal=all(x["repeat_equal"] for x in timings),
        instrumentation_equal=audit["instrumentation_equal"],
    )


if __name__ == "__main__":
    main()
