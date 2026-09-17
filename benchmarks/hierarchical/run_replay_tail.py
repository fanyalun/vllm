# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure baseline/replay-tail pairs and audit the native tail-state control."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-method", choices=("mtp", "dspark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grouped-gdn", action="store_true")
    parser.add_argument("--three-level", action="store_true")
    parser.add_argument("--drain-device", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--capture-inputs", type=Path)
    parser.add_argument("--cost-output", type=Path)
    parser.add_argument("--action-audit", action="store_true")
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "benchmark_results/sampling_acceptance_t1_p095_4x256_d16_d32_"
            "20260908/datasets/qwen36_first4.jsonl"
        ),
    )
    args = parser.parse_args()
    cases = (
        [
            f"{state}:{group}"
            for state in ("none", "replay_tail")
            for group in ("none", "projection", "full")
        ]
        if args.grouped_gdn
        else ["none", "replay_tail", "tail_only"]
    )
    timed_cases = cases if args.grouped_gdn else ["none", "replay_tail"]
    if args.three_level:
        cases = args.cases or (
            ["none:carry:low_error"]
            + [
                f"{update}:{tail}:{stop}"
                for stop in ("low_error", "balanced", "aggressive", "none")
                for update, tail in (
                    ("exact", "carry"),
                    ("three_level", "carry"),
                    ("three_level", "repair_on_reject"),
                )
            ]
        )
        timed_cases = cases
    root = Path(__file__).resolve().parents[2]
    os.environ["PATH"] = str(root / ".venv/bin") + os.pathsep + os.environ["PATH"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams

    samples = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    samples = samples[: args.samples]
    assert len(samples) == args.samples
    for sample in samples:
        assert (
            hashlib.sha256(sample["prompt"].encode()).hexdigest()
            == sample["prompt_sha256"]
        )
    spec = dict(
        method="hierarchical",
        inner_method=args.inner_method,
        inner_num_speculative_tokens=4,
        inner_num_rounds=4,
        moe_skip_top_h=4,
        draft_sample_method="greedy",
        preverify_gdn_mode="replay_tail",
    )
    if args.inner_method == "dspark":
        spec["model"] = "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark"
    if args.grouped_gdn:
        spec["preverify_gdn_group_mode"] = "projection"
    config = dict(
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
        speculative_config=spec,
        disable_log_stats=True,
        seed=args.seed,
        worker_extension_cls="replay_tail_worker.ReplayTailWorker",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.grouped_gdn:
        config["worker_extension_cls"] = "grouped_gdn_worker.GroupedGDNWorker"
    if args.three_level:
        config["worker_extension_cls"] = "three_level_worker.ThreeLevelWorker"

    def save(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")

    save(
        "contract.json",
        {
            "llm": config,
            "samples": samples,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "drain_device_before_stopping_timer": args.drain_device,
            "cases": cases,
            "timed_cases": timed_cases,
            "gates": "per_token",
            "tail_kernel": "register_recurrence_final_store",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "command": sys.argv,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in (
                    "vllm/config/speculative.py",
                    "vllm/v1/worker/gpu/spec_decode/hierarchical/state.py",
                    "vllm/v1/worker/gpu/spec_decode/hierarchical/speculator.py",
                    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                    "vllm/model_executor/layers/mamba/gdn/replay_tail_update.py",
                    "benchmarks/hierarchical/run_replay_tail.py",
                    "benchmarks/hierarchical/replay_tail_worker.py",
                    "benchmarks/hierarchical/grouped_gdn_worker.py",
                    "benchmarks/hierarchical/three_level_worker.py",
                    "vllm/model_executor/layers/mamba/gdn/grouped_input.py",
                    "vllm/v1/worker/gpu/spec_decode/hierarchical/grouped_gdn.py",
                )
            },
        },
    )
    contract = json.loads((args.output / "contract.json").read_text())
    for path in contract["source_sha256"]:
        target = args.output / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / path).read_bytes())
    llm = LLM(**config)
    if args.capture_inputs:
        llm.collective_rpc(
            "capture_three_level_inputs", args=(str(args.capture_inputs.resolve()),)
        )
    params = SamplingParams(
        temperature=0, max_tokens=args.max_tokens, ignore_eos=True, seed=args.seed
    )
    rows, memory = [], {}
    references = {}

    def generate(case, phase, repeat, index, sample):
        params.seed = args.seed if args.three_level else args.seed + index
        if phase == "audit":
            llm.collective_rpc("begin_replay_audit")
        elif phase == "actions":
            llm.collective_rpc("begin_action_audit")
        start = time.perf_counter()
        result = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
        if args.drain_device:
            llm.collective_rpc("drain_replay_device")
        elapsed = time.perf_counter() - start
        tokens = list(result.outputs[0].token_ids)
        assert len(tokens) == args.max_tokens
        assert len(result.prompt_token_ids) == sample["prompt_token_count"]
        if "prompt_token_ids" in sample:
            assert list(result.prompt_token_ids) == sample["prompt_token_ids"]
        key = case, index
        if key in references:
            assert tokens == references[key], ("nonrepeatable output", key, phase)
        references[key] = tokens
        audit = (
            llm.collective_rpc("collect_replay_audit")[0] if phase == "audit" else {}
        )
        if phase == "actions":
            audit = llm.collective_rpc("collect_action_audit")[0]
        if phase != "warmup":
            rows.append(
                dict(
                    case=case,
                    phase=phase,
                    repeat=repeat,
                    sample=index,
                    prompt_sha256=sample["prompt_sha256"],
                    seconds=elapsed,
                    token_ids=tokens,
                    text=result.outputs[0].text,
                    **audit,
                )
            )
            save("results.json", rows)
        print(f"COMPLETE {case} {phase} {repeat} {index} {elapsed:.3f}s", flush=True)

    for case in cases:
        memory[case] = llm.collective_rpc("set_replay_case", args=(case,))[0]
        if args.cost_output and case == cases[0]:
            llm.collective_rpc(
                "begin_three_level_cost", args=(str(args.cost_output.resolve()),)
            )
        for index, sample in enumerate(samples):
            generate(case, "warmup", 0, index, sample)
        memory[case] = llm.collective_rpc("set_replay_case", args=(case,))[0]
    save("private_state.json", memory)
    if args.three_level:
        llm.collective_rpc("set_three_level_jit_guard", args=(True,))
    for repeat in range(args.repeats):
        order = timed_cases if repeat % 2 == 0 else timed_cases[::-1]
        for case in order:
            before = llm.collective_rpc("set_replay_case", args=(case,))[0]
            for index, sample in enumerate(samples):
                generate(case, "e2e", repeat, index, sample)
            after = llm.collective_rpc("set_replay_case", args=(case,))[0]
            assert before["graphs"] == after["graphs"], "new graph in timed run"
    if args.three_level:
        llm.collective_rpc("set_three_level_jit_guard", args=(False,))
    for case in cases:
        llm.collective_rpc("set_replay_case", args=(case,))
        for index, sample in enumerate(samples):
            generate(case, "audit", 0, index, sample)
    expected = args.samples * (len(timed_cases) * args.repeats + len(cases))
    if args.action_audit:
        for case in cases:
            if not case.startswith(("three_level", "windowed")):
                continue
            llm.collective_rpc("set_replay_case", args=(case,))
            for index, sample in enumerate(samples):
                generate(case, "actions", 0, index, sample)
            expected += args.samples
    assert len(rows) == expected
    if args.profile_output:
        llm.collective_rpc("set_replay_case", args=(cases[-1],))
        params.max_tokens = 32
        reference = llm.generate([samples[0]["prompt"]], params, use_tqdm=False)[0]
        llm.collective_rpc("begin_three_profile")
        profiled = llm.generate([samples[0]["prompt"]], params, use_tqdm=False)[0]
        llm.collective_rpc(
            "end_three_profile", args=(str(args.profile_output.resolve()),)
        )
        assert list(reference.outputs[0].token_ids) == list(
            profiled.outputs[0].token_ids
        )
        save(
            "profile_parity.json",
            dict(
                case=cases[-1],
                tokens=32,
                token_ids=list(profiled.outputs[0].token_ids),
                identical=True,
            ),
        )
    if args.three_level:
        llm.collective_rpc(
            "export_three_level_kernels",
            args=(str((args.output / "kernels").resolve()),),
        )
    save(
        "measurement_complete.json",
        {
            "rows": len(rows),
            "expected": expected,
            "repeatable_tokens": True,
            "ar_correctness_checked": False,
        },
    )


if __name__ == "__main__":
    main()
