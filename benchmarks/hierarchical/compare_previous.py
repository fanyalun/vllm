# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match the previous Qwen3.6 performance experiments with D=4, variable N."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "benchmark_results/moe_skip_e2e_16x512_b1_20260908_run2"
DATASET = (
    ROOT
    / "benchmark_results/moe_skip_qwen36_multicategory_128x512_20260906"
    / "dataset/multicategory_128.jsonl"
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_queued_cell(directory, config, env):
    with (directory / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"BUSY {directory}", flush=True)
            return
        if (directory / "CELL_COMPLETE").exists():
            return
        write_json(directory / "config.json", config)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            str(directory / "config.json"),
        ]
        write_json(directory / "command.json", command)
        print(f"START {directory}", flush=True)
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(
                command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            write_json(
                directory / "failed.json",
                {"cell": str(directory), "exit_code": result.returncode},
            )
            raise RuntimeError(f"Failed: {directory}")
        print(f"COMPLETE {directory}", flush=True)


def cell(path):
    from vllm import LLM, SamplingParams

    config = json.loads(path.read_text())
    spec = {
        "method": "hierarchical",
        "inner_method": config["inner_method"],
        "inner_num_speculative_tokens": 4,
        "inner_num_rounds": config["n"],
        "moe_skip_top_h": 4,
        "draft_sample_method": "greedy",
    }
    if config["inner_method"] == "dspark":
        spec["model"] = "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark"
    extra = {}
    if config["phase"] != "e2e":
        extra["worker_extension_cls"] = "measurement_worker.MeasurementWorker"
    started = time.perf_counter()
    llm = LLM(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024 if config["phase"] == "acceptance" else 4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        async_scheduling=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config=spec,
        per_request_spec_decode_metrics=(
            "detailed" if config["phase"] == "acceptance" else "none"
        ),
        disable_log_stats=True,
        seed=0,
        **extra,
    )
    init_seconds = time.perf_counter() - started
    samples = [
        json.loads(line) for line in Path(config["dataset"]).read_text().splitlines()
    ]
    assert len(samples) == config["samples"]
    sampling = SamplingParams(temperature=0, max_tokens=512, ignore_eos=True)
    warmup_seconds = 0.0
    if config["phase"] != "acceptance":
        started = time.perf_counter()
        llm.generate([samples[0]["prompt"]], sampling, use_tqdm=False)
        warmup_seconds = time.perf_counter() - started
    print(
        "MEASUREMENT_START" if config["phase"] == "acceptance" else "WARMUP_COMPLETE",
        flush=True,
    )
    outputs, proposals, verification = [], [], []
    for sample in samples:
        assert (
            hashlib.sha256(sample["prompt"].encode()).hexdigest()
            == sample["prompt_sha256"]
        )
        if config["phase"] != "e2e":
            llm.collective_rpc("begin_measurement", args=(config["phase"],))
        started = time.perf_counter()
        output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        tokens = list(output.outputs[0].token_ids)
        assert len(tokens) == 512
        assert len(output.prompt_token_ids) == sample["prompt_token_count"]
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "prompt_sha256": sample["prompt_sha256"],
                "token_ids": tokens,
                "e2e_seconds": elapsed,
                "spec_decode_metrics": (
                    output.outputs[0].spec_decode_metrics.to_dict()
                    if output.outputs[0].spec_decode_metrics is not None
                    else None
                ),
            }
        )
        if config["phase"] != "e2e":
            measured = llm.collective_rpc("collect_measurement")[0]
            proposals.extend(
                {"sample_index": sample["sample_index"], **r}
                for r in measured["proposals"]
            )
            verification.extend(
                {
                    "sample_index": sample["sample_index"],
                    "accepted": count - 1,
                    "scheduled": scheduled,
                }
                for count, scheduled in zip(
                    measured["num_sampled"], measured["scheduled"], strict=True
                )
            )
        write_json(
            path.parent / "progress.json",
            {"completed": len(outputs), "expected": len(samples)},
        )
        print(
            f"SAMPLE_COMPLETE {len(outputs)}/{len(samples)} {elapsed:.3f}s", flush=True
        )
    result = {
        **config,
        "speculative_config": spec,
        "init_seconds": init_seconds,
        "warmup_seconds": warmup_seconds,
        "outputs": outputs,
        "proposals": proposals,
        "verification": verification,
    }
    write_json(path.parent / "result.json", result)
    (path.parent / "CELL_COMPLETE").write_text("All requested output counts verified\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["e2e", "timing", "acceptance"],
        default=["e2e", "timing", "acceptance"],
    )
    parser.add_argument("--acceptance-samples", type=int, default=128)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument(
        "--inner-methods",
        nargs="+",
        choices=["mtp", "dspark"],
        default=["mtp", "dspark"],
    )
    parser.add_argument(
        "--n-values", nargs="+", type=int, choices=[1, 2, 4, 8], default=[1, 2, 4, 8]
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cell:
        cell(args.cell)
        return
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    env = os.environ.copy()
    for key in (
        "VLLM_MOE_SKIP_TRACE_DIR",
        "VLLM_DRAFT_TOPK_TRACE_DIR",
        "VLLM_HIERARCHICAL_TRACE_DIR",
        "VLLM_HIERARCHICAL_CHECK_PREVERIFY",
        "CUDA_LAUNCH_BLOCKING",
        "VLLM_BATCH_INVARIANT",
    ):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.cuda_device),
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + str(ROOT)
    contract = {
        "inner_d": 4,
        "n_values": [1, 2, 4, 8],
        "nominal_budgets": [4, 8, 16, 32],
        "inner_methods": ["mtp", "dspark"],
        "phases": args.phases,
        "e2e_and_timing_samples": 16,
        "acceptance_samples": args.acceptance_samples,
        "output_length": 512,
        "phase_settings": {
            "e2e": {"max_num_batched_tokens": 4096, "warmup_requests": 1},
            "timing": {"max_num_batched_tokens": 4096, "warmup_requests": 1},
            "acceptance": {"max_num_batched_tokens": 1024, "warmup_requests": 0},
        },
        "gpu": args.cuda_device,
        "phase_devices": dict.fromkeys(args.phases, args.cuda_device),
        "baseline_e2e": str(PREVIOUS),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "known_strict_equivalence_gate": (
            "failed; performance measurement explicitly requested"
        ),
        "differences_from_previous": [
            "multimodal input limits are zero (text-only implementation)",
            "async scheduling explicitly disabled (required by implementation)",
            "prefix caching disabled; historical acceptance used automatic default",
            "acceptance records counts only; historical acceptance also traced logits",
        ],
    }
    contract_path = root / "contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        for key in (
            "inner_d",
            "n_values",
            "inner_methods",
            "acceptance_samples",
            "output_length",
            "source_commit",
        ):
            assert previous[key] == contract[key], (key, previous[key], contract[key])
        devices = previous.setdefault(
            "phase_devices", dict.fromkeys(previous["phases"], previous["gpu"])
        )
        for phase, device in contract["phase_devices"].items():
            old = devices.get(phase, device)
            used = sorted(set((old if isinstance(old, list) else [old]) + [device]))
            devices[phase] = used[0] if len(used) == 1 else used
        previous["phases"] = list(dict.fromkeys(previous["phases"] + args.phases))
        previous["phase_settings"] = contract["phase_settings"]
        previous["differences_from_previous"] = contract["differences_from_previous"]
        contract = previous
    write_json(contract_path, contract)
    for phase in args.phases:
        samples = args.acceptance_samples if phase == "acceptance" else 16
        dataset = root / f"samples_{samples}.jsonl"
        lines = DATASET.read_text().splitlines()[:samples]
        content = "\n".join(lines) + "\n"
        if dataset.exists():
            assert dataset.read_text() == content, dataset
        else:
            dataset.write_text(content)
        for method in args.inner_methods:
            for n in args.n_values:
                directory = root / phase / f"{method}_d4_n{n}"
                if (directory / "CELL_COMPLETE").exists():
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                config = {
                    "phase": phase,
                    "inner_method": method,
                    "d": 4,
                    "n": n,
                    "nominal_budget": 4 * n,
                    "capacity": 5 * n,
                    "cuda_device": args.cuda_device,
                    "samples": samples,
                    "dataset": str(dataset),
                }
                run_queued_cell(directory, config, env)


if __name__ == "__main__":
    main()
