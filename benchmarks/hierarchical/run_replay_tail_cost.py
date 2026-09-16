# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKERS = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(WORKERS) + os.pathsep + str(REPO)
os.environ["PATH"] = str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(
        description="Paired fixed-input pre-verifier timing"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ROOT = args.output
    ROOT.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams

    dataset = (
        REPO
        / "benchmark_results/sampling_acceptance_t1_p095_4x256_d16_d32_20260908"
        / "datasets/qwen36_first4.jsonl"
    )
    samples = [json.loads(line) for line in dataset.read_text().splitlines()][:3]
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
        seed=20260908,
        worker_extension_cls="replay_tail_cost_worker.ReplayTailCostWorker",
        speculative_config=dict(
            method="hierarchical",
            inner_method="mtp",
            inner_num_speculative_tokens=4,
            inner_num_rounds=1,
            moe_skip_top_h=4,
            draft_sample_method="greedy",
            preverify_gdn_mode="replay_tail",
        ),
    )
    (ROOT / "contract.json").write_text(
        json.dumps(
            {
                "config": config,
                "samples": samples,
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "repeats": 30,
                "warmup_replays": 5,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "fixed_inputs": True,
                "state_reset_outside_timing": True,
                "l2_flush_bytes": 128 * 1024 * 1024,
                "timing": "CUDA events around graph replay; no internal events",
                "gates": "per_token",
                "rounds": 3,
                "cases": ["baseline", "recurrent", "optimized"],
                "source_sha256": {
                    path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
                    for path in (
                        "vllm/model_executor/layers/mamba/gdn/replay_tail_update.py",
                        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                        "vllm/v1/worker/gpu/spec_decode/hierarchical/state.py",
                        "benchmarks/hierarchical/replay_tail_cost_worker.py",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    llm = LLM(**config)
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    llm.generate([samples[0]["prompt"]], params, use_tqdm=False)
    rows = []
    for trial, sample in enumerate(samples * 3):
        i = trial % len(samples)
        assert (
            hashlib.sha256(sample["prompt"].encode()).hexdigest()
            == sample["prompt_sha256"]
        )
        baseline = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
        llm.collective_rpc("begin_reuse")
        result = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
        measured = llm.collective_rpc("collect_reuse")[0]
        assert list(result.outputs[0].token_ids) == list(baseline.outputs[0].token_ids)
        assert len(measured["rows"]) == 3 * 4 * 30
        rows.append(
            dict(
                sample=i,
                trial=trial // 3,
                prompt_sha256=sample["prompt_sha256"],
                **measured,
            )
        )
        (ROOT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("COMPLETE", trial, flush=True)
    (ROOT / "measurement_complete.json").write_text(
        json.dumps(
            {"prefixes": 3, "rounds": 3, "rows": 3240, "control_outputs_equal": True}
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
