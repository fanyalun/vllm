# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the matched Gemma D4/N4 distribution diagnostic on an idle GPU."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_token_importance import ASSISTANT, MODEL, ROOT, digest, write_json


def worker(path):
    from vllm import LLM, SamplingParams

    config = json.loads(path.read_text())
    samples = [json.loads(s) for s in Path(config["dataset"]).read_text().splitlines()]
    assert digest(Path(config["dataset"])) == config["dataset_sha256"]
    assert config["h"] == 4 and samples
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=False,
        async_scheduling=False,
        speculative_config={
            "method": "hierarchical",
            "model": ASSISTANT,
            "inner_method": "mtp",
            "inner_num_speculative_tokens": 4,
            "inner_num_rounds": 4,
            "num_speculative_tokens": 20,
            "moe_skip_top_h": config["h"],
            "draft_sample_method": "greedy",
        },
        per_request_spec_decode_metrics="detailed",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls=config.get(
            "worker_extension_cls", "disagreement_worker.DisagreementWorker"
        ),
    )
    params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, seed=0)
    for sample in samples:
        llm.generate([sample["prompt"]], params, use_tqdm=False)
    print("WARMUP_COMPLETE", flush=True)

    def generate(directory):
        outputs = []
        for sample in samples:
            result = llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
            assert len(result.outputs[0].token_ids) == 128
            assert len(result.prompt_token_ids) == sample["prompt_token_count"]
            outputs.append(
                {
                    "category": sample["category"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "token_ids": list(result.outputs[0].token_ids),
                    "spec_decode_metrics": result.outputs[
                        0
                    ].spec_decode_metrics.to_dict(),
                }
            )
            write_json(directory / "progress.json", outputs)
            print(
                f"{directory.name} SAMPLE_COMPLETE {len(outputs)}/{len(samples)}",
                flush=True,
            )
        return outputs

    if config.get("confidence_only"):
        control = path.parent.parent / "h4_control"
        control.mkdir(exist_ok=False)
        reference = generate(control)
        write_json(
            control / "result.json",
            {**config, "instrumented": False, "outputs": reference},
        )
        (control / "CELL_COMPLETE").write_text(f"{len(samples)}x128 uninstrumented\n")
    rpc = "begin_confidence" if config.get("confidence_only") else "begin_disagreement"
    llm.collective_rpc(rpc, args=(str(path.parent / "trace"),))
    outputs = generate(path.parent)
    write_json(path.parent / "result.json", {**config, "outputs": outputs})
    (path.parent / "CELL_COMPLETE").write_text(
        f"{len(samples)}x128; distribution audit pending\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--confidence-only", action="store_true")
    args = parser.parse_args()
    if args.cell:
        worker(args.cell)
        return
    root = args.root.resolve()
    root.mkdir(exist_ok=False)
    prior = ROOT / "benchmark_results/.sources/gemma_round_decay_4x128_20260915"
    dataset = args.dataset or prior / "dataset.jsonl"
    (root / "dataset.jsonl").write_bytes(dataset.read_bytes())
    hypotheses = dataset.with_name("hypotheses.json")
    if hypotheses.exists():
        (root / "hypotheses.json").write_bytes(hypotheses.read_bytes())
    num_samples = len(dataset.read_text().splitlines())
    sources = {}
    for name in (
        "run_disagreement.py",
        "disagreement_worker.py",
        "run_token_importance.py",
        "analyze_disagreement.py",
        "confidence_worker.py",
        "analyze_confidence.py",
    ):
        source = Path(__file__).parent / name
        (root / name).write_bytes(source.read_bytes())
        sources[name] = digest(source)
    runtime = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "vllm/v1/worker/gpu/spec_decode").rglob("*.py")
    }
    write_json(
        root / "contract.json",
        {
            "model": MODEL,
            "assistant": ASSISTANT,
            "model_config_sha256": digest(Path(MODEL) / "config.json"),
            "assistant_config_sha256": digest(Path(ASSISTANT) / "config.json"),
            "source_sha256": sources,
            "runtime_sha256": runtime,
            "dataset_sha256": digest(root / "dataset.jsonl"),
            "warmup_requests": num_samples * (2 if args.confidence_only else 1),
            "warmup_only_requests": num_samples,
            "in_process_control_requests": num_samples if args.confidence_only else 0,
            "measured_requests": num_samples,
            "confidence_only": args.confidence_only,
            "protocol": "Gemma B1 TP1 greedy MTP D4 N4; "
            f"{num_samples} warmup + {num_samples} measured x128; h4 and h4 control",
            "probability_temperature": 1,
            "near_definition": "mutual top2 and both opposing-top1 logit gaps <= 1 nat",
            "far_definition": "either opposing-top1 rank > 8 or logit gap > 2 nats",
            "middle_definition": "remaining disagreements; continuous metrics retained",
            "performance_claim": False,
        },
    )
    (root / "git_head.txt").write_bytes(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
    )
    (root / "gpu.txt").write_bytes(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv"]
        )
    )
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            (
                "VLLM_HIERARCHICAL",
                "VLLM_DRAFT_TOPK_TRACE",
                "VLLM_MOE_SKIP_TRACE",
                "PREVERIFY_EXPERT_POOL",
            )
        ):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        HIERARCHICAL_DISAGREEMENT="1",
        OMP_NUM_THREADS="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        PYTHONPATH=str(root)
        + os.pathsep
        + str(Path(__file__).parent.resolve())
        + os.pathsep
        + env.get("PYTHONPATH", ""),
    )
    cells = (
        [("h4", 4, True)]
        if args.confidence_only
        else [("h4_control", 4, False), ("h4", 4, True)]
    )
    for name, h, trace in cells:
        folder = root / name
        folder.mkdir()
        config = {
            "h": h,
            "instrumented": trace,
            "dataset": str(root / "dataset.jsonl"),
            "dataset_sha256": digest(root / "dataset.jsonl"),
            "confidence_only": args.confidence_only,
        }
        write_json(folder / "config.json", config)
        env["VLLM_HIERARCHICAL_TRACE_DIR"] = str(folder / "trace")
        env["HIERARCHICAL_DISAGREEMENT"] = (
            "1" if trace and not args.confidence_only else "0"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            str(folder / "config.json"),
        ]
        write_json(folder / "command.json", command)
        print(f"START {name}", flush=True)
        with (folder / "run.log").open("w") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"COMPLETE {name}", flush=True)
    (root / "MEASUREMENTS_COMPLETE").write_text(
        "h4 diagnostic and h4 control; analysis pending\n"
    )


if __name__ == "__main__":
    main()
