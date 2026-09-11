# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, fixed C4 continuation matrix for long-context speculative acceptance."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/data1/fanya/Qwen/Qwen3.6-35B-A3B"
SOURCE = Path(
    "/home/fanya/data1/fanya/hf_datasets_cache/processed_datasets/"
    "c4/c4_data_10000.jsonl"
)
LENGTHS = (16384, 32768)
WIDTHS = (4, 8, 16, 32)
SEED = 20260911


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(data):
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def validate(metrics, width):
    accepted = metrics["per_step_accepted"]
    drafted = metrics["per_step_drafted"]
    histogram = metrics["acceptance_histogram"]
    steps = metrics["num_spec_steps"]
    assert steps > 0 and len(accepted) == len(drafted) == steps
    assert metrics["num_spec_tokens"] == width
    assert len(histogram) == width + 1 and sum(histogram) == steps
    assert all(0 <= a <= d <= width for a, d in zip(accepted, drafted, strict=True))
    assert sum(accepted) == metrics["num_accepted_draft_tokens"]
    assert sum(drafted) == metrics["num_draft_tokens"]
    assert sum(i * n for i, n in enumerate(histogram)) == sum(accepted)
    assert histogram == [accepted.count(i) for i in range(width + 1)]
    assert abs(metrics["mean_acceptance_length"] - 1 - sum(accepted) / steps) < 1e-12
    assert abs(metrics["draft_acceptance_rate"] - sum(accepted) / sum(drafted)) < 1e-12


def prepare(run):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    documents = [json.loads(line)["text"] for line in SOURCE.read_text().splitlines()]
    cursor = 0
    samples = []
    for index in range(4):
        tokens = []
        first = cursor
        while len(tokens) < max(LENGTHS):
            tokens.extend(
                tokenizer.encode(documents[cursor] + "\n\n", add_special_tokens=False)
            )
            cursor += 1
        full = tokens[: max(LENGTHS)]
        for length in LENGTHS:
            ids = full[-length:]
            samples.append(
                {
                    "sample_index": index,
                    "context_tokens": length,
                    "source_document_range": [first, cursor],
                    "prompt_token_ids": ids,
                    "prompt_sha256": digest(ids),
                }
            )
    write_json(run / "dataset.json", samples)
    contract = {
        "model": MODEL,
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "dataset_sha256": digest(samples),
        "context_lengths": LENGTHS,
        "draft_lengths": WIDTHS,
        "methods": ["mtp", "moe_skip"],
        "samples_per_cell": 4,
        "max_tokens": 256,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": SEED,
        "moe_skip_top_h": 4,
        "max_model_len": 33792,
        "batch_size": 1,
        "tensor_parallel_size": 1,
        "prefix_caching": False,
        "cuda_graph": True,
        "dataset_design": (
            "Disjoint concatenated C4 blocks; 16K is the suffix of 32K. "
            "Raw token continuation, no chat template."
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
    }
    write_json(run / "contract.json", contract)


def worker(run, method, width):
    from vllm import LLM, SamplingParams

    spec = {"method": method, "num_speculative_tokens": width}
    if method == "moe_skip":
        spec["moe_skip_top_h"] = 4
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=33792,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed",
        disable_log_stats=True,
        seed=SEED,
    )
    samples = json.loads((run / "dataset.json").read_text())
    for length in LENGTHS:
        cell = run / "cells" / f"{method}_d{width}_c{length}"
        cell.mkdir(parents=True, exist_ok=True)
        outputs = []
        for sample in samples:
            if sample["context_tokens"] != length:
                continue
            sampling = SamplingParams(
                temperature=1.0,
                top_p=0.95,
                max_tokens=256,
                ignore_eos=True,
                seed=SEED + sample["sample_index"],
            )
            result = llm.generate(
                [{"prompt_token_ids": sample["prompt_token_ids"]}],
                sampling,
                use_tqdm=False,
            )[0]
            output = result.outputs[0]
            assert result.prompt_token_ids == sample["prompt_token_ids"]
            assert len(output.token_ids) == 256
            metrics = output.spec_decode_metrics.to_dict()
            validate(metrics, width)
            outputs.append(
                {
                    "sample_index": sample["sample_index"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "prompt_tokens": len(result.prompt_token_ids),
                    "seed": SEED + sample["sample_index"],
                    "token_ids": list(output.token_ids),
                    "metrics": metrics,
                }
            )
            write_json(cell / "progress.json", outputs)
            print(
                f"SAMPLE_COMPLETE {method} D={width} C={length} {len(outputs)}/4",
                flush=True,
            )
        write_json(cell / "result.json", outputs)
        (cell / "CELL_COMPLETE").write_text("4 x 256 tokens; counters checked\n")


def summarize(run):
    samples = json.loads((run / "dataset.json").read_text())
    contract = json.loads((run / "contract.json").read_text())
    assert digest(samples) == contract["dataset_sha256"]
    rows, requests, hashes = [], [], {}
    for length in LENGTHS:
        expected = [s for s in samples if s["context_tokens"] == length]
        for method in ("mtp", "moe_skip"):
            for width in WIDTHS:
                cell = run / "cells" / f"{method}_d{width}_c{length}"
                assert (cell / "CELL_COMPLETE").exists(), cell
                outputs = json.loads((cell / "result.json").read_text())
                assert len(outputs) == 4
                steps = accepted = drafted = 0
                for sample, output in zip(expected, outputs, strict=True):
                    assert output["prompt_sha256"] == digest(sample["prompt_token_ids"])
                    assert output["sample_index"] == sample["sample_index"]
                    assert output["seed"] == SEED + sample["sample_index"]
                    assert output["prompt_tokens"] == length
                    assert len(output["token_ids"]) == 256
                    m = output["metrics"]
                    validate(m, width)
                    steps += m["num_spec_steps"]
                    accepted += m["num_accepted_draft_tokens"]
                    drafted += m["num_draft_tokens"]
                    requests.append(
                        {
                            "context_tokens": length,
                            "method": method,
                            "d": width,
                            "sample_index": output["sample_index"],
                            "prompt_sha256": output["prompt_sha256"],
                            "mean_acceptance_length": m["mean_acceptance_length"],
                            "verify_steps": m["num_spec_steps"],
                            "accepted": m["num_accepted_draft_tokens"],
                            "drafted": m["num_draft_tokens"],
                        }
                    )
                rows.append(
                    {
                        "context_tokens": length,
                        "method": method,
                        "d": width,
                        "requests": 4,
                        "output_tokens": 1024,
                        "verify_steps": steps,
                        "accepted_draft_tokens": accepted,
                        "drafted_tokens": drafted,
                        "mean_accepted_draft_tokens": accepted / steps,
                        "mean_acceptance_length": 1 + accepted / steps,
                        "draft_acceptance_rate": accepted / drafted,
                        "zero_acceptance_fraction": sum(
                            o["metrics"]["acceptance_histogram"][0] for o in outputs
                        )
                        / steps,
                        "actual_output_tokens_per_spec_step": 1024 / steps,
                    }
                )
                hashes[cell.name] = digest(outputs)
    for name, values in (
        ("acceptance_summary", rows),
        ("acceptance_by_request", requests),
    ):
        with (run / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_json(
        run / "audit.json",
        {
            "status": "passed",
            "cells": len(rows),
            "requests": len(requests),
            "output_tokens": len(requests) * 256,
            "result_hashes": hashes,
            "scope": (
                "Prompt identity, output length, seed and integer acceptance "
                "accounting; no distributional correctness or AR parity claim."
            ),
        },
    )
    (run / "RUN_COMPLETE").write_text(
        "16 cells, 64 requests, 16384 output tokens audited\n"
    )


def lane(run, method, gpu):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    for key in ("VLLM_MOE_SKIP_TRACE_DIR", "VLLM_DRAFT_TOPK_TRACE_DIR"):
        env.pop(key, None)
    for width in WIDTHS:
        if all(
            (run / "cells" / f"{method}_d{width}_c{n}" / "CELL_COMPLETE").exists()
            for n in LENGTHS
        ):
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(run),
            "--worker",
            method,
            "--width",
            str(width),
        ]
        with (run / f"{method}_d{width}.log").open("a") as log:
            log.write("COMMAND " + json.dumps(command) + "\n")
            log.flush()
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("mtp", "moe_skip"))
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(run, args.worker, args.width)
    elif args.summarize:
        summarize(run)
    else:
        if not (run / "contract.json").exists():
            prepare(run)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(lane, run, method, gpu)
                for gpu, method in enumerate(("mtp", "moe_skip"))
            ]
            for job in jobs:
                job.result()
        summarize(run)


if __name__ == "__main__":
    main()
