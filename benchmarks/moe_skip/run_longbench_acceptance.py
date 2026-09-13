# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed LongBench greedy acceptance matrix with an AR output reference."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_long_context_acceptance import MODEL, WIDTHS, digest, validate, write_json

DATASET = "sfc-gh-goliaro/longbench-longctx"
REVISION = "8e44d485957d753310a4af8096491d16c3f37ba2"
BUCKETS = (16384, 32768)
SEED = 20260913


def prepare(run, source):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = pq.read_table(source).to_pylist()
    samples = []
    selected = [(i, r) for i, r in enumerate(rows) if r["bucket"] >= 32768]
    assert len(selected) == 16
    for sample_index, (row_index, row) in enumerate(selected):
        document_with_header, question = row["user"].rsplit("\n\nQuestion:", 1)
        instruction = (
            "You are given a long document followed by a multiple-choice question. "
            "Read the document carefully and answer.\n\n"
        )
        assert document_with_header.startswith(instruction)
        document = document_with_header[len(instruction) :]
        marker = "LONG_BENCH_DOCUMENT_PLACEHOLDER"
        shell = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": instruction + marker + "\n\nQuestion:" + question,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        before, after = shell.split(marker)
        prefix_ids = tokenizer.encode(before, add_special_tokens=False)
        suffix_ids = tokenizer.encode(after, add_special_tokens=False)
        doc_ids = tokenizer.encode(document, add_special_tokens=False)
        for bucket in BUCKETS:
            document_budget = bucket - len(prefix_ids) - len(suffix_ids)
            assert 0 < document_budget <= len(doc_ids)
            ids = prefix_ids + doc_ids[:document_budget] + suffix_ids
            assert len(ids) == bucket
            prompt = tokenizer.decode(ids, skip_special_tokens=False)
            samples.append(
                {
                    "sample_index": sample_index,
                    "source_row_index": row_index,
                    "bucket": bucket,
                    "source_bucket": row["bucket"],
                    "ref_prompt_tokens": row["ref_prompt_tokens"],
                    "source_output_len": row["output_len"],
                    "user": row["user"],
                    "prompt": prompt,
                    "prompt_token_ids": ids,
                    "prompt_sha256": digest(ids),
                    "prefix_token_ids": prefix_ids,
                    "suffix_token_ids": suffix_ids,
                    "document_tokens_kept": document_budget,
                    "source_document_tokens": len(doc_ids),
                }
            )
    write_json(run / "dataset.json", samples)
    shutil.copyfile(source, run / "source.parquet")
    generation = json.loads((Path(MODEL) / "generation_config.json").read_text())
    eos = generation.get("eos_token_id", tokenizer.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    max_length = max(len(s["prompt_token_ids"]) for s in samples)
    contract = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": "train",
        "selection": "All 16 source rows with nominal bucket >= 32K, source order",
        "dataset_digest": digest(samples),
        "model": MODEL,
        "methods": ["mtp", "moe_skip"],
        "draft_lengths": list(WIDTHS),
        "buckets": list(BUCKETS),
        "requests_per_cell": 32,
        "samples_per_context_length": 16,
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": SEED,
        "ignore_eos": True,
        "eos_token_ids": eos,
        "moe_skip_top_h": 4,
        "max_model_len": ((max_length + 512 + max(WIDTHS) + 1023) // 1024) * 1024,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.97,
        "batch_size": 1,
        "tensor_parallel_size": 1,
        "prefix_caching": False,
        "cuda_graph": True,
        "chat_template": "Model default, add_generation_prompt=True; no override",
        "thinking": "Default template emits an opening think tag",
        "truncation": (
            "Token-prefix truncate only document to exact 16K/32K total prompt; "
            "instruction, question, choices and chat suffix remain identical"
        ),
        "gpu_assignment": {"ar": 0, "mtp": 0, "moe_skip": 1},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_tracked_diff": subprocess.check_output(["git", "diff"], text=True),
    }
    write_json(run / "contract.json", contract)
    write_json(
        run / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "triton", "transformers", "vllm", "pyarrow")
            },
            "runner": "V2",
            "multimodal_limits": {"image": 1, "video": 0},
        },
    )
    print(
        f"PREPARED 16 paired samples, 32 requests; prompt range "
        f"{min(len(s['prompt_token_ids']) for s in samples)}..{max_length}; "
        f"max_model_len={contract['max_model_len']}",
        flush=True,
    )


def worker(run, method, width):
    from vllm import LLM, SamplingParams

    contract = json.loads((run / "contract.json").read_text())
    samples = json.loads((run / "dataset.json").read_text())
    assert digest(samples) == contract["dataset_digest"]
    spec = None
    if method != "ar":
        spec = {"method": method, "num_speculative_tokens": width}
        if method == "moe_skip":
            spec["moe_skip_top_h"] = contract["moe_skip_top_h"]
    llm = LLM(
        model=contract["model"],
        tensor_parallel_size=1,
        max_model_len=contract["max_model_len"],
        max_num_seqs=1,
        max_num_batched_tokens=contract["max_num_batched_tokens"],
        gpu_memory_utilization=contract["gpu_memory_utilization"],
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=False,
        speculative_config=spec,
        per_request_spec_decode_metrics="detailed" if spec else "none",
        disable_log_stats=True,
        seed=SEED,
    )
    cell = run / "cells" / f"{method}_d{width}"
    cell.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sample in samples:
        sampling = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=512,
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
        ids = list(output.token_ids)
        assert len(ids) == 512
        metrics = None
        if method != "ar":
            metrics = output.spec_decode_metrics.to_dict()
            validate(metrics, width)
        first_eos = next(
            (i for i, token in enumerate(ids) if token in contract["eos_token_ids"]),
            None,
        )
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "source_row_index": sample["source_row_index"],
                "bucket": sample["bucket"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_tokens": len(result.prompt_token_ids),
                "seed": SEED + sample["sample_index"],
                "token_ids": ids,
                "text": output.text,
                "first_eos_index": first_eos,
                "metrics": metrics,
            }
        )
        write_json(cell / "progress.json", outputs)
        al = metrics["mean_acceptance_length"] if metrics else None
        print(
            f"SAMPLE_COMPLETE {method} D={width} {len(outputs)}/32 "
            f"row={sample['source_row_index']} tokens={len(result.prompt_token_ids)} "
            f"AL={al} first_eos={first_eos}",
            flush=True,
        )
    write_json(cell / "result.json", outputs)
    (cell / "CELL_COMPLETE").write_text("32 requests x 512 tokens checked\n")


def run_lane(run, method, gpu):
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
    jobs = [(method, width) for width in WIDTHS]
    if method == "mtp":
        jobs.insert(0, ("ar", 0))
    for name, width in jobs:
        cell = run / "cells" / f"{name}_d{width}"
        if (cell / "CELL_COMPLETE").exists():
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(run),
            "--worker",
            name,
            "--width",
            str(width),
        ]
        with (run / f"{name}_d{width}.log").open("a") as log:
            log.write("COMMAND " + json.dumps(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if completed.returncode:
            write_json(
                run / f"{name}_d{width}_failed.json",
                {"command": command, "returncode": completed.returncode},
            )
            raise RuntimeError(f"Worker failed: {name} D={width}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", choices=("ar", "mtp", "moe_skip"))
    parser.add_argument("--width", type=int, choices=(0, *WIDTHS))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(run, args.worker, args.width)
        return
    if not (run / "contract.json").exists():
        if args.source is None:
            parser.error("--source is required to prepare a new dataset")
        prepare(run, args.source)
    if args.prepare_only:
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(run_lane, run, method, gpu)
            for gpu, method in enumerate(("mtp", "moe_skip"))
        ]
        for job in jobs:
            job.result()
    (run / "GENERATION_COMPLETE").write_text(
        "8 speculative cells and one AR reference generated\n"
    )


if __name__ == "__main__":
    main()
