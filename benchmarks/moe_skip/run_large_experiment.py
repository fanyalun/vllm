# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmark_integrity import preserve_contract, validate_target

DEFAULT_DRAFT_LENGTHS = (4, 8, 16, 32)
EXPECTED_CATEGORIES = {
    "human_eval": 32,
    "alpaca": 32,
    "gsm8k": 32,
    "ultra_feedback": 32,
}
NUM_SAMPLES = 128
MAX_TOKENS = 512
MAX_MODEL_LEN = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument(
        "--draft-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_DRAFT_LENGTHS,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=MAX_MODEL_LEN,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_contract(dataset: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != NUM_SAMPLES:
        raise ValueError(f"Expected {NUM_SAMPLES} dataset rows, found {len(rows)}")
    if [row["sample_index"] for row in rows] != list(range(NUM_SAMPLES)):
        raise ValueError("Dataset sample_index values must be contiguous from zero")
    category_counts = Counter(row["category"] for row in rows)
    if dict(category_counts) != EXPECTED_CATEGORIES:
        raise ValueError(f"Unexpected category counts: {dict(category_counts)}")
    dataset_sha256 = sha256_file(dataset)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Dataset SHA256 does not match its manifest")
    if max(row["prompt_token_count"] for row in rows) + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError("A prompt plus requested output exceeds MAX_MODEL_LEN")
    return {
        "dataset": str(dataset),
        "dataset_manifest": str(manifest_path),
        "dataset_sha256": dataset_sha256,
        "num_samples": NUM_SAMPLES,
        "category_counts": EXPECTED_CATEGORIES,
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "prompt_token_counts": manifest["prompt_token_counts"],
    }


def validate_cell(
    path: Path,
    dataset_sha256: str,
    draft_length: int,
    max_num_batched_tokens: int,
    model: str,
) -> None:
    cell = json.loads(path.read_text(encoding="utf-8"))
    validate_target(cell, model)
    expected = {
        "method": "moe_skip",
        "mode": "graph",
        "draft_length": draft_length,
        "top_h": 4,
        "num_samples": NUM_SAMPLES,
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": max_num_batched_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "batch_size": 1,
        "category_counts": EXPECTED_CATEGORIES,
        "prompt_format": "raw_text_no_chat_template",
    }
    for key, value in expected.items():
        if cell.get(key) != value:
            raise RuntimeError(
                f"{path}: expected {key}={value!r}, got {cell.get(key)!r}"
            )
    if sha256_file(Path(cell["dataset"])) != dataset_sha256:
        raise RuntimeError(f"{path}: dataset SHA256 changed")
    outputs = cell.get("outputs", [])
    if [output.get("sample_index") for output in outputs] != list(range(NUM_SAMPLES)):
        raise RuntimeError(f"{path}: sample indices are incomplete or reordered")
    if dict(Counter(output.get("category") for output in outputs)) != (
        EXPECTED_CATEGORIES
    ):
        raise RuntimeError(f"{path}: output categories do not match the contract")
    for output in outputs:
        if len(output.get("token_ids", [])) != MAX_TOKENS:
            raise RuntimeError(
                f"{path}: sample {output.get('sample_index')} is not "
                f"{MAX_TOKENS} tokens"
            )
        if output.get("spec_decode_metrics") is None:
            raise RuntimeError(f"{path}: missing speculative decoding metrics")


def main() -> None:
    args = parse_args()
    draft_lengths = tuple(args.draft_lengths)
    if not draft_lengths or any(length <= 0 for length in draft_lengths):
        raise ValueError("--draft-lengths must contain positive integers")
    if len(set(draft_lengths)) != len(draft_lengths):
        raise ValueError("--draft-lengths must not contain duplicates")
    if args.max_num_batched_tokens < MAX_MODEL_LEN:
        raise ValueError(
            f"--max-num-batched-tokens must be at least max_model_len={MAX_MODEL_LEN}"
        )
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset).resolve()
    dataset_manifest = Path(args.dataset_manifest).resolve()
    contract = load_dataset_contract(dataset, dataset_manifest)
    contract.update(
        {
            "model": str(Path(args.model).resolve()),
            "draft_lengths": list(draft_lengths),
            "top_h": 4,
            "target_top_k": 8,
            "mode": "graph",
            "tensor_parallel_size": 1,
            "batch_size": 1,
            "temperature": 0,
            "ignore_eos": True,
            "cuda_visible_devices": args.cuda_device,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
    )
    preserve_contract(run_dir, contract)

    script_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    commands_path = run_dir / "commands.json"
    prior_commands = (
        json.loads(commands_path.read_text(encoding="utf-8"))
        if args.resume and commands_path.exists()
        else []
    )
    commands = {entry["cell"]: entry for entry in prior_commands}

    num_cells = len(draft_lengths)
    for index, draft_length in enumerate(draft_lengths, start=1):
        name = f"moe_skip_top4_graph_d{draft_length}"
        cell_dir = run_dir / "cells" / name
        output_path = cell_dir / "cell_output.json"
        trace_path = cell_dir / "trace" / "raw_trace.jsonl"
        if args.resume and output_path.exists():
            validate_cell(
                output_path,
                contract["dataset_sha256"],
                draft_length,
                args.max_num_batched_tokens,
                contract["model"],
            )
            print(f"[{index}/{num_cells}] SKIP {name}", flush=True)
            continue
        if trace_path.exists() and trace_path.stat().st_size:
            raise RuntimeError(
                f"Refusing to append to incomplete trace: {trace_path}. "
                "Preserve this run and use a fresh --run-dir."
            )
        cell_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(script_dir / "run_cell.py"),
            "--model",
            args.model,
            "--dataset",
            str(dataset),
            "--output",
            str(output_path),
            "--mode",
            "graph",
            "--method",
            "moe_skip",
            "--draft-length",
            str(draft_length),
            "--top-h",
            "4",
            "--num-samples",
            str(NUM_SAMPLES),
            "--max-tokens",
            str(MAX_TOKENS),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--trace-dir",
            str(cell_dir / "trace"),
        ]
        commands[name] = {
            "cell": name,
            "cuda_visible_devices": args.cuda_device,
            "command": command,
        }
        commands_path.write_text(
            json.dumps(list(commands.values()), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{index}/{num_cells}] START {name}", flush=True)
        with (cell_dir / "run.log").open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            failure = {"cell": name, "returncode": result.returncode}
            (run_dir / "RUN_FAILED.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(f"Cell {name} failed; see {cell_dir / 'run.log'}")
        validate_cell(
            output_path,
            contract["dataset_sha256"],
            draft_length,
            args.max_num_batched_tokens,
            contract["model"],
        )
        print(f"[{index}/{num_cells}] DONE {name}", flush=True)

    subprocess.run(
        [
            sys.executable,
            str(script_dir / "analyze_large_experiment.py"),
            "--run-dir",
            str(run_dir),
            "--dataset-manifest",
            str(dataset_manifest),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
