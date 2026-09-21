# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect per-round hierarchical traces without changing the measured worker."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_token_importance import ASSISTANT, MODEL, ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    previous = (
        ROOT / "benchmark_results/.sources/gemma_token_importance_4x128_20260915_run3"
    )
    (root / "dataset.jsonl").write_bytes((previous / "dataset.jsonl").read_bytes())
    sources = {}
    for name in (
        "run_round_decay.py",
        "run_token_importance.py",
        "token_importance_worker.py",
    ):
        path = Path(__file__).parent / name
        sources[name] = digest(path)
        (root / name).write_bytes(path.read_bytes())
    cells = [
        {"name": "h4", "h": 4, "pool": "none"},
        {"name": "h6", "h": 6, "pool": "none"},
        {"name": "h8", "h": 8, "pool": "none"},
        {"name": "routing60", "h": 8, "pool": "routing"},
        {"name": "attention60", "h": 8, "pool": "attention"},
    ]
    write_json(
        root / "contract.json",
        {
            "cells": cells,
            "source_sha256": sources,
            "dataset_sha256": digest(root / "dataset.jsonl"),
            "model": MODEL,
            "model_config_sha256": digest(Path(MODEL) / "config.json"),
            "assistant": ASSISTANT,
            "assistant_config_sha256": digest(Path(ASSISTANT) / "config.json"),
            "gpu": args.gpu,
            "protocol": "B1 greedy, 4 warmup + 4 measured requests x 128; "
            "MTP D4, four rounds, outer capacity20; trace timing is diagnostic only",
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
                "VLLM_MOE_SKIP_TRACE",
                "VLLM_DRAFT_TOPK_TRACE",
                "MOE_SKIP_BENCH",
            )
        ):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    )
    for cell in cells:
        folder = root / cell["name"]
        folder.mkdir()
        config = {
            **cell,
            "model_path": MODEL,
            "dataset": str(root / "dataset.jsonl"),
            "dataset_sha256": digest(root / "dataset.jsonl"),
            "source_sha256": sources,
        }
        write_json(folder / "config.json", config)
        command = [
            sys.executable,
            str(root / "run_token_importance.py"),
            "--cell",
            str(folder / "config.json"),
        ]
        write_json(folder / "command.json", command)
        env["VLLM_HIERARCHICAL_TRACE_DIR"] = str(folder / "trace")
        print("START", cell["name"], flush=True)
        with (folder / "run.log").open("w") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("COMPLETE", cell["name"], flush=True)
    (root / "MEASUREMENTS_COMPLETE").write_text("5 cells; per-round audit pending\n")


if __name__ == "__main__":
    main()
