# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed Qwen DSpark D=3/F=3/B=1 branch-backbone validation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from async_ssd_eagle3_matrix import Cell, run_cell, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["ar", "sync", "async_jit", "async_cache"], required=True
    )
    parser.add_argument("--engine", choices=["eager", "graph"], default="eager")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--correctness-audit", type=Path)
    parser.add_argument("--port", type=int, default=46700)
    options = parser.parse_args()
    if options.performance:
        if options.correctness_audit is None:
            raise ValueError("Performance requires a passed strict correctness audit")
        if json.loads(options.correctness_audit.read_text())["status"] != "passed":
            raise ValueError("Strict correctness failed; performance remains gated")
    root = options.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(options.prompts.read_text())
    assert len(prompts) == 16
    if not options.performance:
        prompts = [prompts[i] for i in [0, 4, 8, 12]]
    write_json(root / "prompts.json", prompts)
    write_json(
        root / "prompt_fingerprint.json",
        {
            "source": str(options.prompts.resolve()),
            "source_sha256": hashlib.sha256(options.prompts.read_bytes()).hexdigest(),
            "ordered_prompts": [
                {k: p[k] for k in ("prompt_index", "dataset", "token_sha256") if k in p}
                for p in prompts
            ],
        },
    )
    os.environ["ASYNC_DRAFT_DSPARK_FAN_OUT"] = "3"
    for name in (
        "ASYNC_DRAFT_FORCE_JIT",
        "ASYNC_DRAFT_DSPARK_SHADOW_PATH",
        "REPLAYSSM_SPEC_DECODE_TRACE_LOGITS",
    ):
        os.environ.pop(name, None)
    args = SimpleNamespace(
        target="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        draft="/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
        method="dspark",
        dtype="bfloat16",
        target_device=0,
        draft_device=1,
        target_tensor_parallel_size=1,
        draft_tensor_parallel_size=1,
        attention_backend=None,
        gdn_recurrent_reference=False,
        qwen_gdn_mode="replayssm",
        replayssm_buffer_len=16,
        output_length=128,
        num_speculative_tokens=3,
        max_model_len=512,
        gpu_memory_utilization=0.9,
        warmup_seconds=30,
        startup_timeout=900,
        resume=True,
        trace_async_timing=options.performance,
        diagnostic_shadow=options.shadow,
    )
    write_json(root / "run_contract.json", vars(args))
    cell = Cell(
        "performance" if options.performance else "correctness",
        options.mode,
        options.engine,
        1,
    )
    if options.shadow:
        if options.performance or options.mode != "async_cache":
            raise ValueError("Shadow JIT is restricted to separate cache diagnostics")
        os.environ["ASYNC_DRAFT_DSPARK_SHADOW_PATH"] = str(root / "shadow.jsonl")
    if options.performance and options.mode != "ar":
        os.environ["REPLAYSSM_SPEC_DECODE_TRACE_PATH"] = str(
            root / "cells" / cell.name / "proposals.jsonl"
        )
    cell_dir = root / "cells" / cell.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        cell_dir / "source_fingerprint.json",
        {
            "cwd": str(Path.cwd()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "python": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "vllm", "transformers", "triton"]
            },
            "vllm_source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("vllm").rglob("*.py"))
            },
        },
    )
    run_cell(args, root, prompts, cell, options.port)


if __name__ == "__main__":
    main()
