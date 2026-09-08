# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP B=1 eager Sync/Async validation with a fixed prompt manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from async_ssd_eagle3_matrix import Cell, run_cell, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["ar", "sync", "async_jit", "async_cache"], required=True
    )
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--draft-length", type=int, default=6)
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--fan-out", type=int, default=3)
    parser.add_argument("--port", type=int, default=46740)
    options = parser.parse_args()
    root = options.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(options.prompts.read_text())
    if len(prompts) != 4:
        raise ValueError("This smoke requires exactly four tokenized prompts")
    args = SimpleNamespace(
        target=options.target,
        draft=options.draft,
        method="mtp",
        dtype="bfloat16",
        target_device=0,
        draft_device=1,
        target_tensor_parallel_size=1,
        draft_tensor_parallel_size=1,
        attention_backend="TRITON_ATTN",
        gdn_recurrent_reference=False,
        output_length=options.output_length,
        num_speculative_tokens=options.draft_length,
        max_model_len=512,
        gpu_memory_utilization=0.9,
        warmup_seconds=30 if options.performance else 10,
        startup_timeout=900,
        resume=True,
        trace_async_timing=True,
    )
    os.environ["ASYNC_DRAFT_MTP_FAN_OUT"] = str(options.fan_out)
    for name in ("ASYNC_DRAFT_FORCE_JIT", "REPLAYSSM_SPEC_DECODE_TRACE_LOGITS"):
        os.environ.pop(name, None)
    cell = Cell(
        "performance" if options.performance else "correctness",
        options.mode,
        "eager",
        1,
    )
    os.environ["REPLAYSSM_SPEC_DECODE_TRACE_PATH"] = str(
        root / "cells" / cell.name / "proposals.jsonl"
    )
    write_json(root / "prompts.json", prompts)
    write_json(
        root / f"{cell.name}_contract.json",
        {
            **vars(args),
            "fan_out": options.fan_out,
            "prompt_manifest_sha256": hashlib.sha256(
                options.prompts.read_bytes()
            ).hexdigest(),
            "cuda_launch_blocking": os.getenv("CUDA_LAUNCH_BLOCKING"),
        },
    )
    run_cell(args, root, prompts, cell, options.port)


if __name__ == "__main__":
    main()
