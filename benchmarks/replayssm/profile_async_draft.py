# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the existing Llama EAGLE3 D4 workload after an untimed warmup."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("sync", "async_cache"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--port", type=int, default=54210)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=repo / "SSSD_results/vllm_llama_eagle3_d4_16x256_20260909/prompts.json",
    )
    parser.add_argument(
        "--target", default="/home/fanya/data1/fanya/models/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--draft",
        default="/home/fanya/data1/fanya/models/EAGLE3-LLaMA3.1-Instruct-8B",
    )
    options = parser.parse_args()
    root = options.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "ssd_matrix", Path(__file__).with_name("async_ssd_eagle3_matrix.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    prompt_source = options.prompts.resolve()

    prompts = json.loads(prompt_source.read_text())
    if not options.all_prompts:
        prompts = [prompts[index] for index in (0, 4, 8, 12)]
    sys.argv = [
        str(__file__),
        "--target",
        options.target,
        "--draft",
        options.draft,
        "--dataset-root",
        str(root),
        "--output-root",
        str(root),
        "--output-length",
        "256",
        "--num-speculative-tokens",
        "4",
        "--max-model-len",
        "512",
    ]
    args = module.parse_args()
    module.write_json(root / "prompts.json", prompts)
    module.prepare_manifest(args, root, prompts)
    module.write_json(
        root / "diagnostic_contract.json",
        {
            "mode": options.mode,
            "profile": options.profile,
            "prompts": len(prompts),
            "output_tokens_per_request": 256,
            "D": 4,
            "F": 3,
            "B": 1,
            "warmup_requests": 1,
            "prompt_source": str(prompt_source),
            "scope": "diagnostic; profiled timing is not benchmark throughput",
        },
    )
    original_command = module.server_command
    original_requests = module.run_requests

    def command(args, cell, port):
        argv = original_command(args, cell, port)
        if not options.profile:
            return argv
        argv += ["--profiler-config", '{"profiler":"cuda"}']
        return [
            "/usr/local/cuda-12.9/bin/nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--cuda-event-trace=false",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--trace-fork-before-exec=true",
            "--output",
            str(root / "timeline"),
            *argv,
        ]

    def warmup(args, port, cell, prompt):
        result = module.stream_request(
            port=port,
            request_id="warmup-" + cell.name,
            prompt_token_ids=prompt["token_ids"],
            max_tokens=256,
            logprobs=None,
        )
        assert len(result["token_ids"]) == 256
        if options.profile:
            response = module.requests.post(
                f"http://127.0.0.1:{port}/start_profile", timeout=60
            )
            response.raise_for_status()
        return {"requests": 1, "completion_tokens": 256}

    def run_requests(**kwargs):
        try:
            return original_requests(**kwargs)
        finally:
            if options.profile:
                response = module.requests.post(
                    f"http://127.0.0.1:{options.port}/stop_profile", timeout=60
                )
                response.raise_for_status()

    if options.profile:
        os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"
    module.server_command = command
    module.warmup_server = warmup
    module.run_requests = run_requests
    module.run_cell(
        args,
        root,
        prompts,
        module.Cell("performance", options.mode, "graph", 1),
        options.port,
    )


if __name__ == "__main__":
    main()
