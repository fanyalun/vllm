# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run paired AR, native-route and batch-policy generation smoke cells."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-after-fix", action="store_true")
    parser.add_argument("--inner-rounds", type=int, default=1)
    parser.add_argument("--hierarchical-only", action="store_true")
    parser.add_argument("--include-top1", action="store_true")
    parser.add_argument("--policy-family", choices=("half", "max_gap"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    args = parser.parse_args()
    if args.resume_after_fix and not args.resume:
        parser.error("--resume-after-fix requires --resume")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    cells = [("ar", "native")]
    methods = (
        ("hierarchical",) if args.hierarchical_only else ("moe_skip", "hierarchical")
    )
    policies = ["native", "batch_top_half", "batch_max_gap"]
    if args.include_top1:
        policies += ["batch_top_half_top1", "batch_max_gap_top1"]
    if args.policy_family:
        prefix = "batch_top_half" if args.policy_family == "half" else "batch_max_gap"
        policies = [p for p in policies if p == "native" or p.startswith(prefix)]
    cells += [(method, policy) for method in methods for policy in policies]
    source_paths = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--diff-filter=ACM"], text=True
    ).splitlines() + [
        "vllm/model_executor/layers/fused_moe/router/batch_expert_selection.py",
        "benchmarks/kernels/moe_batch_policy_reference.py",
        "benchmarks/moe_skip/run_batch_policy_matrix.py",
        "benchmarks/hierarchical/run_cell.py",
        "vllm/config/speculative.py",
    ]
    source_hashes = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths
    }
    manifest = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources=source_hashes,
        device=args.device,
        batch_size=args.batch_size,
        cells=cells,
        inner_rounds=args.inner_rounds,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
    )
    request_path = args.output / "request.json"
    if args.resume and request_path.exists():
        previous = json.loads(request_path.read_text())
        if previous["sources"] != manifest["sources"]:
            assert args.resume_after_fix, "Source changed; use --resume-after-fix"
            manifest["previous_run"] = previous
            manifest["source_changes"] = sorted(
                p
                for p in set(previous["sources"]) | set(manifest["sources"])
                if previous["sources"].get(p) != manifest["sources"].get(p)
            )
        assert previous["device"] == args.device
        assert previous["batch_size"] == args.batch_size
        assert previous["cells"] == [list(cell) for cell in cells]
        assert previous.get("inner_rounds", 1) == args.inner_rounds
        assert previous.get("cpu_offload_gb", 0) == args.cpu_offload_gb
        assert (
            previous.get("gpu_memory_utilization", 0.95) == args.gpu_memory_utilization
        )
    request_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    ar_path = args.output / "ar_native.json"
    for method, policy in cells:
        name = f"{method}_{policy}"
        output = args.output / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.hierarchical.run_cell",
            "--method",
            method,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-tokens",
            "128",
            "--num-samples",
            "4",
            "--inner-rounds",
            str(args.inner_rounds),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--cpu-offload-gb",
            str(args.cpu_offload_gb),
            "--output",
            str(output),
        ]
        if method == "moe_skip":
            command += ["--draft-tokens", "4"]
        if method != "ar":
            command += ["--ar-reference", str(ar_path)]
        if policy != "native":
            command += ["--batch-policy", policy]
        else:
            command += ["--top-h", "8"]
        complete = (
            json.loads(output.read_text()).get("complete", False)
            if output.exists()
            else False
        )
        if not (args.resume and complete):
            print(f"START {name} B{args.batch_size}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device)
            log_path = args.output / f"{name}.log"
            if args.resume and log_path.exists():
                attempt = len(list(args.output.glob(f"{name}.failed_*.log")))
                log_path.rename(args.output / f"{name}.failed_{attempt}.log")
            with log_path.open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        result = json.loads(output.read_text())
        assert result["complete"] and len(result["outputs"]) == 4
        assert all(len(row["token_ids"]) == 128 for row in result["outputs"])
        assert result["args"]["batch_size"] == args.batch_size
        summaries.append(
            dict(
                method=method,
                policy=policy,
                reused=args.resume and complete,
                throughput=result["returned_token_throughput"],
                acceptance=result.get("acceptance"),
                ar_parity=result.get("ar_parity"),
                result_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            )
        )
        print(f"COMPLETE {name} {summaries[-1]}", flush=True)
    (args.output / "matrix_complete.json").write_text(
        json.dumps(dict(manifest=manifest, results=summaries), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
