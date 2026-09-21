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
    args = parser.parse_args()
    if args.resume_after_fix and not args.resume:
        parser.error("--resume-after-fix requires --resume")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    cells = [("ar", "native")]
    cells += [
        (method, policy)
        for method in ("moe_skip", "hierarchical")
        for policy in ("native", "batch_top_half", "batch_max_gap")
    ]
    source_paths = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--diff-filter=ACM"], text=True
    ).splitlines() + [
        "vllm/model_executor/layers/fused_moe/router/batch_expert_selection.py",
        "benchmarks/kernels/moe_batch_policy_reference.py",
        "benchmarks/moe_skip/run_batch_policy_matrix.py",
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
            "1",
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
