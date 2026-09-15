# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resume a fixed 52-cell matrix, running one fresh process per GPU/cell."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.replayssm.dspark_matrix_cell import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="1")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    gpus = [int(x) for x in args.gpus.split(",")]
    cells = []
    for batch in (1, 4, 8, 16):
        cells.append(dict(method="ar", policy="ar", batch=batch))
        for policy in ("d4", "d8", "p08", "p06"):
            for method in ("sd", "replayssm", "dual"):
                cells.append(dict(method=method, policy=policy, batch=batch))
    for cell in cells:
        cell["name"] = f"{cell['method']}_{cell['policy']}_b{cell['batch']}"
    sources = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], text=True
    ).splitlines()
    sources += [str(p) for p in Path("benchmarks/replayssm").glob("dspark_matrix_*.py")]
    hashes = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in sorted(set(sources))
        if Path(p).is_file() and p.startswith(("vllm/", "tests/", "benchmarks/"))
    }
    contract = dict(
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        sources=hashes,
        cells=cells,
        repeats=1,
        warmup=dict(first_cohort_tokens=32, remaining_prompt_tokens=1),
        gpu_ids=gpus,
        samples=16,
        output_tokens=256,
        threshold_rule=(
            "longest contiguous prefix with every confidence >= p, "
            "max 8, zero permitted"
        ),
        async_scheduling=False,
        atomic_cohort_admission=True,
        buffer_setting=16,
        kv_cache_gib=10,
        prompts_sha256=hashlib.sha256((root / "prompts.json").read_bytes()).hexdigest(),
    )
    path = root / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == contract, (
            "Matrix source/contract changed"
        )
    else:
        write_json(path, contract)
    pending = []
    completed = []
    for c in cells:
        p = root / "cells" / c["name"] / "result.json"
        if p.exists():
            r = json.loads(p.read_text())
            if (
                r.get("complete")
                and r.get("jit_clean")
                and len(r.get("repeats", [])) == 1
                and r.get("cohort_admission_exact")
            ):
                completed.append(c["name"])
                continue
        pending.append(c)
    active = {}
    failures = []
    while pending or active:
        for gpu, item in list(active.items()):
            process, cell, log = item
            if process.poll() is None:
                continue
            log.close()
            p = root / "cells" / cell["name"] / "result.json"
            result = json.loads(p.read_text()) if p.exists() else {}
            ok = (
                process.returncode == 0
                and result.get("complete")
                and result.get("jit_clean")
                and len(result.get("repeats", [])) == 1
                and result.get("cohort_admission_exact")
            )
            if ok:
                completed.append(cell["name"])
            else:
                failures.append(
                    dict(
                        cell=cell["name"],
                        exit_code=process.returncode,
                        complete=result.get("complete"),
                        jit_clean=result.get("jit_clean"),
                        repeat_tokens_equal=result.get("repeat_tokens_equal"),
                        cohort_admission_exact=result.get("cohort_admission_exact"),
                    )
                )
            del active[gpu]
            print("DONE" if ok else "FAILED", cell["name"], flush=True)
        if not failures:
            memory = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            free = {
                int(line.split(",")[0]): int(line.split(",")[1]) < 500
                for line in memory.splitlines()
            }
            for gpu in gpus:
                if gpu in active or not pending or not free[gpu]:
                    continue
                cell = pending.pop(0)
                directory = root / "cells" / cell["name"]
                directory.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "benchmarks/replayssm/dspark_matrix_cell.py",
                    "--output",
                    str(directory),
                    "--prompts",
                    str(root / "prompts.json"),
                    "--method",
                    cell["method"],
                    "--policy",
                    cell["policy"],
                    "--batch",
                    str(cell["batch"]),
                ]
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    VLLM_USE_V2_MODEL_RUNNER="1",
                    PYTHONPATH=str(Path.cwd()),
                )
                env["PATH"] = (
                    str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
                )
                write_json(
                    directory / "launch.json",
                    dict(
                        command=cmd,
                        gpu=gpu,
                        started_unix=time.time(),
                        source_sha256=hashes,
                    ),
                )
                log = (directory / "run.log").open("w")
                process = subprocess.Popen(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                active[gpu] = (process, cell, log)
                print("START", cell["name"], "GPU", gpu, flush=True)
        write_json(
            root / "status.json",
            dict(
                completed=completed,
                active={str(g): v[1]["name"] for g, v in active.items()},
                pending=[c["name"] for c in pending],
                failures=failures,
                updated_unix=time.time(),
            ),
        )
        if failures and not active:
            raise RuntimeError(f"Matrix stopped on failed gates: {failures}")
        time.sleep(2)
    write_json(
        root / "measurement_complete.json", dict(cells=len(completed), names=completed)
    )


if __name__ == "__main__":
    main()
