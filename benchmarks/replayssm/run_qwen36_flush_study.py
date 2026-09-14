# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resume independent single-GPU stages of the flush crossover study."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from qwen36_flush_crossover import save


def launch(args, command, destination, marker):
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        OMP_NUM_THREADS="8",
        PATH=str(Path(sys.executable).parent) + ":" + env["PATH"],
    )
    script = Path(command[1])
    source = script.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    archive = Path(args.output) / "source" / f"{script.stem}_{digest}.py.txt"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(source)
    save(
        destination / "launch.json",
        dict(
            command=command,
            gpu=args.gpu,
            script_sha256=digest,
            source_archive=str(archive),
            runtime_diff=subprocess.check_output(
                ["git", "diff", "--", "vllm"], text=True
            ),
        ),
    )
    print("START", destination, flush=True)
    with (destination / "run.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        save(destination / "failed.json", dict(exit_code=result.returncode))
        raise SystemExit(result.returncode)
    if not marker.exists():
        raise RuntimeError(f"Missing completion marker: {marker}")
    print("DONE", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument("--candidates")
    parser.add_argument(
        "--stage",
        choices=["default", "references", "distance", "policy", "sensitivity", "tuned"],
        required=True,
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    here = Path(__file__).resolve().parent
    for batch in map(int, args.batches.split(",")):
        if args.stage in ("default", "references", "tuned"):
            cells = [(0, "ar")] if args.stage == "references" else []
            mode = "standard" if args.stage == "references" else "replayssm"
            cells += [(draft, mode) for draft in (4, 8, 16, 32)]
            for draft, mode in cells:
                suffix = "tuned" if args.stage == "tuned" else mode
                dest = root / f"b{batch}_d{draft}_{suffix}"
                command = [
                    sys.executable,
                    str(here / "qwen36_a100_matrix.py"),
                    "--worker",
                    "--output",
                    str(dest),
                    "--batch",
                    str(batch),
                    "--draft",
                    str(draft),
                    "--mode",
                    mode,
                    "--skip-profile",
                ]
                if mode == "replayssm":
                    command.append("--flush-trace")
                if args.stage == "tuned":
                    choice = json.loads((root / "selected_intervals.json").read_text())
                    interval = choice[f"b{batch}_d{draft}"]["interval"]
                    control = root / f"b{batch}_d{draft}_control"
                    control_command = list(command)
                    control_command[control_command.index("--output") + 1] = str(
                        control
                    )
                    launch(args, control_command, control, control / "complete.json")
                    if interval is not None:
                        command += ["--flush-interval", str(interval)]
                launch(args, command, dest, dest / "complete.json")
        else:
            for draft in (4, 8, 16, 32):
                seeds = (
                    [0, 1]
                    if args.stage == "policy"
                    else [1]
                    if args.stage == "sensitivity"
                    else [0]
                )
                for seed in seeds:
                    dest = root / "jobs" / f"{args.stage}_b{batch}_d{draft}_s{seed}"
                    marker = dest / "complete.json"
                    if marker.exists():
                        continue
                    command = [
                        sys.executable,
                        str(here / "qwen36_flush_crossover.py"),
                        "--output",
                        str(root / "kernels"),
                        "--batch",
                        str(batch),
                        "--draft",
                        str(draft),
                        "--seed",
                        str(seed),
                        "--mode",
                        args.stage,
                        "--completion",
                        str(marker),
                    ]
                    if args.stage in ("policy", "sensitivity"):
                        if args.candidates:
                            command += ["--candidates", args.candidates]
                        command += [
                            "--trace",
                            str(
                                root
                                / f"b{batch}_d{draft}_replayssm"
                                / "flush_trace.json"
                            ),
                        ]
                        if args.stage == "sensitivity":
                            command += [
                                "--selected",
                                str(root / "selected_intervals.json"),
                            ]
                    launch(args, command, dest, marker)


if __name__ == "__main__":
    main()
