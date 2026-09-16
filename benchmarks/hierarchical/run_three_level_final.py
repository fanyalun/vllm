# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze tuning winners and run five paired trials sequentially on GPU 0."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    final = root / "final"
    final.mkdir(parents=True, exist_ok=False)
    selected = {}
    for method in ("mtp", "dspark"):
        source = root / "tuning_final_validated" / method
        assert (source / "measurement_complete.json").is_file()
        rows = json.loads((source / "results.json").read_text())
        costs = {}
        for case in {r["case"] for r in rows if r["case"].startswith("three_level:")}:
            timed = [r for r in rows if r["phase"] == "e2e" and r["case"] == case]
            assert len(timed) == 8
            costs[case] = sum(r["seconds"] for r in timed) / sum(
                len(r["token_ids"]) for r in timed
            )
        selected[method] = min(costs, key=costs.get)
    sources = list(
        json.loads((root / "tuning_final_validated/mtp/contract.json").read_text())[
            "source_sha256"
        ]
    )
    sources += ["benchmarks/hierarchical/three_level_cost.py"]

    def fingerprints():
        return {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in sources
        }

    frozen = fingerprints()
    (final / "freeze.json").write_text(
        json.dumps(
            dict(
                selected=selected,
                source_sha256=frozen,
                gpu=0,
                repeats=5,
                samples=16,
                max_tokens=256,
                seed=42,
                selection="total tuning seconds / returned tokens",
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            ),
            indent=2,
        )
    )
    for method, candidate in selected.items():
        assert fingerprints() == frozen, "Source changed after freeze"
        stop = candidate.split(":")[-1]
        cases = ["exact:carry:low_error"]
        if stop != "low_error":
            cases.append(f"exact:carry:{stop}")
        cases.append(candidate)
        output = final / method
        command = [
            sys.executable,
            str(repo / "benchmarks/hierarchical/run_replay_tail.py"),
            "--inner-method",
            method,
            "--three-level",
            "--cases",
            *cases,
            "--samples",
            "16",
            "--max-tokens",
            "256",
            "--repeats",
            "5",
            "--seed",
            "42",
            "--dataset",
            str(root / "final_prompts.jsonl"),
            "--action-audit",
            "--profile-output",
            str(output / "profile.json"),
            "--output",
            str(output),
        ]
        print("START", method, candidate, flush=True)
        with (final / f"{method}.log").open("x") as log:
            subprocess.run(
                command,
                cwd=repo,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        assert fingerprints() == frozen, "Source changed during measurements"
        assert (output / "measurement_complete.json").is_file()
        print("COMPLETE", method, flush=True)
    (final / "paired_runs_complete.json").write_text(
        json.dumps(
            dict(
                selected=selected,
                methods=2,
                sequential_gpu=0,
                performance_accepted=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
