# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_integrity import refuse_incomplete_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-near-tie-audit")
    return parser.parse_args()


def cell_matrix() -> list[dict]:
    cells = [
        {"name": f"ar_{mode}", "method": "ar", "mode": mode}
        for mode in ("eager", "graph")
    ]
    cells.extend(
        {
            "name": f"moe_skip_top4_{mode}_d{draft_length}",
            "method": "moe_skip",
            "mode": mode,
            "draft_length": draft_length,
            "top_h": 4,
            "trace": mode == "graph",
        }
        for mode in ("eager", "graph")
        for draft_length in (4, 8, 16, 32)
    )
    cells.extend(
        {
            "name": f"moe_skip_top8_{mode}_d{draft_length}",
            "method": "moe_skip",
            "mode": mode,
            "draft_length": draft_length,
            "top_h": 8,
            "trace": False,
        }
        for mode in ("eager", "graph")
        for draft_length in (4, 32)
    )
    return cells


def load_near_tie_allowance(path: str | None) -> dict | None:
    if path is None:
        return None
    audit_path = Path(path).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "validated" or not audit.get("allowances"):
        raise ValueError(f"Invalid near-tie audit: {audit_path}")
    required = {
        "sample_index",
        "prompt_sha256",
        "position",
        "ar_token",
        "spec_token",
        "ar_top1_minus_top2",
        "spec_top1_minus_top2",
        "lifecycle",
        "evidence",
    }
    for entry in audit["allowances"]:
        if required - entry.keys():
            raise ValueError(f"Incomplete near-tie entry in {audit_path}")
    return {
        "audit_path": str(audit_path),
        "final_correctness_label": audit["final_correctness_label"],
        "allowances": audit["allowances"],
    }


def is_allowed_near_tie(
    allowance: dict | None,
    spec_output: dict,
    position: int,
    ar_token: int,
    spec_token: int,
) -> bool:
    if allowance is None:
        return False
    observed = {
        "sample_index": spec_output["sample_index"],
        "prompt_sha256": spec_output["prompt_sha256"],
        "position": position,
        "ar_token": ar_token,
        "spec_token": spec_token,
    }
    return any(
        all(observed[key] == entry[key] for key in observed)
        for entry in allowance["allowances"]
    )


def fail_on_parity_mismatch(
    run_dir: Path,
    cell: dict,
    output_path: Path,
    allowance: dict | None,
) -> None:
    if cell["method"] != "moe_skip":
        return
    baseline_path = run_dir / "cells" / f"ar_{cell['mode']}" / "cell_output.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = json.loads(output_path.read_text(encoding="utf-8"))
    for ar_output, spec_output in zip(
        baseline["outputs"], result["outputs"], strict=True
    ):
        for position, (ar_token, spec_token) in enumerate(
            zip(ar_output["token_ids"], spec_output["token_ids"], strict=True)
        ):
            if ar_token == spec_token:
                continue
            if is_allowed_near_tie(
                allowance, spec_output, position, ar_token, spec_token
            ):
                print(
                    f"ALLOW diagnosed near-tie at {cell['name']}, sample "
                    f"{spec_output['sample_index']}, position {position}",
                    flush=True,
                )
                break
            failure = {
                "status": "failed",
                "cell": cell["name"],
                "sample_index": spec_output["sample_index"],
                "prompt_sha256": spec_output["prompt_sha256"],
                "position": position,
                "ar_token": ar_token,
                "spec_token": spec_token,
            }
            (run_dir / "correctness_audit.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"AR parity failed at {cell['name']}, sample "
                f"{spec_output['sample_index']}, position {position}"
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    allowance = load_near_tie_allowance(args.allow_near_tie_audit)
    commands_path = run_dir / "commands.json"
    prior_commands = (
        json.loads(commands_path.read_text(encoding="utf-8"))
        if args.resume and commands_path.exists()
        else []
    )
    commands = {entry["cell"]: entry for entry in prior_commands}
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    for index, cell in enumerate(cell_matrix(), start=1):
        cell_dir = run_dir / "cells" / cell["name"]
        output_path = cell_dir / "cell_output.json"
        if args.resume and output_path.exists():
            fail_on_parity_mismatch(run_dir, cell, output_path, allowance)
            print(f"[{index}/14] SKIP {cell['name']}", flush=True)
            continue
        if cell["trace"]:
            refuse_incomplete_trace(cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(script_dir / "run_cell.py"),
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--output",
            str(output_path),
            "--mode",
            cell["mode"],
            "--method",
            cell["method"],
            "--num-samples",
            "4",
            "--max-tokens",
            "128",
        ]
        if cell["method"] == "moe_skip":
            command.extend(
                [
                    "--draft-length",
                    str(cell["draft_length"]),
                    "--top-h",
                    str(cell["top_h"]),
                ]
            )
            if cell["trace"]:
                command.extend(["--trace-dir", str(cell_dir / "trace")])
        commands[cell["name"]] = {
            "cell": cell["name"],
            "cuda_visible_devices": args.cuda_device,
            "command": command,
        }
        commands_path.write_text(
            json.dumps(list(commands.values()), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{index}/14] START {cell['name']}", flush=True)
        with (cell_dir / "run.log").open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            failure = {"cell": cell["name"], "returncode": result.returncode}
            (run_dir / "RUN_FAILED.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"Cell {cell['name']} failed; see {cell_dir / 'run.log'}"
            )
        fail_on_parity_mismatch(run_dir, cell, output_path, allowance)
        print(f"[{index}/14] DONE {cell['name']}", flush=True)

    subprocess.run(
        [
            sys.executable,
            str(script_dir / "summarize.py"),
            "--run-dir",
            str(run_dir),
            *(
                ["--allow-near-tie-audit", args.allow_near_tie_audit]
                if args.allow_near_tie_audit
                else []
            ),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
