# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import csv
import json
import math
from pathlib import Path

DRAFT_LENGTHS = (4, 8, 16, 32)
MODES = ("eager", "graph")
EXPECTED_STATUS = "completed_with_user_accepted_precision_divergence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_cells() -> set[str]:
    cells = {f"ar_{mode}" for mode in MODES}
    cells.update(
        f"moe_skip_top4_{mode}_d{draft_length}"
        for mode in MODES
        for draft_length in DRAFT_LENGTHS
    )
    cells.update(
        f"moe_skip_top8_{mode}_d{draft_length}"
        for mode in MODES
        for draft_length in (4, 32)
    )
    return cells


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    errors: list[str] = []
    expected = expected_cells()
    cell_paths = sorted((run_dir / "cells").glob("*/cell_output.json"))
    observed = {path.parent.name for path in cell_paths}
    if observed != expected:
        errors.append(
            f"cell set mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )

    prompt_hashes = None
    output_count = 0
    for path in cell_paths:
        cell_name = path.parent.name
        cell = load_json(path)
        outputs = cell.get("outputs", [])
        if len(outputs) != 4:
            errors.append(f"{cell_name}: expected 4 outputs, got {len(outputs)}")
        indices = [output.get("sample_index") for output in outputs]
        if indices != [0, 1, 2, 3]:
            errors.append(f"{cell_name}: unexpected sample indices {indices}")
        current_hashes = [output.get("prompt_sha256") for output in outputs]
        if prompt_hashes is None:
            prompt_hashes = current_hashes
        elif current_hashes != prompt_hashes:
            errors.append(f"{cell_name}: prompt hashes differ from the matrix")
        for output in outputs:
            output_count += 1
            token_count = len(output.get("token_ids", []))
            if token_count != 128:
                errors.append(
                    f"{cell_name}: sample {output.get('sample_index')} has "
                    f"{token_count} output tokens"
                )
        log_path = path.parent / "run.log"
        if not log_path.is_file():
            errors.append(f"{cell_name}: missing run.log")
            continue
        log = log_path.read_text(encoding="utf-8")
        if log.count("CELL_COMPLETE") != 1:
            errors.append(f"{cell_name}: CELL_COMPLETE count is not one")
        if log.count("Loading model from scratch") != 1:
            errors.append(f"{cell_name}: model-load count is not one")
        if "Traceback (most recent call last)" in log:
            errors.append(f"{cell_name}: successful log contains a traceback")

    commands = load_json(run_dir / "commands.json")
    command_cells = [entry.get("cell") for entry in commands]
    if len(commands) != 14 or set(command_cells) != expected:
        errors.append("commands.json does not contain 14 unique matrix cells")

    correctness = load_json(run_dir / "correctness_audit.json")
    if correctness.get("status") != EXPECTED_STATUS:
        errors.append("unexpected correctness status")
    if correctness.get("num_checks") != 48:
        errors.append("correctness audit does not contain 48 sample checks")
    if correctness.get("num_failures") != 0:
        errors.append("correctness audit contains unapproved failures")
    checks = correctness.get("checks", [])
    exact_checks = sum(bool(check.get("exact_match")) for check in checks)
    allowed_checks = sum(bool(check.get("allowed_near_tie")) for check in checks)
    if exact_checks + allowed_checks != 48:
        errors.append("correctness checks are not all exact or explicitly allowed")

    acceptance_rows = list(
        csv.DictReader((run_dir / "acceptance_summary.csv").open(encoding="utf-8"))
    )
    if [int(row["draft_length"]) for row in acceptance_rows] != list(DRAFT_LENGTHS):
        errors.append("acceptance summary does not cover D=4,8,16,32")
    for row in acceptance_rows:
        expected_mean = 1 + int(row["accepted_draft_tokens"]) / int(row["verify_steps"])
        if not math.isclose(float(row["mean_acceptance_length"]), expected_mean):
            errors.append(
                f"D={row['draft_length']}: mean acceptance length formula mismatch"
            )

    position_rows = list(
        csv.DictReader((run_dir / "position_metrics.csv").open(encoding="utf-8"))
    )
    if len(position_rows) != sum(DRAFT_LENGTHS):
        errors.append("position metrics do not contain all 60 D-position cells")
    for row in position_rows:
        draft_length = int(row["draft_length"])
        draft_position = int(row["draft_position"])
        if not 1 <= draft_position <= draft_length:
            errors.append(f"invalid D-position row: {row}")
        if int(row["n"]) <= 0:
            errors.append(f"position metric has a non-positive denominator: {row}")
        for metric in (
            "top1_precision",
            "top2_recall",
            "top3_recall",
            "top8_recall",
        ):
            if not 0 <= float(row[metric]) <= 1:
                errors.append(f"position metric is outside [0, 1]: {row}")

    trace = [
        json.loads(line)
        for line in (run_dir / "raw_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    valid_trace = [row for row in trace if row["valid_mask"]]
    invalid_trace = [row for row in trace if not row["valid_mask"]]
    if any(not 0 <= row["output_position"] < 128 for row in valid_trace):
        errors.append("valid trace contains a position outside the 128-token boundary")
    if any(row["output_position"] < 128 for row in invalid_trace):
        errors.append("boundary-filtered trace contains a valid output position")
    if sum(int(row["n"]) for row in position_rows) != len(valid_trace):
        errors.append("position metric denominators do not sum to valid trace rows")

    rank_rows = list(
        csv.DictReader((run_dir / "rank_recall_summary.csv").open(encoding="utf-8"))
    )
    if [row["draft_length"] for row in rank_rows] != ["4", "8", "16", "32", "all"]:
        errors.append("rank recall summary does not cover D=4,8,16,32 and all")
    for row in rank_rows:
        values = (
            valid_trace
            if row["draft_length"] == "all"
            else [
                trace_row
                for trace_row in valid_trace
                if trace_row["draft_length"] == int(row["draft_length"])
            ]
        )
        if int(row["n"]) != len(values):
            errors.append(f"rank recall denominator mismatch: {row}")
        for rank, metric in (
            (1, "top1_precision"),
            (2, "top2_recall"),
            (3, "top3_recall"),
            (8, "top8_recall"),
        ):
            hits = sum(
                value["target_top1_token_id"]
                in value["draft_argmax_ordered_top8_token_ids"][:rank]
                for value in values
            )
            if int(row[f"top{rank}_hits"]) != hits or not math.isclose(
                float(row[metric]), hits / len(values)
            ):
                errors.append(f"rank-{rank} recall mismatch: {row}")

    required_files = (
        "RUN_COMPLETE",
        "RESULTS.md",
        "PRECISION_POLICY_OVERRIDE.json",
        "DEVICE_ASSIGNMENT.json",
        "outputs.json",
        "raw_trace.jsonl",
        "acceptance_summary.csv",
        "position_metrics.csv",
        "rank_recall_summary.csv",
        "acceptance_length_vs_draft_length.png",
        "token_quality_by_position.png",
        "environment.json",
    )
    for name in required_files:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required artifact: {name}")
    marker = (run_dir / "RUN_COMPLETE").read_text(encoding="utf-8").strip()
    if marker != EXPECTED_STATUS:
        errors.append("RUN_COMPLETE does not match the audited status")
    if (run_dir / "RUN_FAILED.json").exists():
        errors.append("RUN_FAILED.json is present at the run root")

    audit = {
        "status": "passed" if not errors else "failed",
        "run_status": marker,
        "expected_cell_count": len(expected),
        "completed_cell_count": len(observed),
        "output_count": output_count,
        "output_tokens_per_sample": 128,
        "prompt_sha256": prompt_hashes,
        "spec_vs_ar_checks": len(checks),
        "exact_match_checks": exact_checks,
        "allowed_near_tie_checks": allowed_checks,
        "unexpected_mismatch_checks": correctness.get("num_failures"),
        "raw_trace_rows": len(trace),
        "valid_trace_rows": len(valid_trace),
        "boundary_filtered_trace_rows": len(invalid_trace),
        "position_metric_rows": len(position_rows),
        "errors": errors,
    }
    (run_dir / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if errors:
        raise RuntimeError(f"Artifact audit failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
