# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare baseline-GDN and ReplaySSM async-SSD diagnostic artifacts."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from benchmarks.replayssm.async_ssd_eagle3_matrix import (
        audit_request_pair,
        request_map,
        write_json,
    )
except ModuleNotFoundError:
    from async_ssd_eagle3_matrix import (
        audit_request_pair,
        request_map,
        write_json,
    )


CELL_BY_MODE = {
    "ar": "correctness_ar_eager_b1",
    "sync": "correctness_sync_eager_b1",
    "async_jit": "correctness_async_jit_eager_b1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replayssm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-logprob-tolerance", type=float, default=0.1)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mode_requests(root: Path, mode: str) -> dict[int, dict[str, Any]]:
    requests = load_json(root / "cells" / CELL_BY_MODE[mode] / "requests.json")
    return request_map(requests)


def top_candidates(
    top_logprobs: dict[str, float] | None, limit: int = 5
) -> list[dict[str, float | int]]:
    if top_logprobs is None:
        return []
    candidates = []
    for name, logprob in top_logprobs.items():
        if not name.startswith("token_id:"):
            continue
        candidates.append(
            {"token_id": int(name.removeprefix("token_id:")), "logprob": logprob}
        )
    return sorted(candidates, key=lambda value: value["logprob"], reverse=True)[:limit]


def compare_requests(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    if baseline.keys() != candidate.keys():
        raise ValueError(
            "request index mismatch: "
            f"baseline={sorted(baseline)} candidate={sorted(candidate)}"
        )
    comparisons = []
    for prompt_index in sorted(baseline):
        result = audit_request_pair(
            baseline[prompt_index], candidate[prompt_index], tolerance
        )
        result = {"prompt_index": prompt_index, **result}
        offset = result.get("offset")
        if isinstance(offset, int):
            result["baseline_top_candidates"] = top_candidates(
                baseline[prompt_index]["top_logprobs"][offset]
            )
            result["candidate_top_candidates"] = top_candidates(
                candidate[prompt_index]["top_logprobs"][offset]
            )
        comparisons.append(result)

    statuses = Counter(result["status"] for result in comparisons)
    reasons = Counter(
        result["reason"] for result in comparisons if result.get("reason")
    )
    divergent = [result for result in comparisons if "offset" in result]
    offsets = [result["offset"] for result in divergent]
    gaps = [
        max(result["baseline_logprob_gap"], result["candidate_logprob_gap"])
        for result in divergent
        if "baseline_logprob_gap" in result and "candidate_logprob_gap" in result
    ]
    gap_threshold_counts = {
        str(threshold): sum(gap <= threshold + 1e-6 for gap in gaps)
        for threshold in (0.1, 0.125, 0.25, 0.5, 1.0)
    }
    exact_or_tie = statuses["exact"] + statuses["target_top1_tie_equivalent"]
    return {
        "request_count": len(comparisons),
        "exact_or_tie_count": exact_or_tie,
        "exact_or_tie_fraction": exact_or_tie / len(comparisons),
        "status_counts": dict(statuses),
        "failure_reason_counts": dict(reasons),
        "first_divergence_offset_min": min(offsets) if offsets else None,
        "first_divergence_offset_median": (
            statistics.median(offsets) if offsets else None
        ),
        "two_sided_max_gap_median": statistics.median(gaps) if gaps else None,
        "two_sided_max_gap_threshold_counts": gap_threshold_counts,
        "comparisons": comparisons,
    }


def metric_total(result: dict[str, Any], metric: str) -> float:
    matches = [
        value
        for name, value in result["metrics_delta"].items()
        if name.startswith(metric) and "_created" not in name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {metric!r} metric, found {len(matches)}")
    return float(matches[0])


def mode_summary(root: Path, mode: str) -> dict[str, Any]:
    cell_dir = root / "cells" / CELL_BY_MODE[mode]
    result = load_json(cell_dir / "result.json")
    cell_complete = load_json(cell_dir / "cell_complete.json")
    summary = {
        "cell_complete": cell_complete["status"] == "complete",
        "completed_request_count": result["summary"]["completed_request_count"],
        "completion_tokens": result["summary"]["completion_tokens"],
        "completion_throughput_tok_s": result["summary"]["completion_throughput_tok_s"],
        "command": load_json(cell_dir / "command.json"),
    }
    if mode != "ar":
        accepted = metric_total(result, "vllm:spec_decode_num_accepted_tokens_total")
        drafts = metric_total(result, "vllm:spec_decode_num_drafts_total")
        drafted_tokens = metric_total(result, "vllm:spec_decode_num_draft_tokens_total")
        summary.update(
            {
                "accepted_draft_tokens": accepted,
                "draft_rounds": drafts,
                "drafted_tokens": drafted_tokens,
                "mean_accepted_draft_length": accepted / drafts,
                "mean_acceptance_length": 1.0 + accepted / drafts,
            }
        )
    if mode == "async_jit":
        summary["dspark_bank_audit"] = load_json(cell_dir / "dspark_bank_audit.json")
    return summary


def flag_contract(command: list[str], mode: str, gdn_mode: str) -> bool:
    has_ar = "--use-replayssm" in command
    has_spec = "--use-replayssm-spec" in command
    if gdn_mode == "baseline":
        return not has_ar and not has_spec
    if mode == "ar":
        return has_ar and not has_spec
    return has_spec and not has_ar


def without_gdn_flags(command: list[str]) -> list[str]:
    normalized = []
    skip_value = False
    for value in command:
        if skip_value:
            skip_value = False
            continue
        if value == "--replayssm-buffer-len":
            skip_value = True
            continue
        if value in ("--use-replayssm", "--use-replayssm-spec"):
            continue
        normalized.append(value)
    return normalized


def build_report(
    baseline_root: Path,
    replayssm_root: Path,
    tolerance: float,
) -> dict[str, Any]:
    roots = {"baseline": baseline_root.resolve(), "replayssm": replayssm_root.resolve()}
    manifests = {
        name: load_json(root / "manifest.json") for name, root in roots.items()
    }
    requests = {
        gdn_mode: {mode: load_mode_requests(root, mode) for mode in CELL_BY_MODE}
        for gdn_mode, root in roots.items()
    }
    modes = {
        gdn_mode: {mode: mode_summary(root, mode) for mode in CELL_BY_MODE}
        for gdn_mode, root in roots.items()
    }
    within = {
        gdn_mode: {
            "ar_vs_sync": compare_requests(
                requests[gdn_mode]["ar"], requests[gdn_mode]["sync"], tolerance
            ),
            "ar_vs_async_jit": compare_requests(
                requests[gdn_mode]["ar"],
                requests[gdn_mode]["async_jit"],
                tolerance,
            ),
            "sync_vs_async_jit": compare_requests(
                requests[gdn_mode]["sync"],
                requests[gdn_mode]["async_jit"],
                tolerance,
            ),
        }
        for gdn_mode in roots
    }
    across = {
        mode: compare_requests(
            requests["baseline"][mode], requests["replayssm"][mode], tolerance
        )
        for mode in CELL_BY_MODE
    }
    ar_reference_pairs = ("ar_vs_sync", "ar_vs_async_jit")
    ar_reference_scores = {
        gdn_mode: sum(
            within[gdn_mode][pair]["exact_or_tie_count"] for pair in ar_reference_pairs
        )
        for gdn_mode in roots
    }
    all_pair_scores = {
        gdn_mode: sum(
            comparison["exact_or_tie_count"] for comparison in within[gdn_mode].values()
        )
        for gdn_mode in roots
    }
    if all_pair_scores["baseline"] > all_pair_scores["replayssm"]:
        assessment = "supports_replayssm_as_additional_divergence_source"
    elif all_pair_scores["baseline"] < all_pair_scores["replayssm"]:
        assessment = "does_not_support_replayssm_hypothesis"
    else:
        assessment = "not_distinguished_by_primary_pair_count"

    source_checks = {
        "same_git_head": manifests["baseline"]["git"]["head"]
        == manifests["replayssm"]["git"]["head"],
        "same_working_tree_diff_sha256": manifests["baseline"]["git"][
            "working_tree_diff_sha256"
        ]
        == manifests["replayssm"]["git"]["working_tree_diff_sha256"],
        "same_prompt_suite_sha256": manifests["baseline"]["workload"][
            "prompt_suite_sha256"
        ]
        == manifests["replayssm"]["workload"]["prompt_suite_sha256"],
        "same_workload": manifests["baseline"]["workload"]
        == manifests["replayssm"]["workload"],
        "same_commands_except_gdn_flags": {
            mode: without_gdn_flags(modes["baseline"][mode]["command"])
            == without_gdn_flags(modes["replayssm"][mode]["command"])
            for mode in CELL_BY_MODE
        },
        "flag_contract": {
            gdn_mode: {
                mode: flag_contract(summary["command"], mode, gdn_mode)
                for mode, summary in mode_summaries.items()
            }
            for gdn_mode, mode_summaries in modes.items()
        },
    }
    source_checks["passed"] = (
        all(
            value
            for key, value in source_checks.items()
            if key not in ("flag_contract", "same_commands_except_gdn_flags")
        )
        and all(
            passed
            for contract in source_checks["flag_contract"].values()
            for passed in contract.values()
        )
        and all(source_checks["same_commands_except_gdn_flags"].values())
    )
    return {
        "status": "complete" if source_checks["passed"] else "invalid_comparison",
        "diagnostic_scope": {
            "batch_size": 1,
            "prompt_count": 16,
            "output_tokens_per_request": 512,
            "dtype": "bfloat16",
            "verify_width": 4,
            "sync_execution_width": 4,
            "async_bank_width": 8,
            "tie_logprob_tolerance": tolerance,
        },
        "roots": {name: str(root) for name, root in roots.items()},
        "source_and_flag_checks": source_checks,
        "mode_summaries": modes,
        "within_gdn_mode_comparisons": within,
        "cross_gdn_mode_comparisons": across,
        "replayssm_hypothesis": {
            "ar_reference_pairs": list(ar_reference_pairs),
            "ar_reference_exact_or_tie_score_out_of_32": ar_reference_scores,
            "all_pair_exact_or_tie_score_out_of_48": all_pair_scores,
            "sync_vs_async_exact_count_out_of_16": {
                gdn_mode: within[gdn_mode]["sync_vs_async_jit"]["status_counts"].get(
                    "exact", 0
                )
                for gdn_mode in roots
            },
            "assessment": assessment,
            "interpretation_boundary": (
                "This diagnostic localizes numerical divergence; it does not by "
                "itself prove a ReplaySSM state-management bug."
            ),
        },
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        args.baseline_root, args.replayssm_root, args.tie_logprob_tolerance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report["replayssm_hypothesis"], indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
