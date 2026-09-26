# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize completed full-model runs without conflating kernel and E2E costs."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def same_contract(reference, candidate, *, native_spec=False):
    keys = (
        "batch",
        "tp",
        "eager",
        "tokens",
        "cpu_offload_gb",
        "prompt_sha256",
        "runtime_diff_sha256",
        "token_budget",
        "cuda_visible_devices",
        "model",
        "batch_sharded_sampling",
        "launch_blocking",
    )
    if native_spec:
        keys += ("length",)
    return all(reference[key] == candidate[key] for key in keys)


def summarize_cell(path):
    manifest = read(path / "manifest.json")
    args = manifest["args"]
    result = {
        "path": str(path),
        **{
            k: args[k]
            for k in (
                "variant",
                "batch",
                "tp",
                "length",
                "eager",
                "tokens",
                "cpu_offload_gb",
            )
        },
        "prompt_sha256": manifest["prompt_sha256"],
        "runtime_diff_sha256": manifest["runtime_diff_sha256"],
        "token_budget": manifest["config"]["max_num_batched_tokens"],
        "cuda_visible_devices": manifest["cuda_visible_devices"],
        "model": manifest["config"].get("model"),
        "batch_sharded_sampling": manifest["config"].get(
            "enable_batch_sharded_sampling", False
        ),
        "draft_block_graph": args.get("draft_block_graph", False),
        "launch_blocking": manifest.get("diagnostic_environment", {}).get(
            "CUDA_LAUNCH_BLOCKING"
        ),
        "completed": (path / "complete.json").exists(),
    }
    if not result["completed"]:
        return result
    specification = manifest["config"].get("speculative_config")
    if specification:
        result["length"] = specification["num_speculative_tokens"]
    elif args["variant"] == "ar":
        result["length"] = 0
    result.update(read(path / "complete.json"))
    timings = read(path / "timings.json")
    result["repeat_tokens_per_second"] = [r["tokens_per_second"] for r in timings]
    audit = read(path / "audit.json")
    worker = audit["workers"][0]
    accepted = []
    submitted = []
    full_width = []
    full_batch_width = []
    batches = []
    for step in worker["steps"]:
        if step["has_prefill"]:
            continue
        batch = len(step["request_ids"])
        batches.append(batch)
        for proposed, sampled in zip(step["proposed"], step["sampled"], strict=True):
            if not proposed:
                continue
            count = max(0, sampled - 1)
            assert count <= proposed
            accepted.append(count)
            submitted.append(proposed)
            if proposed == result["length"]:
                full_width.append(count)
                if batch == args["batch"]:
                    full_batch_width.append(count)
    result["actual_peak_decode_batch"] = max(batches, default=0)
    result["full_batch_decode_steps"] = sum(b == args["batch"] for b in batches)
    result["decode_steps"] = len(batches)
    result["mean_decode_batch"] = statistics.mean(batches) if batches else 0
    result["candidate_acceptance"] = (
        sum(accepted) / sum(submitted) if submitted else None
    )
    result["accepted_candidates"] = sum(accepted)
    result["submitted_candidates"] = sum(submitted)
    result["mean_accepted"] = statistics.mean(accepted) if accepted else None
    result["mean_submitted"] = statistics.mean(submitted) if submitted else None
    result["full_width_mean_accepted"] = (
        statistics.mean(full_width) if full_width else None
    )
    result["full_batch_full_width_mean_accepted"] = (
        statistics.mean(full_batch_width) if full_batch_width else None
    )
    result["accepted_histogram"] = dict(sorted(Counter(accepted).items()))
    result["full_width_survival"] = (
        [
            sum(a >= i for a in full_width) / len(full_width)
            for i in range(result["length"] + 1)
        ]
        if full_width
        else []
    )
    stages = defaultdict(list)
    for row in worker["stages"]:
        stages[row["stage"]].append(row["gpu_ms"])
    target_times = stages["target_execute"]
    # The engine can execute a final empty scheduling step without sampling.
    assert len(target_times) in (len(worker["steps"]), len(worker["steps"]) + 1)
    full_batch_target = [
        elapsed
        for elapsed, step in zip(
            target_times[: len(worker["steps"])], worker["steps"], strict=True
        )
        if not step["has_prefill"] and len(step["request_ids"]) == args["batch"]
    ]
    result["full_batch_target_median_ms"] = (
        statistics.median(full_batch_target) if full_batch_target else None
    )
    proposals = stages.get("proposal", [])
    if proposals:
        assert len(proposals) == len(worker["steps"])
        proposal_times = [
            elapsed
            for elapsed, step in zip(proposals, worker["steps"], strict=True)
            if not step["has_prefill"] and len(step["request_ids"]) == args["batch"]
        ]
        if worker["paired_forward"]:
            proposal_times = proposal_times[1:]
        result["full_batch_proposal_median_ms"] = (
            statistics.median(proposal_times) if proposal_times else None
        )
    result["audit_stages"] = {
        name: {
            "count": len(values),
            "total_gpu_ms": sum(values),
            "mean_gpu_ms": statistics.mean(values),
            "median_gpu_ms": statistics.median(values),
        }
        for name, values in stages.items()
    }
    result["peak_allocated_gib_per_rank"] = [
        w["peak_allocated_bytes"] / 2**30 for w in audit["workers"]
    ]
    result["checks_per_rank"] = [w["checks"] for w in audit["workers"]]
    result["block_graph_checks_per_rank"] = [
        w.get("block_graph_checks", []) for w in audit["workers"]
    ]
    if worker["paired_forward"]:
        probe = worker["paired_forward"]
        result["paired_forward_median_ms"] = {
            key: statistics.median(probe[key]["full_forward_gpu_ms"])
            for key in ("native_draft", "full", "v2")
        }
        native = result["paired_forward_median_ms"]["native_draft"]
        approximate = result["paired_forward_median_ms"]["v2"]
        result["paired_r"] = approximate / native
        result["paired_full_forward_speedup"] = native / approximate
        if "oracle_gdn_reuse" in probe:
            oracle = statistics.median(probe["oracle_gdn_reuse"]["full_forward_gpu_ms"])
            result["oracle_gdn_reuse_median_ms"] = oracle
            result["oracle_gdn_reuse_speedup"] = native / oracle
        result["forced_full_checks_per_rank"] = [
            w["paired_forward"]["forced_full_check"] for w in audit["workers"]
        ]
        mean_a = result["full_width_mean_accepted"]
        if mean_a is not None:
            result["idealized_verify_maintenance_budget_ar_steps"] = (
                mean_a + 1 - args["length"] * approximate / native
            )
            result["idealized_all_accepted_budget_ar_steps"] = (
                args["length"] + 1 - args["length"] * approximate / native
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = [
        summarize_cell(path.parent)
        for path in sorted(args.root.glob("*/manifest.json"))
    ]
    for row in rows:
        if not row["completed"]:
            continue
        references = [
            ref
            for ref in rows
            if ref["completed"] and ref["variant"] == "ar" and same_contract(ref, row)
        ]
        if references:
            reference = references[-1]
            row["ar_reference"] = reference["path"]
            row["speedup_vs_ar"] = (
                row["tokens_per_second"] / reference["tokens_per_second"]
            )
            actual = read(Path(row["path"]) / "timings.json")[-1]["token_ids"]
            expected = read(Path(reference["path"]) / "timings.json")[-1]["token_ids"]
            row["ar_equal_requests"] = sum(
                a == b for a, b in zip(actual, expected, strict=True)
            )
            row["strict_ar_parity"] = actual == expected
            ar_step = reference["full_batch_target_median_ms"]
            verify = row["full_batch_target_median_ms"]
            draft = row.get("paired_forward_median_ms", {}).get("v2")
            mean_a = row["full_batch_full_width_mean_accepted"]
            if ar_step and verify and draft and mean_a is not None:
                row["gpu_cost_bound"] = {
                    "ar_step_ms": ar_step,
                    "verify_ms": verify,
                    "draft_step_ms": draft,
                    "mean_accepted": mean_a,
                    "draft_ratio_to_ar": draft / ar_step,
                    "verify_ar_steps": verify / ar_step,
                    "max_draft_ratio_without_maintenance": (
                        mean_a + 1 - verify / ar_step
                    )
                    / row["length"],
                    "required_accepted_without_maintenance": (
                        row["length"] * draft + verify
                    )
                    / ar_step
                    - 1,
                }
                proposal = row.get("full_batch_proposal_median_ms")
                if proposal:
                    maintenance = max(0, proposal - row["length"] * draft)
                    row["gpu_cost_bound"].update(
                        proposal_ms=proposal,
                        proposal_minus_paired_forwards_ms=maintenance,
                        break_even_draft_ms_with_current_other_costs=(
                            (mean_a + 1) * ar_step - verify - maintenance
                        )
                        / row["length"],
                        draft_ms_for_1_2x_with_current_other_costs=(
                            (mean_a + 1) * ar_step / 1.2 - verify - maintenance
                        )
                        / row["length"],
                        full_batch_stage_speedup_bound=(
                            (mean_a + 1) * ar_step / (proposal + verify)
                        ),
                        free_verify_stage_speedup_bound=(
                            (mean_a + 1) * ar_step / proposal
                        ),
                    )
            row["first_difference"] = [
                next(
                    (i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y),
                    None,
                )
                for a, b in zip(actual, expected, strict=True)
            ]
        controls = [
            ref
            for ref in rows
            if ref["completed"]
            and ref["variant"] == "native_draft"
            and same_contract(ref, row, native_spec=True)
        ]
        if controls:
            control = controls[-1]
            actual = read(Path(row["path"]) / "timings.json")[-1]["token_ids"]
            expected = read(Path(control["path"]) / "timings.json")[-1]["token_ids"]
            row["native_spec_reference"] = control["path"]
            row["native_spec_equal_requests"] = sum(
                a == b for a, b in zip(actual, expected, strict=True)
            )
            row["native_spec_parity"] = actual == expected
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# GDN V2 complete-model feasibility",
        "",
        "Times include generation, exact verification, state maintenance, scheduling, "
        "and a final device drain. Acceptance excludes correction/bonus; terminal "
        "candidate counts can exceed returned tokens. Paired-forward probes are "
        "warm-cache single-input-position CUDA graphs, not end-to-end timings. "
        "Eager and graph cells, TP1 and TP2, and CPU-offload configurations are "
        "separate contracts.",
        "",
        "| Cell | TP | B | K | tok/s | vs AR | Mean accepted | "
        "Actual peak decode B | AR equal requests |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not row["completed"]:
            continue
        speedup = f"{row['speedup_vs_ar']:.3f}" if "speedup_vs_ar" in row else "N/A"
        accepted = (
            f"{row['mean_accepted']:.3f}" if row["mean_accepted"] is not None else "N/A"
        )
        lines.append(
            f"| [{Path(row['path']).name}]({Path(row['path']).name}/complete.json) "
            f"| {row['tp']} | {row['batch']} | {row['length']} "
            f"| {row['tokens_per_second']:.2f} | {speedup} | {accepted} "
            f"| {row['actual_peak_decode_batch']} "
            f"| {row.get('ar_equal_requests', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "Incomplete cells are preserved in summary.json; absence of "
            "complete.json is not a performance result.",
            "",
        ]
    )
    (args.root / "readme.md").write_text("\n".join(lines))
    print(
        json.dumps(
            [
                {
                    k: row[k]
                    for k in (
                        "path",
                        "completed",
                        "tokens_per_second",
                        "speedup_vs_ar",
                        "mean_accepted",
                        "actual_peak_decode_batch",
                        "strict_ar_parity",
                    )
                    if k in row
                }
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
