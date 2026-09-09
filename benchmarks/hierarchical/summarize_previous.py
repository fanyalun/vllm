# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and compare complete hierarchical outer-loop measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmark_results"
AR_OUTPUTS = BASE / "moe_skip_e2e_16x512_b1_20260908_run2/qwen36/ar_d0/result.json"
BASELINE_FILES = {
    "e2e": BASE / "moe_skip_e2e_16x512_b1_20260908_final/performance.csv",
    "timing": BASE / "draft_timing_16x512_b1_20260908_final/draft_timing.csv",
    "acceptance": (
        BASE
        / "qwen36_gemma4_mtp_vs_moe_skip_128x512_20260907_final"
        / "model_method_metrics.csv"
    ),
}


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def summarize(root):
    contract = json.loads((root / "contract.json").read_text())
    (root / "baselines").mkdir(exist_ok=True)
    ar_path = root / "baselines/ar_outputs.json"
    if not ar_path.exists():
        ar_path.write_bytes(AR_OUTPUTS.read_bytes())
    ar_outputs = json.loads(ar_path.read_text())["outputs"]
    rows, hashes, cells, parity = [], {}, [], []
    for method in ("mtp", "dspark"):
        for n in (1, 2, 4, 8):
            results = {}
            for phase in ("e2e", "timing", "acceptance"):
                directory = root / phase / f"{method}_d4_n{n}"
                assert (directory / "CELL_COMPLETE").exists(), directory
                path = directory / "result.json"
                result = json.loads(path.read_text())
                expected = (
                    contract["acceptance_samples"] if phase == "acceptance" else 16
                )
                assert result["samples"] == expected
                assert len(result["outputs"]) == expected
                assert all(len(r["token_ids"]) == 512 for r in result["outputs"])
                assert result["d"] == 4 and result["n"] == n
                dataset = root / f"samples_{expected}.jsonl"
                samples = [
                    json.loads(line) for line in dataset.read_text().splitlines()
                ]
                assert [r["prompt_sha256"] for r in result["outputs"]] == [
                    r["prompt_sha256"] for r in samples
                ]
                for name in ("result.json", "config.json", "command.json", "run.log"):
                    evidence = directory / name
                    hashes[str(evidence.relative_to(root))] = hashlib.sha256(
                        evidence.read_bytes()
                    ).hexdigest()
                log = (directory / "run.log").read_text()
                marker = (
                    "MEASUREMENT_START" if phase == "acceptance" else "WARMUP_COMPLETE"
                )
                assert marker in log, directory
                tail = log.split(marker, 1)[-1]
                if phase != "acceptance":
                    assert "JIT compilation during inference" not in tail, directory
                cells.append(
                    {
                        "phase": phase,
                        "method": method,
                        "n": n,
                        "cuda_device": result.get(
                            "cuda_device", contract["phase_devices"][phase]
                        ),
                        "requests": expected,
                        "tokens": expected * 512,
                        "post_warmup_jit_warnings": tail.count(
                            "JIT compilation during inference"
                        ),
                    }
                )
                results[phase] = result
            elapsed = sum(r["e2e_seconds"] for r in results["e2e"]["outputs"])
            first_differences = []
            for current, reference in zip(
                results["e2e"]["outputs"], ar_outputs, strict=True
            ):
                assert current["prompt_sha256"] == reference["prompt_sha256"]
                first_differences.append(
                    next(
                        (
                            i
                            for i, (a, b) in enumerate(
                                zip(
                                    current["token_ids"],
                                    reference["token_ids"],
                                    strict=True,
                                )
                            )
                            if a != b
                        ),
                        None,
                    )
                )
            parity.append(
                {
                    "method": method,
                    "n": n,
                    "first_different_token_indices": first_differences,
                }
            )
            calls = results["acceptance"]["verification"]
            assert calls and all(
                0 <= r["accepted"] <= r["scheduled"] <= 5 * n for r in calls
            )
            accepted = sum(r["accepted"] for r in calls)
            scheduled = sum(r["scheduled"] for r in calls)
            for request in results["acceptance"]["outputs"]:
                request_calls = [
                    r for r in calls if r["sample_index"] == request["sample_index"]
                ]
                metrics = request["spec_decode_metrics"]
                assert metrics["per_step_accepted"] == [
                    r["accepted"] for r in request_calls
                ]
                assert metrics["per_step_drafted"] == [
                    r["scheduled"] for r in request_calls
                ]
                assert (
                    511 <= sum(r["accepted"] + 1 for r in request_calls) <= 511 + 5 * n
                )
            proposals = results["timing"]["proposals"]
            assert sum(r["has_prefill"] for r in proposals) == 16
            assert all(
                r["draft_width"] == 5 * n
                and r["num_reqs"] == 1
                and r["inner_rounds"] == n
                for r in proposals
            )
            decode = [r for r in proposals if not r["has_prefill"]]
            times = [r["stream_elapsed_ms"] for r in decode]
            assert times and min(times) > 0
            row = {
                "method": f"hierarchical_{method}",
                "inner_d": 4,
                "n": n,
                "nominal_budget": 4 * n,
                "capacity": 5 * n,
                "e2e_tokens_per_second": 8192 / elapsed,
                "e2e_seconds": elapsed,
                "e2e_vs_historical_ar_exact_requests": first_differences.count(None),
                "acceptance_samples": contract["acceptance_samples"],
                "verify_calls": len(calls),
                "accepted_draft_tokens": accepted,
                "scheduled_candidates": scheduled,
                "mean_accepted_drafts": accepted / len(calls),
                "mean_acceptance_length": 1 + accepted / len(calls),
                "mean_scheduled_candidates": scheduled / len(calls),
                "acceptance_fraction": accepted / scheduled,
                "decode_proposal_calls": len(times),
                "mean_loop_ms": float(np.mean(times)),
                "median_loop_ms": float(np.median(times)),
                "p90_loop_ms": float(np.percentile(times, 90)),
                "mean_cpu_submit_ms": float(
                    np.mean([r["cpu_submit_ms"] for r in decode])
                ),
                "mean_prefill_loop_ms": float(
                    np.mean(
                        [r["stream_elapsed_ms"] for r in proposals if r["has_prefill"]]
                    )
                ),
                "mean_materialized_candidates": float(
                    np.mean([r["actual_candidates"] for r in decode])
                ),
                "timing_vs_e2e_exact_requests": sum(
                    a["token_ids"] == b["token_ids"]
                    for a, b in zip(
                        results["timing"]["outputs"],
                        results["e2e"]["outputs"],
                        strict=True,
                    )
                ),
            }
            rows.append(row)
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "measurement_status": "complete",
        "completion_scope": "performance_and_acceptance_measurement_only",
        "baselines_rerun_contemporaneously": False,
        "historical_ar_comparison": parity,
        "historical_ar_reference": {
            "source": str(AR_OUTPUTS),
            "sha256": hashlib.sha256(ar_path.read_bytes()).hexdigest(),
        },
        "strict_model_equivalence_passed": False,
        "acceptance_worker_matches_scheduler_per_step": True,
        "post_warmup_timing_jit_warnings": 0,
        "cells": cells,
        "cell_count": len(cells),
        "requests": sum(c["requests"] for c in cells),
        "generated_tokens": sum(c["tokens"] for c in cells),
        "result_sha256": hashes,
        "dataset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("samples_*.jsonl")
        },
    }
    audit["baseline_sources"] = {}
    for phase, path in BASELINE_FILES.items():
        snapshot = root / "baselines" / f"{phase}.csv"
        if not snapshot.exists():
            snapshot.write_bytes(path.read_bytes())
        audit["baseline_sources"][phase] = {
            "source": str(path),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        }
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot(root, rows)
    report(root, rows, contract, audit)
    return rows


def report(root, rows, contract, audit):
    previous = read_csv(root / "baselines/e2e.csv")
    ar = next(
        float(r["output_tokens_per_second"])
        for r in previous
        if r["model"] == "qwen36" and r["method"] == "ar"
    )
    lines = [
        "# Hierarchical decoding: previous-configuration measurements",
        "",
        "The measurement matrix is complete. **Strict numerical-equivalence gates "
        "failed in the implementation validation and remain failed.** These are "
        "performance and acceptance measurements of that experimental implementation.",
        "",
        "## Configuration and measurement boundaries",
        "",
        "Qwen3.6-35B-A3B, TP1, B=1, greedy, seed=0, ignore_eos=True, "
        "512 output tokens/request, max_model_len=1024, GPU memory utilization=0.95. "
        "Target top-k=8 and shared-weight MoE-Skip pre-verifier top-h=4. "
        "MTP/DSpark inner D=4 and N=1/2/4/8. CUDA Graph execution.",
        "",
        "Nominal budget D*N=4/8/16/32 aligns the old D. Materialized candidates "
        "also contain each inner round's recovery/bonus token, so capacity is "
        "N*(D+1)=5/10/20/40. Actual candidate counts are reported separately.",
        "",
        "- E2E: the original 16 prompts, one full 512-token warmup, GPU1, "
        "max_num_batched_tokens=4096, no measurement instrumentation. "
        "8192 / sum(request wall time); includes prefill, decode, and offline API.",
        "- Drafting time: the same 16 prompts and warmup on GPU1. CUDA events "
        "enclose one complete outer propose(), including all N small-draft/"
        "pre-verifier rounds and state handling. Target verification is excluded. "
        "Decode-only call-weighted mean, with prefill, median and P90 in summary.csv. "
        "Events are collected after each request. CPU submission time is separate "
        "and overlaps the GPU stream interval; do not add the two.",
        f"- Acceptance: {contract['acceptance_samples']} original prompts, "
        "no extra warmup, max_num_batched_tokens=1024, on A100 80GB PCIe device(s) "
        f"{contract['phase_devices']['acceptance']}. Mean length = 1 + "
        "sum(final-Target accepted drafts) / Target verify "
        "calls, including the final Target recovery/bonus. Worker counts are "
        "cross-checked against every scheduler per-step metric. Counts are before "
        "the final max_tokens truncation, matching the old metric.",
        "",
        "## Results",
        "",
        "| Inner model | N | D*N | Tokens/s | vs AR | Acceptance length | "
        "Mean scheduled candidates | Full loop ms | CPU submit ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['n']} | {r['nominal_budget']} | "
            f"{r['e2e_tokens_per_second']:.3f} | "
            f"{r['e2e_tokens_per_second'] / ar:.3f}x | "
            f"{r['mean_acceptance_length']:.4f} | "
            f"{r['mean_scheduled_candidates']:.4f} | {r['mean_loop_ms']:.3f} | "
            f"{r['mean_cpu_submit_ms']:.3f} |"
        )
    best = max(rows, key=lambda r: r["e2e_tokens_per_second"])
    lines += [
        "",
        "Ratios against historical MoE-Skip at the same nominal budget:",
        "",
        "| Inner model | N | Throughput ratio | Acceptance-length ratio | "
        "Full-loop-time ratio (lower is faster) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    previous_metrics = read_csv(root / "comparison.csv")
    for r in rows:
        baseline = next(
            p
            for p in previous_metrics
            if p["method"] == "moe_skip"
            and int(p["nominal_budget"]) == r["nominal_budget"]
        )
        throughput_ratio = r["e2e_tokens_per_second"] / float(
            baseline["e2e_tokens_per_second"]
        )
        acceptance_ratio = r["mean_acceptance_length"] / float(
            baseline["mean_acceptance_length"]
        )
        lines.append(
            f"| {r['method']} | {r['n']} | "
            f"{throughput_ratio:.3f}x | {acceptance_ratio:.3f}x | "
            f"{r['mean_loop_ms'] / float(baseline['mean_loop_ms']):.3f}x |"
        )
    lines += [
        "",
        f"Best measured hierarchical cell: {best['method']}, N={best['n']}, "
        f"{best['e2e_tokens_per_second']:.3f} tokens/s. "
        f"Historical AR: {ar:.3f} tokens/s.",
        "",
        "## Comparability and limitations",
        "",
        "Historical baselines are snapshots in baselines/ with source paths/hashes "
        "in audit.json. They were not rerun contemporaneously. One measurement "
        "per cell; no confidence intervals or same-prefix pairing. Acceptance "
        "and performance queues ran concurrently on separate GPUs. Per-cell "
        "device indices are in audit.json. Historical Qwen3.6 and Gemma4 "
        "performance queues also ran on separate GPUs concurrently.",
        "",
        "The implementation requires disabled multimodal inputs, disabled async "
        "scheduling, and disabled prefix caching. Historical MTP/DSpark could use "
        "automatic async scheduling; old acceptance used the automatic prefix "
        "caching default and additional logit tracing. These restrictions prevent "
        "an exact match of all resolved runtime settings. Old acceptance MTP used "
        "GPU0 and MoE-Skip GPU1. All three experimental passes preserve their "
        "respective historical sample and scheduler-token-budget settings.",
        "",
        "Historical DSpark E2E/timing exists only at D=4/8. The available Qwen3.6 "
        "128x512 acceptance summary has no DSpark baseline; no missing points "
        "are interpolated or inferred from throughput.",
        "",
        "The instrumented timing pass and uninstrumented E2E pass do not always "
        "follow identical token trajectories. Exact-request counts against each "
        "other and the historical AR reference are in summary.csv; zero-based "
        "first differences against AR are in audit.json. These measurements "
        "cannot isolate component costs by subtracting values from different passes.",
        "",
        f"Audited {audit['cell_count']} cells, {audit['requests']} measured requests, "
        f"{audit['generated_tokens']} output tokens. Excluded startup/warmup work "
        "and the interrupted acceptance pilot are outside these totals. "
        "Token IDs, raw loop events, final verification counts and logs are retained.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python benchmarks/hierarchical/compare_previous.py "
        "--run-dir benchmark_results/hierarchical_previous_config_new "
        "--phases e2e timing --cuda-device 1",
        ".venv/bin/python benchmarks/hierarchical/compare_previous.py "
        "--run-dir benchmark_results/hierarchical_previous_config_new "
        "--phases acceptance --cuda-device 0 --resume",
        ".venv/bin/python benchmarks/hierarchical/summarize_previous.py "
        "--run-dir benchmark_results/hierarchical_previous_config_new",
        "```",
    ]
    (root / "results.md").write_text("\n".join(lines) + "\n")


def plot(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    historical_e2e = read_csv(root / "baselines/e2e.csv")
    historical_time = read_csv(root / "baselines/timing.csv")
    historical_accept = read_csv(root / "baselines/acceptance.csv")
    series = []
    labels = {"moe_skip": "MoE-Skip", "mtp": "MTP", "dspark": "DSpark"}
    for method in labels:
        for d in (4, 8, 16, 32):
            e2e = next(
                (
                    r
                    for r in historical_e2e
                    if r["model"] == "qwen36"
                    and r["method"] == method
                    and int(r["d"]) == d
                ),
                None,
            )
            timing = next(
                (
                    r
                    for r in historical_time
                    if r["model"] == "qwen36"
                    and r["method"] == method
                    and int(r["d"]) == d
                    and r["phase"] == "decode_proposal"
                ),
                None,
            )
            accept = next(
                (
                    r
                    for r in historical_accept
                    if r["model"] == "Qwen3.6"
                    and r["method"] == labels[method]
                    and int(r["draft_length"]) == d
                ),
                None,
            )
            if e2e:
                series.append(
                    {
                        "method": method,
                        "nominal_budget": d,
                        "e2e_tokens_per_second": float(e2e["output_tokens_per_second"]),
                        "mean_loop_ms": float(timing["mean_block_ms"])
                        if timing
                        else None,
                        "mean_acceptance_length": float(
                            accept["mean_acceptance_length"]
                        )
                        if accept
                        else None,
                    }
                )
    series.extend(rows)
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "method",
                "nominal_budget",
                "e2e_tokens_per_second",
                "mean_acceptance_length",
                "mean_loop_ms",
            ],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(series)
    output = root / "figures/previous_comparison"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 22,
            "axes.labelsize": 23,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    labels.update(
        hierarchical_mtp="Hierarchical MTP", hierarchical_dspark="Hierarchical DSpark"
    )
    colors = ["#59A14F", "#4C78A8", "#F2B447", "#E45756", "#B279A2"]
    metrics = [
        ("e2e_tokens_per_second", "Output tokens/s", "(a) Throughput"),
        ("mean_acceptance_length", "Accepted length", "(b) Acceptance"),
        ("mean_loop_ms", "Drafting time (ms)", "(c) Full-loop drafting"),
    ]
    handles = []
    for axis, (metric, ylabel, panel) in zip(axes, metrics, strict=True):
        for (method, label), color, marker in zip(
            labels.items(), colors, ["s", "o", "^", "D", "v"], strict=True
        ):
            values = [
                r for r in series if r["method"] == method and r.get(metric) is not None
            ]
            (line,) = axis.plot(
                [r["nominal_budget"] for r in values],
                [r[metric] for r in values],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=7,
                label=label,
            )
            if metric == "e2e_tokens_per_second":
                handles.append(line)
        axis.set_xscale("log", base=2)
        axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        axis.set_xlabel("Nominal draft budget")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.text(0.5, -0.3, panel, transform=axis.transAxes, ha="center", va="top")
    ar = next(
        float(r["output_tokens_per_second"])
        for r in historical_e2e
        if r["model"] == "qwen36" and r["method"] == "ar"
    )
    handles.append(axes[0].axhline(ar, color="0.4", linestyle="--", label="AR"))
    axes[2].set_yscale("log")
    fig.legend(
        handles=[handles[i] for i in (0, 3, 1, 4, 2, 5)],
        loc="upper center",
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.5,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.26, top=0.76, wspace=0.32)
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"previous_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)
    (output / "previous_comparison.md").write_text(
        "# Previous experiment comparison\n\n"
        "Qwen3.6, one A100 80GB PCIe per cell, B=1, TP1, greedy, CUDA Graph. "
        "New hierarchical inner D=4; N=1/2/4/8. "
        "Horizontal axis is nominal budget D*N, not materialized candidate count. "
        "Each inner round can append a recovery/bonus token.\n\n"
        "E2E and CUDA-event drafting use 16 fixed prompts x 512 output tokens. "
        "Acceptance uses the sample count in contract.json and includes the final "
        "Target bonus/recovery. Drafting time is the call-weighted decode-only "
        "mean around the full N-round propose(), including pre-verification and "
        "CPU-induced stream idle gaps, excluding Target verification. "
        "One warmup request is excluded from the two timing passes; acceptance "
        "has no extra warmup. No error bars: one run per cell.\n\n"
        "Historical lines come from the 2026-09-08 E2E/timing experiments and "
        "the 128x512 acceptance summary. The available Qwen3.6 128x512 acceptance "
        "summary has no DSpark baseline; no value was invented. "
        "New samples are not paired on identical generated prefixes across methods. "
        "Strict numerical equivalence gates failed. New runs disable async scheduling "
        "and multimedia inputs as required by the hierarchical implementation.\n\n"
        "Sources: [summary.csv](../../summary.csv), "
        "[comparison.csv](../../comparison.csv), original prompts in "
        "`../../samples_16.jsonl` and `../../samples_128.jsonl`. Reproduce: "
        "`.venv/bin/python benchmarks/hierarchical/summarize_previous.py "
        f"--run-dir {root}`.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_dir.resolve())
