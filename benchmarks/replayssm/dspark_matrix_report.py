# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit integer acceptance counters and render the complete matrix."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import regex as re


def acceptance_metrics(histogram, requests):
    drafted = accepted = rounds = positive = full = zero = 0
    windows = {}
    for key, count in histogram.items():
        match = re.fullmatch(r"window_(\d+)_accepted_(\d+)", key)
        assert match is not None
        n, a = map(int, match.groups())
        assert 0 <= a <= n and count >= 0
        drafted += n * count
        accepted += a * count
        rounds += count
        positive += count if n else 0
        full += count if n and a == n else 0
        zero += count if not n else 0
        windows[n] = windows.get(n, 0) + count
    assert zero >= requests
    decode_rounds = rounds - requests
    zero_decode = zero - requests
    return dict(
        drafted_tokens=drafted,
        accepted_tokens=accepted,
        verification_rounds=positive,
        full_accept_rounds=full,
        decode_rounds=decode_rounds,
        zero_draft_decode_rounds=zero_decode,
        acceptance_rate=accepted / drafted if drafted else None,
        mean_acceptance_length=1 + accepted / positive if positive else None,
        mean_effective_decode_length=1 + accepted / decode_rounds
        if decode_rounds
        else None,
        full_accept_probability=full / positive if positive else None,
        full_accept_probability_all_decode=full / decode_rounds
        if decode_rounds
        else None,
        zero_draft_fraction=zero_decode / decode_rounds if decode_rounds else None,
        mean_verified_draft_length=drafted / decode_rounds if decode_rounds else None,
        window_histogram=windows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    manifest = json.loads((root / "manifest.json").read_text())
    cells = manifest["cells"]
    scope_path = root / "scope_override.json"
    if scope_path.exists():
        scope = json.loads(scope_path.read_text())
        cells = [cell for cell in cells if cell["name"] in scope["cells"]]
        assert {cell["name"] for cell in cells} == set(scope["cells"])
    batches = sorted({cell["batch"] for cell in cells})
    cell_count = len(cells)
    results = {}
    for cell in cells:
        r = json.loads((root / "cells" / cell["name"] / "result.json").read_text())
        assert r["complete"] and r["jit_clean"], cell
        assert r["cohort_admission_exact"], cell
        assert len(r["repeats"]) == 1 and r["samples"] == 16
        assert r["prompts_sha256"] == manifest["prompts_sha256"]
        results[cell["name"]] = r
    summary = []
    parity = []
    for name, r in results.items():
        times = [x["throughput_tps"] for x in r["repeats"]]
        hist = {}
        for rep in r["repeats"]:
            assert rep["output_tokens"] == 4096
            assert all(len(q["token_ids"]) == 256 for q in rep["requests"])
            for key, count in rep["histogram"].items():
                hist[key] = hist.get(key, 0) + count
        row = dict(
            name=name,
            method=r["method"],
            policy=r["policy"],
            batch=r["batch"],
            throughput_tps=statistics.median(times),
            throughput_min=min(times),
            throughput_max=max(times),
            **acceptance_metrics(hist, 16),
        )
        baseline = results[f"ar_ar_b{r['batch']}"]
        row["speedup_vs_ar"] = row["throughput_tps"] / statistics.median(
            x["throughput_tps"] for x in baseline["repeats"]
        )
        match = 0
        for current, ref in zip(
            r["repeats"][0]["requests"], baseline["repeats"][0]["requests"]
        ):
            assert current["sample_id"] == ref["sample_id"]
            equal = current["token_ids"] == ref["token_ids"]
            match += equal
            if not equal:
                first = next(
                    i
                    for i, (a, b) in enumerate(
                        zip(current["token_ids"], ref["token_ids"])
                    )
                    if a != b
                )
                parity.append(
                    dict(
                        cell=name,
                        sample_id=current["sample_id"],
                        first_difference=first,
                        actual_token=current["token_ids"][first],
                        ar_token=ref["token_ids"][first],
                    )
                )
        row["requests_matching_ar"] = match
        summary.append(row)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [k for k in summary[0] if k != "window_histogram"]
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(summary)
    (root / "output_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
        }
    )
    colors = {"sd": "#4C78A8", "replayssm": "#F2B447", "dual": "#59A14F"}
    labels = {"sd": "Baseline SD", "replayssm": "ReplaySSM", "dual": "Dual checkpoint"}
    metrics = [
        ("throughput_tps", "Throughput (tokens/s)", "throughput"),
        ("acceptance_rate", "Draft token acceptance rate", "acceptance_rate"),
        (
            "mean_acceptance_length",
            "Mean accepted length (+ bonus)",
            "acceptance_length",
        ),
        ("full_accept_probability", "Full acceptance probability", "full_acceptance"),
    ]
    for metric, ylabel, figure_name in metrics:
        if len(batches) <= 3:
            fig, axes = plt.subplots(
                1, len(batches), figsize=(3.9 * len(batches), 3.8), squeeze=False
            )
        else:
            fig, axes = plt.subplots(2, 2, figsize=(9, 6.8))
        for i, (ax, batch) in enumerate(zip(axes.flat, batches)):
            for j, method in enumerate(("sd", "replayssm", "dual")):
                values = [
                    next(
                        row[metric]
                        for row in summary
                        if row["batch"] == batch
                        and row["method"] == method
                        and row["policy"] == policy
                    )
                    for policy in ("d4", "d8", "p08", "p06")
                ]
                ax.bar(
                    np.arange(4) + (j - 1) * 0.25,
                    values,
                    width=0.23,
                    color=colors[method],
                    label=labels[method],
                )
            if metric == "throughput_tps":
                ar = next(
                    row[metric]
                    for row in summary
                    if row["batch"] == batch and row["method"] == "ar"
                )
                ax.axhline(
                    ar, color="#555555", linestyle="--", linewidth=1.5, label="AR"
                )
            ax.set_xticks(range(4), ["D=4", "D=8", "p=0.8", "p=0.6"])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f"({chr(97 + i)}) Batch {batch}")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if metric in ("acceptance_rate", "full_accept_probability"):
                ax.set_ylim(0, 1.05)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=len(handles), frameon=False
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90), pad=0.7)
        directory = root / figure_name
        directory.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(
                directory / f"{figure_name}.{ext}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.03,
            )
        (directory / f"{figure_name}.md").write_text(
            f"# {ylabel}\n\nSource: ../summary.csv. A100 TP1, Qwen3.6 + "
            f"DSpark, 16 fixed MMLU subjects, 256 output tokens/request; one "
            f"measurement after warmup. Throughput uses wall time including "
            f"prefill; acceptance uses pooled integer verification counters. "
            f"Each panel is one batch limit. Threshold modes generate at most "
            f"eight drafts and keep a continuous prefix with each confidence "
            f">= p. All methods use synchronous scheduling and buffer "
            f"parameter 16. Full acceptance excludes zero-draft windows; "
            f"accepted length includes one bonus and conditions on nonempty "
            f"draft verification. See ../report.md for zero-draft-adjusted "
            f"metrics, parity limitations and full "
            f"configuration.\n\nReproduce: `.venv/bin/python "
            f"benchmarks/replayssm/dspark_matrix_report.py --output {root}`.\n"
        )
        plt.close(fig)
    report = (
        "# DSpark + Qwen3.6 end-to-end matrix\n"
        "\n"
        "## Contract\n"
        "\n"
        "16 fixed randomly selected MMLU subjects, one prompt per "
        "subject, seed 20260915. Each request produces exactly 256 tokens "
        "(greedy, ignore EOS, thinking disabled). Batch is the request "
        "cohort/concurrency limit; all 16 requests are run in consecutive "
        "cohorts. Each cohort is fully enqueued before scheduler release, "
        "and its actual admission width is audited. A100 TP1, BF16 model, "
        "FP32 SSM, V2 CUDA Graph. All "
        "modes use synchronous scheduling; thresholded drafts require a "
        "CPU-visible length before scheduling. Same 10 GiB KV pool, max "
        "model length 1024 and token budget 2048. Replay buffer parameter "
        "16 retains each method's native allocation semantics.\n"
        "Original ReplaySSM flush thresholds are 21 for D4 and 25 for "
        "D8/threshold modes, with a 32-slot physical ring. Dual-checkpoint "
        "uses a hard threshold of 16 and a 16-slot ring.\n"
        "\n"
        "AR has no draft policy; other methods use D4, D8, or max-eight "
        "per-position confidence >= 0.8/0.6 contiguous-prefix selection. "
        "Zero drafts are allowed. Confidence is computed for all DSpark "
        "methods; only p modes use it to truncate.\n"
        "\n"
        "Each fresh process warms one cohort for 32 output tokens per request, "
        "and the remaining prompts for one token to cover prefill shapes, "
        "with any additional short-cohort warmups recorded separately in "
        "tail_shape_warmup.json and run_notes.md, "
        "then measures the full sample set once. The timed measurement must "
        "contain no monitored JIT. "
        "Throughput counts 4096 emitted tokens per measurement "
        "divided by llm.generate wall time across cohorts, including "
        "prefill. Initialization, warmup, post-run metric reads and JSON "
        "processing are excluded. CPU-side histogram bookkeeping and "
        "request-completion log writes are present equally for all "
        "methods. No GPU tracing is enabled during measurement.\n"
        "\n"
        "## Metric definitions\n"
        "\n"
        "- Draft acceptance rate: sum accepted drafts / sum actually "
        "verified drafts.\n"
        "- Mean acceptance length: 1 + accepted drafts / nonempty draft "
        "verification rounds.\n"
        "- Full acceptance: rounds accepting every actually verified "
        "draft / nonempty draft rounds. Zero-draft rounds are excluded, "
        "not counted as vacuous full acceptance.\n"
        "- Effective decode length: 1 + accepted drafts / all decode "
        "rounds, including zero-draft rounds and excluding the first "
        "prefill-generation round per request.\n"
        "- Zero-draft fraction: zero-draft decode rounds / all decode "
        "rounds.\n"
        "- Counters are verification-level before final output-length "
        "truncation; the numerator of throughput uses only emitted "
        "tokens.\n"
        "\n"
        "## Results\n"
        "\n"
        "| Batch | Method | Policy | Tokens/s | vs AR | Acceptance | Mean "
        "length | Full accept | Zero-draft fraction | Matches AR |\n"
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: |\n"
    )

    def fmt(x):
        return "N/A" if x is None else f"{x:.3f}"

    for row in summary:
        report += f"| {row[('batch')]} | {row[('method')]} | {row[('policy')]} | {
            row[('throughput_tps')]:.2f} | {row[('speedup_vs_ar')]:.3f} | {
            fmt(row[('acceptance_rate')])
        } | {fmt(row[('mean_acceptance_length')])} | {
            fmt(row[('full_accept_probability')])
        } | {fmt(row[('zero_draft_fraction')])} | {
            row[('requests_matching_ar')]
        }/16 |\n"
    report += (
        f"\n## Verification limits\n\nAll {cell_count} requested cells "
        f"(batches {batches}) completed one measurement "
        f"without monitored timed JIT. Repeat reproducibility was not assessed. "
        f"Cross-method greedy output parity has {len(parity)} differing "
        f"request/cell pairs; see output_parity.json for first "
        f"divergences. These comparisons do not by themselves prove "
        f"bitwise equivalence or isolate the cause of any divergence. "
        f"Acceptance is measured along each method's generated trajectory. "
        f"The confidence head predicts distribution-overlap acceptance and "
        f"was not recalibrated for greedy decoding. Small fixed-sample "
        f"results are not a broad serving benchmark.\n\nOriginal ReplaySSM "
        f"V2 prefill lengths were wired into its existing first-decode "
        f"cursor reset before this matrix, preventing stale history on "
        f"request reuse. Adaptive verification transports actual prefix "
        f"lengths to the synchronous scheduler; target verification does "
        f"not run a fixed-width masked tail.\n"
    )
    (root / "report.md").write_text(report)
    (root / "matrix_complete.json").write_text(
        json.dumps(
            dict(
                cells=cell_count,
                batches=batches,
                measurements=cell_count,
                measured_output_tokens=cell_count * 4096,
                repeat_identity_passed=None,
                cohort_admission_exact=True,
                timed_jit_clean=True,
                ar_parity_mismatches=len(parity),
                strict_ar_parity_passed=not parity,
                correctness_qualified_speedup=not parity,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
