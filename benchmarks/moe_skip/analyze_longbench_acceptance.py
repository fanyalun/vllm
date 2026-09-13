# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and plot LongBench acceptance, preserving AR consistency failures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from run_long_context_acceptance import WIDTHS, digest, validate, write_json


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(run):
    contract = json.loads((run / "contract.json").read_text())
    samples = json.loads((run / "dataset.json").read_text())
    assert len(samples) == 32 and digest(samples) == contract["dataset_digest"]
    assert contract["temperature"] == 0 and contract["max_tokens"] == 512
    assert len({s["source_row_index"] for s in samples}) == 16
    for index in range(16):
        pair = [s for s in samples if s["sample_index"] == index]
        assert len(pair) == 2
        small, large = sorted(pair, key=lambda s: s["bucket"])
        assert small["source_row_index"] == large["source_row_index"]
        assert small["prefix_token_ids"] == large["prefix_token_ids"]
        assert small["suffix_token_ids"] == large["suffix_token_ids"]
        for sample in pair:
            assert len(sample["prompt_token_ids"]) == sample["bucket"]
        start = len(small["prefix_token_ids"])
        end = start + small["document_tokens_kept"]
        assert (
            small["prompt_token_ids"][start:end] == large["prompt_token_ids"][start:end]
        )
    assert (run / "GENERATION_COMPLETE").exists()
    loaded = {}
    for method, width in [("ar", 0)] + [
        (m, d) for m in ("mtp", "moe_skip") for d in WIDTHS
    ]:
        cell = run / "cells" / f"{method}_d{width}"
        assert (cell / "CELL_COMPLETE").exists(), cell
        outputs = json.loads((cell / "result.json").read_text())
        assert len(outputs) == 32
        for sample, output in zip(samples, outputs, strict=True):
            for key in ("sample_index", "source_row_index", "bucket"):
                assert output[key] == sample[key]
            assert output["prompt_sha256"] == digest(sample["prompt_token_ids"])
            assert output["prompt_tokens"] == len(sample["prompt_token_ids"])
            assert output["seed"] == contract["seed"] + sample["sample_index"]
            assert len(output["token_ids"]) == 512
            if method != "ar":
                validate(output["metrics"], width)
        loaded[method, width] = outputs

    cross_method = []
    for width in WIDTHS:
        for mtp, skip in zip(
            loaded["mtp", width], loaded["moe_skip", width], strict=True
        ):
            cross_method.append(
                {
                    "d": width,
                    "sample_index": mtp["sample_index"],
                    "bucket": mtp["bucket"],
                    "exact": mtp["token_ids"] == skip["token_ids"],
                    "mtp_output_sha256": digest(mtp["token_ids"]),
                    "moe_skip_output_sha256": digest(skip["token_ids"]),
                }
            )
    cross_status = "passed" if all(c["exact"] for c in cross_method) else "failed"
    cross_exact = sum(c["exact"] for c in cross_method)
    write_json(
        run / "cross_method_consistency.json",
        {
            "status": cross_status,
            "exact_pairs": cross_exact,
            "total_pairs": len(cross_method),
            "comparisons": cross_method,
        },
    )
    reference = loaded["ar", 0]
    requests, comparisons, summaries = [], [], []
    for method in ("mtp", "moe_skip"):
        for width in WIDTHS:
            outputs = loaded[method, width]
            for output, ar in zip(outputs, reference, strict=True):
                common = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(output["token_ids"], ar["token_ids"], strict=True)
                        )
                        if a != b
                    ),
                    512,
                )
                comparison = {
                    "method": method,
                    "d": width,
                    "sample_index": output["sample_index"],
                    "source_row_index": output["source_row_index"],
                    "bucket": output["bucket"],
                    "exact_ar": common == 512,
                    "common_prefix_tokens": common,
                    "first_different_token": None
                    if common == 512
                    else {
                        "ar": ar["token_ids"][common],
                        "speculative": output["token_ids"][common],
                    },
                }
                comparisons.append(comparison)
                m = output["metrics"]
                requests.append(
                    {
                        **{
                            k: comparison[k]
                            for k in (
                                "method",
                                "d",
                                "sample_index",
                                "source_row_index",
                                "bucket",
                                "exact_ar",
                                "common_prefix_tokens",
                            )
                        },
                        "prompt_tokens": output["prompt_tokens"],
                        "prompt_sha256": output["prompt_sha256"],
                        "output_tokens": 512,
                        "first_eos_index": output["first_eos_index"],
                        "verify_steps": m["num_spec_steps"],
                        "accepted_draft_tokens": m["num_accepted_draft_tokens"],
                        "drafted_tokens": m["num_draft_tokens"],
                        "mean_acceptance_length": m["mean_acceptance_length"],
                    }
                )
            for bucket in ("all", *contract["buckets"]):
                subset = [
                    o for o in outputs if bucket == "all" or o["bucket"] == bucket
                ]
                matches = [
                    c
                    for c in comparisons
                    if c["method"] == method
                    and c["d"] == width
                    and (bucket == "all" or c["bucket"] == bucket)
                ]
                steps = sum(o["metrics"]["num_spec_steps"] for o in subset)
                accepted = sum(
                    o["metrics"]["num_accepted_draft_tokens"] for o in subset
                )
                drafted = sum(o["metrics"]["num_draft_tokens"] for o in subset)
                summaries.append(
                    {
                        "method": method,
                        "d": width,
                        "bucket": bucket,
                        "requests": len(subset),
                        "output_tokens": len(subset) * 512,
                        "prompt_tokens_min": min(o["prompt_tokens"] for o in subset),
                        "prompt_tokens_max": max(o["prompt_tokens"] for o in subset),
                        "verify_steps": steps,
                        "accepted_draft_tokens": accepted,
                        "drafted_tokens": drafted,
                        "mean_accepted_draft_tokens": accepted / steps,
                        "mean_acceptance_length": 1 + accepted / steps,
                        "draft_acceptance_rate": accepted / drafted,
                        "zero_acceptance_fraction": sum(
                            o["metrics"]["acceptance_histogram"][0] for o in subset
                        )
                        / steps,
                        "actual_output_tokens_per_spec_step": len(subset) * 512 / steps,
                        "exact_ar_requests": sum(c["exact_ar"] for c in matches),
                        "requests_with_eos": sum(
                            o["first_eos_index"] is not None for o in subset
                        ),
                    }
                )
    write_csv(run / "acceptance_summary.csv", summaries)
    write_csv(run / "acceptance_by_request.csv", requests)
    indexed = {
        (r["method"], r["d"], r["sample_index"], r["bucket"]): r for r in requests
    }
    paired = []
    for method in ("mtp", "moe_skip"):
        for width in WIDTHS:
            for index in range(16):
                short = indexed[method, width, index, 16384]
                long = indexed[method, width, index, 32768]
                paired.append(
                    {
                        "method": method,
                        "d": width,
                        "sample_index": index,
                        "source_row_index": short["source_row_index"],
                        "acceptance_length_16k": short["mean_acceptance_length"],
                        "acceptance_length_32k": long["mean_acceptance_length"],
                        "delta_32k_minus_16k": long["mean_acceptance_length"]
                        - short["mean_acceptance_length"],
                    }
                )
    write_csv(run / "paired_deltas.csv", paired)
    write_csv(
        run / "sample_manifest.csv",
        [
            {
                "sample_index": s["sample_index"],
                "source_row_index": s["source_row_index"],
                "bucket": s["bucket"],
                "ref_prompt_tokens": s["ref_prompt_tokens"],
                "prompt_tokens": len(s["prompt_token_ids"]),
                "prompt_sha256": s["prompt_sha256"],
            }
            for s in samples
        ],
    )
    parity = {
        "status": "passed" if all(c["exact_ar"] for c in comparisons) else "failed",
        "exact_requests": sum(c["exact_ar"] for c in comparisons),
        "total_requests": len(comparisons),
        "cross_method_status": cross_status,
        "cross_method_exact_pairs": cross_exact,
        "cross_method_total_pairs": len(cross_method),
        "comparisons": comparisons,
    }
    write_json(run / "output_consistency.json", parity)
    if parity["status"] == "failed":
        write_json(
            run / "gates_failed.json",
            {
                "gate": "exact_ar_output",
                "details": "output_consistency.json",
                "note": (
                    "Acceptance measurements are complete; "
                    "strict AR parity did not pass."
                ),
            },
        )
    write_json(
        run / "audit.json",
        {
            "status": "passed",
            "scope": "coverage, identity, integer metrics",
            "speculative_cells": 16,
            "speculative_requests": 256,
            "speculative_output_tokens": 131072,
            "ar_requests": 32,
            "ar_output_tokens": 16384,
            "output_consistency_status": parity["status"],
            "cross_method_consistency_status": cross_status,
        },
    )
    return summaries, parity


def plot_and_report(run, summaries, parity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.labelsize": 13,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for axis, bucket, label in zip(
        axes,
        (16384, 32768),
        ("(a) 16K context", "(b) 32K context"),
        strict=True,
    ):
        for method, title, color, marker in (
            ("mtp", "MTP", "#4C78A8", "o"),
            ("moe_skip", "MoE-Skip (top-4)", "#F2B447", "s"),
        ):
            values = [
                r for r in summaries if r["method"] == method and r["bucket"] == bucket
            ]
            axis.plot(
                [r["d"] for r in values],
                [r["mean_acceptance_length"] for r in values],
                label=title,
                color=color,
                marker=marker,
                linewidth=1.8,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(WIDTHS, [str(d) for d in WIDTHS])
        axis.set_xlabel("Draft length D\n" + label)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean acceptance length")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=0.4, w_pad=1)
    directory = run / "longbench_acceptance"
    directory.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            directory / f"longbench_acceptance.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    notes = f"""# LongBench greedy acceptance pilot

Qwen3.6-35B-A3B, MoE-Skip top-h=4 versus native MTP, D=4/8/16/32.
Data: sfc-gh-goliaro/longbench-longctx, pinned revision in contract.json.
Selection: all 16 source rows from nominal buckets >=32K, in source order.
Each source sample produces two inputs of exactly 16,384 and 32,768 Qwen tokens.
Only the document is prefix-truncated; instruction, question, choices and
default chat template are identical between each pair. Tokenized prefix,
document prefix and question/chat suffix are concatenated to fit each budget.
There is no thinking override. The template opens a think block.
Each request generates exactly 512 tokens with temperature=0, top_p=1,
ignore_eos=True. First EOS positions are preserved in per-request artifacts.
TP=1, B=1, CUDA graphs, A100 80GB, prefix caching disabled, text only.

Mean acceptance length = 1 + total accepted draft tokens / total spec steps.
Aggregation is weighted by verification steps. The +1 is conventional and
can differ from exact emitted yield at the generation boundary. CSV files
also report accepted-only means, acceptance fractions and emitted-token ratios.
Panels show paired 16K/32K contexts; each point aggregates 16 requests.
This controls task identity, but shortening a document may remove answer evidence.
No throughput, LongBench answer-accuracy score or confidence interval is claimed.

AR output consistency: **{parity["status"]}**, {parity["exact_requests"]}/256
speculative requests match all 512 AR tokens. Detailed first divergences are
in output_consistency.json; a failed check is retained in gates_failed.json.
Acceptance-count audit and output-equivalence checks are separate.

At matching D, MoE-Skip versus MTP output consistency:
**{parity["cross_method_status"]}**,
{parity["cross_method_exact_pairs"]}/{parity["cross_method_total_pairs"]} pairs match
all 512 tokens. See cross_method_consistency.json. This checks the two methods'
generated trajectories separately from their equivalence to AR.

Artifacts in the run directory: acceptance_summary.csv, acceptance_by_request.csv,
sample_manifest.csv, cells/*/result.json, contract.json and audit.json.
AI assistance was used to prepare the benchmark and report.

Reproduce from the repository root, using the included pinned source.parquet:

```bash
.venv/bin/python benchmarks/moe_skip/run_longbench_acceptance.py \\
  --run-dir RUN_DIR --source SOURCE_PARQUET
MPLCONFIGDIR=/tmp/longbench_mpl .venv/bin/python \\
  benchmarks/moe_skip/analyze_longbench_acceptance.py --run-dir RUN_DIR
```
"""
    (directory / "longbench_acceptance.md").write_text(notes)
    table = ["| Bucket | D | MTP | MoE-Skip |", "| --- | ---: | ---: | ---: |"]
    for bucket in ("all", 16384, 32768):
        for width in WIDTHS:
            values = {
                r["method"]: r["mean_acceptance_length"]
                for r in summaries
                if r["bucket"] == bucket and r["d"] == width
            }
            table.append(
                f"| {bucket} | {width} | {values['mtp']:.4f} | "
                f"{values['moe_skip']:.4f} |"
            )
    (run / "RESULTS.md").write_text(notes + "\n" + "\n".join(table) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries, parity = audit(args.run_dir)
    plot_and_report(args.run_dir, summaries, parity)
    (args.run_dir / "MEASUREMENT_COMPLETE").write_text(
        f"256 spec + 32 AR requests audited; AR parity {parity['status']}\n"
    )
    files = [
        p
        for p in args.run_dir.rglob("*")
        if p.is_file() and p.name != "fingerprints.json"
    ]
    files += [
        Path(__file__),
        Path(__file__).with_name("run_longbench_acceptance.py"),
        Path(__file__).with_name("run_long_context_acceptance.py"),
    ]
    write_json(
        args.run_dir / "fingerprints.json",
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
    )
    print(json.dumps({k: v for k, v in parity.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
