# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and visualize the small top-p expert-budget experiment."""

import argparse
import json
from pathlib import Path

from analyze_static_budget import emitted_accepted_tokens, write_csv
from run_static_budget import sha256, validate_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    rows, layers, parity, timing = [], [], [], []
    for model in ("qwen36", "gemma4"):
        folder = root / model
        contract = json.loads((folder / "contract.json").read_text())
        assert contract["model"] == model
        for name, digest in contract["source_sha256"].items():
            assert sha256(folder / name) == digest
        assert (
            sha256(Path(contract["model_path"]) / "config.json")
            == contract["model_config_sha256"]
        )
        samples = [json.loads(s) for s in contract["dataset_payload"].splitlines()]
        assert (folder / "dataset.jsonl").read_text() == contract["dataset_payload"]
        results = {}
        for cell in contract["cells"]:
            directory = folder / cell["name"]
            assert (directory / "CELL_COMPLETE").exists(), directory
            log_path = sorted(directory.glob("run_*.log"))[-1]
            measured = log_path.read_text().split("WARMUP_COMPLETE", 1)
            assert len(measured) == 2, log_path
            warnings = [
                line
                for line in measured[1].splitlines()
                if "JIT compilation during inference" in line
            ]
            timing.append(
                {
                    "model": model,
                    "method": cell["name"],
                    "post_warmup_jit_warnings": warnings,
                }
            )
            result = json.loads((directory / "result.json").read_text())
            assert all(result[k] == v for k, v in cell.items())
            assert result["model"] == model
            assert result["model_path"] == contract["model_path"]
            assert result["source_sha256"] == contract["source_sha256"]
            assert result["dataset_sha256"] == sha256(folder / "dataset.jsonl")
            assert len(result["outputs"]) == len(samples) == 4
            for output, sample in zip(result["outputs"], samples, strict=True):
                assert output["sample_index"] == sample["sample_index"]
                assert output["prompt_sha256"] == sample["prompt_sha256"]
                assert len(output["token_ids"]) == 128
                if cell["h"]:
                    validate_metrics(output["spec_decode_metrics"], cell["d"])
            assert (
                abs(
                    sum(o["e2e_seconds"] for o in result["outputs"])
                    - result["e2e_seconds"]
                )
                < 1e-8
            )
            results[cell["name"]] = result
        ar_seconds = sum(results[c]["e2e_seconds"] for c in ("ar_start", "ar_end")) / 2
        for name, result in results.items():
            outputs = result["outputs"]
            accepted = steps = emitted = 0
            mean_h = float(result["h"])
            if result["h"]:
                accepted = sum(
                    o["spec_decode_metrics"]["num_accepted_draft_tokens"]
                    for o in outputs
                )
                steps = sum(o["spec_decode_metrics"]["num_spec_steps"] for o in outputs)
                emitted = sum(emitted_accepted_tokens(o) for o in outputs)
            if result["p"] is not None:
                hist = [0] * 8
                for worker in result["budget_histograms"]:
                    for layer, counts in worker.items():
                        if not sum(counts):
                            continue
                        assert len(counts) == 8
                        assert sum(counts) == (steps + len(outputs)) * result["d"]
                        hist = [a + b for a, b in zip(hist, counts, strict=True)]
                        layers.append(
                            {
                                "model": model,
                                "method": name,
                                "layer": layer,
                                **{f"h{i + 1}": c for i, c in enumerate(counts)},
                                "mean_h": sum((i + 1) * c for i, c in enumerate(counts))
                                / sum(counts),
                            }
                        )
                assert sum(hist) > 0
                mean_h = sum((i + 1) * c for i, c in enumerate(hist)) / sum(hist)
                if result["p"] == 1:
                    assert mean_h == 8
            rows.append(
                {
                    "model": model,
                    "method": name,
                    "d": result["d"],
                    "p": result["p"],
                    "mean_h": mean_h,
                    "acceptance_length": accepted / steps if steps else None,
                    "accepted_draft_tokens": accepted,
                    "emitted_accepted_tokens": emitted,
                    "spec_steps": steps,
                    "e2e_seconds": result["e2e_seconds"],
                    "ms_per_output_token": result["e2e_seconds"] * 1000 / 512,
                    "ms_per_emitted_accepted_token": result["e2e_seconds"]
                    * 1000
                    / emitted
                    if emitted
                    else None,
                    "speedup_vs_ar_async": ar_seconds / result["e2e_seconds"],
                }
            )
            for reference in ("ar_start", "h8"):
                matches = sum(
                    a["token_ids"] == b["token_ids"]
                    for a, b in zip(outputs, results[reference]["outputs"], strict=True)
                )
                parity.append(
                    {
                        "model": model,
                        "method": name,
                        "reference": reference,
                        "matching_requests": matches,
                        "total_requests": 4,
                    }
                )
    write_csv(root / "summary.csv", rows)
    write_csv(root / "layer_budgets.csv", layers)
    write_csv(root / "output_parity.csv", parity)
    write_json(
        root / "audit.json",
        {
            "measurement_contract": "passed",
            "cells": len(rows),
            "output_parity": parity,
            "p1_output_identity_gate": {
                p["model"]: "passed" if p["matching_requests"] == 4 else "failed"
                for p in parity
                if p["method"] == "p1_control" and p["reference"] == "h8"
            },
            "timing": timing,
            "timing_jit_gate": "failed"
            if any(t["post_warmup_jit_warnings"] for t in timing)
            else "passed",
        },
    )
    plot(root, rows)
    report(root, rows, parity)
    print(json.dumps(rows, indent=2))


def report(root, rows, parity):
    lines = [
        "# Top-p expert-budget pilot",
        "",
        "Measured Qwen3.6 (GPU 0, D=8) and Gemma4 (GPU 1, D=4), "
        "each on four prompts, one per category, with 128 output tokens each. "
        "B=1, TP=1, greedy, seed=0, ignore_eos, CUDA graphs, no prefix cache. "
        "All configurations use the same model-specific prompts and settings. "
        "The two models run independently on two A100 80GB PCIe GPUs.",
        "",
        "Top-p is applied inside native top-8, separately for every draft "
        "token and MoE layer. It keeps the shortest probability-mass prefix "
        "reaching p, renormalizes retained weights, and preserves Gemma's "
        "per-expert scale. Target routing is unchanged. Target and Draft "
        "share the same model instance and weights. Invalid expert IDs skip "
        "expert GEMMs; the routing tensor retains eight slots. Extra routing "
        "and histogram overhead are included in measured wall time. "
        "This is a benchmark worker extension, not a production configuration.",
        "",
        "Time is the sum of llm.generate wall times, including prefill and "
        "API overhead, excluding model initialization and four full warmup "
        "requests. Acceptance length excludes the recovery/bonus token. "
        "ms/accepted divides the entire request time by actually emitted "
        "accepted draft tokens, clipping the final speculative step. "
        "ms/output uses all 512 output tokens. Mean h includes all generated "
        "draft token-layer invocations, including the final unused proposal. "
        "AR speedup uses the mean of start/end async controls.",
        "",
        "Only four samples per configuration, no repeated speculative timing "
        "trials or confidence intervals. Do not extrapolate to the previous "
        "16x512 matrix or interpret small timing differences as established gains.",
        "",
    ]
    for model in ("qwen36", "gemma4"):
        data = {r["method"]: r for r in rows if r["model"] == model}
        ar = (
            data["ar_start"]["ms_per_output_token"]
            + data["ar_end"]["ms_per_output_token"]
        ) / 2
        drift = (
            data["ar_end"]["ms_per_output_token"]
            / data["ar_start"]["ms_per_output_token"]
            - 1
        ) * 100
        lines += [
            f"## {model}",
            "",
            f"AR (Async): {ar:.3f} ms/output token; "
            f"end/start time change: {drift:+.2f}%.",
            "",
            "| Method | Mean h | Accepted/step | ms/output | "
            "ms/accepted | AR speedup |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in ("h2", "h4", "h6", "h8", "p07", "p08", "p09", "p1_control"):
            r = data[name]
            lines.append(
                f"| {name} | {r['mean_h']:.3f} | "
                f"{r['acceptance_length']:.3f} | "
                f"{r['ms_per_output_token']:.3f} | "
                f"{r['ms_per_emitted_accepted_token']:.3f} | "
                f"{r['speedup_vs_ar_async']:.3f}x |"
            )
        control = next(
            p
            for p in parity
            if p["model"] == model
            and p["method"] == "p1_control"
            and p["reference"] == "h8"
        )
        lines += [
            "",
            "p=1 versus native h=8 exact output matches: "
            f"{control['matching_requests']}/4 requests.",
            "",
        ]
    lines += [
        "## Validation and limitations",
        "",
        "See audit.json for model/dataset/source identity, acceptance counter "
        "recomputation, routing invocation counts, warmup/JIT checks and "
        "per-cell output parity. GPU primitive validation is recorded in "
        "gpu_reference_check.log. It tests unsorted expert slots, tied logits, "
        "budget histograms, per-expert scaling, p=1 identity, and masked "
        "expert GEMMs against a zero-weight reference across 32 cases. "
        "Two unsupported aligned-assignment shapes must raise ValueError.",
        "",
        "The eight-slot invalid-expert representation is restricted to naive "
        "MoE assignment (4 * tokens * 8 <= number of experts), which includes "
        "the measured B=1 draft path. An unsorted-slot test exposed incorrect "
        "results in the larger aligned-assignment path; it is now rejected. "
        "See unsupported_alignment_probe.log and post_measurement_guard.patch. "
        "The main timing matrix uses the frozen worker snapshots before this "
        "Python shape guard; the GPU kernel is unchanged. guard_validation "
        "contains additional p=0.8 model runs with the final guarded worker "
        "and guard_validation_audit.json records their checks. These runs "
        "do not enter the main timing table.",
        "",
        "Gemma output reproducibility is unresolved: prior-attempt unmodified "
        "AR and h8 repeats each matched 0/4 requests against this run. "
        "These controls did not enable top-p. See gemma_repeat_control.json "
        "and prior_attempt_controls for copied results, exact source snapshots "
        "and provenance. This limits Gemma quality and fine-grained performance "
        "interpretation; no correctness-preserving serving claim is made.",
        "",
        "The first attempt stopped on a benchmark extension import error. "
        "The corrected driver ran all 20 cells afresh under this directory. "
        "Only the explicitly archived native Gemma controls are reused for "
        "the reproducibility check; they do not enter the main timing table.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python benchmarks/moe_skip/check_top_p.py",
        ".venv/bin/python benchmarks/moe_skip/run_top_p_probe.py "
        "--model qwen36 --gpu 0 --run-dir <fresh_directory>",
        ".venv/bin/python benchmarks/moe_skip/run_top_p_probe.py "
        "--model gemma4 --gpu 1 --run-dir <fresh_directory>",
        ".venv/bin/python benchmarks/moe_skip/analyze_top_p_probe.py <fresh_directory>",
        "```",
        "",
        "Each figure directory contains matching PNG, PDF and Markdown files. "
        "summary.csv holds aggregate metrics; layer_budgets.csv holds per-layer "
        "h=1..8 counts; output_parity.csv holds exact request-level comparisons. "
        "AI assistance was used for the implementation, experiments and analysis.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")


def plot(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 13,
            "axes.labelsize": 14,
            "pdf.fonttype": 42,
        }
    )
    methods = ["h2", "h4", "h6", "h8", "p07", "p08", "p09", "p1_control"]
    labels = ["h=2", "h=4", "h=6", "h=8", "p=.7", "p=.8", "p=.9", "p=1"]
    metrics = [
        ("ms_per_output_token", "ms / output token"),
        ("ms_per_emitted_accepted_token", "ms / accepted token"),
        ("acceptance_length", "Accepted tokens / step"),
        ("mean_h", "Mean experts / token / layer"),
    ]
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), layout="constrained")
        for index, (ax, model) in enumerate(zip(axes, ("qwen36", "gemma4"))):
            data = {r["method"]: r for r in rows if r["model"] == model}
            ax.bar(
                labels,
                [data[m][metric] for m in methods],
                color=["#4C78A8"] * 4 + ["#F2B447"] * 3 + ["#999999"],
                width=0.72,
            )
            if metric == "ms_per_output_token":
                ar = (data["ar_start"][metric] + data["ar_end"][metric]) / 2
                ax.axhline(ar, color="#E45756", linestyle="--", label="AR (Async)")
            ax.set_ylabel(ylabel)
            ax.set_xlabel(
                f"({chr(97 + index)}) "
                + ("Qwen3.6, D=8" if index == 0 else "Gemma4, D=4")
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
        if metric == "ms_per_output_token":
            fig.legend(
                *axes[0].get_legend_handles_labels(),
                loc="outside upper right",
                frameon=False,
            )
        name = "top_p_" + metric
        folder = root / "figures" / name
        folder.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(
                folder / f"{name}.{extension}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.03,
            )
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# {ylabel}\n\nSource: ../../summary.csv.\n\n"
            "Qwen3.6 D=8; Gemma4 D=4; B=1; greedy; four prompts "
            "(one per category), 128 output tokens each; A100 80GB, TP=1. "
            "All requests are warmed up before measurement. Wall time includes "
            "prefill, decode, API and detailed speculative metrics. "
            "Top-p uses native top-8 probability mass with retained-weight "
            "renormalization, h=1..8. Dynamic routing and histogram overhead "
            "are included. p=1 is an implementation control. "
            "AR is the mean of start/end async measurements. No error bars; "
            "this is a small exploratory run. Accepted-token time divides "
            "the entire wall time by actually emitted accepted draft tokens "
            "after final-step clipping; AR has no such denominator. "
            "Mean h is weighted over measured draft token-layer invocations.\n\n"
            "Reproduce: `.venv/bin/python benchmarks/moe_skip/analyze_top_p_probe.py "
            f"{root}`.\n"
        )


if __name__ == "__main__":
    main()
