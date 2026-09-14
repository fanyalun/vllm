# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit static-budget measurements and plot acceptance versus wall time."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from run_static_budget import sha256, validate_metrics, write_json


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def emitted_accepted_tokens(output):
    """Clip the final speculative step after the single prefill output token."""
    counts = output["spec_decode_metrics"]["per_step_accepted"]
    previous_outputs = 1 + sum(a + 1 for a in counts[:-1])
    remaining = len(output["token_ids"]) - previous_outputs
    if not 1 <= remaining <= counts[-1] + 1:
        raise RuntimeError("Output trace violates single-prefill/final-clip contract")
    return sum(counts[:-1]) + min(counts[-1], remaining)


def analyze(root, models):
    rows, requests, positions, parity = [], [], [], []
    runtime = []
    for model in models:
        folder = root / model
        contract = json.loads((folder / "contract.json").read_text())
        if sha256(folder / "runner_snapshot.py") != contract["script_sha256"]:
            raise RuntimeError("Runner snapshot fingerprint changed")
        if (
            sha256(Path(contract["model_path"]) / "config.json")
            != contract["model_config_sha256"]
        ):
            raise RuntimeError("Model configuration fingerprint changed")
        expected = {f"h{h}_d{d}" for h in (2, 4, 6, 8) for d in (4, 8, 16, 32)} | {
            "ar_start",
            "ar_end",
        }
        if (
            len(contract["cells"]) != 18
            or {cell["name"] for cell in contract["cells"]} != expected
        ):
            raise RuntimeError("Requested budget matrix is incomplete")
        dataset = Path(contract["dataset"])
        if sha256(dataset) != contract["dataset_sha256"]:
            raise RuntimeError("Dataset hash changed")
        samples = [json.loads(s) for s in dataset.read_text().splitlines()]
        if len(samples) != 16:
            raise RuntimeError("Expected exactly 16 samples")
        baseline = json.loads((folder / "ar_start/result.json").read_text())
        end = json.loads((folder / "ar_end/result.json").read_text())
        baseline_seconds = (baseline["e2e_seconds"] + end["e2e_seconds"]) / 2
        for cell in contract["cells"]:
            directory = folder / cell["name"]
            if not (directory / "CELL_COMPLETE").exists():
                raise RuntimeError(f"Incomplete cell: {directory}")
            result = json.loads((directory / "result.json").read_text())
            log = (directory / "run.log").read_text()
            measured_log = log.rsplit("WARMUP_COMPLETE", 1)
            if len(measured_log) != 2:
                raise RuntimeError(f"Missing warmup boundary: {directory}")
            jit = [
                line
                for line in measured_log[1].splitlines()
                if "JIT compilation during inference" in line
            ]
            runtime.append(
                {"model": model, "cell": cell["name"], "post_warmup_jit_warnings": jit}
            )
            outputs = result["outputs"]
            if (
                len(outputs) != 16
                or result["h"] != cell["h"]
                or (result["d"] != cell["d"])
            ):
                raise RuntimeError(f"Cell contract mismatch: {directory}")
            steps = accepted = emitted_accepted = drafted = matches = 0
            all_a, all_n = [], []
            for sample, output, ref in zip(
                samples, outputs, baseline["outputs"], strict=True
            ):
                if (
                    len(output["token_ids"]) != 512
                    or output["prompt_sha256"] != sample["prompt_sha256"]
                    or output["sample_index"] != sample["sample_index"]
                    or output["prompt_tokens"] != sample["prompt_token_count"]
                    or hashlib.sha256(sample["prompt"].encode()).hexdigest()
                    != sample["prompt_sha256"]
                    or not math.isfinite(output["e2e_seconds"])
                    or output["e2e_seconds"] <= 0
                ):
                    raise RuntimeError(f"Invalid request: {directory}")
                equal = output["token_ids"] == ref["token_ids"]
                matches += equal
                first = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(output["token_ids"], ref["token_ids"], strict=True)
                        )
                        if a != b
                    ),
                    None,
                )
                parity.append(
                    {
                        "model": model,
                        "cell": cell["name"],
                        "sample_index": sample["sample_index"],
                        "exact_ar_match": equal,
                        "first_mismatch_index": first,
                    }
                )
                request = {
                    "model": model,
                    "cell": cell["name"],
                    "h": cell["h"],
                    "d": cell["d"],
                    "sample_index": sample["sample_index"],
                    "category": sample["category"],
                    "e2e_seconds": output["e2e_seconds"],
                    "output_tokens": 512,
                    "exact_ar_match": equal,
                    "spec_steps": 0,
                    "accepted_draft_tokens": 0,
                    "emitted_accepted_draft_tokens": 0,
                    "draft_tokens": 0,
                }
                if cell["h"]:
                    metrics = output["spec_decode_metrics"]
                    validate_metrics(metrics, cell["d"])
                    steps += metrics["num_spec_steps"]
                    accepted += metrics["num_accepted_draft_tokens"]
                    actual_accepted = emitted_accepted_tokens(output)
                    emitted_accepted += actual_accepted
                    drafted += metrics["num_draft_tokens"]
                    all_a.extend(metrics["per_step_accepted"])
                    all_n.extend(metrics["per_step_drafted"])
                    request.update(
                        spec_steps=metrics["num_spec_steps"],
                        accepted_draft_tokens=metrics["num_accepted_draft_tokens"],
                        emitted_accepted_draft_tokens=actual_accepted,
                        draft_tokens=metrics["num_draft_tokens"],
                    )
                requests.append(request)
                request["ms_per_output_token"] = output["e2e_seconds"] * 1000 / 512
                request["ms_per_accepted_draft_token"] = (
                    output["e2e_seconds"]
                    * 1000
                    / request["emitted_accepted_draft_tokens"]
                    if request["emitted_accepted_draft_tokens"]
                    else None
                )
            total = sum(o["e2e_seconds"] for o in outputs)
            if not math.isclose(total, result["e2e_seconds"], abs_tol=1e-9):
                raise RuntimeError("Wall-time aggregate mismatch")
            for j in range(1, cell["d"] + 1):
                reached = sum(
                    a >= j - 1 and n >= j for a, n in zip(all_a, all_n, strict=True)
                )
                accepted_here = sum(a >= j for a in all_a)
                first_reject = sum(
                    a == j - 1 and n >= j for a, n in zip(all_a, all_n, strict=True)
                )
                positions.append(
                    {
                        "model": model,
                        "h": cell["h"],
                        "d": cell["d"],
                        "position": j,
                        "reached": reached,
                        "accepted": accepted_here,
                        "first_nonaccepted": first_reject,
                        "conditional_acceptance": accepted_here / reached
                        if reached
                        else None,
                    }
                )
            rows.append(
                {
                    "model": model,
                    "cell": cell["name"],
                    "h": cell["h"],
                    "d": cell["d"],
                    "requests": 16,
                    "output_tokens": 8192,
                    "e2e_seconds": total,
                    "mean_request_seconds": total / 16,
                    "output_tokens_per_second": 8192 / total,
                    "speedup_vs_ar": baseline_seconds / total,
                    "spec_steps": steps,
                    "accepted_draft_tokens": accepted,
                    "emitted_accepted_draft_tokens": emitted_accepted,
                    "accepted_tokens_discarded_by_output_cap": accepted
                    - emitted_accepted,
                    "draft_tokens": drafted,
                    "ms_per_verified_accepted_draft_token": total * 1000 / accepted
                    if accepted
                    else None,
                    "ms_per_accepted_draft_token": total * 1000 / emitted_accepted
                    if emitted_accepted
                    else None,
                    "ms_per_output_token": total * 1000 / 8192,
                    "draft_acceptance_rate": accepted / drafted if drafted else None,
                    "accepted_draft_per_step": accepted / steps if steps else None,
                    "mean_acceptance_length": 1 + accepted / steps if steps else None,
                    "output_tokens_per_spec_step": 8192 / steps if steps else None,
                    "exact_ar_requests": matches,
                    "ar_end_vs_start_speed_ratio": baseline["e2e_seconds"]
                    / end["e2e_seconds"],
                }
            )
    write_csv(root / "summary.csv", rows)
    write_csv(root / "requests.csv", requests)
    write_csv(root / "position_acceptance.csv", positions)
    write_csv(root / "output_consistency.csv", parity)
    write_json(root / "summary.json", rows)
    write_json(root / "runtime_audit.json", runtime)
    failures = [r for r in rows if r["exact_ar_requests"] != 16]
    write_json(
        root / "audit.json",
        {
            "measurement_coverage": "passed",
            "cells": len(rows),
            "requests": len(requests),
            "output_tokens": len(requests) * 512,
            "counter_consistency": "passed",
            "emitted_accepted_reconstruction": "passed",
            "ar_repeat_controls": [
                {
                    k: r[k]
                    for k in (
                        "model",
                        "exact_ar_requests",
                        "ar_end_vs_start_speed_ratio",
                    )
                }
                for r in rows
                if r["cell"] == "ar_end"
            ],
            "strict_greedy_ar_parity": "failed" if failures else "passed",
            "parity_failed_cells": [
                {k: r[k] for k in ("model", "cell", "exact_ar_requests")}
                for r in failures
            ],
            "scope": "standalone MoE-Skip only; no Pre-Verify measurements",
            "timing_jit_gate": "failed"
            if any(r["post_warmup_jit_warnings"] for r in runtime)
            else "passed",
        },
    )
    return rows


def report(root, rows, models):
    lines = [
        "# Static expert budget results",
        "",
        "All costs below use summed measured request wall time. "
        "Accepted-token cost excludes correction/bonus from its denominator. "
        "Output-token cost includes all 8192 final tokens. "
        "See audit.json for coverage, timing and strict output parity.",
        "",
    ]
    failed = [r for r in rows if r["h"] and r["exact_ar_requests"] != 16]
    lines += [
        f"Strict greedy AR parity failed in {len(failed)} budget cells. "
        "The curves describe measured costs and do not establish lossless "
        "speedups. Each budget cell ran once; small differences should not be "
        "interpreted as stable optima.",
        "",
    ]
    for model in models:
        selected = sorted(
            (r for r in rows if r["model"] == model and r["h"]),
            key=lambda r: (r["d"], r["h"]),
        )
        lines += [
            f"## {model}",
            "",
            "| D | h | Accept rate | Accept length (with bonus) | "
            "ms/actual accepted draft token | ms/output token | tok/s | "
            "Speedup/sync AR | Exact AR requests |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in selected:
            cost = r["ms_per_accepted_draft_token"]
            cost_text = f"{cost:.4f}" if cost is not None else "undefined"
            lines.append(
                f"| {r['d']} | {r['h']} | {r['draft_acceptance_rate']:.2%} | "
                f"{r['mean_acceptance_length']:.4f} | {cost_text} | "
                f"{r['ms_per_output_token']:.4f} | "
                f"{r['output_tokens_per_second']:.3f} | "
                f"{r['speedup_vs_ar']:.3f} | {r['exact_ar_requests']}/16 |"
            )
        best_output = min(selected, key=lambda r: r["ms_per_output_token"])
        best_accepted = min(
            (r for r in selected if r["ms_per_accepted_draft_token"] is not None),
            key=lambda r: r["ms_per_accepted_draft_token"],
        )
        lines += [
            "",
            f"Lowest output cost: {best_output['cell']}; "
            f"lowest accepted-draft cost: {best_accepted['cell']}.",
            "",
            "AR end/start throughput ratio: "
            f"{selected[0]['ar_end_vs_start_speed_ratio']:.4f}.",
            "",
        ]
    (root / "results.md").write_text("\n".join(lines))


def plot(root, rows, models):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#4C78A8", "#F2B447", "#59A14F", "#E45756"]
    markers = ["o", "s", "^", "D"]
    for model in models:
        folder = root / f"{model}_static_budget"
        folder.mkdir(exist_ok=True)
        name = folder.name
        fig, axes = plt.subplots(3, 2, figsize=(8, 8.7), layout="constrained")
        for d, color, marker in zip((4, 8, 16, 32), colors, markers, strict=True):
            selected = sorted(
                (r for r in rows if r["model"] == model and r["d"] == d),
                key=lambda r: r["h"],
            )
            h = [r["h"] for r in selected]
            for axis, metric in zip(
                axes.flat,
                (
                    "output_tokens_per_second",
                    "draft_acceptance_rate",
                    "mean_acceptance_length",
                    "output_tokens_per_second",
                    "ms_per_accepted_draft_token",
                    "ms_per_output_token",
                ),
                strict=True,
            ):
                x = (
                    [r["mean_acceptance_length"] for r in selected]
                    if (axis is axes[1, 1])
                    else h
                )
                axis.plot(
                    x,
                    [r[metric] for r in selected],
                    color=color,
                    marker=marker,
                    label=f"D={d}",
                    linewidth=1.6,
                )
            for r in selected:
                if r["h"] not in (2, 8):
                    continue
                axes[1, 1].annotate(
                    str(r["h"]),
                    (r["mean_acceptance_length"], r["output_tokens_per_second"]),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=9,
                )
        ar = [r for r in rows if r["model"] == model and r["h"] == 0]
        ar_tps = 8192 / (sum(r["e2e_seconds"] for r in ar) / len(ar))
        for axis in (axes[0, 0], axes[1, 1]):
            axis.axhline(ar_tps, color="0.45", linestyle="--", label="AR (sync)")
            axis.axhspan(
                min(r["output_tokens_per_second"] for r in ar),
                max(r["output_tokens_per_second"] for r in ar),
                color="0.5",
                alpha=0.12,
            )
        axes[2, 1].axhline(1000 / ar_tps, color="0.45", linestyle="--")
        axes[2, 1].axhspan(
            min(r["ms_per_output_token"] for r in ar),
            max(r["ms_per_output_token"] for r in ar),
            color="0.5",
            alpha=0.12,
        )
        labels = [
            ("(a) Throughput", "Output tokens/s"),
            ("(b) Draft acceptance", "Accepted / drafted"),
            ("(c) Acceptance length", "1 + accepted / steps"),
            ("(d) Budget trade-off", "Output tokens/s"),
            ("(e) Accepted-token cost", "ms / accepted draft token"),
            ("(f) Final-output cost", "ms / output token"),
        ]
        for index, (axis, (label, ylabel)) in enumerate(
            zip(axes.flat, labels, strict=True)
        ):
            axis.set_ylabel(ylabel)
            axis.set_xlabel(
                ("Expert budget h" if index != 3 else "Mean acceptance length")
                + "\n"
                + label
            )
            if index != 3:
                axis.set_xticks([2, 4, 6, 8])
            axis.grid(alpha=0.2)
        axes[0, 0].legend(ncol=2)
        axes[0, 1].set_ylim(0, 1.03)
        axes[1, 1].margins(y=0.12)
        for extension in ("png", "pdf"):
            fig.savefig(
                folder / f"{name}.{extension}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        plt.close(fig)
        (folder / f"{name}.md").write_text(
            f"# {model}: static expert budget\n\n"
            "Source: ../summary.csv and per-cell result.json.\n\n"
            "16 identical ordered prompts (4 per category), 512 output tokens, "
            "B=1, TP1, greedy, seed 0, CUDA Graph, prefix cache off, async "
            "scheduling off. Two warmup requests excluded. Each cell runs once. "
            "Detailed acceptance collection is included in wall time.\n\n"
            "(a) Total output tokens divided by summed request wall time, "
            "including prefill/decode/API. (b) Accepted draft tokens / drafted "
            "tokens. (c) 1 + accepted draft tokens / speculative steps. "
            "(d) Throughput versus acceptance length; endpoint labels are h. "
            "AR uses mean elapsed time of fresh start/end baselines. "
            "Gray bands span the two AR endpoint measurements, not confidence "
            "intervals. AR repeat output agreement is reported in audit.json. "
            "(e) Summed request wall time / summed actually emitted accepted "
            "draft tokens in ms, excluding correction/bonus and accepted tokens "
            "discarded by the final output cap from the denominator. "
            "(f) Summed request wall time / 8192 final output tokens in ms. "
            "Both costs include draft, verification, state, prefill and API "
            "overhead; they are amortized costs, not timestamps of individual "
            "accepted tokens, and are ratios of totals, not means of ratios. "
            "No error bars; no repeated-cell uncertainty estimate.\n\n"
            "Acceptance length is a counter convention, not exact final yield: "
            "output limits may truncate the final step. First-nonaccepted "
            "positions in position_acceptance.csv use pre-clipping counters. "
            "Consult ../audit.json and ../output_consistency.csv for strict "
            "greedy parity; differing outputs prevent a lossless speedup claim.\n\n"
            "Reproduce: `.venv/bin/python benchmarks/moe_skip/"
            f"analyze_static_budget.py --run-dir {root} "
            f"--models {' '.join(models)}`\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen36", "gemma4"])
    args = parser.parse_args()
    root = args.run_dir.resolve()
    rows = analyze(root, args.models)
    report(root, rows, args.models)
    plot(root, rows, args.models)
    plot_costs(root, rows, args.models)
    (root / "MEASUREMENTS_AUDITED").write_text(
        "Coverage and counters passed; see audit.json for output parity\n"
    )


def plot_costs(root, rows, models):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    folder = root / "token_costs"
    folder.mkdir(exist_ok=True)
    metrics = ["ms_per_accepted_draft_token", "ms_per_output_token"]
    labels = ["ms / accepted token", "ms / output token"]
    fig, axes = plt.subplots(
        len(models),
        2,
        figsize=(8, 3.3 * len(models)),
        squeeze=False,
        layout="constrained",
    )
    selected = [r for r in rows if r["h"]]
    for column, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
        values = [r[metric] for r in selected if r[metric] is not None]
        for index, model in enumerate(models):
            axis = axes[index, column]
            data = np.array(
                [
                    [
                        next(
                            r[metric]
                            for r in selected
                            if r["model"] == model and r["h"] == h and r["d"] == d
                        )
                        for d in (4, 8, 16, 32)
                    ]
                    for h in (2, 4, 6, 8)
                ],
                dtype=float,
            )
            im = axis.imshow(
                data,
                cmap="viridis_r",
                vmin=min(values),
                vmax=max(values),
                aspect="auto",
            )
            for y in range(4):
                for x in range(4):
                    value = data[y, x]
                    dark = im.norm(value) > 0.55
                    axis.text(
                        x,
                        y,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if dark else "black",
                        fontsize=11,
                    )
            y, x = np.unravel_index(np.nanargmin(data), data.shape)
            axis.add_patch(
                Rectangle(
                    (x - 0.48, y - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )
            axis.set_xticks(range(4), [4, 8, 16, 32])
            axis.set_yticks(range(4), [2, 4, 6, 8])
            axis.set_ylabel("Expert budget h")
            display = {"qwen36": "Qwen3.6", "gemma4": "Gemma4"}[model]
            axis.set_xlabel(
                f"Draft width D\n({chr(97 + index * 2 + column)}) {display}"
            )
            colorbar = fig.colorbar(im, ax=axis, fraction=0.045, pad=0.03)
            colorbar.set_label(label)
    for extension in ("png", "pdf"):
        fig.savefig(
            folder / f"token_costs.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)
    (folder / "token_costs.md").write_text(
        "# Unit token costs\n\n"
        "Source: ../summary.csv. Each cell is 16 fixed short multicategory "
        "prompts x 512 output tokens, B=1, TP1, greedy, CUDA Graph. "
        "Left: summed request wall time / actual emitted accepted draft tokens. "
        "Right: summed request wall time / all 8192 final output tokens. "
        "Both are milliseconds and include prefill, draft, target verification, "
        "state and API overhead. Final accepted-tail clipping is accounted for; "
        "the left denominator excludes correction/bonus tokens.\n\n"
        "Lower is better. Color limits are shared across models within each "
        "column. Red outlines identify the minimum measured cost for that "
        "model/metric. Each cell ran once; no uncertainty estimate is implied. "
        "Strict AR output-parity failures remain in ../audit.json and do not "
        "support a lossless speedup claim. See ../readme.md for full contract.\n\n"
        "Reproduce with benchmarks/moe_skip/analyze_static_budget.py and the "
        "--run-dir argument pointing to the parent of this directory.\n"
    )


if __name__ == "__main__":
    main()
