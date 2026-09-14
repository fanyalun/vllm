# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and plot the completed AR/full-budget performance diagnosis."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_performance_paths import summarize


def report(root):
    summarize(root)
    analysis = json.loads((root / "analysis.json").read_text())
    audit = {"cells": [], "ar_sync_async_exact": {}, "h8_vs_ar_sync_exact": {}}
    modes = ("ar_sync", "h8_d8", "ar_async")
    models = ("qwen36", "gemma4")
    names = ("Qwen3.6", "Gemma4")
    for model in models:
        outputs = {}
        for mode in modes:
            cell = root / model / mode
            assert (cell / "complete.json").exists()
            config = json.loads((cell / "config.json").read_text())
            dataset = Path(config["dataset"])
            assert (
                hashlib.sha256(dataset.read_bytes()).hexdigest()
                == config["dataset_sha256"]
            )
            outputs[mode] = json.loads((cell / "e2e.json").read_text())
            rows = outputs[mode]
            assert len(rows) == 6 and all(len(r["token_ids"]) == 512 for r in rows)
            assert [(r["repeat"], r["sample_index"]) for r in rows] == [
                (repeat, sample) for repeat in range(3) for sample in range(2)
            ]
            assert all(
                r["token_ids"] == rows[r["sample_index"]]["token_ids"] for r in rows
            )
            tail = (cell / "run.log").read_text().split("WARMUP_COMPLETE", 1)[1]
            assert "JIT compilation during inference" not in tail
            diagnostics = json.loads((cell / "diagnostics.json").read_text())
            assert [d["mode"] for d in diagnostics] == ["events", "profile"]
            assert diagnostics[0]["token_ids"] == rows[0]["token_ids"]
            assert diagnostics[1]["token_ids"] == rows[0]["token_ids"][:64]
            audit["cells"].append({"model": model, "mode": mode, "passed": True})
        for mode, key in [
            ("ar_async", "ar_sync_async_exact"),
            ("h8_d8", "h8_vs_ar_sync_exact"),
        ]:
            audit[key][model] = sum(
                a["token_ids"] == b["token_ids"]
                for a, b in zip(outputs["ar_sync"], outputs[mode], strict=True)
            )
    audit.update(
        requests=36,
        output_tokens=18432,
        post_warmup_jit="none",
        scope="CPU/CUDA diagnostic; same prompts, not forced identical continuations",
    )
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lookup = {(r["model"], r["mode"]): r for r in analysis["costs"]}
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for index, (ax, model, name) in enumerate(zip(axes, models, names, strict=True)):
        values = [lookup[model, mode]["ms_per_output"] for mode in modes]
        ax.bar(range(3), values, width=0.64, color=["#4C78A8", "#F2B447", "#59A14F"])
        for x, value in enumerate(values):
            ax.text(x, value + 0.12, f"{value:.2f}", ha="center", fontsize=12)
        ax.set_xticks(range(3), ["AR\nsync", "h=8\nD=8", "AR\nasync"])
        ax.set_ylim(0, 12)
        ax.set_xlabel(f"({chr(97 + index)}) {name}")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Time per output token (ms)")
    fig.tight_layout(pad=0.5, w_pad=1.2)
    folder = root / "ar_h8_cost"
    folder.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            folder / f"ar_h8_cost.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)
    (folder / "ar_h8_cost.md").write_text(
        "# AR and full-budget draft output cost\n\n"
        "Source: ../costs.csv and per-cell e2e.json. Lower is better. "
        "Panels show Qwen3.6 and Gemma4 on one A100 80GB, TP1/B1, greedy, "
        "two archived prompts x 512 outputs x three repetitions per mode. "
        "Bars are summed uninstrumented request wall time divided by actual "
        "output count; initialization and warmup are excluded. No uncertainty "
        "interval is claimed. Modes were run sequentially, not interleaved. "
        "The controlled AR comparison is exact for all six Qwen requests and "
        "three of six Gemma requests. This is not a new 16-prompt matrix.\n\n"
        "Reproduce with `.venv/bin/python benchmarks/moe_skip/"
        "report_performance_paths.py --output <run-directory>`.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)
