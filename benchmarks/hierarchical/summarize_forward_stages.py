# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and summarize graph interval partitions without double counting."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def partition(row):
    points = [(0.0, 0, ""), (row["total_ms"], 0, "")]
    for span in row["spans"]:
        assert -0.001 <= span["start_ms"] <= span["end_ms"]
        assert span["end_ms"] <= row["total_ms"] + 0.001
        points.extend(
            [
                (span["start_ms"], 1, span["phase"]),
                (span["end_ms"], -1, span["phase"]),
            ]
        )
    active = Counter()
    durations = defaultdict(float)
    previous = 0.0
    for timestamp, delta, phase in sorted(points):
        selected = "other"
        for candidate in ("lm_head", "gdn", "attention", "dense_mlp", "embedding"):
            if active[candidate]:
                selected = candidate
                break
        else:
            if active["shared"] and (active["routed"] or active["routing"]):
                selected = "shared_overlap"
            elif active["shared"]:
                selected = "shared"
            elif active["routed"]:
                selected = "routed"
            elif active["routing"]:
                selected = "routing"
            elif active["norm"]:
                selected = "norm"
            elif active["moe_envelope"]:
                selected = "moe_other"
        durations[selected] += timestamp - previous
        active[phase] += delta
        previous = timestamp
    assert abs(sum(durations.values()) - row["total_ms"]) < 1e-5
    return dict(durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    layer_groups = defaultdict(list)
    for model in ("qwen36", "gemma4"):
        directory = args.root / model
        assert (directory / "MEASUREMENT_COMPLETE").exists()
        results = json.loads((directory / "result.json").read_text())
        assert len(results) == 3
        for sample in results:
            assert sample["token_ids"] == sample["control_token_ids"]
            assert len(sample["rows"]) == 160
            for row in sample["rows"]:
                key = (model, row["width"], row["top_h"], row["detailed"])
                groups[key].append(row)
                if row["detailed"]:
                    row["partition"] = partition(row)
                    for span in row["spans"]:
                        layer_groups[key[:3] + (span["phase"], span["name"])].append(
                            span["end_ms"] - span["start_ms"]
                        )
    totals, stages = [], []
    for (model, width, top_h, detailed), rows in sorted(groups.items()):
        assert len(rows) == 60
        values = [row["total_ms"] for row in rows]
        base = dict(model=model, width=width, top_h=top_h, detailed=detailed)
        totals.append(
            {
                **base,
                "calls": len(rows),
                "mean_ms": np.mean(values),
                "median_ms": np.median(values),
                "p10_ms": np.percentile(values, 10),
                "p90_ms": np.percentile(values, 90),
            }
        )
        if detailed:
            phases = sorted({k for row in rows for k in row["partition"]})
            for phase in phases:
                mean = np.mean([row["partition"].get(phase, 0) for row in rows])
                stages.append(
                    {
                        **base,
                        "phase": phase,
                        "mean_ms": mean,
                        "percent": mean / np.mean(values) * 100,
                    }
                )
    layers = [
        dict(
            model=k[0],
            width=k[1],
            top_h=k[2],
            phase=k[3],
            name=k[4],
            mean_ms=np.mean(v),
            calls=len(v),
        )
        for k, v in sorted(layer_groups.items())
    ]
    for filename, rows in (
        ("totals.csv", totals),
        ("stages.csv", stages),
        ("layers.csv", layers),
    ):
        with (args.root / filename).open("w") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    (args.root / "audit.json").write_text(
        json.dumps(
            {
                "cases": 6,
                "measured_graph_replays": 960,
                "instrumentation_output_controls": "6/6 passed",
                "partition_sum_check": "passed",
                "measurement_boundary": (
                    "backbone + lm_head + argmax; metadata and state restore excluded"
                ),
                "limitations": [
                    "Private preverify metadata for top-8 and top-4; "
                    "not native Target execute_model latency.",
                    "Eager operators captured into CUDA Graph; "
                    "no torch.compile fusion.",
                    "Stage times are CUDA event intervals, not kernel-only activity.",
                    "Stage percentages use instrumented total; "
                    "uninstrumented total is separately reported.",
                    "Three fixed prefixes; replays measure local variability, "
                    "not workload confidence intervals.",
                    "Instrumentation on/off output control does not certify "
                    "hierarchical decoder correctness.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "STIXGeneral", "font.size": 12, "pdf.fonttype": 42}
    )
    phases = [
        "attention",
        "gdn",
        "dense_mlp",
        "norm",
        "embedding",
        "lm_head",
        "routing",
        "shared",
        "shared_overlap",
        "routed",
        "moe_other",
        "other",
    ]
    labels = {
        "attention": "Attention",
        "gdn": "GDN",
        "dense_mlp": "Dense MLP",
        "shared": "Shared only",
        "routed": "Routed only",
        "shared_overlap": "Shared overlap",
        "routing": "Router",
        "norm": "Norm",
        "embedding": "Embedding",
        "lm_head": "LM head",
        "moe_other": "MoE other",
        "other": "Other",
    }
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)
    for ax, model, panel in zip(axes, ("qwen36", "gemma4"), ("a", "b")):
        bottom = np.zeros(4)
        cells = [(4, 8), (4, 4), (5, 8), (5, 4)]
        for i, phase in enumerate(phases):
            values = [
                sum(
                    r["mean_ms"]
                    for r in stages
                    if r["model"] == model
                    and r["width"] == w
                    and r["top_h"] == h
                    and r["phase"] == phase
                )
                for w, h in cells
            ]
            ax.bar(
                range(4), values, bottom=bottom, color=colors[i], label=labels[phase]
            )
            bottom += values
        ax.set_xticks(
            range(4),
            ["4 tokens\nFull", "4 tokens\nSkip", "5 tokens\nFull", "5 tokens\nSkip"],
        )
        ax.set_xlabel(f"({panel}) {'Qwen3.6' if model == 'qwen36' else 'Gemma4'}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Instrumented forward time (ms)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.79))
    directory = args.root / "forward_stage_comparison"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"forward_stage_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    (directory / "forward_stage_comparison.md").write_text(
        "# Forward stage comparison\n\nSource: ../stages.csv and ../totals.csv. "
        "B=1, TP=1, A100 80GB, BF16, top-8/full versus top-4/skip. "
        "Three fixed prefixes per model, 20 alternating replays per mode and prefix. "
        "Bars show the instrumented CUDA Graph interval partition in ms. "
        "Attention and GDN include their projections and internal normalization. "
        "Shared overlap is the union of shared/routed or shared/router overlap; "
        "inclusive layer timings are in ../layers.csv. "
        "Other includes residuals, argmax, unwrapped operators, "
        "graph/event overhead and gaps. "
        "No error bars; replays are not independent prompts. "
        "See ../audit.json for measurement boundaries.\n\nReproduce: "
        "`.venv/bin/python benchmarks/hierarchical/summarize_forward_stages.py "
        + str(args.root)
        + "`\n"
    )


if __name__ == "__main__":
    main()
