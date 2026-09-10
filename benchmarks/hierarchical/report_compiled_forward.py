# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative name/order attribution of compiled graph kernel traces."""

import argparse
import csv
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from analyze_forward_kernels import read_trace
from summarize_forward_stages import partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows, totals, audit = [], [], []
    for model in ("qwen36", "gemma4"):
        directory = args.root / f"{model}_compiled"
        results = json.loads((directory / "result.json").read_text())
        assert (directory / "MEASUREMENT_COMPLETE").exists()
        assert len(results) == 3
        for sample in results:
            assert len(sample["rows"]) == 80
            assert sample["token_ids"] == sample["control_token_ids"]
        for width in (4, 5):
            for top_h in (8, 4):
                values = [
                    r["total_ms"]
                    for s in results
                    for r in s["rows"]
                    if r["width"] == width and r["top_h"] == top_h
                ]
                totals.append(
                    dict(
                        model=model,
                        width=width,
                        top_h=top_h,
                        mean_ms=np.mean(values),
                        median_ms=np.median(values),
                        p10_ms=np.percentile(values, 10),
                        p90_ms=np.percentile(values, 90),
                        calls=len(values),
                    )
                )
                prefix = f"w{width}_h{top_h}"
                source = json.loads(
                    (args.root / model / f"{prefix}_kernel_mapping.json").read_text()
                )
                name_phases = defaultdict(set)
                for event in source:
                    name_phases[event["kernel"]].add(event["phase"])
                trace = read_trace(directory / f"{prefix}_graph_trace.json.gz")
                a, b = defaultdict(list), defaultdict(list)
                for event in source:
                    a[event["stream"]].append(event)
                for event in trace:
                    if event.get("cat") == "kernel":
                        b[event["args"]["stream"]].append(event)
                baseline = sorted(a.values(), key=len, reverse=True)
                compiled = sorted(b.values(), key=len, reverse=True)
                assert len(baseline) == len(compiled)
                mapped = []
                for old, new in zip(baseline, compiled, strict=True):
                    old.sort(key=lambda e: e["start_us"])
                    new.sort(key=lambda e: e["ts"])
                    names_a = [e["kernel"] for e in old]
                    names_b = [e["name"] for e in new]
                    matches = SequenceMatcher(None, names_a, names_b, autojunk=False)
                    phases = ["other"] * len(new)
                    for block in matches.get_matching_blocks():
                        for offset in range(block.size):
                            phases[block.b + offset] = old[block.a + offset]["phase"]
                    for phase, event in zip(phases, new, strict=True):
                        candidates = name_phases[event["name"]]
                        if len(candidates) == 1:
                            phase = next(iter(candidates))
                        elif "rms_norm" in event["name"]:
                            phase = "norm"
                        mapped.append(
                            dict(
                                phase=phase,
                                kernel=event["name"],
                                start_us=event["ts"],
                                duration_us=event["dur"],
                                stream=event["args"]["stream"],
                            )
                        )
                if model == "gemma4":
                    anchors = [
                        i for i, e in enumerate(mapped) if e["phase"] == "dense_mlp"
                    ]
                    assert len(anchors) == 30
                    for index in anchors:
                        block = mapped[index : index + 4]
                        assert "fused_gelu_mul_slice" in block[1]["kernel"]
                        assert "dense_mlp" in name_phases[block[2]["kernel"]]
                        assert "dense_mlp" in name_phases[block[3]["kernel"]]
                        for event in block:
                            event["phase"] = "dense_mlp"
                (directory / f"{prefix}_kernel_mapping.json").write_text(
                    json.dumps(mapped) + "\n"
                )
                origin = min(e["start_us"] for e in mapped)
                end = max(e["start_us"] + e["duration_us"] for e in mapped)
                wall_ms = (end - origin) / 1000
                spans = [
                    {
                        "phase": e["phase"],
                        "start_ms": (e["start_us"] - origin) / 1000,
                        "end_ms": (e["start_us"] + e["duration_us"] - origin) / 1000,
                    }
                    for e in mapped
                ]
                parts = partition({"total_ms": wall_ms, "spans": spans})
                for phase, duration in parts.items():
                    rows.append(
                        dict(
                            model=model,
                            width=width,
                            top_h=top_h,
                            phase=phase,
                            trace_ms=duration,
                            percent=duration / wall_ms * 100,
                        )
                    )
                unknown = sum(e["duration_us"] for e in mapped if e["phase"] == "other")
                audit.append(
                    dict(
                        model=model,
                        width=width,
                        top_h=top_h,
                        kernels=len(mapped),
                        trace_span_ms=wall_ms,
                        event_mean_ms=np.mean(values),
                        unattributed_kernel_ms=unknown / 1000,
                        method=(
                            "Unique names and ordered name subsequences; "
                            "asserted Gemma dense block; named RMS fusion; "
                            "unmatched kernels remain other"
                        ),
                    )
                )
    for name, data in (("compiled_totals.csv", totals), ("compiled_stages.csv", rows)):
        with (args.root / name).open("w") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(data[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(data)
    (args.root / "compiled_trace_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "STIXGeneral", "font.size": 13, "pdf.fonttype": 42}
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
    labels = [
        "Attention",
        "GDN",
        "Dense MLP",
        "Norm / residual",
        "Embedding",
        "LM head",
        "Router",
        "Shared only",
        "Shared overlap",
        "Routed only",
        "MoE other",
        "Fused / other / gaps",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, model, panel in zip(axes, ("qwen36", "gemma4"), ("a", "b")):
        base = np.zeros(4)
        for i, (phase, label) in enumerate(zip(phases, labels, strict=True)):
            if not any(r["phase"] == phase for r in rows):
                continue
            values = [
                sum(
                    r["trace_ms"]
                    for r in rows
                    if r["model"] == model
                    and r["width"] == w
                    and r["top_h"] == h
                    and r["phase"] == phase
                )
                for w, h in ((4, 8), (4, 4), (5, 8), (5, 4))
            ]
            ax.bar(
                range(4),
                values,
                bottom=base,
                color=plt.get_cmap("tab20").colors[i],
                label=label,
            )
            base += values
        ax.set_xticks(
            range(4),
            ["4 tokens\nFull", "4 tokens\nSkip", "5 tokens\nFull", "5 tokens\nSkip"],
        )
        ax.set_xlabel(f"({panel}) {'Qwen3.6' if model == 'qwen36' else 'Gemma4'}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Compiled graph trace time (ms)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.79))
    directory = args.root / "compiled_forward_comparison"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"compiled_forward_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    (directory / "compiled_forward_comparison.md").write_text(
        "# Compiled forward comparison\n\n"
        "Source: ../compiled_stages.csv and ../compiled_trace_audit.json. "
        "Each bar partitions one warmed CUDA graph trace at the first prompt prefix. "
        "Actual kernel durations are measured with CUPTI; attribution transfers "
        "unique names and exact names in stream order from the annotated reference. "
        "Gemma dense MLP four-kernel blocks are asserted against reference names; "
        "named RMS kernels include fused residual operations. "
        "Unmatched fused kernels remain in Other; attribution is approximate. "
        "Shared overlap is counted once. Graph gaps remain in Other. "
        "Unprofiled totals use 3 prefixes x 20 replays; see ../compiled_totals.csv. "
        "B=1, TP=1, BF16, A100 80GB, top-8 versus top-4. "
        "Both methods use identical private preverify metadata; no scheduling, "
        "state restore, metadata building or proposal generation is timed.\n\n"
        "Reproduce: `.venv/bin/python "
        "benchmarks/hierarchical/report_compiled_forward.py " + str(args.root) + "`\n"
    )


if __name__ == "__main__":
    main()
