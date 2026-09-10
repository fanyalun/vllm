# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partition fixed-input mean-GDN stage intervals without overlap double counting."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_forward_kernels import annotate, map_graph, read_trace
from summarize_forward_stages import partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (args.directory / "MEASUREMENT_COMPLETE").exists()
    samples = json.loads((args.directory / "result.json").read_text())
    assert len(samples) == 3
    groups = defaultdict(list)
    for sample in samples:
        assert sample["token_ids"] == sample["control_token_ids"]
        assert len(sample["rows"]) == 160
        identities = {tuple(r["input_ids"]) for r in sample["rows"]}
        assert len(identities) == 1
        for row in sample["rows"]:
            assert row["width"] == 5
            groups[row["mode"], row["top_h"], row["detailed"]].append(row)
    totals, stages = [], []
    for (mode, top_h, detailed), rows in sorted(groups.items()):
        assert len(rows) == 60
        totals.append(
            {
                "mode": mode,
                "top_h": top_h,
                "detailed": detailed,
                "calls": len(rows),
                "mean_ms": mean(r["total_ms"] for r in rows),
            }
        )
        if detailed:
            parts = [partition(row) for row in rows]
            for phase in sorted({k for part in parts for k in part}):
                value = mean(part.get(phase, 0) for part in parts)
                stages.append(
                    {
                        "mode": mode,
                        "top_h": top_h,
                        "phase": phase,
                        "mean_ms": value,
                        "percent": value / totals[-1]["mean_ms"] * 100,
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    kernel_stages, timeline_stages, gdn_components = [], [], []
    for mode, top_h in (("none", 8), ("none", 4), ("ssm_mean", 4), ("input_mean", 4)):
        case = f"{mode}_h{top_h}"
        directory = args.directory / case
        events = read_trace(directory / f"w5_h{top_h}_eager_trace.json.gz")
        for event in events:
            if event.get("cat") == "gpu_memcpy":
                assert event["name"] == "Memcpy DtoD (Device -> Device)"
                assert event["args"]["bytes"] == 20480
                event.update(cat="kernel", name="memcpy32_post")
        eager = annotate(events)
        gdn_ranges = [e for e in events if e.get("name", "").startswith("stage/gdn/")]
        for event in eager:
            if any(
                r["ts"] <= event["launch_ts"] <= r["ts"] + r["dur"] for r in gdn_ranges
            ):
                event["phase"] = "gdn"
        mapped = map_graph(
            eager, read_trace(directory / f"w5_h{top_h}_graph_trace.json.gz")
        )
        (args.output / f"{case}_kernel_mapping.json").write_text(
            json.dumps(mapped) + "\n"
        )
        work = defaultdict(float)
        components = defaultdict(list)
        for event in mapped:
            work[event["phase"]] += event["duration_us"] / 1000
            if event["phase"] == "gdn":
                kernel = event["kernel"].lower()
                if "gemm" in kernel or "gemv" in kernel:
                    component = "projection"
                elif "causal_conv1d" in kernel:
                    component = "convolution"
                elif "gdn_decode_post_conv" in kernel:
                    component = "ssm_and_norm_fused"
                elif "_mean_update" in kernel:
                    component = "pooled_ssm"
                elif "layer_norm" in kernel:
                    component = "normalization"
                else:
                    component = "other"
                components[component].append(event["duration_us"] / 1000)
        if mode != "none":
            assert len(components["pooled_ssm"]) == 30
        for component, values in sorted(components.items()):
            gdn_components.append(
                {
                    "mode": mode,
                    "top_h": top_h,
                    "component": component,
                    "kernels": len(values),
                    "kernel_ms": sum(values),
                }
            )
        for phase, value in sorted(work.items()):
            kernel_stages.append(
                {
                    "mode": mode,
                    "top_h": top_h,
                    "phase": phase,
                    "kernel_ms": value,
                    "kernel_work_percent": value / sum(work.values()) * 100,
                }
            )
        start = min(e["start_us"] for e in mapped)
        end = max(e["start_us"] + e["duration_us"] for e in mapped)
        row = {
            "total_ms": (end - start) / 1000,
            "spans": [
                {
                    "phase": e["phase"],
                    "start_ms": (e["start_us"] - start) / 1000,
                    "end_ms": (e["start_us"] + e["duration_us"] - start) / 1000,
                }
                for e in mapped
            ],
        }
        for phase, value in sorted(partition(row).items()):
            timeline_stages.append(
                {
                    "mode": mode,
                    "top_h": top_h,
                    "phase": phase,
                    "duration_ms": value,
                    "percent": value / row["total_ms"] * 100,
                }
            )
    for name, rows in (
        ("stage_totals.csv", totals),
        ("stages.csv", stages),
        ("kernel_stages.csv", kernel_stages),
        ("kernel_timeline.csv", timeline_stages),
        ("gdn_components.csv", gdn_components),
    ):
        with (args.output / name).open("w") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
