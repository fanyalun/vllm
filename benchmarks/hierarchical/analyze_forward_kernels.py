# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer annotated eager kernel identities to event-free graph traces."""

import argparse
import csv
import gzip
import json
from collections import defaultdict


def read_trace(path):
    with gzip.open(path, "rt") as source:
        return json.load(source)["traceEvents"]


def annotate(events):
    ranges = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("stage/")
    ]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    result = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        launch = launches[event["args"]["correlation"]]
        parents = [
            r
            for r in ranges
            if r["tid"] == launch["tid"]
            and r["ts"] <= launch["ts"]
            and r["ts"] + r["dur"] >= launch["ts"] + launch["dur"]
        ]
        parent = min(parents, key=lambda r: r["dur"]) if parents else None
        phase = parent["name"].split("/", 2)[1] if parent else "other"
        result.append({**event, "phase": phase, "launch_ts": launch["ts"]})
    return sorted(result, key=lambda e: e["launch_ts"])


def map_graph(eager, graph):
    streams = defaultdict(list)
    for event in graph:
        if event.get("cat") == "kernel":
            streams[event["args"]["stream"]].append(event)
    streams = sorted(streams.values(), key=len, reverse=True)
    for values in streams:
        values.sort(key=lambda e: e["ts"])
    if len(streams) == 1:
        expected = [eager]
    else:
        main = [e for e in eager if e["phase"] not in ("routing", "routed")]
        branch = [e for e in eager if e["phase"] in ("routing", "routed")]
        expected = [main]
        offset = 0
        for stream in streams[1:]:
            expected.append(branch[offset : offset + len(stream)])
            offset += len(stream)
        assert offset == len(branch)
    result = []
    for source, stream in zip(expected, streams, strict=True):
        assert len(source) == len(stream), (len(source), len(stream))
        for a, b in zip(source, stream, strict=True):
            assert a["name"] == b["name"], (a["name"], b["name"])
            result.append(
                {
                    "phase": a["phase"],
                    "kernel": b["name"],
                    "start_us": b["ts"],
                    "duration_us": b["dur"],
                    "stream": b["args"]["stream"],
                }
            )
    assert len(result) == len(eager)
    return result


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = []
    for model in ("qwen36", "gemma4"):
        for width in (4, 5):
            for top_h in (8, 4):
                directory = args.root / model
                prefix = f"w{width}_h{top_h}"
                eager = annotate(
                    read_trace(directory / f"{prefix}_eager_trace.json.gz")
                )
                graph = read_trace(directory / f"{prefix}_graph_trace.json.gz")
                mapped = map_graph(eager, graph)
                (directory / f"{prefix}_kernel_mapping.json").write_text(
                    json.dumps(mapped) + "\n"
                )
                totals = defaultdict(float)
                counts = defaultdict(int)
                for event in mapped:
                    totals[event["phase"]] += event["duration_us"] / 1000
                    counts[event["phase"]] += 1
                for phase, value in totals.items():
                    summary.append(
                        dict(
                            model=model,
                            width=width,
                            top_h=top_h,
                            phase=phase,
                            kernel_ms=value,
                            kernel_work_percent=value / sum(totals.values()) * 100,
                            kernels=counts[phase],
                        )
                    )
                print(model, prefix, len(mapped), dict(totals))
    with (args.root / "kernel_stages.csv").open("w") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
