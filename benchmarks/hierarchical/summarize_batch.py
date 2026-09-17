# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize true active batches, matched acceptance, and CUDA latency samples."""

import argparse
import bisect
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=values[int(0.95 * (len(values) - 1))],
        total=sum(values),
    )


def profile(path):
    events = json.loads(path.read_text())["traceEvents"]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    ranges = sorted(
        [e for e in events if e.get("name") == "vllm::qwen_gdn_mean_forward"],
        key=lambda e: e["ts"],
    )
    starts = [e["ts"] for e in ranges]
    gdn = defaultdict(float)
    windowed_ranges = set()
    kernels = defaultdict(list)
    for event in events:
        if event.get("cat") != "kernel":
            continue
        name = event["name"]
        kernels[name].append(event["dur"])
        launch = launches.get(event["args"].get("correlation"))
        if launch is None:
            continue
        i = bisect.bisect_right(starts, launch["ts"]) - 1
        if (
            i >= 0
            and ranges[i]["tid"] == launch["tid"]
            and launch["ts"] + launch["dur"] <= starts[i] + ranges[i]["dur"]
        ):
            gdn[i] += event["dur"]
            if "_windowed_update" in name:
                windowed_ranges.add(i)
    return dict(
        kernels={k: stats(v) for k, v in kernels.items()},
        full_gdn_eager_gpu_busy_us=stats(list(gdn.values())),
        preverify_gdn_eager_gpu_busy_us=stats(
            [v for i, v in gdn.items() if i in windowed_ranges]
        ),
        note=(
            "GDN busy time uses CPU launch correlation; "
            "graph-only traces have no such attribution."
        ),
    )


def summarize(directory):
    timings = json.loads((directory / "timings.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    stages = defaultdict(list)
    for row in audit["stages"]:
        stages[row["stage"]].append(row["ms"])
    outer = []
    matched = []
    for row in audit["outer"]:
        for i, (scheduled, sampled) in enumerate(
            zip(row["scheduled"], row["sampled"], strict=True)
        ):
            if scheduled:
                accepted = max(0, sampled - 1)
                assert accepted <= scheduled
                outer.append((scheduled, accepted))
                matched.append(
                    dict(
                        request_id=row["request_ids"][i],
                        proposal_cycle=row.get(
                            "proposal_cycles", [None] * len(row["scheduled"])
                        )[i],
                        scheduled=scheduled,
                        accepted=accepted,
                    )
                )
    inner = audit["inner"]
    return dict(
        throughput_tokens_s=sum(x["output_tokens"] for x in timings)
        / sum(x["elapsed_s"] for x in timings),
        repeats=len(timings),
        returned_tokens=sum(x["output_tokens"] for x in timings),
        elapsed_s=sum(x["elapsed_s"] for x in timings),
        repeat_equal=all(x["repeat_equal"] for x in timings),
        instrumentation_equal=audit["instrumentation_equal"],
        inner_accepted=stats([x["accepted"] for x in inner]),
        inner_acceptance_histogram=dict(Counter(x["accepted"] for x in inner)),
        inner_proposed=sum(x["proposed"] for x in inner),
        inner_active_request_round_histogram=dict(
            Counter(x.get("active_batch", 1) for x in inner)
        ),
        outer_active_batch_histogram=dict(
            Counter(len(x["request_ids"]) for x in audit["outer"])
        ),
        outer_scheduled=stats([x[0] for x in outer]),
        outer_accepted=stats([x[1] for x in outer]),
        outer_matched=matched,
        stage_ms={k: stats(v) for k, v in stages.items()},
        complete_cycle_ms=stats([x["ms"] for x in audit.get("cycles", [])]),
        private_bytes=audit["private_bytes"],
        note=(
            "Latency and acceptance come from the separate audit pass; "
            "proposal includes its nested stages."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = {}
    for path in sorted(args.root.rglob("audit.json")):
        directory = path.parent
        if not (directory / "timings.json").exists():
            continue
        result = summarize(directory)
        if (directory / "profile.json").exists():
            result["profile"] = profile(directory / "profile.json")
        rows[str(directory.relative_to(args.root))] = result
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
