# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit coverage, pooled output cost, paired uncertainty, and token controls."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import regex as re


def interval_union(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    end = -float("inf")
    total = 0.0
    for start, stop in intervals:
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_trace(path):
    trace = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in trace if e.get("cat") == "kernel" and "dur" in e]
    groups, streams = defaultdict(list), defaultdict(list)
    for event in kernels:
        name = event["name"]
        known = re.search(
            r"(?:_replay_tail_update|_begin_private_states|_advance_conv_many|_causal_conv1d_update_kernel|fused_moe_kernel|_rejection_kernel|_resample_kernel)",
            name,
        )
        key = known.group() if known else name.split("<")[0][:120]
        groups[key].append(event)
        streams[str(event.get("args", {}).get("stream", event.get("tid")))].append(
            event
        )
    rows = []
    for name, events in groups.items():
        rows.append(
            dict(
                name=name,
                launches=len(events),
                union_us=interval_union(events),
                median_us=float(np.median([e["dur"] for e in events])),
            )
        )
    gaps = {}
    for stream, events in streams.items():
        ordered = sorted(events, key=lambda e: e["ts"])
        values = [
            max(0.0, b["ts"] - a["ts"] - a["dur"]) for a, b in zip(ordered, ordered[1:])
        ]
        gaps[stream] = dict(
            kernels=len(events),
            median_gap_us=float(np.median(values)) if values else 0.0,
            p95_gap_us=float(np.quantile(values, 0.95)) if values else 0.0,
        )
    sync = [
        e
        for e in trace
        if e.get("cat") == "cuda_runtime"
        and "dur" in e
        and "Synchronize" in e.get("name", "")
    ]
    return dict(
        kernels=len(kernels),
        device_union_us=interval_union(kernels),
        groups=sorted(rows, key=lambda r: r["union_us"], reverse=True),
        streams=gaps,
        host_synchronization_count=len(sync),
        host_synchronization_median_us=float(np.median([e["dur"] for e in sync]))
        if sync
        else 0.0,
        scope=(
            "Separate 32-token trace. Group interval unions may overlap; "
            "do not add them or interpret profiling wall time as generation cost."
        ),
    )


def paired(base, candidate):
    assert base.shape == candidate.shape == (5, 16)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 16, size=(10000, 16))
    ratios = base.mean(0)[indices].sum(1) / candidate.mean(0)[indices].sum(1) - 1
    trials = base.sum(1) / candidate.sum(1) - 1
    return dict(
        gain=float(base.sum() / candidate.sum() - 1),
        request_paired_ci95=np.quantile(ratios, [0.025, 0.975]).tolist(),
        trial_gains=trials.tolist(),
        all_five_positive=bool((trials > 0).all()),
        bootstrap="10000 paired prompt-cluster resamples; five-trial mean per prompt",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    freeze = json.loads((root / "final/freeze.json").read_text())
    assert (root / "final/paired_runs_complete.json").is_file()
    ar_rows = json.loads((root / "ar/results.json").read_text())
    assert len(ar_rows) == 32
    ar = {r["sample"]: r for r in ar_rows if r["repeat"] == 0}
    for row in ar_rows:
        assert row["token_ids"] == ar[row["sample"]]["token_ids"]
    summaries, comparisons, cycles_out, traces = [], {}, [], {}
    raw_times = []
    for method, candidate in freeze["selected"].items():
        path = root / "final" / method
        complete = json.loads((path / "measurement_complete.json").read_text())
        assert json.loads((path / "profile_parity.json").read_text())["identical"]
        traces[method] = summarize_trace(path / "profile.json")
        contract = json.loads((path / "contract.json").read_text())
        rows = json.loads((path / "results.json").read_text())
        assert len(rows) == complete["rows"] == complete["expected"]
        assert contract["repeats"] == 5 and contract["max_tokens"] == 256
        assert contract["cuda_visible_devices"] == "0"
        for i, sample in enumerate(contract["samples"]):
            assert sample["prompt_sha256"] == ar[i]["prompt_sha256"]
        for name, digest in contract["source_sha256"].items():
            assert digest == freeze["source_sha256"][name]
        matrices = {}
        for case in contract["timed_cases"]:
            timed = [r for r in rows if r["case"] == case and r["phase"] == "e2e"]
            audit = [r for r in rows if r["case"] == case and r["phase"] == "audit"]
            assert len(timed) == 80 and len(audit) == 16
            matrix = np.full((5, 16), np.nan)
            references = {}
            for row in timed:
                assert len(row["token_ids"]) == 256
                key = row["repeat"], row["sample"]
                assert np.isnan(matrix[key])
                matrix[key] = row["seconds"]
                references.setdefault(row["sample"], row["token_ids"])
                assert references[row["sample"]] == row["token_ids"]
                raw_times.append(
                    dict(
                        method=method,
                        case=case,
                        repeat=row["repeat"],
                        sample=row["sample"],
                        seconds=row["seconds"],
                        tokens=256,
                    )
                )
            assert np.isfinite(matrix).all()
            matrices[case] = matrix
            stage_ms = defaultdict(float)
            inner = Counter()
            stop_reasons = Counter()
            widths = Counter()
            round_hist = Counter()
            cycle_count = outer_accepted = outer_proposed = 0
            target_sampling_scheduling_ms = 0.0
            for row in audit:
                assert row["token_ids"] == references[row["sample"]]
                inner.update(row["policy_metrics"])
                for span in row["spans"]:
                    stage_ms[span["phase"]] += span["ms"]
                proposals = [s["ms"] for s in row["spans"] if s["phase"] == "proposal"]
                remaining = 255
                for index, cycle in enumerate(row["cycles"]):
                    target_sampling_scheduling_ms += cycle["ms"] - proposals[index]
                    emitted = min(remaining, cycle["emitted"])
                    remaining -= emitted
                    cycle_count += 1
                    outer_accepted += cycle["emitted"] - 1
                    outer_proposed += cycle["scheduled"]
                    round_hist[len(cycle["inner"])] += 1
                    for window in cycle["inner"]:
                        widths[window["proposed"] + 1] += 1
                        stop_reasons[window["stop_reason"]] += 1
                    cycles_out.append(
                        dict(
                            method=method,
                            case=case,
                            sample=row["sample"],
                            cycle=index,
                            returned_tokens=emitted,
                            raw_emitted=cycle["emitted"],
                            milliseconds=cycle["ms"],
                            inner_rounds=len(cycle["inner"]),
                            scheduled=cycle["scheduled"],
                        )
                    )
                assert remaining == 0, "Unaccounted output tokens in cycle audit"
            actions = [r for r in rows if r["case"] == case and r["phase"] == "actions"]
            counts = np.zeros(3, dtype=np.int64)
            for row in actions:
                assert row["token_ids"] == references[row["sample"]]
                reference_audit = next(r for r in audit if r["sample"] == row["sample"])
                metrics = reference_audit["policy_metrics"]
                expected = 960 * (metrics["inner_rounds"] + metrics["inner_proposed"])
                assert sum(row["forward_action_counts"]) == expected
                counts += row["forward_action_counts"]
            ar_matches = sum(references[i] == ar[i]["token_ids"] for i in range(16))
            ar_positions = sum(
                sum(
                    a == b
                    for a, b in zip(references[i], ar[i]["token_ids"], strict=True)
                )
                for i in range(16)
            )
            summaries.append(
                dict(
                    method=method,
                    case=case,
                    total_tokens=20480,
                    total_seconds=float(matrix.sum()),
                    tokens_per_second=float(20480 / matrix.sum()),
                    ms_per_token=float(matrix.sum() * 1000 / 20480),
                    trial_tokens_per_second=(4096 / matrix.sum(1)).tolist(),
                    ar_exact_requests=ar_matches,
                    ar_equal_token_positions=ar_positions,
                    ar_total_positions=4096,
                    repeatable=True,
                    inner_counts=dict(inner),
                    outer_accepted=outer_accepted,
                    outer_proposed=outer_proposed,
                    observed_cycles=cycle_count,
                    returned_tokens_per_cycle=4080 / cycle_count,
                    initial_prefill_output_tokens=16,
                    round_histogram=dict(round_hist),
                    width_histogram=dict(widths),
                    stop_histogram=dict(stop_reasons),
                    action_counts=counts.tolist() if actions else None,
                    action_order=["full", "decay", "skip"],
                    stage_ms_per_returned_token={
                        k: v / 4096 for k, v in stage_ms.items()
                    },
                    target_sampling_scheduling_ms_per_token=target_sampling_scheduling_ms
                    / 4096,
                )
            )
        same_stop = f"exact:carry:{candidate.split(':')[-1]}"
        comparisons[method] = dict(
            candidate=candidate,
            original=paired(matrices["exact:carry:low_error"], matrices[candidate]),
            same_stop=paired(matrices[same_stop], matrices[candidate]),
        )
    out = root / "report"
    out.mkdir(exist_ok=True)
    payload = dict(
        summary=summaries,
        comparisons=comparisons,
        stage_boundary=(
            "Nested stage timers are not additive. "
            "Target+sampling+scheduling is the difference of intervals "
            "sharing the same start event."
        ),
        cycle_boundary=(
            "256 returned tokens include one initial prefill token; "
            "speculative cycle emissions are clipped at the request limit."
        ),
    )
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    (out / "trace_summary.json").write_text(json.dumps(traces, indent=2))
    for name, data in (("raw_times", raw_times), ("cycles", cycles_out)):
        with (out / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    flat = [
        {
            k: row[k]
            for k in (
                "method",
                "case",
                "tokens_per_second",
                "ms_per_token",
                "total_seconds",
                "ar_exact_requests",
                "observed_cycles",
                "returned_tokens_per_cycle",
            )
        }
        for row in summaries
    ]
    with (out / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Three-Level p50 final paired evaluation",
        "",
        (
            "Qwen3.6-35B-A3B; A100 80GB PCIe GPU 0; TP1/B1; D4/h4; "
            "at most four inner rounds; greedy seed 42; GSM8K first 16 prompts, "
            "256 tokens each, five paired trials."
        ),
        "",
        "| Method | Case | tokens/s | ms/token | Exact AR requests |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['case']} | "
            f"{row['tokens_per_second']:.3f} | {row['ms_per_token']:.4f} | "
            f"{row['ar_exact_requests']}/16 |"
        )
    for method, result in comparisons.items():
        lines.extend(["", f"## {method}", ""])
        for label in ("original", "same_stop"):
            value = result[label]
            lo, hi = value["request_paired_ci95"]
            gains = ", ".join(f"{g:+.2%}" for g in value["trial_gains"])
            lines.append(
                f"Relative to {label}: {value['gain']:+.2%}; "
                f"paired prompt-cluster bootstrap 95% CI [{lo:+.2%}, {hi:+.2%}]. "
                f"Five paired gains: {gains}."
            )
    lines.extend(
        [
            "",
            (
                "All modes are internally repeatable. AR agreement is reported "
                "separately; these results do not establish lossless acceleration."
            ),
            "",
            payload["stage_boundary"],
            "",
            payload["cycle_boundary"],
            "",
            (
                "Raw tokens and text: ../final/{mtp,dspark}/results.json. "
                "Timed runs reject new JIT compilation and graph capture. "
                "Action counters and profiling run separately after timing. "
                "Source fingerprints and selection are frozen in ../final/freeze.json."
            ),
        ]
    )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    accepted = all(
        r["original"]["all_five_positive"]
        and r["original"]["request_paired_ci95"][0] > 0
        for r in comparisons.values()
    )
    (out / "coverage_complete.json").write_text(
        json.dumps(
            dict(
                methods=2,
                prompts=16,
                trials=5,
                total_timed_rows=len(raw_times),
                ar_checked=True,
                repeatability_checked=True,
                action_counts_checked=True,
                paired_positive_gain_accepted=accepted,
            ),
            indent=2,
        )
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
