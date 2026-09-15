# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit inner-round acceptance and final Target yield on matched cycles."""

import argparse
import json
from pathlib import Path

from analyze_token_importance import write_csv
from run_token_importance import digest, write_json


def overlap(prefix, offset, length):
    return max(0, min(length, prefix - offset))


def summarize(rows):
    n = len(rows)
    return {
        "cycles": n,
        "inner_accepted": sum(r["inner_accepted"] for r in rows) / n,
        "candidates": sum(r["candidates"] for r in rows) / n,
        "target_accepted": sum(r["target_accepted"] for r in rows) / n,
        "returned_accepted": sum(r["returned_accepted"] for r in rows) / n,
        "zero_target_fraction": sum(r["target_accepted"] == 0 for r in rows) / n,
        "zero_returned_fraction": sum(r["returned_accepted"] == 0 for r in rows) / n,
        "target_retention": sum(r["target_accepted"] for r in rows)
        / sum(r["candidates"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    contract = json.loads((root / "contract.json").read_text())
    assert digest(root / "dataset.jsonl") == contract["dataset_sha256"]
    dataset = [json.loads(s) for s in (root / "dataset.jsonl").read_text().splitlines()]
    for name, fingerprint in contract["source_sha256"].items():
        assert digest(root / name) == fingerprint
    for model in ("model", "assistant"):
        assert (
            digest(Path(contract[model]) / "config.json")
            == contract[f"{model}_config_sha256"]
        )
    details, audits = [], []
    for cell in contract["cells"]:
        folder = root / cell["name"]
        assert (folder / "CELL_COMPLETE").exists()
        result = json.loads((folder / "result.json").read_text())
        assert all(result[k] == v for k, v in cell.items())
        assert result["model_path"] == contract["model"]
        assert result["source_sha256"] == contract["source_sha256"]
        assert result["dataset_sha256"] == contract["dataset_sha256"]
        groups = {}
        for line in (folder / "trace/rounds.jsonl").read_text().splitlines():
            trace = json.loads(line)
            groups.setdefault(trace["request_id"], []).append(trace)
        assert len(groups) == 8, (cell, list(groups))
        measured = list(groups.items())[4:]
        assert len(result["outputs"]) == 4
        used_rounds = used_inner = 0
        for sample, ((request_id, cycles), output) in enumerate(
            zip(measured, result["outputs"], strict=True)
        ):
            assert len(output["token_ids"]) == 128
            assert output["prompt_sha256"] == dataset[sample]["prompt_sha256"]
            assert int(request_id.split("-", 1)[0]) == sample + 4
            metrics = output["spec_decode_metrics"]
            a = [r["outer_accepted"] for r in cycles]
            assert a == metrics["per_step_accepted"], (cell, request_id)
            assert [r["outer_scheduled"] for r in cycles] == metrics["per_step_drafted"]
            assert sum(a) == metrics["num_accepted_draft_tokens"]
            produced = 1
            for step, cycle in enumerate(cycles):
                remaining = 128 - produced
                returned = min(cycle["outer_accepted"], remaining)
                assert remaining > 0
                inner = cycle["inner_rounds"]
                assert len(inner) == 4
                offset = 0
                accepted_sum = returned_sum = 0
                for index, row in enumerate(inner):
                    assert row["inner_round"] == index
                    assert row["offset"] == offset
                    assert row["proposed"] == 4
                    assert 0 <= row["accepted"] <= 4
                    assert row["emitted"] == row["accepted"] + 1
                    accepted = overlap(cycle["outer_accepted"], offset, row["emitted"])
                    final = overlap(returned, offset, row["emitted"])
                    details.append(
                        {
                            "method": cell["name"],
                            "sample": sample,
                            "category": output["category"],
                            "request_id": request_id,
                            "cycle": step,
                            "round": index + 1,
                            "inner_accepted": row["accepted"],
                            "candidates": row["emitted"],
                            "offset": offset,
                            "target_accepted": accepted,
                            "returned_accepted": final,
                            "last_cycle": step == len(cycles) - 1,
                        }
                    )
                    offset += row["emitted"]
                    accepted_sum += accepted
                    returned_sum += final
                    used_inner += row["accepted"]
                    used_rounds += 1
                assert offset == cycle["outer_proposed"]
                assert cycle["outer_accepted"] <= cycle["outer_scheduled"] <= offset
                assert accepted_sum == cycle["outer_accepted"]
                assert returned_sum == returned
                produced += min(cycle["outer_accepted"] + 1, remaining)
            assert produced == 128
        aggregate = result["measurement"][0]["inner_counts"]
        assert aggregate[2] >= used_rounds and aggregate[1] >= used_inner
        audits.append(
            {
                "method": cell["name"],
                "matched_requests": 4,
                "verified_cycles": used_rounds // 4,
                "unverified_rounds_excluded": aggregate[2] - used_rounds,
                "unverified_inner_accepted_excluded": aggregate[1] - used_inner,
            }
        )
    summary, samples = [], []
    for cell in contract["cells"]:
        for round_number in range(1, 5):
            selected = [
                r
                for r in details
                if r["method"] == cell["name"] and r["round"] == round_number
            ]
            summary.append(
                {"method": cell["name"], "round": round_number, **summarize(selected)}
            )
            for sample in range(4):
                samples.append(
                    {
                        "method": cell["name"],
                        "round": round_number,
                        "sample": sample,
                        **summarize([r for r in selected if r["sample"] == sample]),
                    }
                )
    write_csv(root / "cycle_rounds.csv", details)
    write_csv(root / "round_summary.csv", summary)
    write_csv(root / "request_round_summary.csv", samples)
    write_json(root / "round_audit.json", {"status": "passed", "cells": audits})
    plot(root, summary)
    report = [
        "# Gemma four-round decay diagnostic",
        "",
        "B1, greedy, 4x128, MTP D4, four rounds, outer capacity20. "
        "Four warmup requests are excluded. Every measured trace is matched "
        "step by step to per-request Target counters. Only cycles that reach "
        "Target verification are included; final unused proposals are counted "
        "separately in round_audit.json. All rounds use the same matched cycles.",
        "",
        "Target yield includes correction/bonus tokens appended by Pre-Verify. "
        "Returned yield also clips the last cycle to the 128-token output limit. "
        "A zero Target contribution can result from an earlier-round rejection; "
        "it does not alone prove that this round has worse local quality. "
        "The trace writes synchronize and add I/O, so these runs do not establish "
        "performance or an optimal number of rounds. Existing Gemma AR repeat "
        "and speculative output parity limitations remain unresolved.",
        "",
        "| Method | Round | Inner accepted | Target accepted | "
        "Returned accepted | Zero Target |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        report.append(
            f"| {r['method']} | {r['round']} | {r['inner_accepted']:.3f} | "
            f"{r['target_accepted']:.3f} | {r['returned_accepted']:.3f} | "
            f"{r['zero_target_fraction']:.1%} |"
        )
    report += [
        "",
        "Reproduce: `.venv/bin/python benchmarks/hierarchical/run_round_decay.py "
        "<fresh_directory> --gpu 1`, then `.venv/bin/python "
        "benchmarks/hierarchical/analyze_round_decay.py <fresh_directory>`. "
        "AI assistance was used.",
    ]
    (root / "README.md").write_text("\n".join(report) + "\n")
    (root / "ANALYSIS_COMPLETE").write_text("5 cells, 20 measured requests audited\n")
    print(json.dumps(summary, indent=2))


def plot(root, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 13})
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), layout="constrained")
    panels = [
        ("inner_accepted", "Inner accepted / round"),
        ("target_accepted", "Target accepted / round"),
        ("zero_target_fraction", "Zero Target contribution"),
    ]
    colors = ["#4C78A8", "#76B7B2", "#B279A2", "#59A14F", "#F2B447"]
    methods = ["h4", "h6", "h8", "routing60", "attention60"]
    for panel, (ax, (metric, label)) in enumerate(zip(axes, panels, strict=True)):
        for method, color in zip(methods, colors, strict=True):
            values = [r[metric] for r in summary if r["method"] == method]
            ax.plot(range(1, 5), values, marker="o", color=color, label=method)
        ax.set_xticks(range(1, 5))
        ax.set_ylim(bottom=0)
        ax.set_ylabel(label)
        ax.set_xlabel(f"({chr(97 + panel)}) Inner round")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    axes[-1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=5,
        frameon=False,
        columnspacing=1,
    )
    folder = root / "figures/round_decay"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            folder / f"round_decay.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    (folder / "round_decay.md").write_text(
        "# Round decay\n\nSource: ../../round_summary.csv. Gemma4 on A100 80GB PCIe, "
        "B1/TP1, greedy, 4x128, MTP D4 and four rounds. Panels show inner acceptance, "
        "final Target contribution and zero-contribution frequency by inner round. "
        "Cycle-weighted means over matching verified cycles, no error bars. "
        "Output-capped yield is separately available in the CSV. See ../../README.md "
        "for audit and correctness limitations. Reproduce using "
        "benchmarks/hierarchical/analyze_round_decay.py.\n"
    )


if __name__ == "__main__":
    main()
