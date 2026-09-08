# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize paired Top-1/Top-2 draft continuations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = []
    offset_rows = []
    audit = {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    examples = {}
    for index, model in enumerate(("qwen36", "gemma4")):
        directory = args.root / model
        assert (directory / "RUN_COMPLETE").exists()
        result = json.loads((directory / "result.json").read_text())
        outputs, events = result["outputs"], result["events"]
        assert len(outputs) == 4
        assert all(len(o["token_ids"]) == 256 for o in outputs)
        assert events, "No eligible events; cannot estimate branch similarity"
        for event in events:
            j = event["draft_position"] - 1
            a, b = event["baseline"], event["branch"]
            assert a[:j] == b[:j]
            assert [a[j], b[j]] == event["top2_token_ids"]
            assert a[j] != b[j]
            mask = [x == y for x, y in zip(a[j + 1 :], b[j + 1 :], strict=True)]
            assert mask == event["suffix_matches"] and mask
            assert len(mask) == event["suffix_length"]
            assert event["emitted_tokens"] + len(a) <= 256
            assert 0 <= event["margin"] < 1
            assert event["all_equal"] == all(mask)
            assert event["all_different"] == (not any(mask))
            assert event["consecutive_matches"] == next(
                (i for i, same in enumerate(mask) if not same), len(mask)
            )
        control_dir = args.root / f"{model}_control"
        audit[model] = {"branch_token_audit": "passed", "control_complete": False}
        if (control_dir / "RUN_COMPLETE").exists():
            control = json.loads((control_dir / "result.json").read_text())["outputs"]
            assert (directory / "samples.json").read_bytes() == (
                control_dir / "samples.json"
            ).read_bytes()
            matches = [
                a["token_ids"] == b["token_ids"]
                for a, b in zip(outputs, control, strict=True)
            ]
            audit[model].update(
                control_complete=True,
                exact_control_requests=sum(matches),
                expected_control_requests=4,
                per_request_exact_match=matches,
            )
        n = len(events)
        full = sum(e["all_equal"] for e in events)
        different = sum(e["all_different"] for e in events)
        row = {
            "model": model,
            "samples": len(outputs),
            "proposals": sum(o["proposals"] for o in outputs),
            "events": n,
            "all_equal": full,
            "partial_match": n - full - different,
            "all_different": different,
            "mean_suffix_length": sum(e["suffix_length"] for e in events) / n,
            "mean_consecutive_matches": sum(e["consecutive_matches"] for e in events)
            / n,
            "any_consecutive_match_fraction": sum(
                e["consecutive_matches"] > 0 for e in events
            )
            / n,
            "aligned_token_match_fraction": sum(
                sum(e["suffix_matches"]) for e in events
            )
            / sum(e["suffix_length"] for e in events),
        }
        summary.append(row)
        bottom = 0
        for label, count, color in (
            ("Entire suffix equal", full, "#4c956c"),
            ("Partial aligned match", n - full - different, "#e9b44c"),
            ("No aligned match", different, "#cf5c60"),
        ):
            height = 100 * count / n
            axes[0].bar(
                model,
                height,
                bottom=bottom,
                color=color,
                label=label if index == 0 else None,
            )
            if height > 5:
                axes[0].text(
                    index,
                    bottom + height / 2,
                    f"{height:.1f}%\n({count}/{n})",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            bottom += height
        xs, survival, aligned = [], [], []
        for k in range(1, 16):
            eligible = [e for e in events if e["suffix_length"] >= k]
            if not eligible:
                continue
            xs.append(k)
            survival.append(
                100
                * sum(e["consecutive_matches"] >= k for e in eligible)
                / len(eligible)
            )
            aligned.append(
                100 * sum(e["suffix_matches"][k - 1] for e in eligible) / len(eligible)
            )
            offset_rows.append(
                {
                    "model": model,
                    "offset": k,
                    "eligible_events": len(eligible),
                    "continuous_match_percent": survival[-1],
                    "aligned_match_percent": aligned[-1],
                }
            )
        axes[1].plot(xs, survival, "o-", label=f"{model} (n={n})")
        axes[2].plot(xs, aligned, "o-", label=model)
        examples[model] = {}
        for kind in ("all_equal", "all_different", "partial"):
            if kind == "partial":
                candidates = [
                    e for e in events if not e["all_equal"] and not e["all_different"]
                ]
            else:
                candidates = [e for e in events if e[kind]]
            examples[model][kind] = sorted(
                candidates, key=lambda e: e["suffix_length"], reverse=True
            )[:3]
    axes[0].set(title="Suffix outcome after forced Top-2", ylabel="Events (%)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
    for ax, title in zip(
        axes[1:],
        ("Continuous prefix survival", "Individual aligned-token agreement"),
        strict=True,
    ):
        ax.set(
            title=title,
            xlabel="Offset after branching token",
            ylabel="Match (%) among suffixes reaching this offset",
            ylim=(0, 105),
        )
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        "MoE-Skip draft branch sensitivity | 4 prompts/model x 256 tokens | "
        "B=1, D=16, top-h=4, margin < 1\n"
        "Both continuations use draft greedy decoding; this is not Target acceptance"
    )
    fig.savefig(args.root / "branch_comparison.png", dpi=180)
    fig.savefig(args.root / "branch_comparison.pdf")
    with (args.root / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.root / "offset_metrics.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(offset_rows[0]))
        writer.writeheader()
        writer.writerows(offset_rows)
    (args.root / "examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n"
    )
    (args.root / "aggregate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
