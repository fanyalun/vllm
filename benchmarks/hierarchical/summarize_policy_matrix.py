# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit the complete policy matrix and compare throughput and acceptance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(root):
    contract = json.loads((root / "contract.json").read_text())
    assert contract["samples"] == 16 and contract["output_length"] == 512
    assert {tuple(cell) for cell in contract["cells"]} == {
        (batch, mode)
        for batch in (1, 4, 8, 16)
        for mode in ("ar", "mtp", "low_error", "balanced", "aggressive")
    }
    assert len(contract["cells"]) == 20
    for name, digest in contract["source_sha256"].items():
        assert (
            hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
        )
    dataset = root / "dataset.jsonl"
    assert (
        hashlib.sha256(dataset.read_bytes()).hexdigest() == contract["dataset_sha256"]
    )
    samples = [json.loads(line) for line in dataset.read_text().splitlines()]
    results = {}
    for batch, mode in contract["cells"]:
        folder = root / f"b{batch}_{mode}"
        assert (folder / "CELL_COMPLETE").exists(), str(folder)
        result = json.loads((folder / "result.json").read_text())
        assert result["source_sha256"] == contract["source_sha256"]
        assert result["batch_size"] == batch and result["mode"] == mode
        assert len(result["outputs"]) == 16 and result["output_tokens"] == 8192
        assert all(row["size"] == batch for row in result["batches"])
        assert result["counters"].get("preverify_graphs", 0) == result["warmup_graphs"]
        for sample, output in zip(samples, result["outputs"], strict=True):
            assert sample["prompt_sha256"] == output["prompt_sha256"]
            assert len(output["token_ids"]) == 512
            metrics = output["spec_decode_metrics"]
            if metrics:
                accepted = metrics["per_step_accepted"]
                drafted = metrics["per_step_drafted"]
                assert len(accepted) == len(drafted) == metrics["num_spec_steps"]
                assert sum(accepted) == metrics["num_accepted_draft_tokens"]
                assert sum(drafted) == metrics["num_draft_tokens"]
                assert all(0 <= a <= d for a, d in zip(accepted, drafted, strict=True))
        results[batch, mode] = result
    summaries, comparisons = [], []
    for (batch, mode), result in results.items():
        metrics = [r["spec_decode_metrics"] for r in result["outputs"]]
        accepted = sum(r["num_accepted_draft_tokens"] for r in metrics if r)
        drafted = sum(r["num_draft_tokens"] for r in metrics if r)
        steps = sum(r["num_spec_steps"] for r in metrics if r)
        matches = 0
        for index, (output, reference) in enumerate(
            zip(result["outputs"], results[batch, "ar"]["outputs"], strict=True)
        ):
            first = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(output["token_ids"], reference["token_ids"], strict=True)
                    )
                    if a != b
                ),
                None,
            )
            matches += first is None
            comparisons.append(
                {
                    "batch": batch,
                    "mode": mode,
                    "sample": index,
                    "equal_ar": first is None,
                    "first_difference": first,
                }
            )
        counters = result["counters"]
        summaries.append(
            {
                "batch": batch,
                "mode": mode,
                "seconds": result["seconds"],
                "tokens_per_second": 8192 / result["seconds"],
                "speedup_ar": results[batch, "ar"]["seconds"] / result["seconds"],
                "speedup_mtp": results[batch, "mtp"]["seconds"] / result["seconds"],
                "target_engine_steps": counters["engine_steps"],
                "outer_request_steps": steps,
                "outer_proposed": drafted,
                "outer_accepted": accepted,
                "outer_acceptance_rate": accepted / drafted if drafted else None,
                "mean_accepted_draft": accepted / steps if steps else None,
                "mean_emitted_including_bonus": 1 + accepted / steps if steps else None,
                "inner_rounds": counters.get("inner_rounds"),
                "inner_acceptance_rate": counters.get("inner_accepted", 0)
                / counters["inner_proposed"]
                if counters.get("inner_proposed")
                else None,
                "early_stops": counters.get("early_stops"),
                "skipped_request_rounds": counters.get("skipped_rounds"),
                "batch_round_calls": counters.get("batch_round_calls"),
                "exact_ar_requests": matches,
            }
        )
    for name, rows in (("summary", summaries), ("output_comparisons", comparisons)):
        (root / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (root / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    (root / "MATRIX_AUDIT_COMPLETE").write_text(
        "20 cells; exact requested coverage, counts and warm graphs audited; "
        "AR output equality reported separately\n"
    )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
