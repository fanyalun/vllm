# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit complete batch-policy smoke matrices and report paired results."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(root):
    rows = []
    for batch in (1, 4):
        folder = root / f"b{batch}"
        matrix = json.loads((folder / "matrix_complete.json").read_text())
        ar_outputs = json.loads((folder / "ar_native.json").read_text())["outputs"]
        assert len(matrix["results"]) == 7
        expected = {("ar", "native")} | {
            (m, p)
            for m in ("moe_skip", "hierarchical")
            for p in ("native", "batch_top_half", "batch_max_gap")
        }
        assert {(r["method"], r["policy"]) for r in matrix["results"]} == expected
        for cell in matrix["results"]:
            name = f"{cell['method']}_{cell['policy']}"
            path = folder / f"{name}.json"
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == cell["result_sha256"]
            )
            data = json.loads(path.read_text())
            assert data["complete"] and len(data["outputs"]) == 4
            spec = data["speculative_config"] or {}
            assert spec.get("method", "ar") == cell["method"]
            assert spec.get("moe_skip_batch_policy") == (
                None if cell["policy"] == "native" else cell["policy"]
            )
            if cell["method"] != "ar" and cell["policy"] == "native":
                assert spec["moe_skip_top_h"] == 8
            assert data["args"]["batch_size"] == batch
            assert data["args"]["gdn_mode"] == "none"
            assert data["args"]["temperature"] == 0
            assert data["args"]["repeats"] == 1
            assert data["args"]["max_tokens"] == 128
            assert not data["args"]["eager"]
            if cell["method"] == "moe_skip":
                assert spec["num_speculative_tokens"] == 4
            elif cell["method"] == "hierarchical":
                assert spec["inner_method"] == "mtp"
                assert spec["inner_num_speculative_tokens"] == 4
                assert spec["inner_num_rounds"] == 1
                assert spec["num_speculative_tokens"] == 5
            assert len(data["batches"]) == 4 // batch
            assert sum(b["returned_tokens"] for b in data["batches"]) == 512
            assert all(b["elapsed_seconds"] > 0 for b in data["batches"])
            assert math.isclose(
                data["returned_token_throughput"],
                512 / sum(b["elapsed_seconds"] for b in data["batches"]),
            )
            assert data["returned_token_throughput"] == cell["throughput"]
            matched = 0
            for output, reference in zip(data["outputs"], ar_outputs):
                for key in ("repeat", "sample_index", "prompt_sha256", "seed"):
                    assert output[key] == reference[key]
                matched += output["token_ids"] == reference["token_ids"]
                assert len(output["token_ids"]) == 128
                metrics = output["spec_decode_metrics"]
                if cell["method"] != "ar":
                    assert metrics is not None
                    hist = metrics["histogram"]
                    assert sum(hist) == len(metrics["per_step_accepted"])
                    assert sum(i * n for i, n in enumerate(hist)) == sum(
                        metrics["per_step_accepted"]
                    )
                    assert metrics["num_draft_tokens"] == sum(
                        metrics["per_step_drafted"]
                    )
            if cell["method"] != "ar":
                assert data["ar_parity"] == dict(matched=matched, total=4)
                assert data["ar_parity"] == cell["ar_parity"]
                metrics = [o["spec_decode_metrics"] for o in data["outputs"]]
                accepted = sum(sum(m["per_step_accepted"]) for m in metrics)
                drafted = sum(m["num_draft_tokens"] for m in metrics)
                steps = sum(len(m["per_step_accepted"]) for m in metrics)
                assert data["acceptance"] == dict(
                    drafted=drafted,
                    accepted=accepted,
                    steps=steps,
                    acceptance_rate=accepted / drafted,
                    mean_acceptance_length=1 + accepted / steps,
                )
                assert data["acceptance"] == cell["acceptance"]
            log = (folder / f"{name}.log").read_text()
            assert "WARMUP_COMPLETE" in log
            late_jit = (
                "JIT compilation during inference" in log.split("WARMUP_COMPLETE", 1)[1]
            )
            rows.append(dict(batch=batch, **cell, late_jit=late_jit))
    for row in rows:
        native = next(
            r
            for r in rows
            if r["batch"] == row["batch"]
            and r["method"] == row["method"]
            and r["policy"] == "native"
        )
        ar = next(r for r in rows if r["batch"] == row["batch"] and r["method"] == "ar")
        row["speedup_vs_native"] = row["throughput"] / native["throughput"]
        row["speedup_vs_ar"] = row["throughput"] / ar["throughput"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.root)
    (args.root / "generation_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    text = [
        "# Batch-policy generation smoke",
        "",
        "Four fixed prompts, 128 greedy returned tokens each, exact GDN. "
        "Each batch has its own same-device AR and native-route controls.",
        "",
        "| B | Method | Policy | tok/s | vs native | vs AR | "
        "Accept length | AR parity | Late JIT |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        length = (
            f"{row['acceptance']['mean_acceptance_length']:.3f}"
            if row["acceptance"]
            else "-"
        )
        parity = row["ar_parity"]
        parity = f"{parity['matched']}/{parity['total']}" if parity else "reference"
        text.append(
            f"| {row['batch']} | {row['method']} | {row['policy']} | "
            f"{row['throughput']:.2f} | {row['speedup_vs_native']:.3f}x | "
            f"{row['speedup_vs_ar']:.3f}x | {length} | {parity} | "
            f"{row['late_jit']} |"
        )
    text += [
        "",
        "A mismatch is not lossless parity. Rows with late JIT are not "
        "steady-state throughput evidence.",
        "",
    ]
    (args.root / "generation_summary.md").write_text("\n".join(text))
    (args.root / "generation_audit.json").write_text(
        json.dumps(
            dict(
                complete=True,
                cells=len(rows),
                returned_tokens=14 * 512,
                late_jit_cells=sum(r["late_jit"] for r in rows),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
