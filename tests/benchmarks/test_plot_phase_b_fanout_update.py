# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

import pytest

from benchmarks.replayssm import plot_phase_b_fanout_update as plot_update


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_baseline(root: Path) -> None:
    write_json(root / "matrix_complete.json", {"status": "complete"})
    write_json(
        root / "performance_measurement_complete.json",
        {"status": "complete"},
    )
    rows = []
    for model_index, model_key in enumerate(plot_update.MODEL_KEYS):
        for mode_index, mode in enumerate(("AR", "Sync", "Async")):
            rows.append(
                {
                    "model_key": model_key,
                    "decode_mode": mode,
                    "completion_throughput_tok_s": 10 + model_index + mode_index,
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "throughput.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for model_index, model_key in enumerate(plot_update.MODEL_KEYS):
        cell = root / "models" / model_key / "cells" / "performance_sync_eager_b1"
        write_json(
            cell / "result.json",
            {
                "status": "complete",
                "metrics_delta": {
                    'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': (
                        10 + model_index
                    ),
                    'vllm:spec_decode_num_drafts_total{engine="0"}': 5,
                },
            },
        )


def make_tuned(root: Path, model_key: str, d: int, fan_out: int) -> None:
    write_json(
        root / "manifest.json",
        {
            "contract": {
                "batch_size": 1,
                "prompt_count": 16,
                "output_length": 128,
                "warmup_seconds_minimum": 30.0,
            },
            "models": [{"key": model_key}],
        },
    )
    cell = root / model_key / "cells" / "performance_async_cache_eager_b1"
    write_json(cell / "cell_complete.json", {"status": "complete"})
    write_json(
        cell / "result.json",
        {
            "status": "complete",
            "expected_completion_tokens": 2048,
            "summary": {
                "completion_tokens": 2048,
                "completed_request_count": 16,
                "completion_throughput_tok_s": 20.0,
            },
            "warmup": {"seconds": 30.5},
        },
    )
    write_json(
        root / "fanout_analysis.json",
        {
            "rows": [
                {
                    "model": model_key,
                    "status": "complete",
                    "verify_width": d,
                    "fan_out": fan_out,
                    "branches_per_round": (d + 1) * fan_out,
                    "cache": {
                        "hits": 90,
                        "eligible_rounds": 100,
                        "hit_rate": 0.9,
                    },
                    "acceptance": {"mean_accepted_draft_count": 1.5},
                    "runtime_branch_audit": {"passed": True},
                }
            ]
        },
    )


def make_mtp_tuned(root: Path) -> None:
    write_json(
        root / "manifest.json",
        {
            "artifact_kind": "async_mtp_fanout_calibration",
            "contract": {
                "batch_size": 1,
                "prompt_count": 16,
                "output_length": 128,
                "verify_width": 3,
            },
        },
    )
    cell = root / "qwen36_mtp" / "cells" / "performance_async_cache_eager_b1"
    write_json(cell / "cell_complete.json", {"status": "complete"})
    write_json(
        cell / "result.json",
        {
            "status": "complete",
            "expected_completion_tokens": 2048,
            "summary": {
                "completion_tokens": 2048,
                "completed_request_count": 16,
                "completion_throughput_tok_s": 19.0,
            },
            "warmup": {"seconds": 30.5},
        },
    )
    write_json(
        root / "fanout_analysis.json",
        {
            "rows": [
                {
                    "status": "complete",
                    "verify_width": 3,
                    "fan_out": 96,
                    "branches_per_round": 384,
                    "cache": {
                        "hits": 98,
                        "eligible_rounds": 100,
                        "hit_rate": 0.98,
                    },
                    "acceptance": {"mean_accepted_draft_count": 1.75},
                    "runtime_branch_audit": {"passed": True},
                }
            ]
        },
    )


def test_render_preserves_provenance_and_writes_all_formats(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    qwen = tmp_path / "qwen"
    gemma = tmp_path / "gemma"
    make_baseline(baseline)
    make_tuned(qwen, "qwen36_dspark", 3, 24)
    make_tuned(gemma, "gemma4_dspark", 2, 48)

    args = plot_update.parse_args(
        [
            "--baseline-root",
            str(baseline),
            "--qwen-tuned-root",
            str(qwen),
            "--gemma-tuned-root",
            str(gemma),
        ]
    )
    rows = plot_update.render_outputs(args)

    assert len(rows) == 3
    assert rows[0]["async_tuned_tok_s"] is None
    assert rows[1]["tuned_branches_per_round"] == 96
    assert rows[2]["tuned_branches_per_round"] == 144
    for suffix in (".csv", ".json", ".md", ".png", ".svg"):
        assert (baseline / f"throughput_fanout_optimized{suffix}").stat().st_size > 0
        assert (
            baseline / f"throughput_cache_acceptance_tuned{suffix}"
        ).stat().st_size > 0
    summary = json.loads((baseline / "throughput_fanout_optimized.json").read_text())
    assert summary["correctness_status"] == "not_claimed_by_performance_plot"


def test_rejects_incomplete_tuned_cell(tmp_path) -> None:
    root = tmp_path / "qwen"
    make_tuned(root, "qwen36_dspark", 3, 24)
    marker = next(root.rglob("cell_complete.json"))
    write_json(marker, {"status": "failed"})

    with pytest.raises(ValueError, match="not complete"):
        plot_update.load_tuned(root, "qwen36_dspark")


def test_render_includes_tuned_mtp_when_supplied(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    mtp = tmp_path / "mtp"
    qwen = tmp_path / "qwen"
    gemma = tmp_path / "gemma"
    make_baseline(baseline)
    make_mtp_tuned(mtp)
    make_tuned(qwen, "qwen36_dspark", 3, 24)
    make_tuned(gemma, "gemma4_dspark", 2, 48)

    args = plot_update.parse_args(
        [
            "--baseline-root",
            str(baseline),
            "--mtp-tuned-root",
            str(mtp),
            "--qwen-tuned-root",
            str(qwen),
            "--gemma-tuned-root",
            str(gemma),
        ]
    )
    rows = plot_update.render_outputs(args)

    assert rows[0]["async_tuned_tok_s"] == 19.0
    assert rows[0]["tuned_fan_out"] == 96
    assert rows[0]["async_tuned_cache_hit_rate"] == 0.98
    assert rows[0]["async_tuned_mean_accepted_draft_count"] == 1.75
    assert rows[0]["sync_mean_accepted_draft_count"] == 2.0


def test_rejects_inconsistent_cache_hit_rate(tmp_path) -> None:
    root = tmp_path / "qwen"
    make_tuned(root, "qwen36_dspark", 3, 24)
    analysis_path = root / "fanout_analysis.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["rows"][0]["cache"]["hit_rate"] = 0.8
    write_json(analysis_path, analysis)

    with pytest.raises(ValueError, match="hit rate is inconsistent"):
        plot_update.load_tuned(root, "qwen36_dspark")
