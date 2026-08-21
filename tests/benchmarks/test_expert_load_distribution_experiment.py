# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    _get_num_experts_per_tok,
)

EXPERIMENT_PATH = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "replayssm"
    / "expert_load_distribution_experiment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "expert_load_distribution_experiment", EXPERIMENT_PATH
)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def model_spec(num_layers: int = 3, num_experts: int = 8):
    return experiment.ModelSpec(
        key="synthetic",
        model_path="unused",
        drafter_path=None,
        draft_method="mtp_replayssm",
        num_layers=num_layers,
        num_experts=num_experts,
        selected_layers=(0, 1, 2),
    )


def synthetic_step(mode: str, *, step: int = 10):
    rows = []
    requests_per_rank = 8 if mode == "spec" else 32
    stages = experiment.SPEC_STAGES if mode == "spec" else ("ar_decode",)
    for rank in range(2):
        for request in range(requests_per_rank):
            sample_id = rank + 2 * request
            for stage_index, stage in enumerate(stages):
                routes = np.empty((3, 8), dtype=np.int64)
                for layer in range(3):
                    routes[layer] = np.arange(8)
                rows.append(
                    {
                        "rank": rank,
                        "scheduler_step": step,
                        "request_id": f"r{rank}-{request}",
                        "sample_id": sample_id,
                        "prompt_token_count": 20,
                        "absolute_position": 36 + stage_index,
                        "generation_position": 16 + stage_index,
                        "token_id": 100 + stage_index,
                        "row_in_request": stage_index,
                        "route_kind": stage,
                        "accepted": stage_index < 2,
                        "routes": routes,
                    }
                )
    return rows


def test_top_k_config_fields_for_qwen_and_gemma():
    assert _get_num_experts_per_tok(SimpleNamespace(num_experts_per_tok=8)) == 8
    assert _get_num_experts_per_tok(SimpleNamespace(top_k_experts=8)) == 8
    with pytest.raises(ValueError, match="neither"):
        _get_num_experts_per_tok(SimpleNamespace())


def test_seeded_sample_manifest_is_unique_stable_and_ranked(tmp_path):
    dataset = tmp_path / "gsm8k.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"question": f"question {index}", "answer": "x"}) + "\n"
            for index in range(1319)
        ),
        encoding="utf-8",
    )
    first = experiment.build_sample_manifest(dataset)
    second = experiment.build_sample_manifest(dataset)
    experiment.validate_sample_manifest(first)
    first_rows = [sample["source_index"] for sample in first["samples"]]
    second_rows = [sample["source_index"] for sample in second["samples"]]
    assert first_rows == second_rows
    assert len(set(first_rows)) == 64
    assert [sample["data_parallel_rank"] for sample in first["samples"]] == [
        index % 2 for index in range(64)
    ]


def test_spec_step_metadata_acceptance_and_assignment_contract():
    rows = synthetic_step("spec")
    summary = experiment.validate_step(rows, mode="spec", model=model_spec())
    assert summary["valid"]
    assert summary["rows"] == 64
    assert summary["stage_rows"] == {stage: 16 for stage in experiment.SPEC_STAGES}
    assert summary["draft_acceptance"]["accepted_rows"] == 16
    ranked = experiment.ranked_loads(rows, layer=0, num_experts=8)
    assert sum(count for _, _, count in ranked) == 512


def test_step_validation_fails_closed_for_missing_rank_and_duplicate_stage():
    rows = synthetic_step("spec")
    missing_rank = [row for row in rows if row["rank"] == 0]
    summary = experiment.validate_step(
        missing_rank, mode="spec", model=model_spec()
    )
    assert not summary["valid"]
    assert "missing DP rank" in summary["reasons"]
    duplicate = rows + [dict(rows[0])]
    summary = experiment.validate_step(duplicate, mode="spec", model=model_spec())
    assert not summary["valid"]
    assert "duplicate request/stage row" in summary["reasons"]


def test_ar_step_requires_64_distinct_requests():
    rows = synthetic_step("ar")
    summary = experiment.validate_step(rows, mode="ar", model=model_spec())
    assert summary["valid"]
    rows[-1] = {**rows[-1], "request_id": rows[-2]["request_id"]}
    summary = experiment.validate_step(rows, mode="ar", model=model_spec())
    assert not summary["valid"]


def test_ranked_loads_descending_with_expert_id_tie_break():
    rows = synthetic_step("ar")
    ranked = experiment.ranked_loads(rows, layer=0, num_experts=10)
    assert [expert for _, expert, _ in ranked[:8]] == list(range(8))
    assert [count for _, _, count in ranked[:8]] == [64] * 8
    assert ranked[-2:] == [(9, 8, 0), (10, 9, 0)]


def test_early_late_selection_is_distinct_and_tie_breaks_by_step():
    candidates = [
        {
            "valid": True,
            "scheduler_step": step,
            "position": {"min": position, "median": position, "max": position},
        }
        for step, position in ((8, 15), (7, 17), (20, 47), (21, 49))
    ]
    early, late = experiment.choose_distinct_steps(candidates, (16, 48))
    assert early["scheduler_step"] == 7
    assert late["scheduler_step"] == 20
    assert early["scheduler_step"] != late["scheduler_step"]


def test_synthetic_trace_plot(tmp_path):
    model = model_spec(num_experts=8)
    rows = {
        (label, mode): synthetic_step(mode)
        for label in ("early", "late")
        for mode in ("spec", "ar")
    }
    experiment.plot_model(model, rows, tmp_path)
    png = tmp_path / "synthetic_expert_load_composite.png"
    pdf = tmp_path / "synthetic_expert_load_composite.pdf"
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
