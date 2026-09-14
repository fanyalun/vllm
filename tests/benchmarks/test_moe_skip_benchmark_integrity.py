# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks" / "moe_skip")
    )
    return importlib.import_module


def test_resume_preserves_original_contract_on_mismatch(tmp_path, scripts):
    preserve = scripts("benchmark_integrity").preserve_contract
    contract: dict = {"model": "/original", "draft_lengths": [4, 8]}
    preserve(tmp_path, contract)
    path = tmp_path / "EXPERIMENT_CONTRACT.json"
    original = path.read_bytes()
    preserve(tmp_path, contract)
    changes: list[dict] = [{"model": "/different"}, {"draft_lengths": [4]}]
    for change in changes:
        with pytest.raises(RuntimeError, match="contract mismatch"):
            preserve(tmp_path, contract | change)
        assert path.read_bytes() == original


def test_resume_rejects_missing_contract_with_existing_results(tmp_path, scripts):
    (tmp_path / "cells").mkdir()
    with pytest.raises(RuntimeError, match="Missing original"):
        scripts("benchmark_integrity").preserve_contract(tmp_path, {"model": "/a"})


@pytest.mark.parametrize(
    "module", ["run_large_experiment", "run_draft_model_experiment"]
)
def test_completed_cell_rejects_wrong_target(tmp_path, scripts, module):
    runner = scripts(module)
    path = tmp_path / "cell_output.json"
    path.write_text(json.dumps({"model": "/wrong"}))
    kwargs = dict(
        path=path,
        model="/expected",
        dataset_sha256="unused",
        draft_length=4,
        max_num_batched_tokens=1024,
    )
    if module == "run_draft_model_experiment":
        kwargs.update(method="mtp", spec_model=None)
    with pytest.raises(RuntimeError, match="Target model mismatch"):
        runner.validate_cell(**kwargs)


def test_analyzer_rejects_wrong_target_before_reading_trace(tmp_path, scripts):
    (tmp_path / "EXPERIMENT_CONTRACT.json").write_text('{"model": "/expected"}')
    cell = tmp_path / "cells" / "moe_skip_top4_graph_d4"
    cell.mkdir(parents=True)
    (cell / "cell_output.json").write_text('{"model": "/wrong"}')
    with pytest.raises(RuntimeError, match="Target model mismatch"):
        scripts("analyze_large_experiment").combine_traces(tmp_path, "moe_skip", (4,))


def test_incomplete_trace_is_preserved_and_not_appended(tmp_path, scripts):
    guard = scripts("benchmark_integrity").refuse_incomplete_trace
    guard(tmp_path)
    trace = tmp_path / "trace" / "raw_trace.jsonl"
    trace.parent.mkdir()
    trace.write_text('{"request_id": "0"}\n')
    with pytest.raises(RuntimeError, match="incomplete trace"):
        guard(tmp_path)
    assert trace.read_text() == '{"request_id": "0"}\n'


def test_duplicate_trace_is_rejected_before_summary(tmp_path, scripts, monkeypatch):
    summary = scripts("summarize")
    monkeypatch.setattr(summary, "DRAFT_LENGTHS", (1,))
    name = "moe_skip_top4_graph_d1"
    trace_dir = tmp_path / "cells" / name / "trace"
    trace_dir.mkdir(parents=True)
    row = dict(request_id="0", verify_step=0, draft_position=1, valid_mask=True)
    (trace_dir / "raw_trace.jsonl").write_text((json.dumps(row) + "\n") * 2)
    cells = {
        name: {
            "outputs": [
                dict(
                    request_id="0",
                    sample_index=0,
                    token_ids=[1, 2],
                    spec_decode_metrics={"per_step_accepted": [1]},
                )
            ]
        }
    }
    with pytest.raises(RuntimeError, match="Duplicate trace"):
        summary.combine_traces(tmp_path, cells)
    assert not (tmp_path / "raw_trace.jsonl").exists()


def test_tied_logits_use_actual_argmax_for_rank_and_position(
    tmp_path, scripts, monkeypatch
):
    summary = scripts("summarize")
    monkeypatch.setattr(summary, "DRAFT_LENGTHS", (1,))
    row = dict(
        draft_length=1,
        draft_position=1,
        valid_mask=True,
        target_top1_token_id=7,
        draft_top8_token_ids=[8, 7, 9],
        draft_argmax_ordered_top8_token_ids=[7, 8, 9],
    )
    for metric in [summary.write_rank_recall_summary, summary.write_position_metrics]:
        rows = metric(tmp_path, [row])
        assert all(r["top1_precision"] == 1 for r in rows)
        assert all(r["top2_recall"] == 1 for r in rows)


def test_performance_retry_archives_failed_attempt_and_can_retry_again(
    tmp_path, scripts
):
    prepare = scripts("benchmark_integrity").prepare_performance_retry
    directory = tmp_path / "cell"
    prepare(directory)
    for attempt in (1, 2):
        (directory / "run.log").write_text(f"failed {attempt}")
        (directory / "result.json").write_text("partial")
        prepare(directory)
        assert (directory / "attempts" / str(attempt) / "run.log").read_text() == (
            f"failed {attempt}"
        )
        assert not (directory / "result.json").exists()
