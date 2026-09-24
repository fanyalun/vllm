# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("all_padding", [False, True])
def test_route_counts_exclude_padding_and_count_unique_experts(all_padding):
    import torch

    from benchmarks.hierarchical.routing_count_worker import route_counts

    native = torch.tensor([[0, 1, 2], [0, 2, 3], [3, 4, 5]])
    kept = torch.tensor([[0, -1, 2], [0, 2, -1], [3, 4, 5]])
    padding = torch.tensor([all_padding, all_padding, True])
    counts = route_counts(native, kept, padding, 6)
    assert counts.tolist() == ([0, 0, 0, 0] if all_padding else [4, 6, 2, 4])


def test_route_counts_update_on_graph_replay():
    import torch

    from benchmarks.hierarchical.routing_count_worker import route_counts

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    native = torch.tensor([[0, 1], [1, 2]], device="cuda")
    kept = native.clone()
    padding = torch.zeros(2, dtype=torch.bool, device="cuda")
    route_counts(native, kept, padding, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counts = route_counts(native, kept, padding, 4)
    kept[:, 1] = -1
    padding[1] = True
    graph.replay()
    assert counts.tolist() == [2, 2, 1, 1]
    padding.fill_(True)
    graph.replay()
    assert counts.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("batch_policy", [True, False])
def test_route_counts_exclude_verify_graph_construction_and_warmup(
    monkeypatch, batch_policy
):
    from types import SimpleNamespace

    import torch

    from benchmarks.hierarchical.routing_count_worker import RoutingCountWorker
    from vllm import forward_context
    from vllm.model_executor.layers.fused_moe.router import batch_expert_selection
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    native = torch.tensor([[0, 1], [1, 2]])
    kept = torch.tensor([[0, -1], [1, -1]]) if batch_policy else native[:, :1]
    monkeypatch.setattr(
        batch_expert_selection, "select_batch_experts", lambda *args: (None, kept)
    )
    monkeypatch.setattr(BaseRouter, "select_routing_top_k", lambda *args: (None, kept))
    monkeypatch.setattr(
        forward_context, "get_forward_context", lambda: SimpleNamespace(is_padding=None)
    )
    config = SimpleNamespace(
        preverify_gdn_mode="none",
        moe_skip_batch_policy="half" if batch_policy else None,
        moe_skip_top_h=1,
        moe_skip_weight_mode="preserve",
        moe_skip_min_weight=None,
    )
    spec = SimpleNamespace(
        config=config,
        device="cpu",
        preverify_graphs={"old": None},
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=1)
            )
        ),
    )

    def eager(*args):
        if batch_policy:
            batch_expert_selection.select_batch_experts(
                None, native, torch.zeros(2, 3), "half", None
            )
        else:
            BaseRouter.select_routing_top_k(
                SimpleNamespace(top_k=2), None, native, torch.zeros(2, 3)
            )

    def verify(*args):
        # Model private graph warmups, capture, and the actual forward.
        for _ in range(5):
            spec._verify_eager(*args)

    spec._verify_eager = eager
    spec._verify = verify
    worker = RoutingCountWorker()
    worker.model_runner = SimpleNamespace(speculator=spec)
    worker.setup_routing_counts()
    batch = SimpleNamespace(num_reqs=2, num_tokens=2)
    spec._verify(batch, None, None)
    worker.begin_routing_counts()
    for _ in range(2):
        spec._verify(batch, None, None)
    measured = worker.collect_routing_counts()
    assert measured["native_expert_invocations"] == 6
    assert measured["retained_expert_invocations"] == 4
    assert measured["native_connections"] == 8
    assert measured["retained_connections"] == 4
    assert measured["preverify_calls"] == [
        dict(active_requests=2, token_rows=2, calls=2)
    ]


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


def test_kernel_busy_time_does_not_double_count_overlapping_streams(scripts):
    union = scripts("analyze_performance_paths").union_duration
    assert union([]) == 0
    assert union([(0, 10), (2, 3), (5, 12), (15, 20)]) == 17
    assert union([(0, 1), (1, 2)]) == 2
