# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

import pytest

from benchmarks.replayssm.async_ssd_eagle3_matrix import (
    Cell,
    audit_draft_trace_pair,
    audit_lifecycle,
    audit_performance,
    audit_request_pair,
    checkpoint_manifest,
    correctness_cells,
    draft_trace_statistics,
    is_batch_shape_numerical_exception,
    lifecycle_cells,
    normalize_template_token_ids,
    parse_prometheus,
    performance_cells,
    server_command,
    write_json,
)


def make_request(tokens, top_logprobs=None):
    if top_logprobs is None:
        top_logprobs = [None] * len(tokens)
    return {"token_ids": tokens, "top_logprobs": top_logprobs}


def make_trace(draft_tokens, accepted=0, top2=None):
    record = {
        "accepted_draft_count": accepted,
        "num_rejected": 7 - accepted,
        "recovery_token": 99,
        "accepted_draft_tokens": [],
        "draft_tokens": draft_tokens,
    }
    if top2 is not None:
        record["draft_top2"] = top2
    return record


def make_args() -> Namespace:
    return Namespace(
        target="/target",
        draft="/draft",
        target_device=0,
        draft_device=1,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        num_speculative_tokens=7,
        draft_tie_logit_tolerance=0.1,
        acceptance_length_relative_tolerance=0.01,
    )


def test_phase_a_matrix_has_all_expected_cells() -> None:
    assert len(correctness_cells()) == 24
    assert len(lifecycle_cells()) == 3
    assert len(performance_cells()) == 18
    assert len({cell.name for cell in performance_cells()}) == 18
    assert Cell("correctness", "ar", "eager", 1).name == ("correctness_ar_eager_b1")


def test_normalize_template_token_ids_accepts_batch_encoding_shape() -> None:
    rendered = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    assert normalize_template_token_ids(rendered) == [1, 2, 3]


def test_normalize_template_token_ids_rejects_mapping_keys() -> None:
    try:
        normalize_template_token_ids(["input_ids", "attention_mask"])
    except TypeError as error:
        assert "flat integer token list" in str(error)
    else:
        raise AssertionError("non-token template output was accepted")


def test_output_audit_accepts_exact_tokens() -> None:
    request = make_request([1, 2, 3])

    assert audit_request_pair(request, request, 1e-4) == {"status": "exact"}


def test_output_audit_accepts_only_a_two_sided_target_tie() -> None:
    baseline = make_request(
        [1, 2],
        [None, {"token_id:2": -1.0, "token_id:3": -1.00005}],
    )
    candidate = make_request(
        [1, 3],
        [None, {"token_id:3": -1.0, "token_id:2": -1.00005}],
    )

    audit = audit_request_pair(baseline, candidate, 1e-4)

    assert audit["status"] == "target_top1_tie_equivalent"
    assert audit["offset"] == 1


def test_output_audit_rejects_a_non_tie_divergence() -> None:
    baseline = make_request(
        [1, 2],
        [None, {"token_id:2": -1.0, "token_id:3": -1.2}],
    )
    candidate = make_request(
        [1, 3],
        [None, {"token_id:3": -1.0, "token_id:2": -1.2}],
    )

    audit = audit_request_pair(baseline, candidate, 1e-4)

    assert audit["status"] == "failed"
    assert audit["reason"] == "non_tie_token_divergence"


def test_draft_trace_audit_accepts_rejected_tail_top1_tie() -> None:
    top2_a = [
        {"token_ids": [1, 8], "logits": [2.0, 1.0], "gap": 1.0},
        {"token_ids": [2, 9], "logits": [2.0, 1.0], "gap": 1.0},
        {"token_ids": [3, 4], "logits": [1.0, 0.99995], "gap": 0.00005},
    ]
    top2_b = [
        {"token_ids": [1, 8], "logits": [2.0, 1.0], "gap": 1.0},
        {"token_ids": [2, 9], "logits": [2.0, 1.0], "gap": 1.0},
        {"token_ids": [4, 3], "logits": [1.0, 0.99995], "gap": 0.00005},
    ]
    baseline = [make_trace([1, 2, 3], top2=top2_a), make_trace([7], accepted=2)]
    candidate = [make_trace([1, 2, 4], top2=top2_b), make_trace([7], accepted=2)]

    audit = audit_draft_trace_pair(baseline, candidate, 1e-4)

    assert audit["status"] == "draft_top1_tie_cascade_equivalent"


def test_draft_trace_audit_accepts_tie_with_different_runner_up() -> None:
    baseline_top2 = [{"token_ids": [3, 4], "logits": [1.0, 1.0], "gap": 0.0}]
    candidate_top2 = [{"token_ids": [4, 9], "logits": [1.01, 1.0], "gap": 0.01}]
    baseline = [make_trace([3], top2=baseline_top2)]
    candidate = [make_trace([4], top2=candidate_top2)]

    audit = audit_draft_trace_pair(baseline, candidate, 0.04)

    assert audit["status"] == "draft_top1_tie_cascade_equivalent"
    tie = audit["first_tie_divergence"]
    assert tie["baseline_runner_up_token"] == 4
    assert tie["candidate_runner_up_token"] == 9


def test_draft_trace_audit_accepts_outcome_cascade_after_first_tie() -> None:
    baseline_top2 = [{"token_ids": [3, 4], "logits": [1.0, 0.99], "gap": 0.01}]
    candidate_top2 = [{"token_ids": [4, 3], "logits": [1.0, 0.99], "gap": 0.01}]
    baseline = [
        make_trace([3], top2=baseline_top2),
        make_trace([7], accepted=1),
    ]
    candidate = [
        make_trace([4], top2=candidate_top2),
        make_trace([8], accepted=0),
    ]

    audit = audit_draft_trace_pair(baseline, candidate, 0.04)

    assert audit["status"] == "draft_top1_tie_cascade_equivalent"
    assert audit["strict_prefix_rounds"] == 1


def test_draft_trace_audit_rejects_outcome_mismatch_before_tie() -> None:
    baseline = [make_trace([1]), make_trace([2], accepted=1)]
    candidate = [make_trace([1]), make_trace([2], accepted=0)]

    audit = audit_draft_trace_pair(baseline, candidate, 0.04)

    assert audit["status"] == "failed"
    assert audit["reason"] == "outcome_mismatch_before_first_draft_tie"


def test_draft_trace_audit_accepts_target_tie_at_outcome_offset() -> None:
    baseline = [make_trace([1]), make_trace([2], accepted=1)]
    candidate = [make_trace([1]), make_trace([2], accepted=0)]
    baseline[1]["accepted_draft_tokens"] = [2]
    candidate[1]["recovery_token"] = 3
    target_audit = {
        "status": "target_top1_tie_equivalent",
        "offset": 1,
        "baseline_logprob_gap": 0.01,
        "candidate_logprob_gap": 0.0,
    }

    audit = audit_draft_trace_pair(
        baseline,
        candidate,
        0.1,
        target_audit,
    )

    assert audit["status"] == "target_top1_tie_cascade_equivalent"
    assert audit["first_outcome_divergence_output_offset"] == 1


def test_draft_trace_audit_rejects_target_tie_at_other_offset() -> None:
    baseline = [make_trace([1]), make_trace([2], accepted=1)]
    candidate = [make_trace([1]), make_trace([2], accepted=0)]
    baseline[1]["accepted_draft_tokens"] = [2]
    candidate[1]["recovery_token"] = 3
    target_audit = {
        "status": "target_top1_tie_equivalent",
        "offset": 2,
    }

    audit = audit_draft_trace_pair(
        baseline,
        candidate,
        0.1,
        target_audit,
    )

    assert audit["status"] == "failed"
    assert audit["outcome_divergence_output_offset"] == 1


def test_draft_trace_audit_rejects_accepted_prefix_divergence() -> None:
    baseline = [make_trace([1, 2, 3]), make_trace([7], accepted=3)]
    candidate = [make_trace([1, 2, 4]), make_trace([7], accepted=3)]

    audit = audit_draft_trace_pair(baseline, candidate, 1e-4)

    assert audit["status"] == "failed"
    assert audit["reason"] == "accepted_draft_prefix_mismatch"


def test_draft_trace_statistics_records_acceptance_behavior() -> None:
    traces = {
        0: [make_trace([1, 2], accepted=2), make_trace([3], accepted=0)],
        1: [make_trace([4, 5], accepted=1)],
    }

    stats = draft_trace_statistics(traces)

    assert stats == {
        "num_drafts": 3,
        "num_draft_tokens": 5,
        "num_accepted_tokens": 3,
        "accepted_counts_per_position": [2, 1, 0, 0, 0, 0, 0],
        "mean_accepted_draft_length": 1.0,
        "mean_acceptance_length": 2.0,
    }


def test_batch_shape_exception_requires_all_evidence() -> None:
    trace_audit = {
        "status": "failed",
        "reason": "non_tie_draft_divergence",
        "baseline_token": 10,
        "baseline_runner_up_token": 20,
        "candidate_token": 20,
        "candidate_runner_up_token": 10,
    }
    smaller = [
        {"batch_size": 1, "status": "exact"},
        {"batch_size": 4, "status": "draft_top1_tie_cascade_equivalent"},
    ]

    assert is_batch_shape_numerical_exception(
        trace_audit,
        {"status": "exact"},
        16,
        smaller,
        0.005,
        0.01,
    )
    assert not is_batch_shape_numerical_exception(
        trace_audit,
        {"status": "target_top1_tie_equivalent"},
        16,
        smaller,
        0.005,
        0.01,
    )
    assert not is_batch_shape_numerical_exception(
        trace_audit,
        {"status": "exact"},
        16,
        smaller[:1],
        0.005,
        0.01,
    )
    assert not is_batch_shape_numerical_exception(
        trace_audit,
        {"status": "exact"},
        16,
        smaller,
        0.02,
        0.01,
    )


def test_server_command_isolates_async_and_preemption_devices() -> None:
    args = make_args()
    cell = Cell(
        "lifecycle",
        "async_cache",
        "eager",
        4,
        variant="preemption",
    )

    command = server_command(args, cell, 43100)

    config = command[command.index("--speculative-config") + 1]
    assert '"async_draft_device": 1' in config
    assert command[command.index("--device-ids") + 1] == "0"
    assert command[command.index("--max-model-len") + 1] == "256"
    assert command[command.index("--num-gpu-blocks-override") + 1] == "32"
    assert "--no-async-scheduling" in command


def test_correctness_cache_cell_preserves_real_cache_environment(
    monkeypatch, tmp_path
) -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    args = make_args()
    args.resume = False
    args.startup_timeout = 1
    args.warmup_seconds = 0
    args.output_length = 1
    monkeypatch.setattr(matrix, "wait_for_server", lambda *args: None)
    monkeypatch.setattr(matrix, "warmup_server", lambda *args: {})
    monkeypatch.setattr(matrix, "scrape_metrics", lambda *args: ("", {}))
    monkeypatch.setattr(matrix, "run_requests", lambda **kwargs: ([], 0.1))
    monkeypatch.setattr(matrix, "stop_server", lambda process: {"forced_kill": False})
    monkeypatch.setattr(matrix.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(matrix.GpuSampler, "start", lambda self: None)
    monkeypatch.setattr(matrix.GpuSampler, "stop", lambda self: None)
    cell = Cell("correctness", "async_cache", "eager", 1)

    with pytest.raises(AssertionError):
        matrix.run_cell(
            args,
            tmp_path,
            [{"prompt_index": 0, "token_ids": [1]}],
            cell,
            43100,
        )

    environment = __import__("json").loads(
        (tmp_path / "cells" / cell.name / "environment.json").read_text()
    )
    assert "ASYNC_DRAFT_FORCE_JIT" not in environment
    assert "ASYNC_DRAFT_VALIDATE_HITS" not in environment
    assert environment["REPLAYSSM_SPEC_DECODE_TRACE_LOGITS"] == "1"


def test_prometheus_parser_keeps_only_phase_a_metrics() -> None:
    metrics = parse_prometheus(
        """
# HELP ignored ignored
vllm:spec_decode_num_drafts{engine=\"0\"} 12
vllm:async_draft_cache_hits_total{engine=\"0\"} 4
vllm:num_preemptions_total{engine=\"0\"} 2
vllm:request_success_total 9
"""
    )

    assert metrics == {
        'vllm:spec_decode_num_drafts{engine="0"}': 12.0,
        'vllm:async_draft_cache_hits_total{engine="0"}': 4.0,
        'vllm:num_preemptions_total{engine="0"}': 2.0,
    }


def test_checkpoint_manifest_includes_pytorch_bin_weights(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    (tmp_path / "pytorch_model.bin.index.json").write_text("{}")

    manifest = checkpoint_manifest(tmp_path)

    assert [record["name"] for record in manifest["files"]] == [
        "config.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]


def _write_complete_cell(cell_dir, cell, result) -> None:
    cell_dir.mkdir(parents=True)
    write_json(cell_dir / "result.json", result)
    write_json(
        cell_dir / "cell_complete.json",
        {"status": "complete", "cell": cell.name},
    )
    write_json(
        cell_dir / "shutdown.json",
        {"exit_code": 0, "forced_kill": False},
    )
    (cell_dir / "server.log").write_text("clean shutdown\n")


def test_lifecycle_audit_checks_request_cache_and_shutdown_semantics(
    tmp_path,
) -> None:
    expected = {
        "mixed_abort": (32, 30, 2, 0),
        "preemption": (8, 8, 0, 3),
        "chunked_prefill": (4, 4, 0, 0),
    }
    for cell in lifecycle_cells():
        requests, completed, aborted, preemptions = expected[cell.variant]
        metrics = {
            'vllm:async_draft_cache_hits_total{engine="0"}': 5,
            'vllm:async_draft_cache_misses_total{engine="0"}': 2,
            'vllm:async_draft_jit_fallbacks_total{engine="0"}': 2,
            'vllm:num_preemptions_total{engine="0"}': preemptions,
        }
        cell_dir = tmp_path / "cells" / cell.name
        _write_complete_cell(
            cell_dir,
            cell,
            {
                "status": "complete",
                "cell": {"name": cell.name},
                "summary": {
                    "request_count": requests,
                    "completed_request_count": completed,
                    "aborted_request_count": aborted,
                },
                "metrics_delta": metrics,
            },
        )
        (cell_dir / "proposals.jsonl").write_text('{"round": 1}\n')

    assert audit_lifecycle(tmp_path)
    preemption = tmp_path / "cells" / lifecycle_cells()[1].name
    write_json(
        preemption / "shutdown.json",
        {"exit_code": -9, "forced_kill": True},
    )
    assert not audit_lifecycle(tmp_path)
    audit = __import__("json").loads((tmp_path / "lifecycle_audit.json").read_text())
    assert any("forced kill" in failure["reason"] for failure in audit["failures"])

    write_json(
        preemption / "shutdown.json",
        {"exit_code": 0, "forced_kill": False},
    )
    (preemption / "server.log").write_text(
        "Async draft child pid=123 did not exit; terminating it.\n"
    )
    assert not audit_lifecycle(tmp_path)
    audit = __import__("json").loads((tmp_path / "lifecycle_audit.json").read_text())
    assert any(
        "child did not shut down cleanly" in failure["reason"]
        for failure in audit["failures"]
    )


def test_performance_audit_separates_semantics_from_speed_gate(
    monkeypatch,
    tmp_path,
) -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    write_json(
        tmp_path / "manifest.json",
        {
            "workload": {"prompt_count": 128, "output_length": 512},
            "topology": {"target_device": 0, "draft_device": 1},
        },
    )
    for cell in performance_cells():
        throughput = 100.0 if cell.mode == "sync" else 110.0
        completion_tokens = 65536
        elapsed = completion_tokens / throughput
        metrics = {}
        if cell.mode == "async_cache":
            metrics = {
                'vllm:async_draft_cache_hits_total{engine="0"}': 5,
                'vllm:async_draft_cache_misses_total{engine="0"}': 2,
                'vllm:async_draft_jit_fallbacks_total{engine="0"}': 2,
                'vllm:async_draft_ipc_bytes_total{engine="0"}': 1000,
                'vllm:async_draft_branch_build_seconds_total{engine="0"}': 1,
            }
        cell_dir = tmp_path / "cells" / cell.name
        _write_complete_cell(
            cell_dir,
            cell,
            {
                "status": "complete",
                "cell": {"name": cell.name},
                "warmup": {"seconds": 30.1},
                "expected_completion_tokens": completion_tokens,
                "metrics_delta": metrics,
                "summary": {
                    "request_count": 128,
                    "completed_request_count": 128,
                    "aborted_request_count": 0,
                    "completion_tokens": completion_tokens,
                    "elapsed_seconds": elapsed,
                    "completion_throughput_tok_s": throughput,
                    "tokens_per_gpu_second": throughput,
                },
            },
        )
        write_json(
            cell_dir / "command.json",
            ["--no-async-scheduling", "--no-enable-prefix-caching"],
        )
        (cell_dir / "gpu_samples.csv").write_text(
            "timestamp,gpu_index,gpu_uuid,utilization_gpu_percent,"
            "memory_used_mib,power_draw_w\n"
            "1,0,gpu0,50,100,200\n"
            "1,1,gpu1,40,100,180\n"
        )
    monkeypatch.setattr(matrix, "render_performance_plot", lambda *args: None)

    assert audit_performance(tmp_path)
    summary = __import__("json").loads(
        (tmp_path / "performance_summary.json").read_text()
    )
    assert summary["semantic_status"] == "passed"
    assert summary["primary_gate_passed"] is True

    bad_cell = tmp_path / "cells" / performance_cells()[1].name
    bad = __import__("json").loads((bad_cell / "result.json").read_text())
    bad["metrics_delta"]['vllm:async_draft_jit_fallbacks_total{engine="0"}'] = 1
    write_json(bad_cell / "result.json", bad)
    assert not audit_performance(tmp_path)
    summary = __import__("json").loads(
        (tmp_path / "performance_summary.json").read_text()
    )
    assert summary["semantic_status"] == "failed"
    assert summary["primary_gate_passed"] is True
