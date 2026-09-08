# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace

import pytest

from benchmarks.replayssm.async_ssd_eagle3_matrix import (
    Cell,
    audit_draft_trace_pair,
    audit_dspark_bank_trace,
    audit_dspark_round_trace,
    audit_lifecycle,
    audit_performance,
    audit_request_pair,
    audit_request_pair_with_policy,
    checkpoint_manifest,
    compare_acceptance_profiles,
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
from benchmarks.replayssm.audit_async_ssd_gdn_modes import (
    compare_requests,
    flag_contract,
    without_gdn_flags,
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
        replayssm_buffer_len=16,
        draft_tie_logit_tolerance=0.1,
        acceptance_length_relative_tolerance=0.01,
    )


def test_phase_a_matrix_has_all_expected_cells() -> None:
    assert len(correctness_cells()) == 24
    assert [cell.name for cell in correctness_cells("b1-eager")] == [
        "correctness_ar_eager_b1",
        "correctness_sync_eager_b1",
        "correctness_async_jit_eager_b1",
        "correctness_async_cache_eager_b1",
    ]
    assert len(lifecycle_cells()) == 3
    assert len(performance_cells()) == 18
    assert len({cell.name for cell in performance_cells()}) == 18
    assert Cell("correctness", "ar", "eager", 1).name == ("correctness_ar_eager_b1")


def test_acceptance_profile_checks_mean_and_each_position() -> None:
    baseline = {
        "num_drafts": 100,
        "mean_acceptance_length": 2.0,
        "accepted_counts_per_position": [80, 50, 20, 10],
    }
    within_tolerance = {
        "num_drafts": 200,
        "mean_acceptance_length": 2.01,
        "accepted_counts_per_position": [161, 100, 40, 20],
    }
    position_regression = {
        "num_drafts": 100,
        "mean_acceptance_length": 2.0,
        "accepted_counts_per_position": [75, 50, 20, 10],
    }

    assert (
        compare_acceptance_profiles(baseline, within_tolerance, 0.01)["status"]
        == "passed"
    )
    audit = compare_acceptance_profiles(baseline, position_regression, 0.01)
    assert audit["status"] == "failed"
    assert audit["accepted_rate_absolute_differences"][0] == pytest.approx(0.05)


def test_limited_correctness_scope_does_not_run_lifecycle_or_performance(
    monkeypatch, tmp_path
) -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    monkeypatch.setattr(matrix, "audit_correctness", lambda args, root: True)

    def unexpected_audit(root):
        raise AssertionError(f"unexpected full-matrix audit for {root}")

    monkeypatch.setattr(matrix, "audit_lifecycle", unexpected_audit)
    monkeypatch.setattr(matrix, "audit_performance", unexpected_audit)

    assert not matrix.finalize_matrix(Namespace(correctness_scope="b1-eager"), tmp_path)
    status = json.loads((tmp_path / "matrix_incomplete.json").read_text())
    assert status["correctness"] is True
    assert status["lifecycle"] == "not_evaluated"
    assert status["performance"] == "not_evaluated"


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


def test_output_audit_accepts_explicit_path_numerical_evidence() -> None:
    baseline = make_request(
        [1, 2],
        [
            {"token_id:1": -0.1, "token_id:9": -1.0},
            {"token_id:2": -0.1, "token_id:3": -2.0},
        ],
    )
    candidate = make_request(
        [1, 3],
        [
            {"token_id:1": -0.2, "token_id:9": -1.1},
            {"token_id:3": -0.1, "token_id:2": -2.0},
        ],
    )

    audit = audit_request_pair_with_policy(
        baseline,
        candidate,
        tolerance=0.1,
        policy="audited-path-numerical",
    )

    assert audit["status"] == "target_path_numerical_divergence"
    assert audit["numerical_evidence"]["first_observed_offset"] == 0


def test_output_audit_does_not_waive_first_token_divergence() -> None:
    baseline = make_request(
        [2],
        [{"token_id:2": -0.1, "token_id:3": -2.0}],
    )
    candidate = make_request(
        [3],
        [{"token_id:3": -0.1, "token_id:2": -2.0}],
    )

    audit = audit_request_pair_with_policy(
        baseline,
        candidate,
        tolerance=0.1,
        policy="audited-path-numerical",
    )

    assert audit["status"] == "failed"
    assert audit["path_numerical_rejection"] == "diverged at first token"


def test_path_numerical_policy_does_not_require_cross_token_in_top20() -> None:
    baseline = make_request(
        [1, 2],
        [
            {"token_id:1": -0.1, "token_id:9": -1.0},
            {"token_id:2": -0.1, "token_id:8": -2.0},
        ],
    )
    candidate = make_request(
        [1, 3],
        [
            {"token_id:1": -0.2, "token_id:9": -1.1},
            {"token_id:3": -0.1, "token_id:7": -2.0},
        ],
    )

    audit = audit_request_pair_with_policy(
        baseline,
        candidate,
        tolerance=0.1,
        policy="audited-path-numerical",
    )

    assert audit["status"] == "target_path_numerical_divergence"
    assert audit["reason"] == "divergent_token_missing_from_top_logprobs"


def test_path_numerical_evidence_can_be_triangulated_through_ar() -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    mutual = {
        "baseline_mode": "sync",
        "candidate_mode": "async_jit",
        "engine": "eager",
        "batch_size": 1,
        "prompt_index": 13,
        "offset": 76,
        "path_numerical_rejection": ("no Target logprob drift before token divergence"),
    }
    comparisons = [
        {
            "baseline_mode": "ar",
            "candidate_mode": mode,
            "engine": "eager",
            "batch_size": 1,
            "prompt_index": 13,
            "status": "target_path_numerical_divergence",
            "numerical_evidence": {
                "first_observed_offset": 1,
                "max_common_top_logprob_delta": delta,
                "max_delta_offset": 36,
            },
        }
        for mode, delta in (("sync", 0.8), ("async_jit", 0.9))
    ]

    evidence = matrix._triangulated_path_numerical_evidence(mutual, comparisons)

    assert evidence is not None
    assert evidence["ar_vs_sync"]["first_observed_offset"] == 1


def test_gdn_diagnostic_summarizes_exact_tie_and_failure() -> None:
    baseline = {
        0: make_request([1]),
        1: make_request([2], [{"token_id:2": -1.0, "token_id:3": -1.01}]),
        2: make_request([4], [{"token_id:4": -1.0, "token_id:5": -2.0}]),
    }
    candidate = {
        0: make_request([1]),
        1: make_request([3], [{"token_id:3": -1.0, "token_id:2": -1.01}]),
        2: make_request([5], [{"token_id:5": -1.0, "token_id:4": -2.0}]),
    }

    report = compare_requests(baseline, candidate, tolerance=0.1)

    assert report["exact_or_tie_count"] == 2
    assert report["status_counts"] == {
        "exact": 1,
        "target_top1_tie_equivalent": 1,
        "failed": 1,
    }
    assert report["comparisons"][1]["baseline_top_candidates"][0] == {
        "token_id": 2,
        "logprob": -1.0,
    }


@pytest.mark.parametrize(
    ("command", "mode", "gdn_mode", "expected"),
    [
        (["server"], "ar", "baseline", True),
        (["server", "--use-replayssm"], "ar", "replayssm", True),
        (["server", "--use-replayssm-spec"], "sync", "replayssm", True),
        (["server", "--use-replayssm"], "sync", "replayssm", False),
    ],
)
def test_gdn_diagnostic_flag_contract(command, mode, gdn_mode, expected) -> None:
    assert flag_contract(command, mode, gdn_mode) is expected


def test_gdn_diagnostic_normalizes_only_replayssm_flags() -> None:
    baseline = ["server", "--dtype", "bfloat16", "--enforce-eager"]
    replayssm = [
        "server",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--replayssm-buffer-len",
        "16",
        "--use-replayssm-spec",
    ]

    assert without_gdn_flags(replayssm) == baseline


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


def test_dspark_bank_trace_audit_enforces_num_sampled_cursor(tmp_path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    records = [
        {
            "request_id": "request",
            "accepted_draft_count": 0,
            "dspark_bank_cursor": 0,
            "dspark_bank_width": 8,
            "dspark_backbone_refreshed": True,
            "dspark_refresh_reason": "initial",
            "dspark_cursor_before_refresh": 1,
        },
        {
            "request_id": "request",
            "accepted_draft_count": 2,
            "dspark_bank_cursor": 3,
            "dspark_bank_width": 8,
            "dspark_backbone_refreshed": False,
        },
        {
            "request_id": "request",
            "accepted_draft_count": 1,
            "dspark_bank_cursor": 0,
            "dspark_bank_width": 8,
            "dspark_backbone_refreshed": True,
            "dspark_refresh_reason": "insufficient_remaining",
            "dspark_cursor_before_refresh": 5,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    audit = audit_dspark_bank_trace(trace_path, verify_width=4)

    assert audit["status"] == "passed"
    assert audit["refresh_reasons"] == {"initial": 1, "insufficient_remaining": 1}


def test_dspark_bank_trace_audit_rejects_accepted_only_cursor(tmp_path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    records = [
        {
            "request_id": "request",
            "accepted_draft_count": 0,
            "dspark_bank_cursor": 0,
            "dspark_bank_width": 8,
            "dspark_backbone_refreshed": True,
            "dspark_refresh_reason": "initial",
            "dspark_cursor_before_refresh": 1,
        },
        {
            "request_id": "request",
            "accepted_draft_count": 2,
            "dspark_bank_cursor": 2,
            "dspark_bank_width": 8,
            "dspark_backbone_refreshed": False,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    audit = audit_dspark_bank_trace(trace_path, verify_width=4)

    assert audit["status"] == "failed"
    assert audit["failures"][0]["errors"] == [
        "bank cursor does not advance by num_sampled"
    ]


def test_dspark_round_trace_audit_requires_fixed_d_width(tmp_path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    records = [
        {
            "request_id": "request",
            "accepted_draft_count": 0,
            "draft_tokens": [1, 2, 3],
            "dspark_target_verify_width": 3,
            "dspark_proposal_execution_width": 3,
            "dspark_branch_backbone_width": 3,
        },
        {
            "request_id": "request",
            "accepted_draft_count": 3,
            "draft_tokens": [4, 5, 6],
            "dspark_target_verify_width": 3,
            "dspark_proposal_execution_width": 3,
            "dspark_branch_backbone_width": 3,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    audit = audit_dspark_round_trace(trace_path, verify_width=3)

    assert audit["status"] == "passed"
    assert audit["record_count"] == 2
    assert audit["branch_backbone_width"] == 3


def test_dspark_round_trace_audit_rejects_native_bank_width(tmp_path) -> None:
    trace_path = tmp_path / "proposals.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "request_id": "request",
                "accepted_draft_count": 1,
                "draft_tokens": list(range(8)),
                "dspark_bank_cursor": 2,
                "dspark_target_verify_width": 3,
                "dspark_proposal_execution_width": 3,
                "dspark_branch_backbone_width": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_dspark_round_trace(trace_path, verify_width=3)

    assert audit["status"] == "failed"
    assert audit["failures"][0]["errors"] == [
        "proposal width differs from Target verification width",
        "fixed-width DSpark trace unexpectedly contains a bank cursor",
    ]


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


def test_server_command_supports_fp32_tp2_diagnostic() -> None:
    args = make_args()
    args.dtype = "float32"
    args.target_tensor_parallel_size = 2
    args.draft_tensor_parallel_size = 2
    cell = Cell("correctness", "sync", "eager", 1)

    command = server_command(args, cell, 43100)

    config = command[command.index("--speculative-config") + 1]
    assert command[command.index("--dtype") + 1] == "float32"
    assert command[command.index("--device-ids") + 1] == "0,1"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert '"draft_tensor_parallel_size": 2' in config


def test_server_command_can_pin_attention_backend() -> None:
    args = make_args()
    args.attention_backend = "TRITON_ATTN"
    cell = Cell("correctness", "ar", "eager", 1)

    command = server_command(args, cell, 43100)

    assert command[command.index("--attention-backend") + 1] == "TRITON_ATTN"


@pytest.mark.parametrize(
    ("mode", "expected_flag"),
    [
        ("ar", "--use-replayssm"),
        ("sync", "--use-replayssm-spec"),
        ("async_jit", "--use-replayssm-spec"),
        ("async_cache", "--use-replayssm-spec"),
    ],
)
def test_qwen36_server_command_always_enables_replayssm(
    tmp_path, mode, expected_flag
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5_moe",'
        '"architectures":["Qwen3_5MoeForConditionalGeneration"]}',
        encoding="utf-8",
    )
    args = make_args()
    args.target = str(tmp_path)
    args.num_speculative_tokens = 4
    cell = Cell("correctness", mode, "eager", 1)

    command = server_command(args, cell, 43100)

    assert expected_flag in command
    unexpected_flag = (
        "--use-replayssm-spec"
        if expected_flag == "--use-replayssm"
        else "--use-replayssm"
    )
    assert unexpected_flag not in command
    assert command[command.index("--replayssm-buffer-len") + 1] == "16"
    assert command[command.index("--mamba-cache-mode") + 1] == "none"
    assert command[command.index("--mamba-backend") + 1] == "triton"


@pytest.mark.parametrize("mode", ["ar", "sync", "async_jit", "async_cache"])
def test_qwen36_baseline_gdn_mode_disables_replayssm_for_every_mode(
    tmp_path, mode
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5_moe",'
        '"architectures":["Qwen3_5MoeForConditionalGeneration"]}',
        encoding="utf-8",
    )
    args = make_args()
    args.target = str(tmp_path)
    args.num_speculative_tokens = 4
    args.qwen_gdn_mode = "baseline"
    cell = Cell("correctness", mode, "eager", 1)

    command = server_command(args, cell, 43100)

    assert "--use-replayssm" not in command
    assert "--use-replayssm-spec" not in command
    assert "--replayssm-buffer-len" not in command
    assert command[command.index("--mamba-cache-mode") + 1] == "none"
    assert command[command.index("--mamba-backend") + 1] == "triton"


def test_qwen36_spec_rejects_too_short_replayssm_buffer(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5_moe"}', encoding="utf-8"
    )
    args = make_args()
    args.target = str(tmp_path)
    args.num_speculative_tokens = 4
    args.replayssm_buffer_len = 4

    with pytest.raises(ValueError, match=r"num_speculative_tokens \+ 1"):
        server_command(args, Cell("correctness", "sync", "eager", 1), 43100)


def test_gemma4_server_command_never_enables_replayssm(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"gemma4","architectures":["Gemma4ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    args = make_args()
    args.target = str(tmp_path)

    for mode in ("ar", "sync", "async_jit", "async_cache"):
        command = server_command(args, Cell("correctness", mode, "eager", 1), 43100)
        assert "--use-replayssm" not in command
        assert "--use-replayssm-spec" not in command
        assert "--replayssm-buffer-len" not in command
        assert "--mamba-backend" not in command


def test_dspark_native_bank_width_prefers_checkpoint_metadata(tmp_path) -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(
        json.dumps(
            {
                "block_size": 7,
                "speculators_config": {"proposal_methods": [{"speculative_tokens": 6}]},
            }
        ),
        encoding="utf-8",
    )
    args = make_args()
    args.draft = str(draft)

    assert matrix._dspark_native_bank_width(args) == 6


def test_stream_request_uses_explicit_zero_seed(monkeypatch) -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    captured_payload = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode):
            assert decode_unicode
            return ["data: [DONE]"]

    def post(url, *, json, stream, timeout):
        assert url.endswith("/v1/completions")
        assert stream
        assert timeout == (30, 900)
        captured_payload.update(json)
        return Response()

    monkeypatch.setattr(matrix.requests, "post", post)

    matrix.stream_request(
        port=43100,
        request_id="seed-audit",
        prompt_token_ids=[1, 2],
        max_tokens=4,
        logprobs=None,
    )

    assert captured_payload["temperature"] == 0.0
    assert captured_payload["seed"] == 0


def test_gdn_recurrent_reference_is_recorded_in_cell_environment(
    monkeypatch, tmp_path
) -> None:
    from benchmarks.replayssm import async_ssd_eagle3_matrix as matrix

    args = make_args()
    args.resume = False
    args.startup_timeout = 1
    args.warmup_seconds = 0
    args.output_length = 1
    args.gdn_recurrent_reference = True
    monkeypatch.setattr(matrix, "wait_for_server", lambda *args: None)
    monkeypatch.setattr(matrix, "warmup_server", lambda *args: {})
    monkeypatch.setattr(matrix, "scrape_metrics", lambda *args: ("", {}))
    monkeypatch.setattr(matrix, "run_requests", lambda **kwargs: ([], 0.1))
    monkeypatch.setattr(
        matrix,
        "summarize_requests",
        lambda *args: {
            "completion_tokens": 0,
            "aborted_request_count": 0,
        },
    )
    monkeypatch.setattr(matrix, "stop_server", lambda *args: {"forced_kill": False})
    monkeypatch.setattr(matrix.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(matrix.GpuSampler, "start", lambda self: None)
    monkeypatch.setattr(matrix.GpuSampler, "stop", lambda self: None)

    cell = Cell("lifecycle", "ar", "eager", 1, variant="chunked_prefill")
    matrix.run_cell(
        args,
        tmp_path,
        [{"prompt_index": 0, "token_ids": [1]}],
        cell,
        43100,
    )

    environment = matrix.json.loads(
        (tmp_path / "cells" / cell.name / "environment.json").read_text()
    )
    assert environment["VLLM_GDN_PREFILL_USE_RECURRENT_REFERENCE"] == "1"


def test_server_command_rejects_overlapping_async_draft_device() -> None:
    args = make_args()
    args.target_tensor_parallel_size = 2
    cell = Cell("correctness", "async_jit", "eager", 1)

    with pytest.raises(ValueError, match="must not overlap"):
        server_command(args, cell, 43100)


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
