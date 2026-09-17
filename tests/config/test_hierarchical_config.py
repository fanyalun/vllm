# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig


def make_config(monkeypatch, family="qwen", **kwargs):
    draft = SimpleNamespace(
        verify_with_parallel_config=Mock(),
        compute_hash=lambda: "draft",
        hf_config=SimpleNamespace(),
    )
    monkeypatch.setattr(
        SpeculativeConfig,
        "make_inner_config",
        lambda self: SimpleNamespace(
            model="small",
            draft_model_config=draft,
            draft_parallel_config=ParallelConfig(),
        ),
    )
    target = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=8,
            shared_expert_intermediate_size=512,
        ),
    )
    if family == "gemma4":
        target.architectures = ["Gemma4ForConditionalGeneration"]
        target.hf_text_config = SimpleNamespace(top_k_experts=8, num_experts=128)
    return SpeculativeConfig(
        method="hierarchical",
        target_model_config=cast(ModelConfig, target),
        target_parallel_config=ParallelConfig(),
        **kwargs,
    )


def test_capacity_includes_each_preverify_recovery_or_bonus(monkeypatch):
    config = make_config(monkeypatch)
    assert config.num_speculative_tokens == 20
    assert config.inner_method == "mtp"
    assert config.moe_skip_top_h == 8
    assert config.moe_skip_min_weight == 0.125
    assert config.moe_skip_weight_mode == "preserve"
    assert config.hierarchical_stop_policy == "low_error"
    assert make_config(monkeypatch, inner_num_rounds=2).num_speculative_tokens == 10


def test_gemma4_uses_shared_moe_preverification_with_full_outer_capacity(monkeypatch):
    config = make_config(monkeypatch, family="gemma4")
    assert config.num_speculative_tokens == 20
    assert config.moe_skip_top_h == 8
    assert config.moe_skip_min_weight == 0.125


def test_graph_hash_distinguishes_inner_method_depth_rounds_and_top_h(monkeypatch):
    configs = [
        make_config(monkeypatch, **overrides)
        for overrides in (
            {},
            {"inner_method": "dspark"},
            {"inner_num_speculative_tokens": 2},
            {"inner_num_rounds": 2},
            {"moe_skip_top_h": 2},
            {"moe_skip_weight_mode": "renormalize"},
            {"preverify_gdn_mode": "ssm_mean"},
            {"preverify_gdn_mode": "input_mean"},
            {"preverify_gdn_mode": "replay_tail"},
            {"preverify_gdn_group_mode": "projection"},
            {"preverify_gdn_group_mode": "full"},
            {"hierarchical_stop_policy": "balanced"},
            {"hierarchical_stop_policy": "aggressive"},
            {"hierarchical_stop_policy": "none"},
        )
    ]
    assert len({config.compute_hash() for config in configs}) == len(configs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_speculative_tokens": 16}, "must equal"),
        ({"draft_sample_method": "probabilistic"}, "greedy drafting"),
        ({"rejection_sample_method": "block"}, "standard rejection"),
        ({"moe_skip_top_h": 9}, "no larger"),
        ({"enable_adaptive_verification": True}, "adaptive/parallel"),
        ({"hierarchical_stop_policy": "invalid"}, "hierarchical_stop_policy"),
    ],
)
def test_unsupported_hierarchical_semantics_fail_closed(monkeypatch, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_config(monkeypatch, **kwargs)


@pytest.mark.parametrize("mode", ["ssm_mean", "input_mean", "replay_tail"])
def test_mean_gdn_rejects_other_models_and_depths(monkeypatch, mode):
    with pytest.raises(ValueError, match="Qwen3.6"):
        make_config(monkeypatch, family="gemma4", preverify_gdn_mode=mode)
    with pytest.raises(ValueError, match="D=4"):
        make_config(
            monkeypatch, inner_num_speculative_tokens=2, preverify_gdn_mode=mode
        )


def test_three_level_requires_replay_and_isolated_tail_policy(monkeypatch):
    with pytest.raises(ValueError, match="ungrouped replay_tail"):
        make_config(monkeypatch, preverify_gdn_update_policy="three_level_p50")
    with pytest.raises(ValueError, match="requires three_level_p50"):
        make_config(monkeypatch, preverify_gdn_tail_policy="repair_on_reject")
    hashes = []
    for tail in ("carry", "repair_on_reject"):
        config = make_config(
            monkeypatch,
            preverify_gdn_mode="replay_tail",
            preverify_gdn_update_policy="three_level_p50",
            preverify_gdn_tail_policy=tail,
        )
        hashes.append(config.compute_hash())
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"inner_method": "dspark"},
        {"preverify_gdn_tail_policy": "repair_on_reject"},
        {"inner_num_rounds": 5},
        {"preverify_gdn_mode_window_size": 0},
        {"preverify_gdn_tau_alpha": float("nan")},
        {"preverify_gdn_tau_beta": 1.1},
    ],
)
def test_windowed_gdn_rejects_unsupported_semantics(monkeypatch, overrides):
    with pytest.raises(ValueError):
        make_config(
            monkeypatch,
            preverify_gdn_mode="replay_tail",
            preverify_gdn_update_policy="windowed_three_level",
            **overrides,
        )


def test_windowed_gdn_hash_includes_window_thresholds_and_optimization(monkeypatch):
    hashes = []
    for overrides in (
        {},
        {"preverify_gdn_mode_window_size": 1},
        {"preverify_gdn_tau_alpha": 0.96},
        {"preverify_gdn_tau_beta": 0.4},
        {"preverify_gdn_optimization": "cumulative_decay"},
        {"preverify_gdn_optimization": "multi_query"},
        {"preverify_gdn_optimization": "combined"},
    ):
        config = make_config(
            monkeypatch,
            preverify_gdn_mode="replay_tail",
            preverify_gdn_update_policy="windowed_three_level",
            **overrides,
        )
        hashes.append(config.compute_hash())
    assert len(set(hashes)) == len(hashes)


def test_windowed_gdn_rejects_stochastic_requests_at_cpu_input_boundary():
    from vllm import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    processor = SimpleNamespace(
        speculative_config=SimpleNamespace(
            preverify_gdn_update_policy="windowed_three_level"
        )
    )
    with pytest.raises(ValueError, match="temperature=0"):
        InputProcessor._validate_params(
            cast(InputProcessor, processor),
            SamplingParams(temperature=0.5),
            ("generate",),
        )


@pytest.mark.parametrize("group", ["projection", "full"])
def test_grouped_gdn_is_independent_but_rejects_incompatible_modes(monkeypatch, group):
    assert (
        make_config(monkeypatch, preverify_gdn_group_mode=group).preverify_gdn_mode
        == "none"
    )
    config = make_config(
        monkeypatch, preverify_gdn_group_mode=group, preverify_gdn_mode="replay_tail"
    )
    assert config.preverify_gdn_group_mode == group
    for kwargs, error in (
        ({"family": "gemma4"}, "Qwen3.6"),
        ({"inner_num_speculative_tokens": 2}, "D=4"),
        ({"preverify_gdn_mode": "ssm_mean"}, "state policy"),
        ({"preverify_gdn_mode": "input_mean"}, "state policy"),
    ):
        with pytest.raises(ValueError, match=error):
            make_config(monkeypatch, preverify_gdn_group_mode=group, **kwargs)
