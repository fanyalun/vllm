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
    assert config.moe_skip_top_h == 4
    assert config.hierarchical_stop_policy == "low_error"
    assert make_config(monkeypatch, inner_num_rounds=2).num_speculative_tokens == 10


def test_gemma4_uses_shared_moe_preverification_with_full_outer_capacity(monkeypatch):
    config = make_config(monkeypatch, family="gemma4")
    assert config.num_speculative_tokens == 20
    assert config.moe_skip_top_h == 4


def test_graph_hash_distinguishes_inner_method_depth_rounds_and_top_h(monkeypatch):
    configs = [
        make_config(monkeypatch, **overrides)
        for overrides in (
            {},
            {"inner_method": "dspark"},
            {"inner_num_speculative_tokens": 2},
            {"inner_num_rounds": 2},
            {"moe_skip_top_h": 2},
            {"preverify_gdn_mode": "ssm_mean"},
            {"preverify_gdn_mode": "input_mean"},
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


@pytest.mark.parametrize("mode", ["ssm_mean", "input_mean"])
def test_mean_gdn_rejects_other_models_and_depths(monkeypatch, mode):
    with pytest.raises(ValueError, match="Qwen3.6"):
        make_config(monkeypatch, family="gemma4", preverify_gdn_mode=mode)
    with pytest.raises(ValueError, match="D=4"):
        make_config(
            monkeypatch, inner_num_speculative_tokens=2, preverify_gdn_mode=mode
        )
