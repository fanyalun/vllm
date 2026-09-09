# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.engine.arg_utils import EngineArgs


def _target_config(
    architecture: str = "Qwen3_5MoeForConditionalGeneration",
    *,
    top_k: int = 8,
    shared_expert_intermediate_size: int = 512,
):
    return SimpleNamespace(
        architectures=[architecture],
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=top_k,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        ),
    )


def _gemma4_target_config(
    architecture: str = "Gemma4ForConditionalGeneration",
    *,
    top_k: int = 8,
    num_experts: int = 128,
):
    return SimpleNamespace(
        architectures=[architecture],
        hf_text_config=SimpleNamespace(
            top_k_experts=top_k,
            num_experts=num_experts,
        ),
    )


def _make_config(**kwargs) -> SpeculativeConfig:
    return SpeculativeConfig(
        method="moe_skip",
        num_speculative_tokens=kwargs.pop("num_speculative_tokens", 8),
        target_model_config=kwargs.pop("target_model_config", _target_config()),
        target_parallel_config=kwargs.pop("target_parallel_config", ParallelConfig()),
        **kwargs,
    )


def test_moe_skip_defaults_and_has_no_draft_model():
    config = _make_config()

    assert config.moe_skip_top_h == 4
    assert config.model is None
    assert config.draft_model_config is None
    assert config.draft_parallel_config is None
    assert repr(config) == (
        "SpeculativeConfig(method='moe_skip', model=None, num_spec_tokens=8)"
    )


@pytest.mark.parametrize("top_h", [1, 4, 8])
def test_moe_skip_accepts_valid_top_h(top_h: int):
    assert _make_config(moe_skip_top_h=top_h).moe_skip_top_h == top_h


def test_moe_skip_rejects_invalid_top_h():
    with pytest.raises(ValidationError):
        _make_config(moe_skip_top_h=0)
    with pytest.raises(ValueError, match="no larger than"):
        _make_config(moe_skip_top_h=9)


def test_qwen_moe_skip_requires_two_gdn_scratch_candidates():
    with pytest.raises(ValueError, match="at least 2 speculative tokens"):
        _make_config(num_speculative_tokens=1)


@pytest.mark.parametrize(
    "architecture",
    ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"],
)
def test_gemma4_moe_skip_uses_native_top_k_without_shared_expert(architecture: str):
    config = _make_config(
        num_speculative_tokens=1,
        target_model_config=_gemma4_target_config(architecture),
    )

    assert config.moe_skip_top_h == 4
    assert config.num_speculative_tokens == 1
    assert config.draft_model_config is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "draft"}, "does not accept a draft model"),
        (
            {"target_model_config": _target_config("LlamaForCausalLM")},
            "only supports Qwen3.6 MoE and Gemma4 MoE",
        ),
        (
            {"target_model_config": _target_config(shared_expert_intermediate_size=0)},
            "requires a shared expert",
        ),
        (
            {"target_parallel_config": ParallelConfig(tensor_parallel_size=2)},
            "requires TP1, PP1, and DP1",
        ),
        (
            {
                "target_model_config": _gemma4_target_config(num_experts=0),
            },
            "requires routed experts",
        ),
    ],
)
def test_moe_skip_rejects_unsupported_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _make_config(**kwargs)


def test_moe_skip_top_h_requires_method():
    with pytest.raises(ValueError, match="only supported"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=4,
            moe_skip_top_h=4,
        )


def test_moe_skip_cli_aliases_and_json_are_mutually_exclusive():
    target = _target_config()
    config = EngineArgs(
        spec_method="moe_skip",
        spec_tokens=8,
        moe_skip_top_h=4,
    ).create_speculative_config(target, ParallelConfig())
    assert config is not None
    assert config.moe_skip_top_h == 4

    args = EngineArgs(
        speculative_config={
            "method": "moe_skip",
            "num_speculative_tokens": 8,
            "moe_skip_top_h": 4,
        },
        moe_skip_top_h=4,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        args.create_speculative_config(target, ParallelConfig())
