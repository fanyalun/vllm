# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils.configs.speculators import SpeculatorsConfig


def _gemma4_26b_dspark_config() -> dict:
    return {
        "speculators_model_type": "dspark",
        "draft_vocab_size": 32000,
        "mask_token_id": 4,
        "markov_rank": 256,
        "block_size": 7,
        "aux_hidden_state_layer_ids": [3, 10, 18, 25, 28],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 2816,
            "intermediate_size": 2112,
            "num_hidden_layers": 5,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "vocab_size": 262144,
        },
        "speculators_config": {
            "proposal_methods": [{"speculative_tokens": 6}],
            "verifier": {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "name_or_path": "google/gemma-4-26B-A4B-it",
            },
        },
    }


def test_gemma4_26b_speculators_dspark_uses_qwen_draft_backbone():
    """The Gemma4 verifier does not imply a native Gemma4 draft backbone."""
    config = _gemma4_26b_dspark_config()

    draft = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert draft["architectures"] == ["Qwen3DSparkModel"]
    assert draft["model_type"] == "qwen3"
    assert draft["hidden_size"] == 2816
    assert draft["draft_vocab_size"] == 32000
    assert draft["eagle_aux_hidden_state_layer_ids"] == [3, 10, 18, 25, 28]
    assert draft["target_layer_ids"] == [2, 9, 17, 24, 27]
    assert draft["sample_from_anchor"] is False


def test_qwen3_6_speculators_dspark_preserves_sample_from_anchor():
    """Qwen3.6 uses the anchor as the first prediction slot."""
    config = _gemma4_26b_dspark_config()
    config["sample_from_anchor"] = True
    config["aux_hidden_state_layer_ids"] = [2, 10, 20, 30, 37]
    config["speculators_config"]["verifier"] = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "name_or_path": "Qwen/Qwen3.6-35B-A3B",
    }

    draft = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert draft["sample_from_anchor"] is True
    assert draft["eagle_aux_hidden_state_layer_ids"] == [2, 10, 20, 30, 37]
    assert draft["target_layer_ids"] == [1, 9, 19, 29, 36]


def test_gemma4_26b_speculators_dspark_extracts_runtime_defaults():
    config = _gemma4_26b_dspark_config()

    speculative = SpeculatorsConfig.extract_vllm_speculative_config(config)

    assert speculative == {
        "method": "dspark",
        "num_speculative_tokens": 6,
    }
