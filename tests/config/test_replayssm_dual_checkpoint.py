# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from vllm.config.cache import CacheConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_dual_checkpoint_validation():
    assert not CacheConfig().replayssm_spec_dual_checkpoint
    with pytest.raises(ValidationError, match="requires use_replayssm_spec"):
        CacheConfig(replayssm_spec_dual_checkpoint=True)
    with pytest.raises(ValidationError, match="cannot be combined"):
        CacheConfig(
            use_replayssm_spec=True,
            replayssm_spec_dual_checkpoint=True,
            replayssm_spec_flush_interval=4,
        )


def test_dual_checkpoint_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--use-replayssm-spec",
                "--replayssm-spec-dual-checkpoint",
            ]
        )
    )
    assert args.replayssm_spec_dual_checkpoint


def test_dual_checkpoint_allocation():
    calculate = MambaStateShapeCalculator.gated_delta_net_replayssm_spec_state_shape
    old = calculate(1, 16, 32, 128, 128, 4, 16, 4)
    new = calculate(1, 16, 32, 128, 128, 4, 16, 4, dual_checkpoint=True)
    assert len(old) == 5 and len(new) == 6
    assert new[0:2] == old[0:2]
    assert new[5] == old[1]
    assert old[2][1] == 32
    assert new[2][1] == 16
    dtypes = MambaStateDtypeCalculator.gated_delta_net_replayssm_spec_state_dtype(
        torch.bfloat16, "auto", "float32", dual_checkpoint=True
    )
    assert len(dtypes) == len(new)
    assert dtypes[1] == dtypes[5] == torch.float32


def test_model_cache_accounting_matches_layer_layout():
    config = SimpleNamespace(
        cache_config=CacheConfig(
            use_replayssm_spec=True,
            replayssm_spec_dual_checkpoint=True,
            replayssm_buffer_len=16,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                linear_num_key_heads=16,
                linear_num_value_heads=32,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
    )
    shapes = Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(config)
    dtypes = Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(config)
    assert len(shapes) == len(dtypes) == 6
    assert shapes[2][1] == 16
    assert shapes[1] == shapes[5]
    assert dtypes[1] == dtypes[5] == torch.float32
