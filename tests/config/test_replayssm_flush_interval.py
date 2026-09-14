# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config.cache import CacheConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_interval_default_and_allocation():
    default = CacheConfig(use_replayssm_spec=True, replayssm_buffer_len=64)
    tuned = CacheConfig(
        use_replayssm_spec=True,
        replayssm_buffer_len=64,
        replayssm_spec_flush_interval=8,
    )
    assert default.replayssm_spec_flush_interval is None
    assert tuned.replayssm_buffer_len == default.replayssm_buffer_len
    assert tuned.replayssm_spec_flush_interval == 8


@pytest.mark.parametrize("interval", [0, -1])
def test_interval_positive(interval):
    with pytest.raises(ValidationError):
        CacheConfig(use_replayssm_spec=True, replayssm_spec_flush_interval=interval)


def test_interval_requires_replayssm():
    with pytest.raises(ValidationError, match="requires use_replayssm_spec"):
        CacheConfig(replayssm_spec_flush_interval=8)


def test_interval_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(
        [
            "--use-replayssm-spec",
            "--replayssm-spec-flush-interval",
            "8",
        ]
    )
    args = EngineArgs.from_cli_args(parsed)
    assert args.replayssm_spec_flush_interval == 8
