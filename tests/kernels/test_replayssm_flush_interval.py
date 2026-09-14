# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_spec,
    reset_gdn_replayssm_spec_cursors,
)


@pytest.mark.parametrize("draft", [4, 8, 16, 32])
@pytest.mark.parametrize("interval", [None, 1, 8, 16, 64])
def test_flush_rollback(draft, interval):
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_replayssm_spec_decode_gdn.py"))
    )
    helpers["_run_rollback"](
        state_dtype=torch.float32,
        act_dtype=torch.bfloat16,
        ring_dtype=torch.float16,
        HQ=16,
        HV=32,
        K=128,
        V=128,
        buffer_len=64,
        max_spec_len=draft + 1,
        num_steps=80,
        flush_interval=interval,
    )


def test_flush_cursor_graph():
    device = "cuda"
    wp = torch.tensor([0, 7, 8, 63, 4], device=device, dtype=torch.int32)
    base = torch.tensor([0, 0, 0, 127, 8], device=device, dtype=torch.int32)
    flags = torch.tensor([0, 0, 0, 1, 0], device=device, dtype=torch.int8)
    accepted = torch.tensor([1, 5, 1, 4, 0], device=device, dtype=torch.int32)
    slots = torch.tensor([1, 2, 3, 4, 0], device=device, dtype=torch.int32)
    initial = [x.clone() for x in (wp, base, flags)]

    def step():
        commit_gdn_replayssm_spec(
            wp,
            base,
            flags,
            accepted,
            slots,
            69,
            5,
            cache_buf_len=128,
            flush_interval=8,
        )

    step()
    expected = [x.clone() for x in (wp, base, flags)]
    assert wp.tolist() == [0, 8, 13, 1, 8]
    assert base.tolist() == [0, 0, 0, 62, 8]
    assert flags.tolist() == [0, 1, 1, 0, 1]
    for dst, src in zip((wp, base, flags), initial):
        dst.copy_(src)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        for dst, src in zip((wp, base, flags), initial):
            dst.copy_(src)
        graph.replay()
        for actual, reference in zip((wp, base, flags), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    reset_gdn_replayssm_spec_cursors(
        wp,
        base,
        flags,
        torch.tensor([1, 0, 0, 0, 1], device=device, dtype=torch.int32),
        slots,
        69,
        5,
    )
    assert wp.tolist() == [0, 0, 13, 1, 8]
    assert base.tolist() == [0, 0, 0, 62, 8]
    assert flags.tolist() == [0, 0, 1, 0, 1]
