# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.base as gdn_base_module
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    HybridSpecStateOffloadWorkspace,
    QwenGatedDeltaNetAttention,
)
from vllm.v1.hybrid_spec_offload import HybridSpecReloadMode


def test_hybrid_spec_workspace_is_temporal_only() -> None:
    workspace = HybridSpecStateOffloadWorkspace(
        temporal_state_cpu_shadow=torch.empty(1, 1, 1, 1, 1),
    )

    assert not hasattr(workspace, "temporal_state_gpu_scratch")
    assert not hasattr(workspace, "conv_state_gpu_scratch")
    assert not hasattr(workspace, "conv_state_cpu_shadow")


def test_hybrid_temporal_scratch_uses_workspace_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_calls: list[tuple[tuple[int, ...], torch.dtype]] = []
    acquire_calls: list[tuple[tuple[int, ...], torch.dtype]] = []

    class FakeWorkspaceManager:

        def reserve_simultaneous_for_all_ubatches(self, *shapes_and_dtypes):
            reserve_calls.extend(shapes_and_dtypes)

        def get_simultaneous(self, *shapes_and_dtypes):
            acquire_calls.extend(shapes_and_dtypes)
            return [
                torch.empty(shape, dtype=dtype)
                for shape, dtype in shapes_and_dtypes
            ]

    monkeypatch.setattr(
        gdn_base_module,
        "current_workspace_manager",
        lambda: FakeWorkspaceManager(),
    )

    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.num_spec = 4
    attn.get_state_shape = lambda: ((9, 9), (2, 3, 5))
    attn.get_state_dtype = lambda: (torch.float16, torch.float32)

    reserve_spec = attn.reserve_hybrid_temporal_scratch(max_num_reqs=8)
    scratch = attn.acquire_hybrid_temporal_scratch(num_rows=7)

    assert reserve_spec == ((40, 2, 3, 5), torch.float32)
    assert reserve_calls == [((40, 2, 3, 5), torch.float32)]
    assert acquire_calls == [((7, 2, 3, 5), torch.float32)]
    assert tuple(scratch.shape) == (7, 2, 3, 5)
    assert scratch.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_stage_temporal_preload_copies_shadow_to_persistent_state() -> None:
    workspace = HybridSpecStateOffloadWorkspace(
        temporal_state_cpu_shadow=torch.zeros(
            1, 2, 1, 1, 1, device="cpu", dtype=torch.float32, pin_memory=True
        ),
        preload_stream=torch.cuda.Stream(),
        preload_done_events=[None],
        preload_generation_per_req=[-1],
        pending_preload_timings=deque(),
        pending_preloaded_timings=deque(),
        pending_fallback_timings=deque(),
    )
    workspace.temporal_state_cpu_shadow[0, 1].fill_(7.0)
    workspace.shadow_copy_done_event = torch.cuda.Event()
    workspace.shadow_copy_done_event.record(torch.cuda.current_stream())

    ssm_state = torch.zeros(3, 1, 1, 1, device="cuda", dtype=torch.float32)
    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.hybrid_spec_state_offload_workspace = workspace
    attn.kv_cache = (torch.empty(0, device="cuda"), ssm_state)

    attn.stage_temporal_preload(
        req_slots=[0],
        reload_slots=[1],
        running_block_ids=[2],
        reload_generations=[5],
    )

    assert workspace.preload_done_events[0] is not None
    torch.cuda.current_stream().wait_event(workspace.preload_done_events[0])
    torch.cuda.synchronize()

    assert workspace.preload_generation_per_req == [5]
    assert ssm_state[2].item() == pytest.approx(7.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_prepare_temporal_initial_state_uses_preloaded_state() -> None:
    preload_done_event = torch.cuda.Event()
    preload_done_event.record(torch.cuda.current_stream())
    workspace = HybridSpecStateOffloadWorkspace(
        temporal_state_cpu_shadow=torch.zeros(
            1, 2, 1, 1, 1, device="cpu", dtype=torch.float32, pin_memory=True
        ),
        preload_done_events=[preload_done_event],
        preload_generation_per_req=[3],
        pending_preload_timings=deque(),
        pending_preloaded_timings=deque(),
        pending_fallback_timings=deque(),
    )
    workspace.temporal_state_cpu_shadow[0, 1].fill_(13.0)

    ssm_state = torch.zeros(2, 1, 1, 1, device="cuda", dtype=torch.float32)
    ssm_state[1].fill_(11.0)

    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.hybrid_spec_state_offload_workspace = workspace

    metadata = SimpleNamespace(
        temporal_reload_mode_cpu=torch.tensor(
            [int(HybridSpecReloadMode.PRELOADED)], dtype=torch.int32
        ),
        spec_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
        reload_slot_cpu=torch.tensor([1], dtype=torch.int32),
        reload_generation_cpu=torch.tensor([3], dtype=torch.int32),
    )
    initial_state = attn._prepare_temporal_initial_state(
        ssm_state=ssm_state,
        running_state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        attn_metadata=metadata,
    )
    torch.cuda.synchronize()

    assert initial_state[0].item() == pytest.approx(11.0)
    assert ssm_state[1].item() == pytest.approx(11.0)
