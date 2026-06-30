# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.base as gdn_base_module
import vllm.model_executor.layers.mamba.gdn.hybrid_temporal_replay as replay_module
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qwen_gdn_module
from vllm.model_executor.layers.mamba.gdn.hybrid_temporal_replay import (
    HybridTemporalReplayHelper,
    HybridTemporalReplayWorkspace,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.hybrid_spec_replay import (
    HybridSpecRepairMode,
    HybridTemporalGroupPlan,
    HybridTemporalWavePlan,
)


def _make_wave_plan(
    *,
    req_ids: list[str] | None = None,
    spec_req_slots: list[int] | None = None,
    spec_query_start_locs: list[int] | None = None,
    predicted_accept_lens: list[int] | None = None,
    next_replay_generations: list[int] | None = None,
) -> HybridTemporalWavePlan:
    return HybridTemporalWavePlan(
        req_ids=["req-a"] if req_ids is None else req_ids,
        spec_req_slots=[0] if spec_req_slots is None else spec_req_slots,
        spec_query_start_locs=(
            [0, 2]
            if spec_query_start_locs is None
            else spec_query_start_locs
        ),
        predicted_accept_lens=(
            [1] if predicted_accept_lens is None else predicted_accept_lens
        ),
        next_replay_generations=(
            [4]
            if next_replay_generations is None
            else next_replay_generations
        ),
    )


def _make_group_plan(
    *,
    wave_plan: HybridTemporalWavePlan | None = None,
    running_block_ids: list[int] | None = None,
    source_block_ids: list[int] | None = None,
    repair_row_indices: list[int] | None = None,
    repair_req_slots: list[int] | None = None,
    repair_target_slots: list[int] | None = None,
    resident_slots: list[int] | None = None,
    repair_modes: list[HybridSpecRepairMode] | None = None,
    repair_generations: list[int] | None = None,
) -> HybridTemporalGroupPlan:
    return HybridTemporalGroupPlan(
        wave_plan=wave_plan or _make_wave_plan(),
        running_block_ids=[2] if running_block_ids is None else running_block_ids,
        source_block_ids=[1] if source_block_ids is None else source_block_ids,
        repair_row_indices=(
            [0] if repair_row_indices is None else repair_row_indices
        ),
        repair_req_slots=[0] if repair_req_slots is None else repair_req_slots,
        repair_target_slots=(
            [1] if repair_target_slots is None else repair_target_slots
        ),
        resident_slots=[0] if resident_slots is None else resident_slots,
        repair_modes=(
            [HybridSpecRepairMode.FROM_START]
            if repair_modes is None
            else repair_modes
        ),
        repair_generations=(
            [5] if repair_generations is None else repair_generations
        ),
    )


def _make_workspace() -> HybridTemporalReplayWorkspace:
    return HybridTemporalReplayWorkspace(
        segment_start_cpu_shadow=torch.empty(2, 1, 1, 1),
        key_tape_cpu_shadow=torch.empty(2, 4, 1, 1),
        value_tape_cpu_shadow=torch.empty(2, 4, 1, 1),
        g_tape_cpu_shadow=torch.empty(2, 4, 1),
        beta_tape_cpu_shadow=torch.empty(2, 4, 1),
        saved_generation_per_req=[-1, -1],
    )


def _make_replay_buffers(
    num_rows: int,
    num_tokens: int,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(num_rows, 1, 1, 1),
        torch.empty(num_tokens, 1, 1),
        torch.empty(num_tokens, 1, 1),
        torch.empty(num_tokens, 1),
        torch.empty(num_tokens, 1),
        torch.empty(num_tokens, 1, 1, 1),
    )


def _install_fake_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_replay_from_tape(
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        final_state_out: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        packed_k = k.squeeze(0)
        packed_v = v.squeeze(0)
        for row, (start, end) in enumerate(
            zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
        ):
            state_idx = int(ssm_state_indices[row, 0].item())
            state = initial_state[state_idx].clone()
            for token_idx in range(start, end):
                delta = (
                    packed_k[token_idx].sum()
                    + packed_v[token_idx].sum()
                    + g[token_idx].sum()
                    + beta[token_idx].sum()
                )
                state = state + delta
                final_state_out[token_idx].copy_(state)
        return final_state_out

    monkeypatch.setattr(
        replay_module,
        "fused_sigmoid_gating_delta_rule_replay_from_tape",
        fake_replay_from_tape,
    )


def test_hybrid_spec_workspace_is_replay_only() -> None:
    workspace = _make_workspace()

    assert not hasattr(workspace, "temporal_state_gpu_scratch")
    assert not hasattr(workspace, "preload_stream")
    assert hasattr(workspace, "segment_start_cpu_shadow")
    assert not hasattr(workspace, "final_state_cpu_shadow")
    assert hasattr(workspace, "key_tape_cpu_shadow")
    assert hasattr(workspace, "value_tape_cpu_shadow")


def test_hybrid_temporal_scratch_uses_workspace_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_calls: list[tuple[tuple[int, ...], torch.dtype]] = []
    acquire_calls: list[tuple[tuple[int, ...], torch.dtype]] = []

    class FakeWorkspaceManager:

        @staticmethod
        def _required_workspace_bytes(*shapes_and_dtypes):
            total = 0
            for shape, dtype in shapes_and_dtypes:
                total += torch.empty((), dtype=dtype).element_size() * int(
                    torch.tensor(shape).prod().item()
                )
            return total

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
    attn.tp_size = 1
    attn.num_k_heads = 2
    attn.num_v_heads = 3
    attn.head_k_dim = 5
    attn.head_v_dim = 7
    attn.model_config = SimpleNamespace(dtype=torch.bfloat16)
    attn.get_state_shape = lambda: ((9, 9), (2, 3, 5))
    attn.get_state_dtype = lambda: (torch.float16, torch.float32)

    reserve_spec = attn.reserve_hybrid_temporal_scratch(max_num_reqs=8)
    verify_scratch = attn.acquire_hybrid_temporal_verify_scratch(num_tokens=7)
    replay_buffers = attn.acquire_hybrid_temporal_replay_buffers(
        num_rows=3,
        num_tokens=6,
    )

    assert reserve_spec == ((40, 2, 3, 5), torch.float32)
    assert reserve_calls
    assert acquire_calls[0] == ((7, 2, 3, 5), torch.float32)
    assert tuple(verify_scratch.shape) == (7, 2, 3, 5)
    assert len(replay_buffers) == 6
    assert tuple(replay_buffers[0].shape) == (3, 2, 3, 5)
    assert tuple(replay_buffers[1].shape) == (6, 2, 5)
    assert tuple(replay_buffers[2].shape) == (6, 3, 7)
    assert tuple(replay_buffers[3].shape) == (6, 3)
    assert tuple(replay_buffers[4].shape) == (6, 3)
    assert tuple(replay_buffers[5].shape) == (6, 2, 3, 5)


def test_layer_group_plan_binding_uses_shared_plan() -> None:
    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.hybrid_temporal_replay_workspace = _make_workspace()
    attn.prefix = "layer.0"
    attn.kv_cache = (torch.empty(0), torch.empty(3, 1, 1, 1))
    attn.acquire_hybrid_temporal_replay_buffers = _make_replay_buffers

    plan = _make_group_plan()
    attn.set_hybrid_temporal_group_plan(plan)

    assert attn._get_hybrid_temporal_replay_helper().group_plan is plan


def test_store_replay_artifacts_writes_predicted_resident_and_tapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = HybridTemporalReplayWorkspace(
        segment_start_cpu_shadow=torch.zeros(1, 1, 1, 1),
        key_tape_cpu_shadow=torch.zeros(1, 3, 1, 1),
        value_tape_cpu_shadow=torch.zeros(1, 3, 1, 1),
        g_tape_cpu_shadow=torch.zeros(1, 3, 1),
        beta_tape_cpu_shadow=torch.zeros(1, 3, 1),
        saved_generation_per_req=[-1],
    )
    marked_events: list[object] = []

    class FakeWorkspaceManager:

        def mark_in_use_until(self, event) -> None:
            marked_events.append(event)

    monkeypatch.setattr(
        replay_module,
        "current_workspace_manager",
        lambda: FakeWorkspaceManager(),
    )

    ssm_state = torch.zeros(2, 1, 1, 1)
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: ssm_state,
        replay_buffer_getter=_make_replay_buffers,
    )
    wave_plan = _make_wave_plan(
        spec_query_start_locs=[0, 2],
        predicted_accept_lens=[1],
        next_replay_generations=[9],
    )

    helper.store_replay_artifacts(
        initial_state=torch.full((1, 1, 1, 1), 3.0),
        running_state_indices=torch.tensor([1], dtype=torch.int32),
        key=torch.tensor([[[[13.0]], [[17.0]]]]),
        value=torch.tensor([[[[19.0]], [[23.0]]]]),
        saved_g=torch.tensor([[7.0], [8.0]]),
        saved_beta=torch.tensor([[11.0], [12.0]]),
        final_states=torch.tensor([[[[29.0]]], [[[31.0]]]]),
        wave_plan=wave_plan,
    )

    assert ssm_state[1].item() == pytest.approx(29.0)
    assert workspace.segment_start_cpu_shadow[0].item() == pytest.approx(3.0)
    assert workspace.key_tape_cpu_shadow[0, :2, 0, 0].tolist() == [13.0, 17.0]
    assert workspace.value_tape_cpu_shadow[0, :2, 0, 0].tolist() == [19.0, 23.0]
    assert workspace.g_tape_cpu_shadow[0, :2, 0].tolist() == [7.0, 8.0]
    assert workspace.beta_tape_cpu_shadow[0, :2, 0].tolist() == [11.0, 12.0]
    assert workspace.saved_generation_per_req == [9]
    assert len(marked_events) == 0


def test_prepare_temporal_state_from_start_replays_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_replay(monkeypatch)
    workspace = _make_workspace()
    workspace.segment_start_cpu_shadow[0].fill_(10.0)
    workspace.key_tape_cpu_shadow[0, :2, 0, 0] = torch.tensor([1.0, 2.0])
    workspace.value_tape_cpu_shadow[0, :2, 0, 0] = torch.tensor([3.0, 4.0])
    workspace.g_tape_cpu_shadow[0, :2, 0] = torch.tensor([5.0, 6.0])
    workspace.beta_tape_cpu_shadow[0, :2, 0] = torch.tensor([7.0, 8.0])
    workspace.saved_generation_per_req = [5, -1]

    ssm_state = torch.zeros(3, 1, 1, 1)
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: ssm_state,
        replay_buffer_getter=_make_replay_buffers,
    )
    helper.set_group_plan(_make_group_plan())

    initial_state = helper.prepare_temporal_state_for_verify(
        ssm_state=ssm_state,
        running_state_indices=torch.tensor([2], dtype=torch.int32),
    )

    assert initial_state[0].item() == pytest.approx(46.0)
    assert ssm_state[2].item() == pytest.approx(46.0)
    stats = helper.snapshot_repair_timing_stats()
    assert stats["repair_row_count"] == 1
    assert stats["repair_from_start_count"] == 1
    assert stats["repair_from_resident_count"] == 0


def test_prepare_temporal_state_rejects_stale_generation() -> None:
    workspace = _make_workspace()
    workspace.saved_generation_per_req = [3, -1]
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: torch.empty(2, 1, 1, 1),
        replay_buffer_getter=_make_replay_buffers,
    )
    helper.set_group_plan(
        _make_group_plan(
            repair_generations=[5],
            running_block_ids=[1],
        )
    )

    with pytest.raises(RuntimeError, match="stale replay tape"):
        helper.prepare_temporal_state_for_verify(
            ssm_state=torch.empty(2, 1, 1, 1),
            running_state_indices=torch.tensor([1], dtype=torch.int32),
        )


def test_prepare_temporal_state_none_only_relocates_checkpoint() -> None:
    workspace = _make_workspace()
    ssm_state = torch.zeros(4, 1, 1, 1)
    ssm_state[1].fill_(17.0)
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: ssm_state,
        replay_buffer_getter=_make_replay_buffers,
    )
    helper.set_group_plan(
        _make_group_plan(
            wave_plan=_make_wave_plan(
                spec_query_start_locs=[0, 1],
                predicted_accept_lens=[1],
            ),
            running_block_ids=[3],
            source_block_ids=[1],
            repair_row_indices=[],
            repair_req_slots=[],
            repair_target_slots=[],
            resident_slots=[],
            repair_modes=[],
            repair_generations=[],
        )
    )

    initial_state = helper.prepare_temporal_state_for_verify(
        ssm_state=ssm_state,
        running_state_indices=torch.tensor([3], dtype=torch.int32),
    )

    assert ssm_state[3].item() == pytest.approx(17.0)
    assert initial_state[0].item() == pytest.approx(17.0)


def test_prepare_temporal_state_from_resident_replays_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_replay(monkeypatch)
    workspace = _make_workspace()
    workspace.key_tape_cpu_shadow[0, :4, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    workspace.value_tape_cpu_shadow[0, :4, 0, 0] = torch.tensor(
        [5.0, 6.0, 7.0, 8.0]
    )
    workspace.g_tape_cpu_shadow[0, :4, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0])
    workspace.beta_tape_cpu_shadow[0, :4, 0] = torch.tensor(
        [13.0, 14.0, 15.0, 16.0]
    )
    workspace.saved_generation_per_req = [6, -1]

    ssm_state = torch.zeros(5, 1, 1, 1)
    ssm_state[1].fill_(19.0)
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: ssm_state,
        replay_buffer_getter=_make_replay_buffers,
    )
    helper.set_group_plan(
        _make_group_plan(
            wave_plan=_make_wave_plan(
                spec_query_start_locs=[0, 4],
                predicted_accept_lens=[2],
                next_replay_generations=[7],
            ),
            running_block_ids=[3],
            source_block_ids=[1],
            repair_target_slots=[3],
            resident_slots=[1],
            repair_modes=[HybridSpecRepairMode.FROM_RESIDENT],
            repair_generations=[6],
        )
    )

    initial_state = helper.prepare_temporal_state_for_verify(
        ssm_state=ssm_state,
        running_state_indices=torch.tensor([3], dtype=torch.int32),
    )

    # Start from resident checkpoint 19, replay token 2 then 3.
    assert initial_state[0].item() == pytest.approx(95.0)
    assert ssm_state[3].item() == pytest.approx(95.0)


def test_forward_core_spec_replay_stores_replay_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.A_log = torch.zeros(1)
    attn.dt_bias = torch.zeros(1)
    attn.hybrid_temporal_replay_workspace = object()

    query_spec = torch.randn(1, 2, 1, 1)
    key_spec = torch.randn(1, 2, 1, 1)
    value_spec = torch.randn(1, 2, 1, 1)
    a = torch.randn(2, 1)
    b = torch.randn(2, 1)
    ssm_state = torch.randn(3, 1, 1, 1)
    running_state_indices = torch.tensor([1], dtype=torch.int32)
    scratch = torch.empty(2, 1, 1, 1)
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs):
        captured.setdefault("prepare", kwargs)
        return torch.full((1, 1, 1, 1), 23.0)

    fake_helper = SimpleNamespace(
        group_plan=_make_group_plan(),
        prepare_temporal_state_for_verify=fake_prepare,
        store_replay_artifacts=lambda **kwargs: captured.setdefault("store", kwargs),
    )
    attn._get_hybrid_temporal_replay_helper = lambda: fake_helper
    attn.acquire_hybrid_temporal_verify_scratch = lambda num_tokens: scratch

    def fake_capture_tape(**kwargs):
        captured.update(kwargs)
        return (
            torch.empty_like(query_spec),
            scratch,
            torch.full((2, 1), 7.0),
            torch.full((2, 1), 11.0),
        )

    monkeypatch.setattr(
        qwen_gdn_module,
        "fused_sigmoid_gating_delta_rule_update_capture_tape",
        fake_capture_tape,
    )

    metadata = SimpleNamespace(
        spec_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        spec_query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        num_spec_decodes=1,
        num_spec_decode_tokens=2,
        num_accepted_tokens=torch.tensor([2], dtype=torch.int32),
        spec_max_query_len=2,
    )
    attn._forward_core_spec_replay(
        query_spec=query_spec,
        key_spec=key_spec,
        value_spec=value_spec,
        a=a,
        b=b,
        ssm_state=ssm_state,
        running_state_indices=running_state_indices,
        attn_metadata=metadata,
    )

    initial_state = captured["initial_state"]
    initial_state_indices = captured["ssm_state_indices"]
    assert isinstance(initial_state, torch.Tensor)
    assert isinstance(initial_state_indices, torch.Tensor)
    assert tuple(initial_state.shape) == (2, 1, 1, 1)
    assert initial_state[0].item() == pytest.approx(0.0)
    assert initial_state[1].item() == pytest.approx(23.0)
    assert initial_state_indices.tolist() == [[1, 1]]
    assert "prepare" in captured
    assert "store" in captured
