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
    ReplayBatch,
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
        segment_start_gpu_shadow=torch.empty(2, 1, 1, 1),
        key_tape_gpu_shadow=torch.empty(2, 4, 1, 1),
        value_tape_gpu_shadow=torch.empty(2, 4, 1, 1),
        g_tape_gpu_shadow=torch.empty(2, 4, 1),
        beta_tape_gpu_shadow=torch.empty(2, 4, 1),
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


def _state_row_id(indices: torch.Tensor, row: int) -> int:
    if indices.ndim == 1:
        return int(indices[row].item())
    return int(indices[row, 0].item())


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
        final_state_out: torch.Tensor | None,
        ssm_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        resident_final_state_out: torch.Tensor | None = None,
        resident_state_indices: torch.Tensor | None = None,
        resident_token_indices: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        packed_k = k.squeeze(0)
        packed_v = v.squeeze(0)
        if final_state_out is None:
            final_state_out = torch.empty(
                int(cu_seqlens[-1].item()),
                *initial_state.shape[1:],
                dtype=initial_state.dtype,
                device=initial_state.device,
            )
        for row, (start, end) in enumerate(
            zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
        ):
            state_idx = _state_row_id(ssm_state_indices, row)
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
            if (
                resident_final_state_out is not None
                and resident_state_indices is not None
                and resident_token_indices is not None
            ):
                resident_final_state_out[
                    int(resident_state_indices[row].item())
                ].copy_(final_state_out[end - 1])
        return final_state_out

    def fake_replay_from_shadow(
        shadow_key: torch.Tensor,
        shadow_value: torch.Tensor,
        shadow_g: torch.Tensor,
        shadow_beta: torch.Tensor,
        *,
        shadow_req_slots: torch.Tensor,
        shadow_src_begin: torch.Tensor,
        shadow_max_seq_len: int,
        initial_state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        final_state_out: torch.Tensor | None = None,
        resident_final_state_out: torch.Tensor | None = None,
        resident_state_indices: torch.Tensor | None = None,
        resident_token_indices: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        del shadow_max_seq_len
        if final_state_out is None:
            final_state_out = torch.empty(
                int(cu_seqlens[-1].item()),
                *initial_state.shape[1:],
                dtype=initial_state.dtype,
                device=initial_state.device,
            )
        for row, (start, end) in enumerate(
            zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
        ):
            state_idx = _state_row_id(ssm_state_indices, row)
            state = initial_state[state_idx].clone()
            req_slot = int(shadow_req_slots[row].item())
            src_begin = int(shadow_src_begin[row].item())
            for token_idx in range(start, end):
                shadow_idx = src_begin + (token_idx - start)
                delta = (
                    shadow_key[req_slot, shadow_idx].sum()
                    + shadow_value[req_slot, shadow_idx].sum()
                    + shadow_g[req_slot, shadow_idx].sum()
                    + shadow_beta[req_slot, shadow_idx].sum()
                )
                state = state + delta
                final_state_out[token_idx].copy_(state)
            if (
                resident_final_state_out is not None
                and resident_state_indices is not None
                and resident_token_indices is not None
            ):
                resident_final_state_out[
                    int(resident_state_indices[row].item())
                ].copy_(final_state_out[end - 1])
        return final_state_out

    monkeypatch.setattr(
        replay_module,
        "fused_sigmoid_gating_delta_rule_replay_from_tape",
        fake_replay_from_tape,
    )
    monkeypatch.setattr(
        replay_module,
        "fused_sigmoid_gating_delta_rule_replay_from_shadow_resident",
        fake_replay_from_shadow,
    )


def test_hybrid_spec_workspace_is_replay_only() -> None:
    workspace = _make_workspace()

    assert not hasattr(workspace, "temporal_state_gpu_scratch")
    assert not hasattr(workspace, "preload_stream")
    assert not hasattr(workspace, "segment_start_cpu_shadow")
    assert hasattr(workspace, "segment_start_gpu_shadow")
    assert not hasattr(workspace, "final_state_cpu_shadow")
    assert hasattr(workspace, "key_tape_gpu_shadow")
    assert hasattr(workspace, "value_tape_gpu_shadow")
    assert hasattr(workspace, "spill_stream")
    assert hasattr(workspace, "last_spill_done")
    assert hasattr(workspace, "last_segment_start_ready")
    assert hasattr(workspace, "initial_state_padded")
    assert hasattr(workspace, "resident_state_indices")


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


def test_snapshot_repair_timing_stats_exposes_replay_phase_breakdown() -> None:
    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.hybrid_temporal_replay_workspace = _make_workspace()
    attn.prefix = "layer.0"
    attn.kv_cache = (torch.empty(0), torch.empty(3, 1, 1, 1))
    attn.acquire_hybrid_temporal_replay_buffers = _make_replay_buffers

    stats = attn.snapshot_repair_timing_stats()

    assert stats["repair_copy_ms"] == 0.0
    assert stats["prepare_copy_ms"] == 0.0
    assert stats["repair_compute_ms"] == 0.0
    assert stats["verify_attention_ms"] == 0.0
    assert stats["spill_copy_ms"] == 0.0
    assert stats["layer_total_ms"] == 0.0
    assert stats["verify_call_count"] == 0


def test_store_replay_artifacts_writes_gpu_shadow_tapes(
) -> None:
    workspace = HybridTemporalReplayWorkspace(
        segment_start_gpu_shadow=torch.zeros(1, 1, 1, 1),
        key_tape_gpu_shadow=torch.zeros(1, 3, 1, 1),
        value_tape_gpu_shadow=torch.zeros(1, 3, 1, 1),
        g_tape_gpu_shadow=torch.zeros(1, 3, 1),
        beta_tape_gpu_shadow=torch.zeros(1, 3, 1),
        saved_generation_per_req=[-1],
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
        final_states=None,
        wave_plan=wave_plan,
    )

    assert ssm_state[1].item() == pytest.approx(0.0)
    assert workspace.segment_start_gpu_shadow[0].item() == pytest.approx(3.0)
    assert workspace.key_tape_gpu_shadow[0, :2, 0, 0].tolist() == [13.0, 17.0]
    assert workspace.value_tape_gpu_shadow[0, :2, 0, 0].tolist() == [19.0, 23.0]
    assert workspace.g_tape_gpu_shadow[0, :2, 0].tolist() == [7.0, 8.0]
    assert workspace.beta_tape_gpu_shadow[0, :2, 0].tolist() == [11.0, 12.0]
    assert workspace.saved_generation_per_req == [9]


def test_store_replay_artifacts_bulk_spill_matches_row_loop_reference() -> None:
    workspace = HybridTemporalReplayWorkspace(
        segment_start_gpu_shadow=torch.zeros(2, 1, 1, 1),
        key_tape_gpu_shadow=torch.zeros(2, 4, 1, 1),
        value_tape_gpu_shadow=torch.zeros(2, 4, 1, 1),
        g_tape_gpu_shadow=torch.zeros(2, 4, 1),
        beta_tape_gpu_shadow=torch.zeros(2, 4, 1),
        saved_generation_per_req=[-1, -1],
    )
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: torch.zeros(3, 1, 1, 1),
        replay_buffer_getter=_make_replay_buffers,
    )
    wave_plan = _make_wave_plan(
        req_ids=["req-a", "req-b"],
        spec_req_slots=[1, 0],
        spec_query_start_locs=[0, 2, 5],
        predicted_accept_lens=[1, 2],
        next_replay_generations=[7, 8],
    )
    initial_state = torch.tensor([[[[3.0]]], [[[5.0]]]])
    key = torch.tensor([[[[11.0]], [[13.0]], [[17.0]], [[19.0]], [[23.0]]]])
    value = torch.tensor(
        [[[[29.0]], [[31.0]], [[37.0]], [[41.0]], [[43.0]]]]
    )
    saved_g = torch.tensor([[47.0], [53.0], [59.0], [61.0], [67.0]])
    saved_beta = torch.tensor([[71.0], [73.0], [79.0], [83.0], [89.0]])

    helper.store_replay_artifacts(
        initial_state=initial_state,
        running_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        key=key,
        value=value,
        saved_g=saved_g,
        saved_beta=saved_beta,
        final_states=None,
        wave_plan=wave_plan,
    )

    ref_segment = torch.zeros_like(workspace.segment_start_gpu_shadow)
    ref_key = torch.zeros_like(workspace.key_tape_gpu_shadow)
    ref_value = torch.zeros_like(workspace.value_tape_gpu_shadow)
    ref_g = torch.zeros_like(workspace.g_tape_gpu_shadow)
    ref_beta = torch.zeros_like(workspace.beta_tape_gpu_shadow)
    ref_generations = [-1, -1]
    flat_key = key.squeeze(0)
    flat_value = value.squeeze(0)
    for row, (start, end) in enumerate(zip([0, 2], [2, 5])):
        req_slot = wave_plan.spec_req_slots[row]
        token_count = end - start
        ref_segment[req_slot].copy_(initial_state[row])
        ref_key[req_slot, :token_count].copy_(flat_key[start:end])
        ref_value[req_slot, :token_count].copy_(flat_value[start:end])
        ref_g[req_slot, :token_count].copy_(saved_g[start:end])
        ref_beta[req_slot, :token_count].copy_(saved_beta[start:end])
        ref_generations[req_slot] = wave_plan.next_replay_generations[row]

    torch.testing.assert_close(workspace.segment_start_gpu_shadow, ref_segment)
    torch.testing.assert_close(workspace.key_tape_gpu_shadow, ref_key)
    torch.testing.assert_close(workspace.value_tape_gpu_shadow, ref_value)
    torch.testing.assert_close(workspace.g_tape_gpu_shadow, ref_g)
    torch.testing.assert_close(workspace.beta_tape_gpu_shadow, ref_beta)
    assert workspace.saved_generation_per_req == ref_generations


def test_prepare_temporal_state_from_start_replays_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_replay(monkeypatch)
    workspace = _make_workspace()
    workspace.segment_start_gpu_shadow[0].fill_(10.0)
    workspace.key_tape_gpu_shadow[0, :2, 0, 0] = torch.tensor([1.0, 2.0])
    workspace.value_tape_gpu_shadow[0, :2, 0, 0] = torch.tensor([3.0, 4.0])
    workspace.g_tape_gpu_shadow[0, :2, 0] = torch.tensor([5.0, 6.0])
    workspace.beta_tape_gpu_shadow[0, :2, 0] = torch.tensor([7.0, 8.0])
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
    assert ssm_state[2].item() == pytest.approx(0.0)
    stats = helper.snapshot_repair_timing_stats()
    assert stats["repair_row_count"] == 1
    assert stats["repair_from_start_count"] == 1
    assert stats["repair_from_resident_count"] == 0


def test_prepare_temporal_state_waits_for_segment_start_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_workspace()
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: torch.zeros(3, 1, 1, 1),
        replay_buffer_getter=_make_replay_buffers,
    )
    helper.set_group_plan(_make_group_plan())
    waited: list[torch.Tensor] = []
    workspace.last_segment_start_ready = object()  # type: ignore[assignment]

    monkeypatch.setattr(
        helper,
        "_wait_for_segment_start_ready",
        lambda ssm_state: waited.append(ssm_state),
    )
    monkeypatch.setattr(
        helper,
        "_build_replay_batch",
        lambda plan, metadata: ReplayBatch([], [], [], [], [0], (0, 0)),
    )

    helper.prepare_temporal_state_for_verify(
        ssm_state=torch.zeros(3, 1, 1, 1),
        running_state_indices=torch.tensor([2], dtype=torch.int32),
    )

    assert len(waited) == 1


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


def test_prepare_temporal_state_none_uses_source_checkpoint_without_relocate() -> None:
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

    assert ssm_state[3].item() == pytest.approx(0.0)
    assert initial_state[0].item() == pytest.approx(17.0)


def test_prepare_temporal_state_from_resident_replays_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_replay(monkeypatch)
    workspace = _make_workspace()
    workspace.key_tape_gpu_shadow[0, :4, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    workspace.value_tape_gpu_shadow[0, :4, 0, 0] = torch.tensor(
        [5.0, 6.0, 7.0, 8.0]
    )
    workspace.g_tape_gpu_shadow[0, :4, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0])
    workspace.beta_tape_gpu_shadow[0, :4, 0] = torch.tensor(
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
    assert ssm_state[3].item() == pytest.approx(0.0)


def test_stage_replay_batch_bulk_gather_matches_row_loop_reference() -> None:
    workspace = _make_workspace()
    workspace.segment_start_gpu_shadow[0].fill_(10.0)
    workspace.key_tape_gpu_shadow[0, :3, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    workspace.value_tape_gpu_shadow[0, :3, 0, 0] = torch.tensor([4.0, 5.0, 6.0])
    workspace.g_tape_gpu_shadow[0, :3, 0] = torch.tensor([7.0, 8.0, 9.0])
    workspace.beta_tape_gpu_shadow[0, :3, 0] = torch.tensor([10.0, 11.0, 12.0])
    workspace.key_tape_gpu_shadow[1, :4, 0, 0] = torch.tensor([13.0, 14.0, 15.0, 16.0])
    workspace.value_tape_gpu_shadow[1, :4, 0, 0] = torch.tensor([17.0, 18.0, 19.0, 20.0])
    workspace.g_tape_gpu_shadow[1, :4, 0] = torch.tensor([21.0, 22.0, 23.0, 24.0])
    workspace.beta_tape_gpu_shadow[1, :4, 0] = torch.tensor([25.0, 26.0, 27.0, 28.0])
    workspace.saved_generation_per_req = [5, 6]

    ssm_state = torch.zeros(5, 1, 1, 1)
    ssm_state[4].fill_(30.0)
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: ssm_state,
        replay_buffer_getter=_make_replay_buffers,
    )
    plan = _make_group_plan(
        wave_plan=_make_wave_plan(
            req_ids=["req-a", "req-b"],
            spec_req_slots=[0, 1],
            spec_query_start_locs=[0, 4, 8],
            predicted_accept_lens=[2, 2],
            next_replay_generations=[6, 7],
        ),
        running_block_ids=[2, 4],
        source_block_ids=[2, 4],
        repair_row_indices=[0, 1],
        repair_req_slots=[0, 1],
        repair_target_slots=[2, 3],
        resident_slots=[0, 1],
        repair_modes=[
            HybridSpecRepairMode.FROM_START,
            HybridSpecRepairMode.FROM_RESIDENT,
        ],
        repair_generations=[5, 6],
    )
    runtime_metadata = helper._get_runtime_metadata(plan)
    replay_batch = helper._build_replay_batch(plan, runtime_metadata)
    (
        initial_state_buf,
        key_buf,
        value_buf,
        g_buf,
        beta_buf,
        _,
    ) = _make_replay_buffers(replay_batch.num_rows, replay_batch.num_tokens)

    helper._stage_replay_batch(
        ssm_state,
        plan,
        replay_batch,
        initial_state_buf,
        key_buf,
        value_buf,
        g_buf,
        beta_buf,
    )

    expected_initial = torch.tensor([[[[10.0]]], [[[30.0]]]])
    expected_key = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[15.0]], [[16.0]]])
    expected_value = torch.tensor(
        [[[4.0]], [[5.0]], [[6.0]], [[19.0]], [[20.0]]]
    )
    expected_g = torch.tensor([[7.0], [8.0], [9.0], [23.0], [24.0]])
    expected_beta = torch.tensor([[10.0], [11.0], [12.0], [27.0], [28.0]])

    torch.testing.assert_close(initial_state_buf, expected_initial)
    torch.testing.assert_close(key_buf, expected_key)
    torch.testing.assert_close(value_buf, expected_value)
    torch.testing.assert_close(g_buf, expected_g)
    torch.testing.assert_close(beta_buf, expected_beta)


def test_direct_shadow_replay_matches_tape_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_replay(monkeypatch)
    workspace = _make_workspace()
    workspace.segment_start_gpu_shadow[0].fill_(10.0)
    workspace.key_tape_gpu_shadow[0, :3, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    workspace.value_tape_gpu_shadow[0, :3, 0, 0] = torch.tensor([4.0, 5.0, 6.0])
    workspace.g_tape_gpu_shadow[0, :3, 0] = torch.tensor([7.0, 8.0, 9.0])
    workspace.beta_tape_gpu_shadow[0, :3, 0] = torch.tensor([10.0, 11.0, 12.0])
    workspace.saved_generation_per_req = [5, -1]

    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: torch.zeros(4, 1, 1, 1),
        replay_buffer_getter=_make_replay_buffers,
    )
    workspace.saved_generation_per_req = [5, -1]
    plan = _make_group_plan(
        running_block_ids=[3],
        source_block_ids=[1],
        repair_target_slots=[2],
        repair_generations=[5],
        wave_plan=_make_wave_plan(
            spec_query_start_locs=[0, 3],
            predicted_accept_lens=[1],
            next_replay_generations=[6],
        ),
    )
    runtime_metadata = helper._get_runtime_metadata(plan)
    replay_batch = helper._build_replay_batch(plan, runtime_metadata)

    tape_ssm_state = torch.zeros(4, 1, 1, 1)
    workspace.initial_state_padded.zero_()
    workspace.initial_state_padded[1].fill_(10.0)
    staged_tensors = helper._stage_replay_batch_direct(
        metadata=runtime_metadata,
        replay_batch=replay_batch,
    )
    initial_state = workspace.initial_state_padded[:2].clone()
    replay_output_row_ids = staged_tensors.replay_output_row_ids.clone()
    resident_token_indices = staged_tensors.resident_token_indices.clone()
    replay_req_slots = staged_tensors.replay_req_slots.clone()
    replay_src_begin = staged_tensors.replay_src_begin.clone()
    replay_cu_seqlens = staged_tensors.replay_cu_seqlens.clone()
    (
        initial_state_buf,
        key_buf,
        value_buf,
        g_buf,
        beta_buf,
        final_state_buf,
    ) = _make_replay_buffers(replay_batch.num_rows, replay_batch.num_tokens)
    helper._stage_replay_batch(
        tape_ssm_state,
        plan,
        replay_batch,
        initial_state_buf,
        key_buf,
        value_buf,
        g_buf,
        beta_buf,
    )
    helper._run_replay_from_tape(
        replay_batch,
        initial_state_buf,
        key_buf,
        value_buf,
        g_buf,
        beta_buf,
        final_state_buf,
    )
    expected = final_state_buf[replay_batch.cu_seqlens[-1] - 1].clone()

    workspace.initial_state_padded[:2].copy_(initial_state)
    helper._run_replay_from_shadow(
        replay_batch,
        replay_module.StagedReplayBatchTensors(
            replay_req_slots=replay_req_slots,
            replay_src_begin=replay_src_begin,
            replay_lengths=staged_tensors.replay_lengths.clone(),
            replay_cu_seqlens=replay_cu_seqlens,
            replay_output_row_ids=replay_output_row_ids,
            initial_state_row_ids=replay_output_row_ids,
            resident_token_indices=resident_token_indices,
        ),
    )

    torch.testing.assert_close(workspace.initial_state_padded[1], expected)


def test_stage_replay_batch_direct_reuses_runtime_metadata_tensors() -> None:
    workspace = _make_workspace()
    helper = HybridTemporalReplayHelper(
        layer_name="layer.0",
        workspace=workspace,
        ssm_state_getter=lambda: torch.zeros(4, 1, 1, 1),
        replay_buffer_getter=_make_replay_buffers,
    )
    workspace.saved_generation_per_req = [5, -1]
    plan = _make_group_plan(
        running_block_ids=[3],
        source_block_ids=[1],
        repair_target_slots=[2],
        repair_generations=[5],
        wave_plan=_make_wave_plan(
            spec_query_start_locs=[0, 3],
            predicted_accept_lens=[1],
            next_replay_generations=[6],
        ),
    )
    runtime_metadata = helper._get_runtime_metadata(plan)
    replay_batch = helper._build_replay_batch(plan, runtime_metadata)

    staged_tensors = helper._stage_replay_batch_direct(
        metadata=runtime_metadata,
        replay_batch=replay_batch,
    )

    assert (
        staged_tensors.replay_req_slots.data_ptr()
        == runtime_metadata.repair_req_slots_cpu.data_ptr()
    )
    assert (
        staged_tensors.replay_src_begin.data_ptr()
        == runtime_metadata.repair_src_begin_cpu.data_ptr()
    )
    assert (
        staged_tensors.replay_lengths.data_ptr()
        == runtime_metadata.repair_lengths_cpu.data_ptr()
    )
    assert (
        staged_tensors.replay_cu_seqlens.data_ptr()
        == runtime_metadata.replay_cu_seqlens_cpu.data_ptr()
    )
    assert (
        staged_tensors.replay_output_row_ids.data_ptr()
        == runtime_metadata.replay_output_row_ids_cpu.data_ptr()
    )
    assert staged_tensors.initial_state_row_ids.tolist() == [1]
    assert staged_tensors.resident_token_indices.tolist() == [2]


def test_forward_core_spec_replay_stores_replay_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = object.__new__(QwenGatedDeltaNetAttention)
    attn.A_log = torch.zeros(1)
    attn.dt_bias = torch.zeros(1)
    attn.hybrid_temporal_replay_workspace = _make_workspace()

    query_spec = torch.randn(1, 2, 1, 1)
    key_spec = torch.randn(1, 2, 1, 1)
    value_spec = torch.randn(1, 2, 1, 1)
    a = torch.randn(2, 1)
    b = torch.randn(2, 1)
    ssm_state = torch.randn(3, 1, 1, 1)
    running_state_indices = torch.tensor([1], dtype=torch.int32)
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

    def fake_capture_shadow(**kwargs):
        captured.update(kwargs)
        return (
            torch.empty_like(query_spec),
            None,
        )

    monkeypatch.setattr(
        qwen_gdn_module,
        "fused_sigmoid_gating_delta_rule_update_capture_shadow_resident",
        fake_capture_shadow,
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
    assert tuple(initial_state.shape) == (1, 1, 1, 1)
    assert initial_state[0].item() == pytest.approx(23.0)
    assert initial_state_indices.tolist() == [0]
    resident_state_out = captured["resident_final_state_out"]
    resident_state_indices = captured["resident_state_indices"]
    assert resident_state_out is ssm_state
    assert isinstance(resident_state_indices, torch.Tensor)
    assert resident_state_indices.tolist() == [1]
    resident_token_indices = captured["resident_token_indices"]
    assert isinstance(resident_token_indices, torch.Tensor)
    assert resident_token_indices.tolist() == [0]
    assert "final_state_out" not in captured
    assert "prepare" in captured
    assert "store" in captured
