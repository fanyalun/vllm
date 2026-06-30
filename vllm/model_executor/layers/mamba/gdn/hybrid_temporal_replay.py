# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import sys
from typing import Callable

import torch

from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_replay_from_tape,
)
from vllm.v1.hybrid_spec_replay import (
    HybridSpecRepairMode,
    HybridTemporalGroupPlan,
    HybridTemporalWavePlan,
)
from vllm.v1.worker.workspace import current_workspace_manager


TimingQueue = deque[tuple[str, torch.cuda.Event, torch.cuda.Event, int]]
ReplayBufferGetter = Callable[[int, int], tuple[torch.Tensor, ...]]


def _trace_predict_last(message: str) -> None:
    if os.getenv("VLLM_PREDICT_LAST_TRACE") != "1":
        return
    print(f"[predict_last][replay] {message}", file=sys.stderr, flush=True)


@dataclass
class HybridTemporalReplayWorkspace:
    segment_start_cpu_shadow: torch.Tensor
    key_tape_cpu_shadow: torch.Tensor
    value_tape_cpu_shadow: torch.Tensor
    g_tape_cpu_shadow: torch.Tensor
    beta_tape_cpu_shadow: torch.Tensor
    saved_generation_per_req: list[int]
    repair_copy_ms: float = 0.0
    repair_compute_ms: float = 0.0
    repair_row_count: int = 0
    repair_from_start_count: int = 0
    repair_from_resident_count: int = 0
    checkpoint_save_ms: float = 0.0
    tape_save_ms: float = 0.0
    pending_timing_events: TimingQueue = field(default_factory=deque)


@dataclass(frozen=True)
class ReplayBatch:
    row_indices: list[int]
    running_block_ids: list[int]
    req_slots: list[int]
    cu_seqlens: list[int]
    mode_counts: tuple[int, int]

    @property
    def num_rows(self) -> int:
        return len(self.row_indices)

    @property
    def num_tokens(self) -> int:
        return self.cu_seqlens[-1] if self.cu_seqlens else 0


class HybridTemporalReplayHelper:

    def __init__(
        self,
        layer_name: str,
        workspace: HybridTemporalReplayWorkspace,
        ssm_state_getter: Callable[[], torch.Tensor],
        replay_buffer_getter: ReplayBufferGetter,
        *,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> None:
        self.layer_name = layer_name
        self.workspace = workspace
        self._ssm_state_getter = ssm_state_getter
        self._replay_buffer_getter = replay_buffer_getter
        self._use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.group_plan: HybridTemporalGroupPlan | None = None

    def set_group_plan(self, plan: HybridTemporalGroupPlan | None) -> None:
        self.group_plan = plan

    def prepare_temporal_state_for_verify(
        self,
        ssm_state: torch.Tensor,
        running_state_indices: torch.Tensor,
        group_plan: HybridTemporalGroupPlan | None = None,
    ) -> torch.Tensor:
        if group_plan is not None:
            self.group_plan = group_plan
        plan = self.group_plan
        if plan is None:
            return ssm_state[running_state_indices].contiguous()

        running_state_indices_list = [int(idx) for idx in running_state_indices.tolist()]
        if running_state_indices_list != plan.running_block_ids:
            raise RuntimeError(
                f"{self.layer_name}: running_state_indices do not match "
                "group-plan block order"
            )
        _trace_predict_last(
            f"prepare layer={self.layer_name} running={plan.running_block_ids} "
            f"source={plan.source_block_ids} repair_rows={plan.repair_row_indices} "
            f"modes={[int(mode) for mode in plan.repair_modes]} "
            f"targets={plan.repair_target_slots} residents={plan.resident_slots} "
            f"gens={plan.repair_generations}"
        )

        workspace = self.workspace
        use_cuda_timing = ssm_state.is_cuda
        copy_start = copy_end = compute_start = compute_end = None
        if use_cuda_timing:
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            copy_start.record(torch.cuda.current_stream())

        self._relocate_predict_checkpoints(ssm_state, plan)
        replay_batch = self._build_replay_batch(ssm_state, plan)
        if replay_batch.num_rows == 0:
            if use_cuda_timing and copy_end is not None:
                copy_end.record(torch.cuda.current_stream())
                workspace.pending_timing_events.append(
                    ("repair_copy_ms", copy_start, copy_end, 0)
                )
                self._drain_timing_events()
            prepared = ssm_state[running_state_indices].contiguous()
            _trace_predict_last(
                f"prepare_done layer={self.layer_name} no_replay "
                f"first_scalar={float(prepared[0].reshape(-1)[0].item())}"
            )
            return prepared

        (
            initial_state_buf,
            key_buf,
            value_buf,
            g_buf,
            beta_buf,
            final_state_buf,
        ) = self._replay_buffer_getter(
            replay_batch.num_rows,
            replay_batch.num_tokens,
        )
        self._stage_replay_batch(
            ssm_state,
            plan,
            replay_batch,
            initial_state_buf,
            key_buf,
            value_buf,
            g_buf,
            beta_buf,
        )
        if use_cuda_timing and copy_end is not None:
            copy_end.record(torch.cuda.current_stream())
            workspace.pending_timing_events.append(
                ("repair_copy_ms", copy_start, copy_end, 0)
            )
            compute_start.record(torch.cuda.current_stream())
        self._run_replay_from_tape(
            replay_batch,
            initial_state_buf,
            key_buf,
            value_buf,
            g_buf,
            beta_buf,
            final_state_buf,
        )
        if use_cuda_timing and compute_end is not None:
            compute_end.record(torch.cuda.current_stream())
            workspace.pending_timing_events.append(
                ("repair_compute_ms", compute_start, compute_end, 0)
            )

        for row_idx, running_block_id, end in zip(
            replay_batch.row_indices,
            replay_batch.running_block_ids,
            replay_batch.cu_seqlens[1:],
        ):
            repaired = final_state_buf[end - 1]
            ssm_state[int(running_block_id)].copy_(repaired, non_blocking=True)
            _trace_predict_last(
                f"repair layer={self.layer_name} row={row_idx} running="
                f"{running_block_id} final_slot={end - 1} "
                f"first_scalar={float(repaired.reshape(-1)[0].item())}"
            )

        workspace.repair_row_count += replay_batch.num_rows
        workspace.repair_from_start_count += replay_batch.mode_counts[0]
        workspace.repair_from_resident_count += replay_batch.mode_counts[1]
        if use_cuda_timing:
            self._drain_timing_events()
        prepared = ssm_state[running_state_indices].contiguous()
        _trace_predict_last(
            f"prepare_done layer={self.layer_name} repaired_rows="
            f"{replay_batch.num_rows} "
            f"first_scalar={float(prepared[0].reshape(-1)[0].item())}"
        )
        return prepared

    def store_replay_artifacts(
        self,
        initial_state: torch.Tensor,
        running_state_indices: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        saved_g: torch.Tensor,
        saved_beta: torch.Tensor,
        final_states: torch.Tensor,
        wave_plan: HybridTemporalWavePlan,
    ) -> None:
        if not wave_plan.req_ids:
            return

        workspace = self.workspace
        ssm_state = self._ssm_state_getter()
        running_state_indices_list = [int(idx) for idx in running_state_indices.tolist()]
        key = key.squeeze(0)
        value = value.squeeze(0)

        use_cuda_timing = ssm_state.is_cuda
        checkpoint_start = checkpoint_end = tape_start = tape_end = None
        if use_cuda_timing:
            checkpoint_start = torch.cuda.Event(enable_timing=True)
            checkpoint_end = torch.cuda.Event(enable_timing=True)
            tape_start = torch.cuda.Event(enable_timing=True)
            tape_end = torch.cuda.Event(enable_timing=True)
            checkpoint_start.record(torch.cuda.current_stream())

        for row, (start, end) in enumerate(
            zip(
                wave_plan.spec_query_start_locs,
                wave_plan.spec_query_start_locs[1:],
            )
        ):
            resident_len = max(
                1,
                min(int(wave_plan.predicted_accept_lens[row]), end - start),
            )
            predicted_slot = start + resident_len - 1
            ssm_state[running_state_indices_list[row]].copy_(
                final_states[predicted_slot],
                non_blocking=True,
            )
            _trace_predict_last(
                f"store layer={self.layer_name} row={row} running="
                f"{running_state_indices_list[row]} predicted_len="
                f"{int(wave_plan.predicted_accept_lens[row])} predicted_slot="
                f"{predicted_slot} first_scalar="
                f"{float(final_states[predicted_slot].reshape(-1)[0].item())}"
            )
        if use_cuda_timing and checkpoint_end is not None:
            checkpoint_end.record(torch.cuda.current_stream())

        if use_cuda_timing and tape_start is not None:
            tape_start.record(torch.cuda.current_stream())
        for row, (start, end) in enumerate(
            zip(
                wave_plan.spec_query_start_locs,
                wave_plan.spec_query_start_locs[1:],
            )
        ):
            req_slot = int(wave_plan.spec_req_slots[row])
            token_count = end - start
            workspace.segment_start_cpu_shadow[req_slot].copy_(
                initial_state[row],
                non_blocking=True,
            )
            workspace.key_tape_cpu_shadow[req_slot, :token_count].copy_(
                key[start:end],
                non_blocking=True,
            )
            workspace.value_tape_cpu_shadow[req_slot, :token_count].copy_(
                value[start:end],
                non_blocking=True,
            )
            workspace.g_tape_cpu_shadow[req_slot, :token_count].copy_(
                saved_g[start:end],
                non_blocking=True,
            )
            workspace.beta_tape_cpu_shadow[req_slot, :token_count].copy_(
                saved_beta[start:end],
                non_blocking=True,
            )
            workspace.saved_generation_per_req[req_slot] = int(
                wave_plan.next_replay_generations[row]
            )
        if use_cuda_timing and tape_end is not None:
            tape_end.record(torch.cuda.current_stream())
            tape_done = torch.cuda.Event()
            tape_done.record(torch.cuda.current_stream())
            current_workspace_manager().mark_in_use_until(tape_done)

        if (
            use_cuda_timing
            and checkpoint_start is not None
            and checkpoint_end is not None
            and tape_start is not None
            and tape_end is not None
        ):
            workspace.pending_timing_events.append(
                ("checkpoint_save_ms", checkpoint_start, checkpoint_end, 0)
            )
            workspace.pending_timing_events.append(
                ("tape_save_ms", tape_start, tape_end, 0)
            )
            self._drain_timing_events()

    def reset_repair_timing_stats(self) -> None:
        self._drain_timing_events(synchronize=True)
        workspace = self.workspace
        workspace.pending_timing_events.clear()
        workspace.repair_copy_ms = 0.0
        workspace.repair_compute_ms = 0.0
        workspace.repair_row_count = 0
        workspace.repair_from_start_count = 0
        workspace.repair_from_resident_count = 0
        workspace.checkpoint_save_ms = 0.0
        workspace.tape_save_ms = 0.0

    def snapshot_repair_timing_stats(self) -> dict[str, float | int]:
        self._drain_timing_events(synchronize=True)
        workspace = self.workspace
        return {
            "repair_copy_ms": float(workspace.repair_copy_ms),
            "repair_compute_ms": float(workspace.repair_compute_ms),
            "repair_row_count": int(workspace.repair_row_count),
            "repair_from_start_count": int(workspace.repair_from_start_count),
            "repair_from_resident_count": int(
                workspace.repair_from_resident_count
            ),
            "checkpoint_save_ms": float(workspace.checkpoint_save_ms),
            "tape_save_ms": float(workspace.tape_save_ms),
        }

    def _relocate_predict_checkpoints(
        self,
        ssm_state: torch.Tensor,
        plan: HybridTemporalGroupPlan,
    ) -> None:
        for source_block_id, running_block_id in zip(
            plan.source_block_ids,
            plan.running_block_ids,
        ):
            if source_block_id == running_block_id:
                continue
            ssm_state[int(running_block_id)].copy_(
                ssm_state[int(source_block_id)],
                non_blocking=True,
            )

    def _build_replay_batch(
        self,
        ssm_state: torch.Tensor,
        plan: HybridTemporalGroupPlan,
    ) -> ReplayBatch:
        del ssm_state
        workspace = self.workspace
        row_indices: list[int] = []
        running_block_ids: list[int] = []
        req_slots: list[int] = []
        cu_seqlens = [0]
        from_start = 0
        from_resident = 0

        for idx, row_idx in enumerate(plan.repair_row_indices):
            mode = HybridSpecRepairMode(plan.repair_modes[idx])
            if mode == HybridSpecRepairMode.NONE:
                continue
            req_slot = int(plan.repair_req_slots[idx])
            generation = int(plan.repair_generations[idx])
            saved_generation = workspace.saved_generation_per_req[req_slot]
            if saved_generation != generation:
                raise RuntimeError(
                    f"{self.layer_name}: stale replay tape for req_slot={req_slot}, "
                    f"expected generation={generation}, got={saved_generation}"
                )

            target_slot = int(plan.repair_target_slots[idx])
            resident_slot = int(plan.resident_slots[idx])
            if mode == HybridSpecRepairMode.FROM_START:
                replay_tokens = target_slot + 1
                from_start += 1
            elif mode == HybridSpecRepairMode.FROM_RESIDENT:
                replay_tokens = target_slot - resident_slot
                from_resident += 1
            else:
                continue
            if replay_tokens <= 0:
                continue
            row_indices.append(int(row_idx))
            running_block_ids.append(int(plan.running_block_ids[row_idx]))
            req_slots.append(req_slot)
            cu_seqlens.append(cu_seqlens[-1] + replay_tokens)

        return ReplayBatch(
            row_indices=row_indices,
            running_block_ids=running_block_ids,
            req_slots=req_slots,
            cu_seqlens=cu_seqlens,
            mode_counts=(from_start, from_resident),
        )

    def _stage_replay_batch(
        self,
        ssm_state: torch.Tensor,
        plan: HybridTemporalGroupPlan,
        replay_batch: ReplayBatch,
        initial_state_buf: torch.Tensor,
        key_buf: torch.Tensor,
        value_buf: torch.Tensor,
        g_buf: torch.Tensor,
        beta_buf: torch.Tensor,
    ) -> None:
        workspace = self.workspace
        repair_idx_by_row = {
            row_idx: idx for idx, row_idx in enumerate(plan.repair_row_indices)
        }
        cursor = 0
        packed_row = 0
        for row_idx in replay_batch.row_indices:
            repair_idx = repair_idx_by_row[row_idx]
            mode = HybridSpecRepairMode(plan.repair_modes[repair_idx])
            req_slot = int(plan.repair_req_slots[repair_idx])
            target_slot = int(plan.repair_target_slots[repair_idx])
            resident_slot = int(plan.resident_slots[repair_idx])
            row_start = int(plan.wave_plan.spec_query_start_locs[row_idx])
            if mode == HybridSpecRepairMode.FROM_START:
                src_begin = 0
                src_end = target_slot + 1
                initial_state_buf[packed_row].copy_(
                    workspace.segment_start_cpu_shadow[req_slot],
                    non_blocking=True,
                )
            else:
                src_begin = resident_slot + 1
                src_end = target_slot + 1
                initial_state_buf[packed_row].copy_(
                    ssm_state[int(plan.running_block_ids[row_idx])],
                    non_blocking=True,
                )
            replay_tokens = src_end - src_begin
            key_buf[cursor : cursor + replay_tokens].copy_(
                workspace.key_tape_cpu_shadow[req_slot, src_begin:src_end],
                non_blocking=True,
            )
            value_buf[cursor : cursor + replay_tokens].copy_(
                workspace.value_tape_cpu_shadow[req_slot, src_begin:src_end],
                non_blocking=True,
            )
            g_buf[cursor : cursor + replay_tokens].copy_(
                workspace.g_tape_cpu_shadow[req_slot, src_begin:src_end],
                non_blocking=True,
            )
            beta_buf[cursor : cursor + replay_tokens].copy_(
                workspace.beta_tape_cpu_shadow[req_slot, src_begin:src_end],
                non_blocking=True,
            )
            _trace_predict_last(
                f"stage layer={self.layer_name} row={row_idx} req_slot={req_slot} "
                f"mode={int(mode)} tape=[{src_begin},{src_end}) row_start={row_start}"
            )
            cursor += replay_tokens
            packed_row += 1

    def _run_replay_from_tape(
        self,
        replay_batch: ReplayBatch,
        initial_state_buf: torch.Tensor,
        key_buf: torch.Tensor,
        value_buf: torch.Tensor,
        g_buf: torch.Tensor,
        beta_buf: torch.Tensor,
        final_state_buf: torch.Tensor,
    ) -> None:
        initial_state_padded = torch.empty(
            (replay_batch.num_rows + 1, *initial_state_buf.shape[1:]),
            dtype=initial_state_buf.dtype,
            device=initial_state_buf.device,
        )
        initial_state_padded[0].zero_()
        initial_state_padded[1:].copy_(initial_state_buf)
        cu_seqlens = torch.tensor(
            replay_batch.cu_seqlens,
            dtype=torch.int32,
            device=key_buf.device,
        )
        initial_state_indices = (
            torch.arange(
                1,
                replay_batch.num_rows + 1,
                dtype=torch.int32,
                device=key_buf.device,
            )
            .unsqueeze(1)
            .expand(-1, 1)
            .contiguous()
        )
        fused_sigmoid_gating_delta_rule_replay_from_tape(
            k=key_buf.unsqueeze(0),
            v=value_buf.unsqueeze(0),
            g=g_buf,
            beta=beta_buf,
            initial_state=initial_state_padded,
            final_state_out=final_state_buf,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=self._use_qk_l2norm_in_kernel,
        )

    def _drain_timing_events(self, *, synchronize: bool = False) -> None:
        workspace = self.workspace
        if synchronize:
            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize()
        while workspace.pending_timing_events:
            metric, start_event, end_event, _ = workspace.pending_timing_events[0]
            if not synchronize and not end_event.query():
                break
            workspace.pending_timing_events.popleft()
            setattr(
                workspace,
                metric,
                float(getattr(workspace, metric))
                + start_event.elapsed_time(end_event),
            )
