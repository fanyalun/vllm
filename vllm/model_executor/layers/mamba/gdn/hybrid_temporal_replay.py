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
    fused_sigmoid_gating_delta_rule_replay_from_shadow_resident,
    fused_sigmoid_gating_delta_rule_replay_from_tape,
)
from vllm.v1.hybrid_spec_replay import (
    HybridSpecRepairMode,
    HybridTemporalGroupPlan,
    HybridTemporalRuntimeMetadataBundle,
    HybridTemporalWavePlan,
)


TimingQueue = deque[tuple[str, torch.cuda.Event, torch.cuda.Event, int]]
ReplayBufferGetter = Callable[[int, int], tuple[torch.Tensor, ...]]


def _trace_predict_last(message: str) -> None:
    if os.getenv("VLLM_PREDICT_LAST_TRACE") != "1":
        return
    print(f"[predict_last][replay] {message}", file=sys.stderr, flush=True)


@dataclass
class HybridTemporalReplayWorkspace:
    segment_start_gpu_shadow: torch.Tensor
    key_tape_gpu_shadow: torch.Tensor
    value_tape_gpu_shadow: torch.Tensor
    g_tape_gpu_shadow: torch.Tensor
    beta_tape_gpu_shadow: torch.Tensor
    saved_generation_per_req: list[int]
    spill_stream: torch.cuda.Stream | None = None
    last_spill_done: torch.cuda.Event | None = None
    last_segment_start_ready: torch.cuda.Event | None = None
    pending_spill_refs: list[tuple[torch.cuda.Event, tuple[torch.Tensor, ...]]] = (
        field(default_factory=list)
    )
    spill_req_slots: torch.Tensor | None = None
    spill_row_starts: torch.Tensor | None = None
    spill_dst_linear_indices: torch.Tensor | None = None
    replay_req_slots: torch.Tensor | None = None
    replay_src_begin: torch.Tensor | None = None
    replay_src_end: torch.Tensor | None = None
    replay_lengths: torch.Tensor | None = None
    replay_cu_seqlens: torch.Tensor | None = None
    replay_src_linear_indices: torch.Tensor | None = None
    replay_running_block_ids: torch.Tensor | None = None
    replay_output_row_ids: torch.Tensor | None = None
    replay_from_start_rows: torch.Tensor | None = None
    replay_from_start_req_slots: torch.Tensor | None = None
    replay_from_resident_rows: torch.Tensor | None = None
    replay_from_resident_running_blocks: torch.Tensor | None = None
    token_offsets: torch.Tensor | None = None
    state_row_ids: torch.Tensor | None = None
    initial_state_padded: torch.Tensor | None = None
    initial_state_indices: torch.Tensor | None = None
    resident_state_indices: torch.Tensor | None = None
    resident_token_indices: torch.Tensor | None = None
    capture_source_block_ids: torch.Tensor | None = None
    checkpoint_src_blocks: torch.Tensor | None = None
    checkpoint_dst_blocks: torch.Tensor | None = None
    repair_copy_ms: float = 0.0
    repair_compute_ms: float = 0.0
    repair_row_count: int = 0
    repair_from_start_count: int = 0
    repair_from_resident_count: int = 0
    verify_attention_ms: float = 0.0
    layer_total_ms: float = 0.0
    verify_call_count: int = 0
    checkpoint_save_ms: float = 0.0
    post_replay_state_gather_ms: float = 0.0
    capture_materialize_ms: float = 0.0
    segment_start_save_ms: float = 0.0
    segment_start_wait_ms: float = 0.0
    tape_save_ms: float = 0.0
    pending_timing_events: TimingQueue = field(default_factory=deque)

    def __post_init__(self) -> None:
        max_num_reqs = int(self.segment_start_gpu_shadow.shape[0])
        max_candidate_states = int(self.key_tape_gpu_shadow.shape[1])
        max_num_tokens = max_num_reqs * max_candidate_states
        device = self.segment_start_gpu_shadow.device

        if device.type == "cuda" and self.spill_stream is None:
            self.spill_stream = torch.cuda.Stream(device=device)
        if self.spill_req_slots is None:
            self.spill_req_slots = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.spill_row_starts is None:
            self.spill_row_starts = torch.empty(
                max_num_reqs + 1,
                dtype=torch.int32,
                device=device,
            )
        if self.spill_dst_linear_indices is None:
            self.spill_dst_linear_indices = torch.empty(
                max_num_tokens,
                dtype=torch.long,
                device=device,
            )
        if self.replay_req_slots is None:
            self.replay_req_slots = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.replay_src_begin is None:
            self.replay_src_begin = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.replay_src_end is None:
            self.replay_src_end = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.replay_cu_seqlens is None:
            self.replay_cu_seqlens = torch.empty(
                max_num_reqs + 1,
                dtype=torch.int32,
                device=device,
            )
        if self.replay_lengths is None:
            self.replay_lengths = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        if self.replay_src_linear_indices is None:
            self.replay_src_linear_indices = torch.empty(
                max_num_tokens,
                dtype=torch.long,
                device=device,
            )
        if self.replay_running_block_ids is None:
            self.replay_running_block_ids = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        if self.replay_output_row_ids is None:
            self.replay_output_row_ids = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        if self.replay_from_start_rows is None:
            self.replay_from_start_rows = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.replay_from_start_req_slots is None:
            self.replay_from_start_req_slots = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.replay_from_resident_rows is None:
            self.replay_from_resident_rows = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.replay_from_resident_running_blocks is None:
            self.replay_from_resident_running_blocks = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.token_offsets is None:
            self.token_offsets = torch.arange(
                max_num_tokens,
                dtype=torch.long,
                device=device,
            )
        if self.state_row_ids is None:
            self.state_row_ids = torch.arange(
                max_num_reqs + 1,
                dtype=torch.int32,
                device=device,
            )
        if self.initial_state_padded is None:
            self.initial_state_padded = torch.empty(
                (max_num_reqs + 1, *self.segment_start_gpu_shadow.shape[1:]),
                dtype=self.segment_start_gpu_shadow.dtype,
                device=device,
            )
        if self.initial_state_indices is None:
            self.initial_state_indices = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        if self.resident_state_indices is None:
            self.resident_state_indices = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        if self.resident_token_indices is None:
            self.resident_token_indices = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        if self.capture_source_block_ids is None:
            self.capture_source_block_ids = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.checkpoint_src_blocks is None:
            self.checkpoint_src_blocks = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )
        if self.checkpoint_dst_blocks is None:
            self.checkpoint_dst_blocks = torch.empty(
                max_num_reqs,
                dtype=torch.long,
                device=device,
            )


@dataclass(frozen=True)
class ReplayBatch:
    repair_plan_indices: list[int]
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


@dataclass(frozen=True)
class StagedReplayBatchTensors:
    replay_req_slots: torch.Tensor
    replay_src_begin: torch.Tensor
    replay_lengths: torch.Tensor
    replay_cu_seqlens: torch.Tensor
    replay_output_row_ids: torch.Tensor
    initial_state_row_ids: torch.Tensor
    resident_token_indices: torch.Tensor


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

    @staticmethod
    def _select_runtime_tensor(
        metadata: HybridTemporalRuntimeMetadataBundle,
        *,
        cpu_name: str,
        gpu_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        gpu_tensor = getattr(metadata, gpu_name, None)
        if gpu_tensor is not None and gpu_tensor.device == device:
            return gpu_tensor
        return getattr(metadata, cpu_name)

    def _get_runtime_metadata(
        self,
        plan: HybridTemporalGroupPlan,
    ) -> HybridTemporalRuntimeMetadataBundle:
        metadata = plan.runtime_metadata
        if metadata is not None:
            return metadata

        resident_token_indices = [
            max(
                0,
                min(
                    int(plan.wave_plan.predicted_accept_lens[row_idx]),
                    int(
                        plan.wave_plan.spec_query_start_locs[row_idx + 1]
                        - plan.wave_plan.spec_query_start_locs[row_idx]
                    ),
                )
                - 1,
            )
            for row_idx in range(len(plan.wave_plan.req_ids))
        ]
        packed_repair_req_slots: list[int] = []
        packed_repair_src_begin: list[int] = []
        packed_repair_lengths: list[int] = []
        packed_replay_cu_seqlens = [0]
        packed_replay_output_row_ids: list[int] = []
        from_start_rows: list[int] = []
        from_start_req_slots: list[int] = []
        from_resident_rows: list[int] = []
        from_resident_source_blocks: list[int] = []

        for repair_idx, row_idx in enumerate(plan.repair_row_indices):
            mode = HybridSpecRepairMode(plan.repair_modes[repair_idx])
            if mode == HybridSpecRepairMode.NONE:
                continue
            target_slot = int(plan.repair_target_slots[repair_idx])
            resident_slot = int(plan.resident_slots[repair_idx])
            if mode == HybridSpecRepairMode.FROM_START:
                src_begin = 0
                from_start_rows.append(int(row_idx))
                from_start_req_slots.append(int(plan.repair_req_slots[repair_idx]))
            else:
                src_begin = resident_slot + 1
                from_resident_rows.append(int(row_idx))
                from_resident_source_blocks.append(int(plan.source_block_ids[row_idx]))
            replay_length = target_slot + 1 - src_begin
            if replay_length <= 0:
                continue
            packed_repair_req_slots.append(int(plan.repair_req_slots[repair_idx]))
            packed_repair_src_begin.append(src_begin)
            packed_repair_lengths.append(replay_length)
            packed_replay_cu_seqlens.append(
                packed_replay_cu_seqlens[-1] + replay_length
            )
            packed_replay_output_row_ids.append(int(row_idx) + 1)

        return HybridTemporalRuntimeMetadataBundle(
            shadow_req_slots_cpu=torch.tensor(
                plan.wave_plan.spec_req_slots,
                dtype=torch.long,
            ).contiguous(),
            resident_token_indices_cpu=torch.tensor(
                resident_token_indices,
                dtype=torch.int32,
            ).contiguous(),
            source_block_ids_cpu=torch.tensor(
                plan.source_block_ids,
                dtype=torch.long,
            ).contiguous(),
            repair_req_slots_cpu=torch.tensor(
                packed_repair_req_slots,
                dtype=torch.long,
            ).contiguous(),
            repair_src_begin_cpu=torch.tensor(
                packed_repair_src_begin,
                dtype=torch.long,
            ).contiguous(),
            repair_lengths_cpu=torch.tensor(
                packed_repair_lengths,
                dtype=torch.int32,
            ).contiguous(),
            replay_cu_seqlens_cpu=torch.tensor(
                packed_replay_cu_seqlens,
                dtype=torch.int32,
            ).contiguous(),
            replay_output_row_ids_cpu=torch.tensor(
                packed_replay_output_row_ids,
                dtype=torch.int32,
            ).contiguous(),
            from_start_rows_cpu=torch.tensor(
                from_start_rows,
                dtype=torch.long,
            ).contiguous(),
            from_start_req_slots_cpu=torch.tensor(
                from_start_req_slots,
                dtype=torch.long,
            ).contiguous(),
            from_resident_rows_cpu=torch.tensor(
                from_resident_rows,
                dtype=torch.long,
            ).contiguous(),
            from_resident_source_blocks_cpu=torch.tensor(
                from_resident_source_blocks,
                dtype=torch.long,
            ).contiguous(),
        )

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
        metadata = self._get_runtime_metadata(plan)
        self._wait_for_shadow_ready(ssm_state)
        use_cuda_timing = ssm_state.is_cuda
        copy_start = copy_end = compute_start = compute_end = None
        if use_cuda_timing:
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            copy_start.record(torch.cuda.current_stream(ssm_state.device))

        if int(metadata.from_start_rows_cpu.numel()) > 0:
            self._wait_for_segment_start_ready(ssm_state)
        prepared = self._materialize_capture_initial_state(
            ssm_state=ssm_state,
            plan=plan,
            metadata=metadata,
        )
        replay_batch = self._build_replay_batch(plan, metadata)
        if replay_batch.num_rows > 0:
            self._wait_for_shadow_ready(ssm_state)
        if replay_batch.num_rows == 0:
            if use_cuda_timing and copy_end is not None:
                copy_end.record(torch.cuda.current_stream(ssm_state.device))
                workspace.pending_timing_events.append(
                    ("repair_copy_ms", copy_start, copy_end, 0)
                )
                self._drain_timing_events()
            _trace_predict_last(
                f"prepare_done layer={self.layer_name} no_replay "
                f"first_scalar={float(prepared[0].reshape(-1)[0].item())}"
            )
            return prepared

        staged_tensors = self._stage_replay_batch_direct(
            metadata=metadata,
            replay_batch=replay_batch,
        )
        if use_cuda_timing and copy_end is not None and compute_start is not None:
            copy_end.record(torch.cuda.current_stream(ssm_state.device))
            workspace.pending_timing_events.append(
                ("repair_copy_ms", copy_start, copy_end, 0)
            )
            compute_start.record(torch.cuda.current_stream(ssm_state.device))
        self._run_replay_from_shadow(replay_batch, staged_tensors)
        if use_cuda_timing and compute_end is not None:
            compute_end.record(torch.cuda.current_stream(ssm_state.device))
            workspace.pending_timing_events.append(
                ("repair_compute_ms", compute_start, compute_end, 0)
            )
        for row_idx, running_block_id, end in zip(
            replay_batch.row_indices,
            replay_batch.running_block_ids,
            replay_batch.cu_seqlens[1:],
        ):
            _trace_predict_last(
                f"repair layer={self.layer_name} row={row_idx} running="
                f"{running_block_id} final_slot={end - 1}"
            )

        workspace.repair_row_count += replay_batch.num_rows
        workspace.repair_from_start_count += replay_batch.mode_counts[0]
        workspace.repair_from_resident_count += replay_batch.mode_counts[1]
        if use_cuda_timing:
            self._drain_timing_events()
        _trace_predict_last(
            f"prepare_done layer={self.layer_name} repaired_rows="
            f"{replay_batch.num_rows} "
            f"first_scalar={float(prepared[0].reshape(-1)[0].item())}"
        )
        return prepared

    def _materialize_capture_initial_state(
        self,
        *,
        ssm_state: torch.Tensor,
        plan: HybridTemporalGroupPlan,
        metadata: HybridTemporalRuntimeMetadataBundle,
    ) -> torch.Tensor:
        workspace = self.workspace
        assert workspace.initial_state_padded is not None
        assert workspace.capture_source_block_ids is not None
        assert workspace.replay_from_start_rows is not None
        assert workspace.replay_from_start_req_slots is not None

        num_rows = len(plan.wave_plan.req_ids)
        capture_state = workspace.initial_state_padded[: num_rows + 1]
        capture_state[0].zero_()
        state_out = capture_state[1:]
        source_block_ids = self._select_runtime_tensor(
            metadata,
            cpu_name="source_block_ids_cpu",
            gpu_name="source_block_ids_gpu",
            device=ssm_state.device,
        )
        if source_block_ids.device != ssm_state.device:
            capture_source_block_ids = workspace.capture_source_block_ids[:num_rows]
            capture_source_block_ids.copy_(
                source_block_ids,
                non_blocking=ssm_state.is_cuda,
            )
            source_block_ids = capture_source_block_ids
        torch.index_select(ssm_state, 0, source_block_ids, out=state_out)

        from_start_count = int(metadata.from_start_rows_cpu.numel())
        if from_start_count > 0:
            from_start_rows = self._select_runtime_tensor(
                metadata,
                cpu_name="from_start_rows_cpu",
                gpu_name="from_start_rows_gpu",
                device=ssm_state.device,
            )
            from_start_req_slots = self._select_runtime_tensor(
                metadata,
                cpu_name="from_start_req_slots_cpu",
                gpu_name="from_start_req_slots_gpu",
                device=ssm_state.device,
            )
            if from_start_rows.device != ssm_state.device:
                replay_from_start_rows = workspace.replay_from_start_rows[
                    :from_start_count
                ]
                replay_from_start_rows.copy_(
                    from_start_rows,
                    non_blocking=ssm_state.is_cuda,
                )
                from_start_rows = replay_from_start_rows
            if from_start_req_slots.device != ssm_state.device:
                replay_from_start_req_slots = workspace.replay_from_start_req_slots[
                    :from_start_count
                ]
                replay_from_start_req_slots.copy_(
                    from_start_req_slots,
                    non_blocking=ssm_state.is_cuda,
                )
                from_start_req_slots = replay_from_start_req_slots
            start_states = torch.index_select(
                workspace.segment_start_gpu_shadow,
                0,
                from_start_req_slots,
            )
            state_out.index_copy_(0, from_start_rows, start_states)

        return state_out

    def store_replay_artifacts(
        self,
        initial_state: torch.Tensor,
        running_state_indices: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        saved_g: torch.Tensor | None,
        saved_beta: torch.Tensor | None,
        final_states: torch.Tensor | None,
        wave_plan: HybridTemporalWavePlan,
    ) -> None:
        if not wave_plan.req_ids:
            return

        workspace = self.workspace
        ssm_state = self._ssm_state_getter()
        running_state_indices_list = [int(idx) for idx in running_state_indices.tolist()]
        use_cuda_timing = ssm_state.is_cuda
        if key is not None and value is not None and saved_g is not None and saved_beta is not None:
            key = key.squeeze(0)
            value = value.squeeze(0)
            self._release_completed_spill_refs()
            tape_start = tape_end = None
            if use_cuda_timing:
                tape_start = torch.cuda.Event(enable_timing=True)
                tape_end = torch.cuda.Event(enable_timing=True)
            del final_states

            if use_cuda_timing and workspace.spill_stream is not None:
                verify_done = torch.cuda.Event()
                verify_done.record(torch.cuda.current_stream(ssm_state.device))
                workspace.spill_stream.wait_event(verify_done)
                with torch.cuda.stream(workspace.spill_stream):
                    if tape_start is not None:
                        tape_start.record(workspace.spill_stream)
                    self._spill_replay_wave(
                        initial_state=initial_state,
                        key=key,
                        value=value,
                        saved_g=saved_g,
                        saved_beta=saved_beta,
                        wave_plan=wave_plan,
                    )
                    if tape_end is not None:
                        tape_end.record(workspace.spill_stream)
                    spill_done = torch.cuda.Event()
                    spill_done.record(workspace.spill_stream)
                workspace.last_spill_done = spill_done
                workspace.last_segment_start_ready = spill_done
                self._record_pending_spill(
                    spill_done,
                    initial_state,
                    key,
                    value,
                    saved_g,
                    saved_beta,
                )
            else:
                self._spill_replay_wave(
                    initial_state=initial_state,
                    key=key,
                    value=value,
                    saved_g=saved_g,
                    saved_beta=saved_beta,
                    wave_plan=wave_plan,
                )
                workspace.last_spill_done = None
                workspace.last_segment_start_ready = None
                workspace.pending_spill_refs.clear()
            if use_cuda_timing and tape_start is not None and tape_end is not None:
                workspace.pending_timing_events.append(
                    ("tape_save_ms", tape_start, tape_end, 0)
                )
                self._drain_timing_events()
        else:
            assert workspace.spill_req_slots is not None
            req_slots = workspace.spill_req_slots[: len(wave_plan.spec_req_slots)]
            metadata = (
                self._get_runtime_metadata(self.group_plan)
                if self.group_plan is not None
                and self.group_plan.wave_plan.req_ids == wave_plan.req_ids
                else None
            )
            if metadata is not None:
                metadata_req_slots = self._select_runtime_tensor(
                    metadata,
                    cpu_name="shadow_req_slots_cpu",
                    gpu_name="shadow_req_slots_gpu",
                    device=ssm_state.device,
                )
                if metadata_req_slots.device == ssm_state.device:
                    req_slots = metadata_req_slots
                else:
                    req_slots.copy_(
                        metadata_req_slots,
                        non_blocking=ssm_state.is_cuda,
                    )
            else:
                for idx, req_slot in enumerate(wave_plan.spec_req_slots):
                    req_slots[idx] = int(req_slot)
            segment_start = segment_end = None
            if use_cuda_timing and workspace.spill_stream is not None:
                segment_start = torch.cuda.Event(enable_timing=True)
                segment_end = torch.cuda.Event(enable_timing=True)
                verify_done = torch.cuda.Event()
                verify_done.record(torch.cuda.current_stream(ssm_state.device))
                workspace.spill_stream.wait_event(verify_done)
                with torch.cuda.stream(workspace.spill_stream):
                    segment_start.record(workspace.spill_stream)
                    workspace.segment_start_gpu_shadow.index_copy_(
                        0, req_slots, initial_state
                    )
                    segment_end.record(workspace.spill_stream)
                    segment_ready = torch.cuda.Event()
                    segment_ready.record(workspace.spill_stream)
                workspace.last_segment_start_ready = segment_ready
                workspace.last_spill_done = None
                self._record_pending_spill(segment_ready, initial_state)
                workspace.pending_timing_events.append(
                    ("segment_start_save_ms", segment_start, segment_end, 0)
                )
                self._drain_timing_events()
            else:
                workspace.segment_start_gpu_shadow.index_copy_(0, req_slots, initial_state)
                workspace.last_segment_start_ready = None
                workspace.last_spill_done = None

        for row, req_slot in enumerate(wave_plan.spec_req_slots):
            token_count = (
                wave_plan.spec_query_start_locs[row + 1]
                - wave_plan.spec_query_start_locs[row]
            )
            workspace.saved_generation_per_req[int(req_slot)] = int(
                wave_plan.next_replay_generations[row]
            )
            _trace_predict_last(
                f"store layer={self.layer_name} row={row} running="
                f"{running_state_indices_list[row]} predicted_len="
                f"{int(wave_plan.predicted_accept_lens[row])} token_count="
                f"{token_count} replay_gen="
                f"{int(wave_plan.next_replay_generations[row])}"
            )

    def reset_repair_timing_stats(self) -> None:
        self._drain_timing_events(synchronize=True)
        workspace = self.workspace
        workspace.pending_timing_events.clear()
        workspace.repair_copy_ms = 0.0
        workspace.repair_compute_ms = 0.0
        workspace.repair_row_count = 0
        workspace.repair_from_start_count = 0
        workspace.repair_from_resident_count = 0
        workspace.verify_attention_ms = 0.0
        workspace.layer_total_ms = 0.0
        workspace.verify_call_count = 0
        workspace.checkpoint_save_ms = 0.0
        workspace.post_replay_state_gather_ms = 0.0
        workspace.capture_materialize_ms = 0.0
        workspace.segment_start_save_ms = 0.0
        workspace.segment_start_wait_ms = 0.0
        workspace.tape_save_ms = 0.0
        workspace.last_spill_done = None
        workspace.last_segment_start_ready = None
        workspace.pending_spill_refs.clear()

    def snapshot_repair_timing_stats(self) -> dict[str, float | int]:
        self._drain_timing_events(synchronize=True)
        workspace = self.workspace
        return {
            "repair_copy_ms": float(workspace.repair_copy_ms),
            "prepare_copy_ms": float(workspace.repair_copy_ms),
            "repair_compute_ms": float(workspace.repair_compute_ms),
            "repair_row_count": int(workspace.repair_row_count),
            "repair_from_start_count": int(workspace.repair_from_start_count),
            "repair_from_resident_count": int(
                workspace.repair_from_resident_count
            ),
            "verify_attention_ms": float(workspace.verify_attention_ms),
            "layer_total_ms": float(workspace.layer_total_ms),
            "verify_call_count": int(workspace.verify_call_count),
            "checkpoint_save_ms": float(workspace.checkpoint_save_ms),
            "post_replay_state_gather_ms": float(
                workspace.post_replay_state_gather_ms
            ),
            "capture_materialize_ms": float(
                workspace.capture_materialize_ms
            ),
            "segment_start_save_ms": float(workspace.segment_start_save_ms),
            "segment_start_wait_ms": float(workspace.segment_start_wait_ms),
            "tape_save_ms": float(workspace.tape_save_ms),
            "spill_copy_ms": float(workspace.tape_save_ms),
        }

    def _relocate_predict_checkpoints(
        self,
        ssm_state: torch.Tensor,
        plan: HybridTemporalGroupPlan,
    ) -> None:
        pairs = [
            (int(source_block_id), int(running_block_id))
            for source_block_id, running_block_id in zip(
                plan.source_block_ids,
                plan.running_block_ids,
            )
            if source_block_id != running_block_id
        ]
        if not pairs:
            return
        workspace = self.workspace
        assert workspace.checkpoint_src_blocks is not None
        assert workspace.checkpoint_dst_blocks is not None
        src_blocks = workspace.checkpoint_src_blocks[: len(pairs)]
        dst_blocks = workspace.checkpoint_dst_blocks[: len(pairs)]
        for idx, (src, dst) in enumerate(pairs):
            src_blocks[idx] = src
            dst_blocks[idx] = dst
        relocated = torch.index_select(ssm_state, 0, src_blocks)
        ssm_state.index_copy_(0, dst_blocks, relocated)

    def _build_replay_batch(
        self,
        plan: HybridTemporalGroupPlan,
        metadata: HybridTemporalRuntimeMetadataBundle,
    ) -> ReplayBatch:
        workspace = self.workspace
        repair_plan_indices: list[int] = []
        row_indices: list[int] = []
        running_block_ids: list[int] = []
        req_slots: list[int] = []
        cu_seqlens = metadata.replay_cu_seqlens_cpu.tolist()
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

            if mode == HybridSpecRepairMode.FROM_START:
                from_start += 1
            elif mode == HybridSpecRepairMode.FROM_RESIDENT:
                from_resident += 1
            else:
                continue
            if len(repair_plan_indices) >= int(metadata.repair_lengths_cpu.numel()):
                continue
            if int(metadata.repair_lengths_cpu[len(repair_plan_indices)].item()) <= 0:
                continue
            repair_plan_indices.append(int(idx))
            row_indices.append(int(row_idx))
            running_block_ids.append(int(plan.running_block_ids[row_idx]))
            req_slots.append(req_slot)

        return ReplayBatch(
            repair_plan_indices=repair_plan_indices,
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
        assert workspace.replay_req_slots is not None
        assert workspace.replay_src_begin is not None
        assert workspace.replay_src_end is not None
        assert workspace.replay_src_linear_indices is not None
        assert workspace.token_offsets is not None

        from_start_rows: list[int] = []
        from_start_req_slots: list[int] = []
        from_resident_rows: list[int] = []
        from_resident_running_blocks: list[int] = []
        cursor = 0
        tokens_per_req = int(workspace.key_tape_gpu_shadow.shape[1])

        for packed_row, (row_idx, repair_idx) in enumerate(
            zip(replay_batch.row_indices, replay_batch.repair_plan_indices)
        ):
            mode = HybridSpecRepairMode(plan.repair_modes[repair_idx])
            req_slot = int(plan.repair_req_slots[repair_idx])
            target_slot = int(plan.repair_target_slots[repair_idx])
            resident_slot = int(plan.resident_slots[repair_idx])
            if mode == HybridSpecRepairMode.FROM_START:
                src_begin = 0
                from_start_rows.append(packed_row)
                from_start_req_slots.append(req_slot)
            else:
                src_begin = resident_slot + 1
                from_resident_rows.append(packed_row)
                from_resident_running_blocks.append(
                    int(plan.running_block_ids[row_idx])
                )
            src_end = target_slot + 1
            replay_tokens = src_end - src_begin
            workspace.replay_req_slots[packed_row] = req_slot
            workspace.replay_src_begin[packed_row] = src_begin
            workspace.replay_src_end[packed_row] = src_end
            linear_slice = workspace.replay_src_linear_indices[
                cursor : cursor + replay_tokens
            ]
            linear_slice.copy_(workspace.token_offsets[:replay_tokens])
            linear_slice.add_(req_slot * tokens_per_req + src_begin)
            _trace_predict_last(
                f"stage layer={self.layer_name} row={row_idx} req_slot={req_slot} "
                f"mode={int(mode)} tape=[{src_begin},{src_end}) row_start="
                f"{int(plan.wave_plan.spec_query_start_locs[row_idx])}"
            )
            cursor += replay_tokens

        if from_start_req_slots:
            start_req_slots = torch.tensor(
                from_start_req_slots,
                dtype=torch.long,
                device=workspace.segment_start_gpu_shadow.device,
            )
            start_states = torch.index_select(
                workspace.segment_start_gpu_shadow,
                0,
                start_req_slots,
            )
            initial_state_buf.index_copy_(
                0,
                torch.tensor(
                    from_start_rows,
                    dtype=torch.long,
                    device=initial_state_buf.device,
                ),
                start_states,
            )
        if from_resident_running_blocks:
            resident_running_blocks = torch.tensor(
                from_resident_running_blocks,
                dtype=torch.long,
                device=ssm_state.device,
            )
            resident_states = torch.index_select(
                ssm_state,
                0,
                resident_running_blocks,
            )
            initial_state_buf.index_copy_(
                0,
                torch.tensor(
                    from_resident_rows,
                    dtype=torch.long,
                    device=initial_state_buf.device,
                ),
                resident_states,
            )

        src_linear_indices = workspace.replay_src_linear_indices[:cursor]
        flat_key_shadow = workspace.key_tape_gpu_shadow.view(
            -1, *workspace.key_tape_gpu_shadow.shape[2:]
        )
        flat_value_shadow = workspace.value_tape_gpu_shadow.view(
            -1, *workspace.value_tape_gpu_shadow.shape[2:]
        )
        flat_g_shadow = workspace.g_tape_gpu_shadow.view(
            -1, *workspace.g_tape_gpu_shadow.shape[2:]
        )
        flat_beta_shadow = workspace.beta_tape_gpu_shadow.view(
            -1, *workspace.beta_tape_gpu_shadow.shape[2:]
        )
        torch.index_select(flat_key_shadow, 0, src_linear_indices, out=key_buf)
        torch.index_select(flat_value_shadow, 0, src_linear_indices, out=value_buf)
        torch.index_select(flat_g_shadow, 0, src_linear_indices, out=g_buf)
        torch.index_select(flat_beta_shadow, 0, src_linear_indices, out=beta_buf)

    def _stage_replay_batch_direct(
        self,
        metadata: HybridTemporalRuntimeMetadataBundle,
        replay_batch: ReplayBatch,
    ) -> StagedReplayBatchTensors:
        workspace = self.workspace
        assert workspace.replay_req_slots is not None
        assert workspace.replay_src_begin is not None
        assert workspace.replay_lengths is not None
        assert workspace.replay_output_row_ids is not None
        assert workspace.replay_cu_seqlens is not None
        assert workspace.resident_token_indices is not None

        def _get_direct_tensor(
            *,
            cpu_name: str,
            gpu_name: str,
            workspace_tensor: torch.Tensor,
            length: int,
        ) -> torch.Tensor:
            selected = self._select_runtime_tensor(
                metadata,
                cpu_name=cpu_name,
                gpu_name=gpu_name,
                device=workspace_tensor.device,
            )
            if selected.device == workspace_tensor.device:
                return selected[:length]
            staged = workspace_tensor[:length]
            staged.copy_(selected, non_blocking=staged.is_cuda)
            return staged

        replay_req_slots = _get_direct_tensor(
            cpu_name="repair_req_slots_cpu",
            gpu_name="repair_req_slots_gpu",
            workspace_tensor=workspace.replay_req_slots,
            length=replay_batch.num_rows,
        )
        replay_src_begin = _get_direct_tensor(
            cpu_name="repair_src_begin_cpu",
            gpu_name="repair_src_begin_gpu",
            workspace_tensor=workspace.replay_src_begin,
            length=replay_batch.num_rows,
        )
        replay_lengths = _get_direct_tensor(
            cpu_name="repair_lengths_cpu",
            gpu_name="repair_lengths_gpu",
            workspace_tensor=workspace.replay_lengths,
            length=replay_batch.num_rows,
        )
        replay_cu_seqlens = _get_direct_tensor(
            cpu_name="replay_cu_seqlens_cpu",
            gpu_name="replay_cu_seqlens_gpu",
            workspace_tensor=workspace.replay_cu_seqlens,
            length=replay_batch.num_rows + 1,
        )
        replay_output_row_ids = _get_direct_tensor(
            cpu_name="replay_output_row_ids_cpu",
            gpu_name="replay_output_row_ids_gpu",
            workspace_tensor=workspace.replay_output_row_ids,
            length=replay_batch.num_rows,
        )
        resident_token_indices = workspace.resident_token_indices[
            : replay_batch.num_rows
        ]
        resident_token_indices.copy_(replay_lengths)
        resident_token_indices.sub_(1)

        for row_idx, req_slot, src_begin, replay_len in zip(
            replay_batch.row_indices,
            replay_batch.req_slots,
            metadata.repair_src_begin_cpu.tolist(),
            metadata.repair_lengths_cpu.tolist(),
        ):
            _trace_predict_last(
                f"stage_direct layer={self.layer_name} row={row_idx} "
                f"req_slot={req_slot} tape=[{src_begin},{src_begin + replay_len})"
            )

        return StagedReplayBatchTensors(
            replay_req_slots=replay_req_slots,
            replay_src_begin=replay_src_begin,
            replay_lengths=replay_lengths,
            replay_cu_seqlens=replay_cu_seqlens,
            replay_output_row_ids=replay_output_row_ids,
            initial_state_row_ids=replay_output_row_ids,
            resident_token_indices=resident_token_indices,
        )

    def _run_replay_from_shadow(
        self,
        replay_batch: ReplayBatch,
        staged_tensors: StagedReplayBatchTensors,
    ) -> None:
        workspace = self.workspace
        assert workspace.initial_state_padded is not None

        fused_sigmoid_gating_delta_rule_replay_from_shadow_resident(
            shadow_key=workspace.key_tape_gpu_shadow,
            shadow_value=workspace.value_tape_gpu_shadow,
            shadow_g=workspace.g_tape_gpu_shadow,
            shadow_beta=workspace.beta_tape_gpu_shadow,
            shadow_req_slots=staged_tensors.replay_req_slots,
            shadow_src_begin=staged_tensors.replay_src_begin,
            shadow_max_seq_len=int(workspace.key_tape_gpu_shadow.shape[1]),
            initial_state=workspace.initial_state_padded[
                : replay_batch.num_rows + 1
            ],
            cu_seqlens=staged_tensors.replay_cu_seqlens,
            ssm_state_indices=staged_tensors.initial_state_row_ids,
            resident_final_state_out=workspace.initial_state_padded,
            resident_state_indices=staged_tensors.replay_output_row_ids,
            resident_token_indices=staged_tensors.resident_token_indices,
            use_qk_l2norm_in_kernel=self._use_qk_l2norm_in_kernel,
        )

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
        workspace = self.workspace
        assert workspace.initial_state_padded is not None
        assert workspace.replay_cu_seqlens is not None
        assert workspace.state_row_ids is not None

        initial_state_padded = workspace.initial_state_padded[
            : replay_batch.num_rows + 1
        ]
        initial_state_padded[0].zero_()
        initial_state_padded[1:].copy_(initial_state_buf)
        cu_seqlens = workspace.replay_cu_seqlens[: replay_batch.num_rows + 1]
        for idx, cu_seq in enumerate(replay_batch.cu_seqlens):
            cu_seqlens[idx] = cu_seq
        fused_sigmoid_gating_delta_rule_replay_from_tape(
            k=key_buf.unsqueeze(0),
            v=value_buf.unsqueeze(0),
            g=g_buf,
            beta=beta_buf,
            initial_state=initial_state_padded,
            final_state_out=final_state_buf,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=workspace.state_row_ids[1 : replay_batch.num_rows + 1],
            use_qk_l2norm_in_kernel=self._use_qk_l2norm_in_kernel,
        )

    def _wait_for_shadow_ready(self, ssm_state: torch.Tensor) -> None:
        self._release_completed_spill_refs()
        event = self.workspace.last_spill_done
        if event is None or not ssm_state.is_cuda:
            return
        torch.cuda.current_stream(ssm_state.device).wait_event(event)

    def _wait_for_segment_start_ready(self, ssm_state: torch.Tensor) -> None:
        self._release_completed_spill_refs()
        event = self.workspace.last_segment_start_ready
        if event is None or not ssm_state.is_cuda:
            return
        wait_start = torch.cuda.Event(enable_timing=True)
        wait_end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(ssm_state.device)
        wait_start.record(current_stream)
        current_stream.wait_event(event)
        wait_end.record(current_stream)
        self.workspace.pending_timing_events.append(
            ("segment_start_wait_ms", wait_start, wait_end, 0)
        )

    def _release_completed_spill_refs(self) -> None:
        workspace = self.workspace
        if not workspace.pending_spill_refs:
            return
        remaining: list[tuple[torch.cuda.Event, tuple[torch.Tensor, ...]]] = []
        for event, refs in workspace.pending_spill_refs:
            if not event.query():
                remaining.append((event, refs))
        workspace.pending_spill_refs = remaining
        if not remaining and workspace.last_spill_done is not None:
            if workspace.last_spill_done.query():
                workspace.last_spill_done = None
        if not remaining and workspace.last_segment_start_ready is not None:
            if workspace.last_segment_start_ready.query():
                workspace.last_segment_start_ready = None

    def _record_pending_spill(
        self,
        event: torch.cuda.Event,
        *refs: torch.Tensor,
    ) -> None:
        workspace = self.workspace
        stream = workspace.spill_stream
        if stream is None:
            return
        kept_refs = tuple(refs)
        for ref in kept_refs:
            ref.record_stream(stream)
        workspace.pending_spill_refs.append((event, kept_refs))

    def _spill_replay_wave(
        self,
        *,
        initial_state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        saved_g: torch.Tensor,
        saved_beta: torch.Tensor,
        wave_plan: HybridTemporalWavePlan,
    ) -> None:
        workspace = self.workspace
        num_rows = len(wave_plan.req_ids)
        if num_rows == 0:
            return

        assert workspace.spill_req_slots is not None
        assert workspace.spill_row_starts is not None
        assert workspace.spill_dst_linear_indices is not None
        assert workspace.token_offsets is not None

        req_slots = workspace.spill_req_slots[:num_rows]
        for row, req_slot in enumerate(wave_plan.spec_req_slots):
            req_slots[row] = int(req_slot)
        workspace.segment_start_gpu_shadow.index_copy_(0, req_slots, initial_state)

        row_starts = workspace.spill_row_starts[: num_rows + 1]
        for idx, row_start in enumerate(wave_plan.spec_query_start_locs):
            row_starts[idx] = int(row_start)

        total_tokens = int(wave_plan.spec_query_start_locs[-1])
        dst_linear_indices = workspace.spill_dst_linear_indices[:total_tokens]
        cursor = 0
        tokens_per_req = int(workspace.key_tape_gpu_shadow.shape[1])
        for row, (start, end) in enumerate(
            zip(
                wave_plan.spec_query_start_locs,
                wave_plan.spec_query_start_locs[1:],
            )
        ):
            del start
            token_count = int(end - wave_plan.spec_query_start_locs[row])
            linear_slice = dst_linear_indices[cursor : cursor + token_count]
            linear_slice.copy_(workspace.token_offsets[:token_count])
            linear_slice.add_(int(wave_plan.spec_req_slots[row]) * tokens_per_req)
            cursor += token_count

        flat_key_shadow = workspace.key_tape_gpu_shadow.view(
            -1, *workspace.key_tape_gpu_shadow.shape[2:]
        )
        flat_value_shadow = workspace.value_tape_gpu_shadow.view(
            -1, *workspace.value_tape_gpu_shadow.shape[2:]
        )
        flat_g_shadow = workspace.g_tape_gpu_shadow.view(
            -1, *workspace.g_tape_gpu_shadow.shape[2:]
        )
        flat_beta_shadow = workspace.beta_tape_gpu_shadow.view(
            -1, *workspace.beta_tape_gpu_shadow.shape[2:]
        )
        flat_key_shadow.index_copy_(0, dst_linear_indices, key[:total_tokens])
        flat_value_shadow.index_copy_(0, dst_linear_indices, value[:total_tokens])
        flat_g_shadow.index_copy_(0, dst_linear_indices, saved_g[:total_tokens])
        flat_beta_shadow.index_copy_(0, dst_linear_indices, saved_beta[:total_tokens])

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
