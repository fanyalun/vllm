# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import IntEnum

import torch


class HybridSpecRepairMode(IntEnum):
    NONE = 0
    FROM_START = 1
    FROM_RESIDENT = 2


@dataclass(frozen=True)
class HybridTemporalWavePlan:
    req_ids: list[str]
    spec_req_slots: list[int]
    spec_query_start_locs: list[int]
    predicted_accept_lens: list[int]
    next_replay_generations: list[int]

    def __post_init__(self) -> None:
        num_rows = len(self.req_ids)
        if len(self.spec_req_slots) != num_rows:
            raise ValueError("spec_req_slots must align with req_ids")
        if len(self.predicted_accept_lens) != num_rows:
            raise ValueError("predicted_accept_lens must align with req_ids")
        if len(self.next_replay_generations) != num_rows:
            raise ValueError(
                "next_replay_generations must align with req_ids"
            )
        if len(self.spec_query_start_locs) != num_rows + 1:
            raise ValueError(
                "spec_query_start_locs must contain one boundary per row"
            )


@dataclass(frozen=True)
class HybridTemporalRuntimeMetadataBundle:
    shadow_req_slots_cpu: torch.Tensor
    resident_token_indices_cpu: torch.Tensor
    source_block_ids_cpu: torch.Tensor
    repair_req_slots_cpu: torch.Tensor
    repair_src_begin_cpu: torch.Tensor
    repair_lengths_cpu: torch.Tensor
    replay_cu_seqlens_cpu: torch.Tensor
    replay_output_row_ids_cpu: torch.Tensor
    from_start_rows_cpu: torch.Tensor
    from_start_req_slots_cpu: torch.Tensor
    from_resident_rows_cpu: torch.Tensor
    from_resident_source_blocks_cpu: torch.Tensor
    shadow_req_slots_gpu: torch.Tensor | None = None
    resident_token_indices_gpu: torch.Tensor | None = None
    source_block_ids_gpu: torch.Tensor | None = None
    repair_req_slots_gpu: torch.Tensor | None = None
    repair_src_begin_gpu: torch.Tensor | None = None
    repair_lengths_gpu: torch.Tensor | None = None
    replay_cu_seqlens_gpu: torch.Tensor | None = None
    replay_output_row_ids_gpu: torch.Tensor | None = None
    from_start_rows_gpu: torch.Tensor | None = None
    from_start_req_slots_gpu: torch.Tensor | None = None
    from_resident_rows_gpu: torch.Tensor | None = None
    from_resident_source_blocks_gpu: torch.Tensor | None = None


@dataclass(frozen=True)
class HybridTemporalGroupPlan:
    wave_plan: HybridTemporalWavePlan
    running_block_ids: list[int]
    source_block_ids: list[int]
    repair_row_indices: list[int]
    repair_req_slots: list[int]
    repair_target_slots: list[int]
    resident_slots: list[int]
    repair_modes: list[HybridSpecRepairMode]
    repair_generations: list[int]
    runtime_metadata: HybridTemporalRuntimeMetadataBundle | None = None

    def __post_init__(self) -> None:
        num_wave_rows = len(self.wave_plan.req_ids)
        if len(self.running_block_ids) != num_wave_rows:
            raise ValueError("running_block_ids must align with wave_plan.req_ids")
        if len(self.source_block_ids) != num_wave_rows:
            raise ValueError("source_block_ids must align with wave_plan.req_ids")

        num_repair_rows = len(self.repair_row_indices)
        if len(self.repair_req_slots) != num_repair_rows:
            raise ValueError("repair_req_slots must align with repair_row_indices")
        if len(self.repair_target_slots) != num_repair_rows:
            raise ValueError(
                "repair_target_slots must align with repair_row_indices"
            )
        if len(self.resident_slots) != num_repair_rows:
            raise ValueError("resident_slots must align with repair_row_indices")
        if len(self.repair_modes) != num_repair_rows:
            raise ValueError("repair_modes must align with repair_row_indices")
        if len(self.repair_generations) != num_repair_rows:
            raise ValueError(
                "repair_generations must align with repair_row_indices"
            )
        if any(
            row_idx < 0 or row_idx >= num_wave_rows
            for row_idx in self.repair_row_indices
        ):
            raise ValueError("repair_row_indices must point into wave_plan.req_ids")
