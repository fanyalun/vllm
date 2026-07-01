# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import IntEnum


class HybridSpecReloadMode(IntEnum):
    NONE = 0
    CPU_SHADOW = 1
    PRELOADED = 2


@dataclass(frozen=True)
class HybridTemporalWavePlan:
    req_ids: list[str]
    spec_req_slots: list[int]
    spec_query_start_locs: list[int]
    predicted_accept_lens: list[int]
    next_shadow_generations: list[int]

    def __post_init__(self) -> None:
        num_rows = len(self.req_ids)
        if len(self.spec_req_slots) != num_rows:
            raise ValueError("spec_req_slots must align with req_ids")
        if len(self.predicted_accept_lens) != num_rows:
            raise ValueError("predicted_accept_lens must align with req_ids")
        if len(self.next_shadow_generations) != num_rows:
            raise ValueError(
                "next_shadow_generations must align with req_ids"
            )
        if len(self.spec_query_start_locs) != num_rows + 1:
            raise ValueError(
                "spec_query_start_locs must contain one boundary per row"
            )


@dataclass(frozen=True)
class HybridTemporalGroupPlan:
    wave_plan: HybridTemporalWavePlan
    reload_row_indices: list[int]
    reload_req_slots: list[int]
    reload_slots: list[int]
    reload_generations: list[int]
    running_block_ids: list[int]

    def __post_init__(self) -> None:
        num_rows = len(self.reload_row_indices)
        if len(self.reload_req_slots) != num_rows:
            raise ValueError("reload_req_slots must align with reload_row_indices")
        if len(self.reload_slots) != num_rows:
            raise ValueError("reload_slots must align with reload_row_indices")
        if len(self.reload_generations) != num_rows:
            raise ValueError(
                "reload_generations must align with reload_row_indices"
            )
        if len(self.running_block_ids) != num_rows:
            raise ValueError(
                "running_block_ids must align with reload_row_indices"
            )
