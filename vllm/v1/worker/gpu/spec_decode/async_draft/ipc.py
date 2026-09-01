# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

ASYNC_DRAFT_FAN_OUT = 3
ASYNC_DRAFT_RING_SLOTS = 2


@dataclass
class AsyncDraftRingSlot:
    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    aux_hidden_states: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    next_prefill_tokens: torch.Tensor
    temperature: torch.Tensor
    seeds: torch.Tensor
    draft_tokens: torch.Tensor


@dataclass
class AsyncDraftBatch:
    generation: int
    slot: int
    engine_instance_id: str
    req_ids: list[str]
    request_epochs: list[int]
    transient: bool
    overlap_budget_seconds: float
    num_reqs: int
    num_tokens: int
    num_tokens_after_padding: int
    num_reqs_after_padding: int
    num_scheduled_tokens: np.ndarray
    query_start_loc_np: np.ndarray
    seq_lens_cpu_upper_bound: np.ndarray
    num_computed_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    num_computed_prefill_tokens_np: np.ndarray
    is_prefilling_np: np.ndarray


@dataclass
class AsyncDraftResponse:
    generation: int
    slot: int
    status: str
    num_reqs: int = 0
    error: str | None = None
    metrics: dict[str, float | int] | None = None
    cache_hit_indices: list[int] | None = None
    trace_top2: list[dict[str, Any] | None] | None = None


def make_ring_slots(
    *,
    max_num_reqs: int,
    max_num_tokens: int,
    num_speculative_tokens: int,
    aux_hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[AsyncDraftRingSlot]:
    slots: list[AsyncDraftRingSlot] = []
    for _ in range(ASYNC_DRAFT_RING_SLOTS):
        slots.append(
            AsyncDraftRingSlot(
                input_ids=torch.empty(max_num_tokens, dtype=torch.int32, device=device),
                positions=torch.empty(max_num_tokens, dtype=torch.int64, device=device),
                query_start_loc=torch.empty(
                    max_num_reqs + 1, dtype=torch.int32, device=device
                ),
                seq_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
                aux_hidden_states=torch.empty(
                    max_num_tokens,
                    aux_hidden_size,
                    dtype=dtype,
                    device=device,
                ),
                num_sampled=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
                num_rejected=torch.empty(
                    max_num_reqs, dtype=torch.int32, device=device
                ),
                last_sampled=torch.empty(
                    max_num_reqs, dtype=torch.int64, device=device
                ),
                next_prefill_tokens=torch.empty(
                    max_num_reqs, dtype=torch.int32, device=device
                ),
                temperature=torch.empty(
                    max_num_reqs, dtype=torch.float32, device=device
                ),
                seeds=torch.empty(max_num_reqs, dtype=torch.int64, device=device),
                draft_tokens=torch.empty(
                    max_num_reqs,
                    num_speculative_tokens,
                    dtype=torch.int64,
                    device=device,
                ),
            )
        )
    return slots


def response_error(generation: int, slot: int, error: BaseException) -> dict[str, Any]:
    return AsyncDraftResponse(
        generation=generation,
        slot=slot,
        status="error",
        error=f"{type(error).__name__}: {error}",
    ).__dict__
