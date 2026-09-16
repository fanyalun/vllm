# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack(IDS, S, E, N, T: tl.constexpr, BM: tl.constexpr, B: tl.constexpr):
    x = tl.arange(0, B)
    ids = tl.load(IDS + x, x < T, other=-1)
    valid = (x < T) & (ids >= 0)
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    invalid_rank = x - rank - 1
    offsets = x[:, None] * BM + tl.arange(0, BM)[None, :]
    tl.store(S + offsets, T, offsets < T * BM)
    tl.store(E + x, -1, x < T)
    tl.debug_barrier()
    destination = tl.where(valid, rank * BM, count * BM + invalid_rank)
    tl.store(S + destination, x, x < T)
    tl.store(E + rank, ids, valid)
    tl.store(N, (count + tl.cdiv(T - count, BM)) * BM)


def aligned_threshold_expert_assignment(ids, block_size, num_experts):
    """Include skipped slots in alignment so both GEMMs write their zeros."""
    from .moe_align_block_size import moe_align_block_size

    sorted_ids, experts, count = moe_align_block_size(
        ids + 1, block_size, num_experts + 1
    )
    return sorted_ids, experts - 1, count


def threshold_expert_assignment(ids, block_size, num_experts):
    """Pack active decode blocks and coalesce skipped slots into zero blocks."""
    size = ids.numel()
    if size > 256 or size * 4 > num_experts:
        return aligned_threshold_expert_assignment(ids, block_size, num_experts)
    sorted_ids = torch.empty(size * block_size, device=ids.device, dtype=torch.int32)
    experts = torch.empty(size, device=ids.device, dtype=torch.int32)
    count = torch.empty(1, device=ids.device, dtype=torch.int32)
    _pack[(1,)](
        ids,
        sorted_ids,
        experts,
        count,
        size,
        block_size,
        triton.next_power_of_2(size),
        num_warps=4,
    )
    return sorted_ids, experts, count
