# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.v1.worker.gpu.spec_decode.async_draft.ipc import ASYNC_DRAFT_FAN_OUT

AsyncDraftCacheKey = tuple[str, str, int, int, int]


@dataclass
class CachedBranch:
    branch_id: str
    tokens: torch.Tensor
    provisional_state: object | None = None
    owned_resource_ids: tuple[str, ...] = ()
    completion_event: torch.cuda.Event | None = None

    @property
    def feedback_hidden_states(self) -> torch.Tensor:
        """Compatibility view for the phase-A EAGLE3 branch payload."""
        if not isinstance(self.provisional_state, torch.Tensor):
            raise TypeError("Cached branch does not contain EAGLE feedback state")
        return self.provisional_state

    def resource_ids(self) -> list[str]:
        return [self.branch_id, *self.owned_resource_ids]


class BranchCache:
    def __init__(self) -> None:
        self.entries: dict[AsyncDraftCacheKey, CachedBranch] = {}

    def pop(self, key: AsyncDraftCacheKey) -> CachedBranch | None:
        return self.entries.pop(key, None)

    def add(self, key: AsyncDraftCacheKey, branch: CachedBranch) -> None:
        if key in self.entries:
            raise ValueError(f"Duplicate async draft cache key: {key!r}")
        self.entries[key] = branch

    def discard_request(self, request_id: str) -> list[str]:
        discarded: list[str] = []
        for key, branch in list(self.entries.items()):
            if key[1] != request_id:
                continue
            del self.entries[key]
            discarded.extend(branch.resource_ids())
        return discarded

    def discard_requests(self, request_ids: list[str]) -> list[str]:
        discarded: list[str] = []
        for request_id in request_ids:
            discarded.extend(self.discard_request(request_id))
        return discarded

    def discard_all(self) -> list[str]:
        discarded = [
            resource_id
            for branch in self.entries.values()
            for resource_id in branch.resource_ids()
        ]
        self.entries.clear()
        return discarded


def select_branches_within_budget(
    required_seq_lens: list[int],
    request_indices: list[int],
    accepted_counts: list[int],
    candidate_indices: list[int],
    *,
    block_size: int,
    available_slots: int,
    available_blocks: int,
    shared_prefix_blocks: list[int] | None = None,
) -> list[int]:
    if block_size <= 0:
        raise ValueError(f"Async draft block size must be positive: {block_size}")
    if available_slots < 0 or available_blocks < 0:
        raise ValueError("Async draft branch budgets must be non-negative")
    if not (
        len(required_seq_lens)
        == len(request_indices)
        == len(accepted_counts)
        == len(candidate_indices)
    ):
        raise ValueError("Async draft branch budget inputs must have equal length")
    if shared_prefix_blocks is None:
        shared_prefix_blocks = [0] * len(required_seq_lens)
    if len(shared_prefix_blocks) != len(required_seq_lens):
        raise ValueError("Async draft shared-prefix counts must match branches")
    priority = sorted(
        range(len(required_seq_lens)),
        key=lambda index: (
            candidate_indices[index],
            accepted_counts[index],
            request_indices[index],
        ),
    )
    selected: list[int] = []
    for index in priority:
        blocks = (required_seq_lens[index] + block_size - 1) // block_size
        blocks -= shared_prefix_blocks[index]
        if blocks < 0:
            raise ValueError("Async draft shared prefix exceeds branch length")
        if available_slots < 1 or available_blocks < blocks:
            continue
        selected.append(index)
        available_slots -= 1
        available_blocks -= blocks
    selected.sort()
    return selected


def select_branches_with_group_budget(
    block_costs: list[list[int]],
    request_indices: list[int],
    accepted_counts: list[int],
    candidate_indices: list[int],
    *,
    available_slots: int,
    available_blocks: list[int],
) -> list[int]:
    """Select branches without oversubscribing any KV-cache group."""
    if not (
        len(block_costs)
        == len(request_indices)
        == len(accepted_counts)
        == len(candidate_indices)
    ):
        raise ValueError("Async draft branch budget inputs must have equal length")
    if available_slots < 0 or any(value < 0 for value in available_blocks):
        raise ValueError("Async draft branch budgets must be non-negative")
    if any(len(costs) != len(available_blocks) for costs in block_costs):
        raise ValueError("Async draft branch costs must cover every KV-cache group")
    if any(cost < 0 for costs in block_costs for cost in costs):
        raise ValueError("Async draft branch block costs must be non-negative")

    priority = sorted(
        range(len(block_costs)),
        key=lambda index: (
            candidate_indices[index],
            accepted_counts[index],
            request_indices[index],
        ),
    )
    selected: list[int] = []
    remaining = available_blocks.copy()
    for index in priority:
        costs = block_costs[index]
        if available_slots < 1 or any(
            cost > remaining[group] for group, cost in enumerate(costs)
        ):
            continue
        selected.append(index)
        available_slots -= 1
        for group, cost in enumerate(costs):
            remaining[group] -= cost
    selected.sort()
    return selected


def select_recovery_candidates(
    logits: torch.Tensor,
    returned_tokens: torch.Tensor,
    fan_out: int = ASYNC_DRAFT_FAN_OUT,
) -> torch.Tensor:
    """Select SSD recovery candidates for every possible accepted depth."""
    if logits.ndim != 3:
        raise ValueError(f"Expected [B, K+1, V] logits, got {logits.shape}")
    batch_size, num_positions, vocab_size = logits.shape
    num_speculative_tokens = returned_tokens.shape[1]
    if returned_tokens.shape != (batch_size, num_positions - 1):
        raise ValueError(
            "Returned token shape must match the first K glue positions: "
            f"logits={logits.shape}, returned_tokens={returned_tokens.shape}"
        )
    if fan_out < 1:
        raise ValueError(f"Async draft fan-out must be positive: {fan_out}")
    if vocab_size < fan_out + 1:
        raise ValueError(
            f"Vocabulary {vocab_size} is too small for fan-out {fan_out}"
        )

    candidate_logits = logits.clone()
    candidate_logits[:, :num_speculative_tokens].scatter_(
        dim=2,
        index=returned_tokens.unsqueeze(-1),
        value=float("-inf"),
    )
    return candidate_logits.topk(fan_out, dim=-1).indices
