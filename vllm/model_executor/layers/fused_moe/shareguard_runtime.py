# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional ShareGuard branch dropping for expert-parallel MoE layers."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None and value.strip() else default


@dataclass
class ShareGuardStats:
    calls: int = 0
    overload_calls: int = 0
    branches_seen: int = 0
    branches_dropped: int = 0
    select_ms: float = 0.0
    max_load_before: list[float] = field(default_factory=list)
    max_load_after: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        calls = max(self.calls, 1)
        return {
            "calls": self.calls,
            "overload_calls": self.overload_calls,
            "branches_seen": self.branches_seen,
            "branches_dropped": self.branches_dropped,
            "drop_rate": (
                self.branches_dropped / self.branches_seen
                if self.branches_seen
                else 0.0
            ),
            "select_ms_total": self.select_ms,
            "select_ms_avg": self.select_ms / calls,
            "max_load_before_mean": (
                sum(self.max_load_before) / len(self.max_load_before)
                if self.max_load_before
                else 0.0
            ),
            "max_load_after_mean": (
                sum(self.max_load_after) / len(self.max_load_after)
                if self.max_load_after
                else 0.0
            ),
        }


@dataclass
class ShareGuardConfig:
    enabled: bool = False
    mode: str = "shareguard"
    capacity: float = 0.85
    rho_path: str = ""
    compensate: bool = True
    layer_index: dict[int, int] = field(default_factory=dict)
    rho: torch.Tensor | None = None
    eps: torch.Tensor | None = None
    device_tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    stats: ShareGuardStats = field(default_factory=ShareGuardStats)
    last_comp_coeff: torch.Tensor | None = None
    next_layer_idx: int = 0

    def load_tables(self) -> None:
        if not self.rho_path:
            raise ValueError(
                "SHAREGUARD_RHO_PATH is required when ShareGuard is enabled"
            )
        path = Path(self.rho_path)
        if not path.is_file():
            raise FileNotFoundError(f"SHAREGUARD_RHO_PATH not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        rho = data["rho_by_expert"].float()
        eps = data["eps_by_expert"].float()
        if rho.ndim != 2 or eps.ndim != 2 or rho.shape != eps.shape:
            raise ValueError(
                "ShareGuard rho/eps tables must have matching [layers, experts] "
                f"shapes, got rho={tuple(rho.shape)} eps={tuple(eps.shape)}"
            )
        self.rho = rho.nan_to_num(0.0)
        finite_eps = eps[torch.isfinite(eps)]
        mean_eps = float(finite_eps.mean()) if finite_eps.numel() else 1.0
        self.eps = eps.nan_to_num(mean_eps)
        self.device_tables.clear()
        print(
            f"[ShareGuard] loaded rho/eps from {path} "
            f"shape={tuple(self.rho.shape)} mode={self.mode} "
            f"capacity={self.capacity}",
            flush=True,
        )

    def table_rows(
        self, layer_idx: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self.device_tables:
            assert self.rho is not None and self.eps is not None
            self.device_tables[key] = (
                self.rho.to(device=device, dtype=torch.float32),
                self.eps.to(device=device, dtype=torch.float32),
            )
        rho, eps = self.device_tables[key]
        return rho[layer_idx], eps[layer_idx]


_STATE = ShareGuardConfig()
_READY = False


def get_shareguard_state() -> ShareGuardConfig:
    return _STATE


def init_shareguard_from_env() -> ShareGuardConfig:
    global _READY, _STATE
    state = ShareGuardConfig(
        enabled=_env_bool("SHAREGUARD_ENABLE", False),
        mode=os.environ.get("SHAREGUARD_MODE", "shareguard").strip().lower(),
        capacity=_env_float("SHAREGUARD_CAPACITY", 0.85),
        rho_path=os.environ.get("SHAREGUARD_RHO_PATH", ""),
        compensate=_env_bool("SHAREGUARD_COMPENSATE", True),
    )
    if state.mode not in ("off", "min_weight", "shareguard"):
        raise ValueError(f"unsupported SHAREGUARD_MODE: {state.mode}")
    if state.capacity <= 0:
        raise ValueError("SHAREGUARD_CAPACITY must be positive")
    if state.enabled and state.mode != "off":
        state.load_tables()
    _STATE = state
    _READY = True
    return _STATE


def ensure_shareguard_ready() -> ShareGuardConfig:
    if not _READY:
        return init_shareguard_from_env()
    return _STATE


def reset_shareguard_stats() -> None:
    state = ensure_shareguard_ready()
    state.stats = ShareGuardStats()
    state.last_comp_coeff = None


def get_shareguard_stats() -> dict[str, Any]:
    return ensure_shareguard_ready().stats.as_dict()


def reset_shareguard_stats_for_model(_model: torch.nn.Module) -> None:
    reset_shareguard_stats()


def get_shareguard_stats_for_model(_model: torch.nn.Module) -> dict[str, Any]:
    return get_shareguard_stats()


def shareguard_drops_enabled() -> bool:
    state = ensure_shareguard_ready()
    return state.enabled and state.mode != "off"


def register_moe_layers(model_or_runner: torch.nn.Module) -> int:
    """Map FusedMoE modules to their calibration-table layer indices."""
    state = ensure_shareguard_ready()
    state.layer_index.clear()
    root = getattr(model_or_runner, "model", model_or_runner)
    for module in root.modules():
        if module.__class__.__name__ == "FusedMoE":
            state.layer_index[id(module)] = len(state.layer_index)
    state.next_layer_idx = len(state.layer_index)
    print(
        f"[ShareGuard] registered {len(state.layer_index)} MoE layers",
        flush=True,
    )
    return len(state.layer_index)


def expert_to_rank(
    expert_ids: torch.Tensor, ep_size: int, num_experts: int
) -> torch.Tensor:
    """Map contiguous expert blocks to EP ranks for linear placement."""
    if ep_size <= 0 or num_experts <= 0 or ep_size > num_experts:
        raise ValueError(
            f"invalid expert topology: ep_size={ep_size}, experts={num_experts}"
        )
    if ep_size == 1:
        return torch.zeros_like(expert_ids)
    base, remainder = divmod(num_experts, ep_size)
    if remainder == 0:
        return (expert_ids // base).clamp(max=ep_size - 1)
    boundaries = []
    end = 0
    for rank in range(ep_size):
        end += base + (1 if rank < remainder else 0)
        boundaries.append(end)
    bounds = torch.tensor(
        boundaries, device=expert_ids.device, dtype=expert_ids.dtype
    )
    return torch.searchsorted(bounds, expert_ids, right=True).clamp(
        max=ep_size - 1
    )


def shareguard_select_and_drop(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    ep_size: int,
    num_experts: int,
    mode: str,
    capacity: float,
    eps_row: torch.Tensor,
    rho_row: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Drop the least important branches from overloaded destination ranks."""
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError("top-k ids and weights must have matching [tokens, k] shapes")
    if eps_row.numel() < num_experts or rho_row.numel() < num_experts:
        raise ValueError("ShareGuard tables contain fewer experts than the model")

    tokens, top_k = topk_ids.shape
    device = topk_ids.device
    ids = topk_ids.clone()
    weights = topk_weights.float().clone()
    valid = ids >= 0
    destinations = expert_to_rank(ids.clamp(min=0), ep_size, num_experts)
    destinations = torch.where(valid, destinations, -1)
    flat_destinations = destinations.reshape(-1)
    flat_valid = valid.reshape(-1)
    loads = torch.stack(
        [((flat_destinations == rank) & flat_valid).sum() for rank in range(ep_size)]
    ).float()

    total = float(valid.sum().item())
    cap = max(1.0, capacity * total / ep_size)
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    if mode == "min_weight":
        scores = flat_weights.clone()
    else:
        expert_ids = flat_ids.clamp(min=0)
        eps = eps_row.to(device=device, dtype=torch.float32)[expert_ids]
        scores = flat_weights.square() * eps
        scores = torch.where(
            torch.isfinite(scores), scores, torch.full_like(scores, 1e9)
        )
    scores = torch.where(flat_valid, scores, torch.full_like(scores, 1e30))

    drop_mask = torch.zeros(tokens * top_k, dtype=torch.bool, device=device)
    for rank in range(ep_size):
        overload = max(0, math.ceil(float(loads[rank].item()) - cap))
        if overload == 0:
            continue
        candidates = (
            (flat_destinations == rank) & flat_valid & ~drop_mask
        ).nonzero(as_tuple=False).flatten()
        count = min(overload, candidates.numel())
        if count:
            chosen = candidates[
                torch.topk(scores[candidates], count, largest=False).indices
            ]
            drop_mask[chosen] = True

    dropped = int(drop_mask.sum().item())
    if dropped:
        ids.reshape(-1)[drop_mask] = -1
        weights.reshape(-1)[drop_mask] = 0.0

    token_drop_mask = drop_mask.view(tokens, top_k)
    original_ids = topk_ids.clamp(min=0)
    rho = rho_row.to(device=device, dtype=torch.float32)[original_ids]
    compensation = (
        topk_weights.float() * token_drop_mask.float() * rho
    ).sum(dim=-1)

    valid_after = ids >= 0
    destinations_after = expert_to_rank(
        ids.clamp(min=0), ep_size, num_experts
    )
    destinations_after = torch.where(valid_after, destinations_after, -1)
    flat_after = destinations_after.reshape(-1)
    valid_after = valid_after.reshape(-1)
    loads_after = torch.stack(
        [((flat_after == rank) & valid_after).sum() for rank in range(ep_size)]
    ).float()
    info = {
        "dropped": float(dropped),
        "cap": cap,
        "max_before": float(loads.max().item()) if loads.numel() else 0.0,
        "max_after": (
            float(loads_after.max().item()) if loads_after.numel() else 0.0
        ),
        "total": total,
    }
    return ids, weights.to(topk_weights.dtype), compensation, info


def apply_shareguard_to_topk(
    layer: torch.nn.Module,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = ensure_shareguard_ready()
    if not state.enabled or state.mode == "off":
        return topk_ids, topk_weights
    placement = getattr(layer, "expert_placement_strategy", "linear")
    if placement != "linear":
        raise RuntimeError(
            "ShareGuard currently requires linear expert placement, "
            f"got {placement}"
        )
    if state.rho is None or state.eps is None:
        raise RuntimeError("ShareGuard calibration tables are not loaded")

    started = time.perf_counter()
    layer_id = id(layer)
    if layer_id not in state.layer_index:
        state.layer_index[layer_id] = state.next_layer_idx
        state.next_layer_idx += 1
    layer_idx = state.layer_index[layer_id]
    if layer_idx >= state.rho.shape[0]:
        raise RuntimeError(
            f"MoE layer index {layer_idx} exceeds ShareGuard table shape "
            f"{tuple(state.rho.shape)}"
        )

    ep_size = int(getattr(layer, "ep_size", 1))
    num_experts = int(getattr(layer, "global_num_experts", 0))
    rho_row, eps_row = state.table_rows(layer_idx, topk_ids.device)
    new_ids, new_weights, compensation, info = shareguard_select_and_drop(
        topk_ids,
        topk_weights,
        ep_size=ep_size,
        num_experts=num_experts,
        mode=state.mode,
        capacity=state.capacity,
        eps_row=eps_row,
        rho_row=rho_row,
    )
    state.last_comp_coeff = compensation
    stats = state.stats
    stats.calls += 1
    stats.select_ms += (time.perf_counter() - started) * 1000.0
    stats.branches_seen += int(info["total"])
    stats.branches_dropped += int(info["dropped"])
    stats.max_load_before.append(info["max_before"])
    stats.max_load_after.append(info["max_after"])
    if info["dropped"]:
        stats.overload_calls += 1
    return new_ids, new_weights


def maybe_apply_compensation(
    shared_output: torch.Tensor | None,
) -> torch.Tensor | None:
    state = ensure_shareguard_ready()
    coefficient = state.last_comp_coeff
    state.last_comp_coeff = None
    if coefficient is None or not state.enabled or not state.compensate:
        return shared_output
    if shared_output is None:
        raise RuntimeError("ShareGuard compensation requires a shared expert output")
    if coefficient.numel() != shared_output.shape[0]:
        raise RuntimeError(
            "ShareGuard compensation token count does not match shared output: "
            f"{coefficient.numel()} != {shared_output.shape[0]}"
        )
    coefficient = coefficient.to(
        device=shared_output.device, dtype=shared_output.dtype
    )
    return shared_output + coefficient.unsqueeze(-1) * shared_output
