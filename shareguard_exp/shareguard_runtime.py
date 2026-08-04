#!/usr/bin/env python3
"""Compatibility exports for the vLLM-integrated ShareGuard runtime."""

from vllm.model_executor.layers.fused_moe.shareguard_runtime import (
    ShareGuardConfig,
    ShareGuardStats,
    apply_shareguard_to_topk,
    ensure_shareguard_ready,
    expert_to_rank,
    get_shareguard_state,
    get_shareguard_stats,
    get_shareguard_stats_for_model,
    init_shareguard_from_env,
    maybe_apply_compensation,
    register_moe_layers,
    reset_shareguard_stats,
    reset_shareguard_stats_for_model,
    shareguard_select_and_drop,
)

__all__ = [
    "ShareGuardConfig",
    "ShareGuardStats",
    "apply_shareguard_to_topk",
    "ensure_shareguard_ready",
    "expert_to_rank",
    "get_shareguard_state",
    "get_shareguard_stats",
    "get_shareguard_stats_for_model",
    "init_shareguard_from_env",
    "maybe_apply_compensation",
    "register_moe_layers",
    "reset_shareguard_stats",
    "reset_shareguard_stats_for_model",
    "shareguard_select_and_drop",
]
