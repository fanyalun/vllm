# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    _get_num_experts_per_tok,
    get_num_experts,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_decode_trace_metadata(
    *,
    num_scheduled: int,
    start_position: int,
    num_prompt_tokens: int,
    num_spec_tokens: int,
    num_accepted_drafts: int,
) -> tuple[list[int], list[str], list[bool]]:
    """Build decode-only row indices, route kinds, and acceptance flags."""
    num_non_spec = num_scheduled - num_spec_tokens
    row_indices = [
        index
        for index in range(num_scheduled)
        if start_position + index >= num_prompt_tokens
    ]
    route_kinds: list[str] = []
    accepted: list[bool] = []
    for index in row_indices:
        if num_spec_tokens and index == num_non_spec - 1:
            route_kinds.append("spec_target")
            accepted.append(True)
        elif num_spec_tokens and index >= num_non_spec:
            draft_index = index - num_non_spec + 1
            route_kinds.append(f"spec_draft_{draft_index}")
            accepted.append(draft_index <= num_accepted_drafts)
        else:
            route_kinds.append("ar_decode")
            accepted.append(True)
    return row_indices, route_kinds, accepted


def is_target_model_moe_module(module_name: str) -> bool:
    """Return whether a static MoE module belongs to the target model."""
    path_parts = module_name.split(".")
    return "draft_model" not in path_parts and "mtp" not in path_parts


class RoutedExpertsTraceWriter:
    """Stream target-model routed-expert rows from the scheduler to disk."""

    def __init__(self, config: dict[str, Any], vllm_config: VllmConfig) -> None:
        output_dir_value = config.get("output_dir")
        if not isinstance(output_dir_value, str) or not output_dir_value:
            raise ValueError("routed_experts_trace.output_dir must be a string")
        if config.get("decode_only") is not True:
            raise ValueError("routed_experts_trace currently requires decode_only=True")

        self.output_dir = Path(output_dir_value).expanduser().resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"routed experts trace directory is not empty: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        hf_config = vllm_config.model_config.hf_text_config
        self.num_layers = int(hf_config.num_hidden_layers)
        self.top_k = int(_get_num_experts_per_tok(hf_config))
        self.num_experts = int(get_num_experts(hf_config))
        self.dtype = np.dtype(np.uint8 if self.num_experts <= 256 else np.uint16)
        self.routes_path = self.output_dir / "routes.bin"
        self.events_path = self.output_dir / "events.jsonl"
        self.manifest_path = self.output_dir / "trace_manifest.json"
        marker_value = config.get("completion_marker", "worker_complete")
        if not isinstance(marker_value, str) or not marker_value:
            raise ValueError(
                "routed_experts_trace.completion_marker must be a string"
            )
        self.completion_marker = self.output_dir / marker_value
        self.routes_file = self.routes_path.open("xb")
        self.events_file = self.events_path.open("x", encoding="utf-8")
        self.num_rows = 0
        self.num_events = 0
        self.closed = False
        self.manifest: dict[str, Any] = {
            "schema_version": 2,
            "state": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "run_name": config.get("run_name"),
            "data_parallel_rank": getattr(
                vllm_config.parallel_config, "data_parallel_rank", 0
            ),
            "model_family": config.get("model_family"),
            "draft_method": config.get("draft_method"),
            "expected_target_moe_layers": config.get(
                "expected_target_moe_layers"
            ),
            "decode_only": True,
            "route_scope": "target_model_decode_verify_only_excludes_drafter",
            "model": vllm_config.model_config.model,
            "tensor_parallel_size": (
                vllm_config.parallel_config.tensor_parallel_size
            ),
            "expert_parallel_size": (
                vllm_config.parallel_config.tensor_parallel_size
                * vllm_config.parallel_config.data_parallel_size
                if vllm_config.parallel_config.enable_expert_parallel
                else 1
            ),
            "data_parallel_size": vllm_config.parallel_config.data_parallel_size,
            "pipeline_parallel_size": (
                vllm_config.parallel_config.pipeline_parallel_size
            ),
            "max_num_batched_tokens": (
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
            "num_speculative_tokens": vllm_config.num_speculative_tokens,
            "use_replayssm": vllm_config.cache_config.use_replayssm,
            "use_replayssm_spec": vllm_config.cache_config.use_replayssm_spec,
            "replayssm_buffer_len": vllm_config.cache_config.replayssm_buffer_len,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "route_dtype": self.dtype.name,
            "route_shape": [0, self.num_layers, self.top_k],
            "num_events": 0,
            "routes_file": self.routes_path.name,
            "events_file": self.events_path.name,
        }
        _write_json_atomic(self.manifest_path, self.manifest)

    def write_request(
        self,
        routes: np.ndarray,
        *,
        scheduler_step: int,
        request_id: str,
        absolute_positions: list[int],
        token_ids: list[int],
        row_in_request: list[int],
        route_kinds: list[str],
        accepted: list[bool],
    ) -> None:
        """Append one request slice from a scheduler step."""
        if self.closed:
            raise RuntimeError("cannot write to a closed routed experts trace")
        routes_array = np.asarray(routes)
        expected_shape = (len(token_ids), self.num_layers, self.top_k)
        if routes_array.shape != expected_shape:
            raise ValueError(
                f"unexpected routed experts shape {routes_array.shape}; "
                f"expected {expected_shape}"
            )
        if not (
            len(absolute_positions)
            == len(row_in_request)
            == len(route_kinds)
            == len(accepted)
            == len(token_ids)
        ):
            raise ValueError("routed experts event metadata lengths do not match")
        if routes_array.size:
            minimum = int(routes_array.min())
            maximum = int(routes_array.max())
            if minimum < 0 or maximum >= self.num_experts:
                raise ValueError(
                    "routed expert IDs out of range: "
                    f"min={minimum}, max={maximum}, num_experts={self.num_experts}"
                )

        binary_row_offset = self.num_rows
        routes_array.astype(self.dtype, copy=False).tofile(self.routes_file)
        event = {
            "scheduler_step": scheduler_step,
            "request_id": request_id,
            "binary_row_offset": binary_row_offset,
            "row_count": len(token_ids),
            "absolute_positions": absolute_positions,
            "token_ids": token_ids,
            "row_in_request": row_in_request,
            "route_kinds": route_kinds,
            "accepted": accepted,
        }
        self.events_file.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.num_rows += len(token_ids)
        self.num_events += 1

    def close(self) -> None:
        """Flush the trace and publish its final manifest."""
        if self.closed:
            return
        self.closed = True
        self.routes_file.flush()
        os.fsync(self.routes_file.fileno())
        self.routes_file.close()
        self.events_file.flush()
        os.fsync(self.events_file.fileno())
        self.events_file.close()

        completed = self.completion_marker.is_file()
        state = "complete" if completed else "failed"
        self.manifest.update(
            {
                "state": state,
                "completed_at": _utc_now(),
                "route_shape": [self.num_rows, self.num_layers, self.top_k],
                "num_events": self.num_events,
                "routes_bytes": self.routes_path.stat().st_size,
                "events_bytes": self.events_path.stat().st_size,
                "routes_sha256": _sha256(self.routes_path),
                "events_sha256": _sha256(self.events_path),
            }
        )
        _write_json_atomic(self.manifest_path, self.manifest)
        if not completed:
            _write_json_atomic(
                self.output_dir / "failure.json",
                {
                    "state": "failed",
                    "reason": "worker completion marker was not created",
                    "timestamp": _utc_now(),
                    "valid_trace_rows": self.num_rows,
                },
            )
