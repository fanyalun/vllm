# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import gc
import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from multiprocessing.connection import Connection
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.async_draft.ipc import AsyncDraftBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


class AsyncDraftSpeculator(BaseSpeculator):
    """Proxy a standalone draft worker on another local CUDA device."""

    supports_mm_inputs = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        # Model loading mutates the target config with runtime-only objects that
        # cannot be serialized by the spawn multiprocessing context. Preserve a
        # pristine config for the standalone Draft before Target construction.
        self._child_vllm_config = copy.deepcopy(vllm_config)
        self.device = device
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        assert isinstance(speculative_config.async_draft_device, int)
        self.draft_device_id = speculative_config.async_draft_device
        self.method = speculative_config.method
        self.num_speculative_steps = speculative_config.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.engine_instance_id = vllm_config.instance_id
        self.startup_timeout = 900.0
        self.request_timeout = 120.0
        self._process = None
        self._connection: Connection | None = None
        self._ring_slots = None
        self._response_events: list[torch.cuda.Event] = []
        self._generation = 0
        self._request_epochs: dict[str, int] = {}
        self._active_requests: set[str] = set()
        self._preempted_requests: set[str] = set()
        self._local_candidates_enabled = (
            os.environ.get("ASYNC_DRAFT_TARGET_CANDIDATES", "0") == "1"
        )
        if self._local_candidates_enabled and (
            self.method != "eagle3" or self.max_num_reqs != 1
        ):
            raise ValueError(
                "Target candidate experiment requires EAGLE3 max_num_seqs=1"
            )
        if (
            self._local_candidates_enabled
            and os.environ.get("REPLAYSSM_SPEC_DECODE_TRACE_LOGITS", "0") == "1"
        ):
            raise ValueError(
                "Target candidate experiment does not support logits tracing"
            )
        self._candidate_connection = None
        self._candidate_tokens = None
        self._candidate_header = None
        self._pending_local_response = None
        self._export_metrics = os.environ.get("ASYNC_DRAFT_EXPORT_METRICS", "1") == "1"
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "jit_fallbacks": 0,
            "cache_evictions": 0,
            "ipc_bytes": 0,
            "wait_seconds": 0.0,
            "branch_build_seconds": 0.0,
            "overlap_seconds": 0.0,
            "canonical_commit_seconds": 0.0,
            "candidate_or_glue_seconds": 0.0,
            "tree_or_block_build_seconds": 0.0,
            "context_kv_projection_seconds": 0.0,
            "dspark_current_backbone_runs": 0,
            "dspark_current_backbone_seconds": 0.0,
            "dspark_backbone_refreshes": 0,
            "dspark_branch_backbone_seconds": 0.0,
            "dspark_markov_branches": 0,
            "fanout_branches": 0,
            "fanout_build_rounds": 0,
            "next_proposal_wait_seconds": 0.0,
            "ipc_latency_seconds": 0.0,
        }
        self._step_metrics = self._metrics.copy()
        self._last_cache_hit_indices: set[int] = set()
        self._last_trace_top2: list[dict[str, Any] | None] = []
        self._active_trace_req_ids: list[str] = []
        self._last_trace_timing: dict[str, Any] = {}
        self._last_response_ready_at: float | None = None
        self.child_metadata: dict[str, Any] = {}
        self.draft_logits = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        return None

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        return None

    def _check_peer_access(self) -> None:
        source_index = self.device.index
        if source_index is None:
            source_index = torch.cuda.current_device()
        draft_visible_index = self.draft_device_id
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            from vllm.platforms import current_platform

            visible_physical_ids = [
                current_platform.device_control_id_to_physical_device_id(value)
                for value in cvd.split(",")
            ]
            if self.draft_device_id not in visible_physical_ids:
                raise RuntimeError(
                    f"Async draft physical GPU {self.draft_device_id} is not "
                    f"visible in CUDA_VISIBLE_DEVICES={cvd}"
                )
            draft_visible_index = visible_physical_ids.index(self.draft_device_id)
        self.draft_visible_device_id = draft_visible_index

        if not torch.cuda.can_device_access_peer(source_index, draft_visible_index):
            raise RuntimeError(
                "CUDA peer access is unavailable between target GPU "
                f"{source_index} and draft GPU {draft_visible_index}"
            )

        target = torch.device(f"cuda:{source_index}")
        draft = torch.device(f"cuda:{draft_visible_index}")
        forward = torch.tensor(
            [0x13579BDF, 0x2468ACE], dtype=torch.int64, device=target
        )
        draft_copy = torch.empty_like(forward, device=draft)
        draft_copy.copy_(forward)
        backward = torch.empty_like(forward, device=target)
        backward.copy_(draft_copy)
        torch.cuda.synchronize(target)
        torch.cuda.synchronize(draft)
        if not torch.equal(forward.cpu(), backward.cpu()):
            raise RuntimeError("Bidirectional CUDA peer sentinel validation failed")

    def load_model(self, target_model: torch.nn.Module) -> None:
        del target_model
        self._check_peer_access()

        import torch.multiprocessing as mp

        from vllm.v1.worker.gpu.spec_decode.async_draft.runtime import (
            run_async_draft_child,
        )

        if self.vllm_config.speculative_config.use_gemma4_mtp():
            from vllm.v1.worker.gpu.spec_decode.async_draft.gemma4 import (
                run_gemma4_child,
                target_kv_layers,
            )

            self._gemma4_target_layers = target_kv_layers(self.vllm_config)
            run_async_draft_child = run_gemma4_child

        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=run_async_draft_child,
            name="vllm-async-draft",
            args=(
                child_connection,
                self._child_vllm_config,
                self.draft_device_id,
                get_open_port(),
            ),
        )
        # Ensure an abnormal target exit cannot leave the draft child alive.
        process.daemon = True
        process.start()
        child_connection.close()
        self._process = process
        self._connection = parent_connection

        message = self._recv(self.startup_timeout, "draft child startup")
        if message.get("status") != "ready":
            self._raise_child_error(message, "draft child startup")
        self._ring_slots = message.pop("ring_slots")
        response_event_handles = message.pop("response_event_handles")
        self._response_events = [
            torch.cuda.Event.from_ipc_handle(self.draft_visible_device_id, event_handle)
            for event_handle in response_event_handles
        ]
        self.child_metadata = message
        conditioning_size = sum(self.child_metadata["conditioning_splits"])
        self._combined_conditioning_states = torch.empty(
            self.max_num_tokens,
            conditioning_size,
            dtype=self.vllm_config.model_config.dtype,
            device=self.device,
        )
        self._draft_tokens = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=self.device,
        )
        if self._local_candidates_enabled:
            assert self._ring_slots is not None
            candidate_reader, candidate_writer = context.Pipe(duplex=False)
            capacity = (self.num_speculative_steps + 1) * message["fan_out"]
            self._candidate_connection = candidate_reader
            parent_connection.send(
                {
                    "command": "target_candidates",
                    "shape": (
                        len(self._ring_slots),
                        capacity,
                        self.num_speculative_steps,
                    ),
                    "device": self.device,
                    "connection": candidate_writer,
                }
            )
            response = self._recv(self.request_timeout, "target candidate setup")
            candidate_writer.close()
            if response.get("status") != "ok":
                self._raise_child_error(response, "target candidate setup")
            self._candidate_tokens = response.pop("tokens")
        logger.info(
            "Async %s draft child ready: pid=%s physical_gpu=%s "
            "kv_blocks=%s block_sizes=%s fan_out=%s verify_width=%s "
            "execution_width=%s branch_backbone_width=%s "
            "checkpoint_native_width=%s "
            "target_replayssm_owned_by_child=%s",
            self.method,
            self.child_metadata.get("pid"),
            self.child_metadata.get("physical_device_id"),
            self.child_metadata.get("kv_num_blocks"),
            self.child_metadata.get("draft_kv_block_sizes"),
            self.child_metadata.get("fan_out"),
            self.child_metadata.get("target_verify_width"),
            self.child_metadata.get("proposal_execution_width"),
            self.child_metadata.get("branch_backbone_width"),
            self.child_metadata.get("proposal_bank_width"),
            self.child_metadata.get("target_replayssm_owned_by_child"),
        )
        logger.info(
            "Async %s standalone materialized weights: %s",
            self.method,
            [
                {
                    "name": item.get("name"),
                    "source": item.get("source"),
                    "shape": item.get("shape"),
                    "sha256": item.get("sha256"),
                }
                for item in self.child_metadata.get("materialized_weights", [])
            ],
        )

    def _recv(self, timeout: float, operation: str) -> dict[str, Any]:
        connection = self._connection
        process = self._process
        if connection is None or process is None:
            raise RuntimeError(f"Async draft child is unavailable during {operation}")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if connection.poll(min(remaining, 0.1)):
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        "Async draft child closed its IPC channel during "
                        f"{operation}; child_pid={process.pid}, "
                        f"exit_code={process.exitcode}"
                    ) from error
                if not isinstance(message, dict):
                    raise RuntimeError(
                        f"Invalid async draft response during {operation}: {message!r}"
                    )
                return message
            if process.exitcode is not None:
                raise RuntimeError(
                    f"Async draft child exited during {operation}; "
                    f"child_pid={process.pid}, exit_code={process.exitcode}"
                )

        if process.exitcode is not None:
            raise RuntimeError(
                f"Async draft child exited during {operation}; "
                f"child_pid={process.pid}, exit_code={process.exitcode}"
            )
        raise TimeoutError(
            f"Timed out after {timeout:.1f}s during {operation}; "
            f"child_pid={process.pid}, exit_code={process.exitcode}"
        )

    def _raise_child_error(self, message: dict[str, Any], operation: str) -> None:
        detail = message.get("error", message)
        child_traceback = message.get("traceback")
        if child_traceback:
            detail = f"{detail}\n{child_traceback}"
        self.shutdown()
        raise RuntimeError(f"Async draft child failed during {operation}: {detail}")

    def _control(self, command: str, request_ids: Iterable[str]) -> None:
        ids = list(request_ids)
        if not ids or self._connection is None:
            return
        self._finish_local_response()
        self._candidate_header = None
        self._connection.send({"command": command, "request_ids": ids})
        response = self._recv(self.request_timeout, command)
        if response.get("status") != "ok":
            self._raise_child_error(response, command)
        response_metrics = response.get("metrics") or {}
        self._record_metrics(response_metrics)

    def _finish_local_response(self) -> None:
        pending = self._pending_local_response
        if pending is None:
            return
        self._pending_local_response = None
        response = self._recv(self.request_timeout, "local candidate acknowledgement")
        if response.get("status") != "ok":
            self._raise_child_error(response, "local candidate acknowledgement")
        self._validate_response_identity(response, *pending)
        if response.get("cache_hit_indices") != [0]:
            raise RuntimeError("Published Target candidate was not a Draft cache hit")
        metrics = dict(response.get("metrics") or {})
        if metrics.pop("cache_hits", 0) != 1:
            raise RuntimeError("Invalid local candidate acknowledgement counters")
        self._record_metrics(metrics)

    def _find_local_candidate(self, batch: AsyncDraftBatch, ring_slot: Any):
        connection = self._candidate_connection
        if connection is None:
            return None
        while connection.poll():
            self._candidate_header = connection.recv()
        header = self._candidate_header
        if (
            header is None
            or header["generation"] != batch.generation - 1
            or batch.transient
            or bool(batch.is_prefilling_np[0])
            or os.environ.get("ASYNC_DRAFT_FORCE_JIT", "0") == "1"
        ):
            return None
        with record_function_or_nullcontext("async_draft: local_outcome_d2h"):
            accepted = int(ring_slot.num_sampled[0].item()) - 1
            recovery = int(ring_slot.last_sampled[0].item())
        key = (
            batch.engine_instance_id,
            batch.req_ids[0],
            batch.request_epochs[0],
            accepted,
            recovery,
        )
        try:
            index = header["keys"].index(key)
        except ValueError:
            return None
        return self._candidate_tokens[header["slot"], index]

    def _record_metrics(self, metrics: dict[str, float | int] | None) -> None:
        metrics = metrics or {}
        for name in (
            "ipc_bytes",
            "cache_hits",
            "cache_misses",
            "jit_fallbacks",
            "cache_evictions",
            "branch_build_seconds",
            "canonical_commit_seconds",
            "candidate_or_glue_seconds",
            "tree_or_block_build_seconds",
            "context_kv_projection_seconds",
            "dspark_current_backbone_runs",
            "dspark_current_backbone_forwards",
            "dspark_current_backbone_seconds",
            "dspark_backbone_refreshes",
            "dspark_branch_backbone_seconds",
            "dspark_branch_backbone_forwards",
            "dspark_candidate_backbone_forwards",
            "dspark_candidate_seconds",
            "dspark_markov_seconds",
            "dspark_markov_branches",
            "fanout_branches",
            "fanout_build_rounds",
        ):
            delta = metrics.get(name, 0)
            self._metrics[name] = self._metrics.get(name, 0) + delta
            self._step_metrics[name] = self._step_metrics.get(name, 0) + delta

    @staticmethod
    def _validate_response_identity(
        response: dict[str, Any], generation: int, slot: int
    ) -> None:
        if response.get("generation") != generation or response.get("slot") != slot:
            raise RuntimeError(
                "Async draft response generation mismatch: "
                f"expected=({generation}, {slot}), response="
                f"({response.get('generation')}, {response.get('slot')})"
            )

    def on_requests_added(self, request_ids: Iterable[str]) -> None:
        reset: list[str] = []
        for req_id in request_ids:
            if req_id in self._preempted_requests:
                self._preempted_requests.remove(req_id)
            elif req_id in self._request_epochs:
                self._request_epochs[req_id] += 1
                reset.append(req_id)
            else:
                self._request_epochs[req_id] = 0
            self._active_requests.add(req_id)
        self._control("reset", reset)

    def on_requests_finished(self, request_ids: Iterable[str]) -> None:
        ids = list(request_ids)
        for req_id in ids:
            self._active_requests.discard(req_id)
            self._preempted_requests.discard(req_id)
        self._control("release", ids)

    def on_requests_preempted(self, request_ids: Iterable[str]) -> None:
        ids = list(request_ids)
        for req_id in ids:
            self._request_epochs[req_id] = self._request_epochs.get(req_id, 0) + 1
            self._active_requests.discard(req_id)
            self._preempted_requests.add(req_id)
        self._control("reset", ids)

    def _copy_payload(
        self,
        ring_slot: Any,
        input_batch: InputBatch,
        conditioning_states: list[torch.Tensor],
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> int:
        if not conditioning_states:
            raise ValueError("Async draft requires target conditioning states")
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        idx_mapping = input_batch.idx_mapping
        ipc_bytes = 0

        copies = (
            (ring_slot.input_ids[:num_tokens], input_batch.input_ids[:num_tokens]),
            (ring_slot.positions[:num_tokens], input_batch.positions[:num_tokens]),
            (
                ring_slot.query_start_loc[: input_batch.num_reqs_after_padding + 1],
                input_batch.query_start_loc,
            ),
            (
                ring_slot.seq_lens[: input_batch.num_reqs_after_padding],
                input_batch.seq_lens,
            ),
            (ring_slot.num_sampled[:num_reqs], num_sampled),
            (ring_slot.num_rejected[:num_reqs], num_rejected),
            (
                ring_slot.last_sampled[:num_reqs],
                last_sampled[idx_mapping, 0],
            ),
            (
                ring_slot.next_prefill_tokens[:num_reqs],
                next_prefill_tokens[idx_mapping],
            ),
            (
                ring_slot.temperature[:num_reqs],
                temperature[idx_mapping],
            ),
            (ring_slot.seeds[:num_reqs], seeds[idx_mapping]),
        )
        with record_function_or_nullcontext("async_draft: copy_metadata"):
            for destination, source in copies:
                destination.copy_(source, non_blocking=True)
                ipc_bytes += destination.numel() * destination.element_size()

        hidden_offset = 0
        expected_splits = tuple(self.child_metadata["conditioning_splits"])
        actual_splits = tuple(state.shape[-1] for state in conditioning_states)
        if actual_splits != expected_splits:
            raise ValueError(
                "Async draft conditioning layout mismatch: "
                f"copied={actual_splits}, expected={expected_splits}"
            )
        for hidden_states in conditioning_states:
            width = hidden_states.shape[-1]
            local_destination = self._combined_conditioning_states[
                :num_tokens, hidden_offset : hidden_offset + width
            ]
            local_destination.copy_(hidden_states[:num_tokens], non_blocking=True)
            hidden_offset += width
        if hidden_offset != ring_slot.conditioning_states.shape[-1]:
            raise ValueError(
                "Async draft conditioning-state width mismatch: "
                f"copied={hidden_offset}, expected="
                f"{ring_slot.conditioning_states.shape[-1]}"
            )
        destination = ring_slot.conditioning_states[:num_tokens]
        with record_function_or_nullcontext("async_draft: copy_conditioning"):
            destination.copy_(
                self._combined_conditioning_states[:num_tokens], non_blocking=True
            )
        ipc_bytes += destination.numel() * destination.element_size()

        with record_function_or_nullcontext("async_draft: payload_synchronize"):
            torch.cuda.synchronize(ring_slot.input_ids.device)
        return ipc_bytes

    def _copy_response(self, ring_slot: Any, slot_index: int, num_reqs: int) -> None:
        # The child records the event before sending the response header.
        # Wait on the source stream; the peer copy orders the Target consumer.
        with record_function_or_nullcontext("async_draft: response_stream_wait"):
            self._response_events[slot_index].wait(
                torch.cuda.current_stream(ring_slot.draft_tokens.device)
            )
        with record_function_or_nullcontext("async_draft: copy_response"):
            self._draft_tokens[:num_reqs].copy_(
                ring_slot.draft_tokens[:num_reqs], non_blocking=True
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            slot_mappings,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        if self._ring_slots is None or self._connection is None:
            raise RuntimeError("Async draft child has not been initialized")

        start = time.perf_counter()
        self._finish_local_response()
        generation = self._generation
        self._generation += 1
        slot_index = generation % len(self._ring_slots)
        ring_slot = self._ring_slots[slot_index]
        overlap_budget_seconds = 0.0
        if self._last_response_ready_at is not None:
            overlap_budget_seconds = max(0.0, start - self._last_response_ready_at)
            self._metrics["overlap_seconds"] += overlap_budget_seconds
            self._step_metrics["overlap_seconds"] += overlap_budget_seconds
        conditioning_states = (
            aux_hidden_states if aux_hidden_states else [last_hidden_states]
        )
        ipc_start = time.perf_counter()
        ipc_bytes = self._copy_payload(
            ring_slot,
            input_batch,
            conditioning_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
        )
        if ring_slot.target_kv is not None:
            from vllm.v1.worker.gpu.spec_decode.async_draft.gemma4 import (
                copy_target_kv,
            )

            ipc_bytes += copy_target_kv(
                self._gemma4_target_layers,
                attn_metadata,
                ring_slot,
                input_batch,
                dummy_run or is_profile,
            )
        ipc_latency_seconds = time.perf_counter() - ipc_start

        request_epochs = [
            self._request_epochs.setdefault(req_id, 0) for req_id in input_batch.req_ids
        ]
        batch = AsyncDraftBatch(
            generation=generation,
            slot=slot_index,
            engine_instance_id=self.engine_instance_id,
            req_ids=list(input_batch.req_ids),
            request_epochs=request_epochs,
            transient=(
                dummy_run
                or is_profile
                or all(req_id.startswith("_warmup_") for req_id in input_batch.req_ids)
            ),
            overlap_budget_seconds=overlap_budget_seconds,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            num_tokens_after_padding=input_batch.num_tokens_after_padding,
            num_reqs_after_padding=input_batch.num_reqs_after_padding,
            num_scheduled_tokens=input_batch.num_scheduled_tokens.copy(),
            query_start_loc_np=input_batch.query_start_loc_np.copy(),
            seq_lens_cpu_upper_bound=(
                input_batch.seq_lens_cpu_upper_bound.numpy().copy()
            ),
            num_computed_tokens_np=input_batch.num_computed_tokens_np.copy(),
            prefill_len_np=input_batch.prefill_len_np.copy(),
            num_computed_prefill_tokens_np=(
                input_batch.num_computed_prefill_tokens_np.copy()
            ),
            is_prefilling_np=input_batch.is_prefilling_np.copy(),
        )
        local_candidate = self._find_local_candidate(batch, ring_slot)
        with record_function_or_nullcontext("async_draft: send_request"):
            self._connection.send({"command": "propose", "batch": asdict(batch)})
        if local_candidate is not None:
            with record_function_or_nullcontext("async_draft: local_candidate"):
                self._draft_tokens[0].copy_(local_candidate)
            self._pending_local_response = (generation, slot_index)
            self._last_cache_hit_indices = {0}
            self._last_trace_top2 = []
            self._active_trace_req_ids = list(batch.req_ids)
            elapsed = time.perf_counter() - start
            for metrics in (self._metrics, self._step_metrics):
                metrics["cache_hits"] += 1
                metrics["target_local_hits"] = metrics.get("target_local_hits", 0) + 1
                metrics["ipc_bytes"] += ipc_bytes
                metrics["wait_seconds"] += elapsed
                metrics["next_proposal_wait_seconds"] += elapsed
                metrics["ipc_latency_seconds"] += ipc_latency_seconds
            self._last_trace_timing = {
                "async_generation": generation,
                "async_target_local_hit": True,
            }
            self._last_response_ready_at = time.perf_counter()
            return self._draft_tokens[:1]
        with record_function_or_nullcontext("async_draft: receive_response"):
            response = self._recv(self.request_timeout, "propose")
        if response.get("status") != "ok":
            del ring_slot
            self._raise_child_error(response, "propose")
        self._validate_response_identity(response, generation, slot_index)
        self._last_cache_hit_indices = set(response.get("cache_hit_indices") or [])
        self._last_trace_top2 = response.get("trace_top2") or []
        self._active_trace_req_ids = list(input_batch.req_ids)

        num_reqs = input_batch.num_reqs
        response_scope = (
            "async_draft: remote_hit"
            if len(self._last_cache_hit_indices) == num_reqs
            else "async_draft: remote_miss"
        )
        with record_function_or_nullcontext(response_scope):
            self._copy_response(ring_slot, slot_index, num_reqs)
        elapsed = time.perf_counter() - start
        response_metrics = response.get("metrics") or {}
        self._record_metrics(response_metrics)
        ipc_bytes += (
            num_reqs * self.num_speculative_steps * self._draft_tokens.element_size()
        )
        self._metrics["ipc_bytes"] += ipc_bytes
        self._step_metrics["ipc_bytes"] += ipc_bytes
        self._metrics["wait_seconds"] += elapsed
        self._step_metrics["wait_seconds"] += elapsed
        self._metrics["next_proposal_wait_seconds"] += elapsed
        self._step_metrics["next_proposal_wait_seconds"] += elapsed
        self._metrics["ipc_latency_seconds"] += ipc_latency_seconds
        self._step_metrics["ipc_latency_seconds"] += ipc_latency_seconds
        previous_branch_build_seconds = float(
            response_metrics.get("branch_build_seconds", 0.0)
        )
        self._last_trace_timing = {
            "async_generation": generation,
            "async_fan_out": self.child_metadata.get("fan_out"),
            "async_timing_pair_valid": (
                generation > 0 and previous_branch_build_seconds > 0.0
            ),
            "async_verify_window_seconds": overlap_budget_seconds,
            "async_previous_branch_build_seconds": (previous_branch_build_seconds),
            "async_branch_hidden_seconds": min(
                overlap_budget_seconds, previous_branch_build_seconds
            ),
            "async_branch_exposed_seconds": max(
                previous_branch_build_seconds - overlap_budget_seconds, 0.0
            ),
            "async_next_proposal_wait_seconds": elapsed,
            "async_batch_num_reqs": num_reqs,
            "async_response_metrics": response_metrics,
        }
        self._last_response_ready_at = time.perf_counter()
        return self._draft_tokens[:num_reqs]

    def take_metrics(self) -> dict[str, float | int]:
        metrics = self._step_metrics
        self._step_metrics = {name: 0 for name in metrics}
        return metrics if self._export_metrics else {}

    def proposal_trace_metadata(self, num_reqs: int) -> list[dict[str, Any]]:
        trace_top2 = getattr(self, "_last_trace_top2", [])
        return [
            {
                "request_epoch": self._request_epochs.get(req_id, 0),
                "cache_hit": index in self._last_cache_hit_indices,
                **getattr(self, "_last_trace_timing", {}),
                **(
                    trace_top2[index]
                    if index < len(trace_top2) and trace_top2[index] is not None
                    else {}
                ),
            }
            for index, req_id in enumerate(list(self._active_trace_req_ids)[:num_reqs])
        ]

    def shutdown(self) -> None:
        connection = self._connection
        process = self._process
        self._connection = None
        self._process = None
        self._ring_slots = None
        self._response_events = []
        self._candidate_tokens = None
        gc.collect()
        if connection is not None and process is not None and process.is_alive():
            try:
                connection.send({"command": "shutdown"})
                if connection.poll(30.0):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        if connection is not None:
            connection.close()
        if process is not None:
            process.join(timeout=30.0)
            if process.is_alive():
                logger.error(
                    "Async draft child pid=%s did not exit; terminating it.",
                    process.pid,
                )
                process.terminate()
                process.join(timeout=10.0)
            if not process.is_alive():
                process.close()
        if self._candidate_connection is not None:
            self._candidate_connection.close()
            self._candidate_connection = None
