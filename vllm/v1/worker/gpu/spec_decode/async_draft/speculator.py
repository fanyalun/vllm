# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.async_draft.ipc import AsyncDraftBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


class AsyncDraftSpeculator(BaseSpeculator):
    """Proxy a standalone EAGLE3 draft worker on another local CUDA device."""

    supports_mm_inputs = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        assert isinstance(speculative_config.async_draft_device, int)
        self.draft_device_id = speculative_config.async_draft_device
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
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "jit_fallbacks": 0,
            "cache_evictions": 0,
            "ipc_bytes": 0,
            "wait_seconds": 0.0,
            "branch_build_seconds": 0.0,
            "overlap_seconds": 0.0,
        }
        self._step_metrics = self._metrics.copy()
        self._last_cache_hit_indices: set[int] = set()
        self._last_trace_top2: list[dict[str, Any] | None] = []
        self._active_trace_req_ids: list[str] = []
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

        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=run_async_draft_child,
            name="vllm-async-draft",
            args=(
                child_connection,
                self.vllm_config,
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
        aux_hidden_size = sum(self.child_metadata["aux_hidden_splits"])
        self._combined_aux_hidden_states = torch.empty(
            self.max_num_tokens,
            aux_hidden_size,
            dtype=self.vllm_config.model_config.dtype,
            device=self.device,
        )
        self._draft_tokens = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=self.device,
        )
        logger.info(
            "Async EAGLE3 draft child ready: pid=%s physical_gpu=%s "
            "kv_blocks=%s fan_out=3",
            self.child_metadata.get("pid"),
            self.child_metadata.get("physical_device_id"),
            self.child_metadata.get("kv_num_blocks"),
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
        self._connection.send({"command": command, "request_ids": ids})
        response = self._recv(self.request_timeout, command)
        if response.get("status") != "ok":
            self._raise_child_error(response, command)
        self._record_metrics(response.get("metrics"))

    def _record_metrics(self, metrics: dict[str, float | int] | None) -> None:
        metrics = metrics or {}
        for name in (
            "cache_hits",
            "cache_misses",
            "jit_fallbacks",
            "cache_evictions",
            "branch_build_seconds",
        ):
            delta = metrics.get(name, 0)
            self._metrics[name] += delta
            self._step_metrics[name] += delta

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
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> int:
        if not aux_hidden_states:
            raise ValueError("Async EAGLE3 draft requires target auxiliary states")
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
        for destination, source in copies:
            destination.copy_(source, non_blocking=True)
            ipc_bytes += destination.numel() * destination.element_size()

        hidden_offset = 0
        for hidden_states in aux_hidden_states:
            width = hidden_states.shape[-1]
            local_destination = self._combined_aux_hidden_states[
                :num_tokens, hidden_offset : hidden_offset + width
            ]
            local_destination.copy_(hidden_states[:num_tokens], non_blocking=True)
            hidden_offset += width
        if hidden_offset != ring_slot.aux_hidden_states.shape[-1]:
            raise ValueError(
                "Async draft auxiliary hidden-state width mismatch: "
                f"copied={hidden_offset}, expected="
                f"{ring_slot.aux_hidden_states.shape[-1]}"
            )
        destination = ring_slot.aux_hidden_states[:num_tokens]
        destination.copy_(
            self._combined_aux_hidden_states[:num_tokens], non_blocking=True
        )
        ipc_bytes += destination.numel() * destination.element_size()

        torch.cuda.synchronize(ring_slot.input_ids.device)
        return ipc_bytes

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
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        if self._ring_slots is None or self._connection is None:
            raise RuntimeError("Async draft child has not been initialized")

        generation = self._generation
        self._generation += 1
        slot_index = generation % len(self._ring_slots)
        ring_slot = self._ring_slots[slot_index]
        start = time.perf_counter()
        overlap_budget_seconds = 0.0
        if self._last_response_ready_at is not None:
            overlap_budget_seconds = max(0.0, start - self._last_response_ready_at)
            self._metrics["overlap_seconds"] += overlap_budget_seconds
            self._step_metrics["overlap_seconds"] += overlap_budget_seconds
        ipc_bytes = self._copy_payload(
            ring_slot,
            input_batch,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
        )

        request_epochs = [
            self._request_epochs.setdefault(req_id, 0) for req_id in input_batch.req_ids
        ]
        batch = AsyncDraftBatch(
            generation=generation,
            slot=slot_index,
            engine_instance_id=self.engine_instance_id,
            req_ids=list(input_batch.req_ids),
            request_epochs=request_epochs,
            transient=dummy_run or is_profile,
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
        self._connection.send({"command": "propose", "batch": asdict(batch)})
        response = self._recv(self.request_timeout, "propose")
        if response.get("status") != "ok":
            del ring_slot
            self._raise_child_error(response, "propose")
        self._validate_response_identity(response, generation, slot_index)
        self._last_cache_hit_indices = set(response.get("cache_hit_indices") or [])
        self._last_trace_top2 = response.get("trace_top2") or []
        self._active_trace_req_ids = list(input_batch.req_ids)

        num_reqs = input_batch.num_reqs
        self._response_events[slot_index].synchronize()
        self._draft_tokens[:num_reqs].copy_(
            ring_slot.draft_tokens[:num_reqs], non_blocking=True
        )
        elapsed = time.perf_counter() - start
        self._record_metrics(response.get("metrics"))
        ipc_bytes += (
            num_reqs * self.num_speculative_steps * self._draft_tokens.element_size()
        )
        self._metrics["ipc_bytes"] += ipc_bytes
        self._step_metrics["ipc_bytes"] += ipc_bytes
        self._metrics["wait_seconds"] += elapsed
        self._step_metrics["wait_seconds"] += elapsed
        self._last_response_ready_at = time.perf_counter()
        return self._draft_tokens[:num_reqs]

    def take_metrics(self) -> dict[str, float | int]:
        metrics = self._step_metrics
        self._step_metrics = {name: 0 for name in metrics}
        return metrics

    def proposal_trace_metadata(self, num_reqs: int) -> list[dict[str, Any]]:
        trace_top2 = getattr(self, "_last_trace_top2", [])
        return [
            {
                "request_epoch": self._request_epochs.get(req_id, 0),
                "cache_hit": index in self._last_cache_hit_indices,
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
