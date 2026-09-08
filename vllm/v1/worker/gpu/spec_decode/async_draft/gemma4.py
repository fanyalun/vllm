# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
import signal
import time
import traceback
from dataclasses import asdict

import torch

from vllm.config import get_layers_from_vllm_config, replace, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.worker.gpu.spec_decode.async_draft.adapters import (
    Gemma4MTPAsyncDraftAdapter,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.cache import BranchCache, CachedBranch
from vllm.v1.worker.gpu.spec_decode.async_draft.ipc import (
    AsyncDraftBatch,
    AsyncDraftResponse,
    make_ring_slots,
)

BLOCK_SIZE = 16


def target_layer_indices(config) -> dict[str, int]:
    text = config.model_config.hf_config.get_text_config()
    count = len(text.layer_types) - text.num_kv_shared_layers
    return {kind: i for i, kind in enumerate(text.layer_types[:count])}


def target_kv_layers(config):
    layers = get_layers_from_vllm_config(config, Attention)
    result = {}
    for kind, index in target_layer_indices(config).items():
        matches = [
            (name, layer)
            for name, layer in layers.items()
            if name.endswith(f".layers.{index}.self_attn.attn")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one Gemma4 KV source for {kind}: {matches}")
        result[kind] = matches[0]
    return result


def pack_target_kv(cache, block_table, length):
    if cache.ndim != 5 or cache.shape[1] != 2:
        raise ValueError("Gemma4 Async requires Triton N,2,B,H,D KV layout")
    positions = torch.arange(length, device=cache.device)
    blocks = block_table[positions // cache.shape[2]].long()
    return cache[blocks, :, positions % cache.shape[2]]


def copy_target_kv(layers, metadata, slot, batch, dummy):
    size = 0
    length = int(slot.seq_lens[0].item())
    for kind, destination in slot.target_kv.items():
        if dummy:
            destination.zero_()
        else:
            if not 0 < length <= destination.shape[0] * destination.shape[2]:
                raise ValueError("Target sequence exceeds Gemma4 KV snapshot capacity")
            name, layer = layers[kind]
            meta = metadata[name]
            if not isinstance(meta, TritonAttentionMetadata):
                raise ValueError("Gemma4 Async requires TRITON_ATTN")
            source = pack_target_kv(layer.kv_cache, meta.block_table[0], length)
            # Ring snapshots have independent storage on the Draft device.
            packed = destination.transpose(1, 2).reshape(-1, 2, *destination.shape[-2:])
            packed[:length].copy_(source, non_blocking=True)
            size += source.numel() * source.element_size()
    torch.cuda.synchronize(slot.input_ids.device)
    return size


class Gemma4Draft:
    def __init__(self, config, device):
        from vllm.model_executor.model_loader import get_model
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        self.adapter = Gemma4MTPAsyncDraftAdapter(config)
        self.width = config.speculative_config.num_speculative_tokens
        self.fan_out = self.adapter.fan_out()
        self.device = device
        self.config = replace(
            config,
            model_config=config.speculative_config.draft_model_config,
            attention_config=replace(
                config.attention_config, backend=AttentionBackendEnum.TRITON_ATTN
            ),
        )
        self.config.kernel_config.ir_op_priority.set_default()
        from vllm.ir import set_default_torch_wrap

        set_default_torch_wrap(config.compilation_config.ir_enable_torch_wrap)
        with set_current_vllm_config(self.config):
            self.model = get_model(
                vllm_config=self.config,
                model_config=config.speculative_config.draft_model_config,
                load_config=config.speculative_config.draft_load_config,
            )
            self.materialized = self.adapter.materialize_shared_weights(self.model)
        self.layers = get_layers_from_vllm_config(self.config, Attention)
        kinds = self.model.config.get_text_config().layer_types
        self.kinds = dict(zip(self.layers, kinds))
        if len(self.layers) != len(kinds):
            raise ValueError("Assistant attention registration does not match layers")
        self.shapes = {}
        self.scratch = {}
        for name, layer in self.layers.items():
            kind = self.kinds[name]
            # This suppresses Q-only dummy K/V writes; cache binding is local.
            layer.kv_sharing_target_layer_name = name
            layer.impl.kv_sharing_target_layer_name = name
            self.shapes[kind] = (layer.num_kv_heads, layer.head_size)
            threshold = 128 // layer.num_kv_heads
            self.scratch[name] = (
                torch.empty(
                    threshold,
                    layer.num_heads,
                    16,
                    layer.head_size,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    threshold, layer.num_heads, 16, dtype=torch.float32, device=device
                ),
                torch.empty(
                    threshold, layer.num_heads, 16, dtype=torch.float32, device=device
                ),
            )

    def forward(self, ids, positions, hidden, snapshots, seq_len, query_len=None):
        n = ids.numel()
        qlen = n if query_len is None else query_len
        # qlen=n is one real query; qlen=1 is a batch of independent branches.
        count = n // qlen
        metadata = {}
        for name, layer in self.layers.items():
            layer.kv_cache = snapshots[self.kinds[name]]
            blocks = layer.kv_cache.shape[0]
            output, maximum, expsum = self.scratch[name]
            metadata[name] = TritonAttentionMetadata(
                num_actual_tokens=n,
                max_query_len=qlen,
                query_start_loc=torch.arange(
                    count + 1, device=self.device, dtype=torch.int32
                )
                * qlen,
                max_seq_len=seq_len,
                seq_lens=torch.full(
                    (count,), seq_len, device=self.device, dtype=torch.int32
                ),
                block_table=torch.arange(blocks, device=self.device, dtype=torch.int32)
                .expand(count, -1)
                .contiguous(),
                slot_mapping=torch.full(
                    (n,), -1, device=self.device, dtype=torch.int64
                ),
                seq_threshold_3D=output.shape[0],
                num_par_softmax_segments=16,
                softmax_segm_output=output,
                softmax_segm_max=maximum,
                softmax_segm_expsum=expsum,
                causal=True,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )
        with set_forward_context(
            metadata,
            self.config,
            num_tokens=n,
            slot_mapping={name: m.slot_mapping for name, m in metadata.items()},
        ):
            return self.model(
                input_ids=ids.contiguous(),
                positions=positions.contiguous(),
                hidden_states=hidden.contiguous(),
            )

    def decode(self, token, position, hidden, snapshots, seq_len, steps):
        if steps == 0:
            return token.new_empty((token.numel(), 0)), [], []
        tokens, logits, feedback = [], [], []
        for _ in range(steps):
            last, hidden = self.forward(
                token, position, hidden, snapshots, seq_len, query_len=1
            )
            scores = self.model.compute_logits(last)
            token = scores.argmax(-1)
            tokens.append(token)
            logits.append(scores)
            feedback.append(hidden)
        return torch.stack(tokens, 1), logits, feedback

    def fresh(self, slot, batch, num_steps=None):
        width = self.width if num_steps is None else num_steps
        n = batch.num_tokens
        rejected = int(slot.num_rejected[0].item())
        last_index = n - rejected - 1
        ids = slot.input_ids[:n].clone()
        ids[:last_index] = slot.input_ids[1 : last_index + 1]
        ids[last_index] = (
            slot.last_sampled[0]
            if slot.num_sampled[0] > 0
            else slot.next_prefill_tokens[0]
        )
        seq_len = int(slot.seq_lens[0].item())
        last, hidden = self.forward(
            ids,
            slot.positions[:n],
            slot.conditioning_states[:n],
            slot.target_kv,
            seq_len,
        )
        scores = self.model.compute_logits(last[last_index : last_index + 1])
        first = scores.argmax(-1)
        position = slot.positions[last_index : last_index + 1]
        feedback = hidden[last_index : last_index + 1]
        rest, logits, states = self.decode(
            first, position, feedback, slot.target_kv, seq_len, width - 1
        )
        tokens = torch.cat((first[:, None], rest), 1)
        return tokens, [scores, *logits], [feedback, *states], position, seq_len

    def build(self, cache, slot, batch, returned):
        guard = (
            {kind: kv.clone() for kind, kv in slot.target_kv.items()}
            if os.getenv("REPLAYSSM_SPEC_DECODE_TRACE_LOGITS") == "1"
            else {}
        )
        # Replay the actually returned prefix against the newest real snapshot.
        _, logits, states, position, seq_len = self.fresh(slot, batch, num_steps=1)
        scores, hidden = logits[0], states[0]
        for depth in range(self.width + 1):
            candidates = scores.clone()
            if depth < self.width:
                candidates.scatter_(1, returned[:, depth : depth + 1], -torch.inf)
            recovery = candidates.topk(self.fan_out, dim=-1).indices.flatten()
            tokens, _, _ = self.decode(
                recovery,
                (position + depth + 1).expand(self.fan_out),
                hidden.expand(self.fan_out, -1),
                slot.target_kv,
                seq_len,
                self.width,
            )
            event = torch.cuda.Event()
            for index, token in enumerate(recovery.tolist()):
                key = (
                    batch.engine_instance_id,
                    batch.req_ids[0],
                    batch.request_epochs[0],
                    depth,
                    token,
                )
                cache.add(
                    key,
                    CachedBranch(
                        str(key),
                        tokens[index : index + 1].clone(),
                        completion_event=event,
                    ),
                )
            event.record()
            if depth < self.width:
                last, hidden = self.forward(
                    returned[:, depth],
                    position,
                    hidden,
                    slot.target_kv,
                    seq_len,
                    query_len=1,
                )
                scores = self.model.compute_logits(last)
        torch.cuda.synchronize(self.device)
        for kind, snapshot in guard.items():
            if not torch.equal(snapshot, slot.target_kv[kind]):
                raise RuntimeError(
                    "Gemma4 Q-only branch mutated its Target KV snapshot"
                )


def run_gemma4_child(connection, config, physical_device_id, init_port):
    from vllm.distributed import destroy_model_parallel
    from vllm.platforms import current_platform
    from vllm.platforms.interface import set_assigned_physical_gpu_ids
    from vllm.v1.worker.gpu.spec_decode.async_draft.runtime import (
        _wait_for_shutdown_after_error,
    )
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        set_assigned_physical_gpu_ids([physical_device_id])
        device = torch.device(
            f"cuda:{current_platform.logical_device_id_to_visible_device_id(0)}"
        )
        torch.cuda.set_device(device)
        config = copy.deepcopy(config)
        config.speculative_config.async_draft_device = None
        config.parallel_config.assigned_physical_gpu_ids = [physical_device_id]
        config.parallel_config.distributed_executor_backend = "uni"
        with set_current_vllm_config(config):
            init_worker_distributed_environment(
                config,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{init_port}",
                backend="nccl",
            )
            init_workspace_manager(device)
            with torch.inference_mode():
                draft = Gemma4Draft(config, device)
        layout = draft.adapter.target_state_layout()
        slots = make_ring_slots(
            max_num_reqs=1,
            max_num_tokens=config.scheduler_config.max_num_batched_tokens,
            num_speculative_tokens=draft.width,
            conditioning_size=layout.width,
            dtype=config.model_config.dtype,
            device=device,
        )
        blocks = (config.model_config.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for slot in slots:
            # Token-major physical layout makes packing a view, not a copy.
            slot.target_kv = {
                kind: torch.zeros(
                    blocks,
                    BLOCK_SIZE,
                    2,
                    heads,
                    dim,
                    dtype=config.model_config.dtype,
                    device=device,
                ).transpose(1, 2)
                for kind, (heads, dim) in draft.shapes.items()
            }
        events = [torch.cuda.Event(interprocess=True) for _ in slots]
        handles = [event.ipc_handle() for event in events]
        torch.cuda.synchronize(device)
        connection.send(
            dict(
                status="ready",
                ring_slots=slots,
                response_event_handles=handles,
                conditioning_splits=layout.splits,
                conditioning_layout=layout.name,
                adapter=type(draft.adapter).__name__,
                fan_out=draft.fan_out,
                target_verify_width=draft.width,
                proposal_execution_width=draft.width,
                branch_backbone_width=draft.width,
                materialized_weights=draft.materialized,
                pid=os.getpid(),
                physical_device_id=physical_device_id,
                kv_num_blocks=blocks,
                draft_kv_block_sizes=[BLOCK_SIZE],
                target_replayssm_owned_by_child=False,
            )
        )
        cache = BranchCache()
        generations = [-1] * len(slots)
        pending = 0.0
        with torch.inference_mode():
            while True:
                message = connection.recv()
                command = message["command"]
                if command == "shutdown":
                    connection.send(dict(status="shutdown"))
                    break
                if command in ("release", "reset"):
                    cache.discard_requests(message["request_ids"])
                    connection.send(
                        dict(status="ok", metrics=dict(branch_build_seconds=pending))
                    )
                    pending = 0.0
                    continue
                if command != "propose":
                    raise ValueError(f"Unknown Gemma4 Async command {command}")
                batch = AsyncDraftBatch(**message["batch"])
                if batch.num_reqs != 1 or not 0 <= batch.slot < len(slots):
                    raise ValueError("Invalid Gemma4 Async batch")
                if batch.generation <= generations[batch.slot]:
                    raise ValueError("Stale Gemma4 Async generation")
                generations[batch.slot] = batch.generation
                slot = slots[batch.slot]
                depth = int(slot.num_sampled[0].item()) - 1
                key = (
                    batch.engine_instance_id,
                    batch.req_ids[0],
                    batch.request_epochs[0],
                    depth,
                    int(slot.last_sampled[0].item()),
                )
                branch = cache.pop(key)
                cache.discard_request(batch.req_ids[0])
                hit = (
                    branch is not None
                    and not batch.transient
                    and os.getenv("ASYNC_DRAFT_FORCE_JIT") != "1"
                )
                start = time.perf_counter()
                trace = None
                if batch.transient:
                    tokens = slot.draft_tokens[:1].zero_()
                elif hit:
                    branch.completion_event.synchronize()
                    tokens = branch.tokens
                else:
                    tokens, logits, *_ = draft.fresh(slot, batch)
                    if os.getenv("REPLAYSSM_SPEC_DECODE_TRACE_LOGITS") == "1":
                        top = []
                        for scores in logits:
                            values, ids = scores[0].float().topk(2)
                            top.append(
                                dict(token_ids=ids.tolist(), logits=values.tolist())
                            )
                        trace = [dict(draft_top2=top)]
                slot.draft_tokens[:1].copy_(tokens)
                events[batch.slot].record()
                connection.send(
                    asdict(
                        AsyncDraftResponse(
                            batch.generation,
                            batch.slot,
                            "ok",
                            num_reqs=1,
                            cache_hit_indices=[0] if hit else [],
                            trace_top2=trace,
                            metrics=dict(
                                cache_hits=int(hit),
                                cache_misses=int(not hit),
                                jit_fallbacks=int(not hit),
                                branch_build_seconds=pending,
                                wait_seconds=time.perf_counter() - start,
                            ),
                        )
                    )
                )
                pending = 0.0
                if not batch.transient and os.getenv("ASYNC_DRAFT_FORCE_JIT") != "1":
                    start = time.perf_counter()
                    draft.build(cache, slot, batch, tokens)
                    pending = time.perf_counter() - start
    except BaseException as error:
        connection.send(
            dict(
                status="fatal",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        )
        _wait_for_shutdown_after_error(connection)
    finally:
        destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
