# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-length GDN-only drafting and paired full-forward measurements."""

import copy
import inspect
from dataclasses import replace

import numpy as np
import torch

from benchmarks.hierarchical.batch_worker import BatchWorker
from vllm.distributed.parallel_state import graph_capture
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.hierarchical.batched import make_batch
from vllm.v1.worker.gpu.spec_decode.hierarchical.batched_state import (
    BatchedPreverifyState,
)
from vllm.v1.worker.gpu.spec_decode.hierarchical.speculator import (
    HierarchicalSpeculator,
    refresh_graph_metadata,
)


class _GdnDraftAdapter:
    supports_mm_inputs = False
    draft_logits = None
    _set_gdn_metadata = HierarchicalSpeculator._set_gdn_metadata
    _verify_eager = HierarchicalSpeculator._verify_eager

    def __init__(self, runner):
        self.vllm_config = runner.vllm_config
        self.config = copy.copy(runner.vllm_config.speculative_config)
        self.config.moe_skip_top_h = 8
        self.device = runner.device
        self.model = getattr(runner.model, "language_model", runner.model)
        self.logits_model = runner.model
        self.model_state = runner.model_state
        self.kv_cache_config = runner.kv_cache_config
        self.block_tables = runner.block_tables
        self.max_num_reqs = runner.max_num_reqs
        self.capacity = runner.num_speculative_steps
        self.buffers = InputBuffers(
            self.max_num_reqs,
            runner.vllm_config.scheduler_config.max_num_batched_tokens,
            self.device,
        )
        self.attn_groups, _, _ = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        self._draft_lengths = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.draft_lengths = self._draft_lengths
        self.draft_tokens = torch.full(
            (self.max_num_reqs, self.capacity),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.check_preverify = False
        self.grouped_gdn = None
        self.preverify_graph_replays = 0
        self.capture_state_on_cpu = False

    def _verify(self, batch, metadata, slots):
        if not self.use_preverify_graphs:
            return self._verify_eager(batch, metadata, slots)
        key = (batch.num_reqs, batch.num_tokens)
        if key not in self.preverify_graphs:
            before = (
                {
                    name: tuple(x.cpu() for x in cache)
                    for name, cache in self.state.caches.items()
                }
                if self.capture_state_on_cpu
                else self.state.snapshot()
            )
            for _ in range(3):
                self._verify_eager(batch, metadata, slots)
                self.state.restore(before)
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(self.device) as capture,
                torch.cuda.graph(graph, stream=capture.stream),
            ):
                output = self._verify_eager(batch, metadata, slots)
            self.state.restore(before)
            self.preverify_graphs[key] = (
                graph,
                output,
                metadata,
                slots,
                self.last_logits,
                self.last_margins,
            )
        graph, output, old_metadata, old_slots, logits, margins = self.preverify_graphs[
            key
        ]
        refresh_graph_metadata(old_metadata, metadata)
        refresh_graph_metadata(old_slots, slots)
        graph.replay()
        self.last_logits, self.last_margins = logits, margins
        self.preverify_graph_replays += 1
        return output


class _LocalHeads:
    def __init__(self, layer):
        self.layer = layer
        for name in ("key_dim", "value_dim", "conv_dim", "num_k_heads", "num_v_heads"):
            setattr(self, name, getattr(layer, name) // layer.tp_size)

    def __getattr__(self, name):
        return getattr(self.layer, name)


class GdnFeasibilityWorker(BatchWorker):
    def setup_feasibility(
        self, variant="v2", length=16, graphs=False, capture_state_on_cpu=False
    ):
        self.feasibility_rows = []
        self.feasibility_events = []
        self.feasibility_audit = False
        self.feasibility_probe = False
        self.feasibility_probe_result = None
        self.feasibility_quality = False
        self.feasibility_checks = []
        runner = self.model_runner
        spec = runner.speculator
        self.feasibility_variant = variant
        if variant in ("v2", "full", "native_draft"):
            import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

            original_projected = gdn.qwen_gdn_mean_projected

            def projected(layer, *args, **kwargs):
                local = _LocalHeads(layer) if layer.tp_size > 1 else layer
                return original_projected(local, *args, **kwargs)

            gdn.qwen_gdn_mean_projected = projected
            original = spec.propose
            spec = _GdnDraftAdapter(runner)
            spec.capture_state_on_cpu = capture_state_on_cpu
            self._select_state(spec, variant)
            runner.speculator = spec
            for layer in spec.state.layers.values():
                layer.enable_mean_preverify = True
            spec.use_preverify_graphs = graphs
            signature = inspect.signature(original)

            def propose(*args, **kwargs):
                values = signature.bind(*args, **kwargs)
                values.apply_defaults()
                values = values.arguments
                batch = values["input_batch"]
                if any(
                    values[k]
                    for k in ("dummy_run", "is_profile", "skip_attn_for_dummy_run")
                ) or any(req.startswith("_warmup_") for req in batch.req_ids):
                    return original(*args, **kwargs)
                return self._fixed_propose(spec, values, length)

            spec.propose = propose
            for owner, name, label in (
                (spec, "propose", "proposal"),
                (spec, "_verify", "draft_forward"),
                (spec.state, "begin", "state_initialize"),
                (spec.state, "advance", "conv_advance"),
            ):
                self._time_method(owner, name, label)
        self._time_method(runner, "execute_model", "target_execute")
        original_sample = runner.sample

        def sample(hidden_states, batch, grammar_output):
            result = original_sample(hidden_states, batch, grammar_output)
            if self.feasibility_audit:
                sampled = result[1][: batch.num_reqs].detach().clone()
                self.feasibility_rows.append(
                    (
                        list(batch.req_ids),
                        batch.num_draft_tokens_per_req.copy()
                        if batch.num_draft_tokens_per_req is not None
                        else np.zeros(batch.num_reqs, dtype=np.int32),
                        sampled,
                        batch.has_prefill,
                        batch.num_tokens,
                    )
                )
            return result

        runner.sample = sample
        return self.feasibility_info()

    def _select_state(self, spec, variant):
        approximate = variant != "native_draft"
        spec.config.preverify_gdn_mode = "replay_tail" if approximate else "none"
        spec.config.preverify_gdn_update_policy = (
            "windowed_three_level" if approximate else "exact"
        )
        spec.state = BatchedPreverifyState(
            spec.model,
            1,
            spec.device,
            mode=spec.config.preverify_gdn_mode,
            update_policy=spec.config.preverify_gdn_update_policy,
            window_size=1,
            tau_alpha=0.95,
            tau_beta=0.0 if variant == "full" else 0.36328125,
            max_num_reqs=spec.max_num_reqs,
        )
        spec.preverify_graphs = {}

    def _time_method(self, owner, name, label):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            if not self.feasibility_audit:
                return original(*args, **kwargs)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.feasibility_events.append((label, start, end))
            return result

        setattr(owner, name, measured)

    @torch.inference_mode()
    def _fixed_propose(self, spec, values, length):
        original = values["input_batch"]
        n = original.num_reqs
        spec.draft_lengths = spec._draft_lengths[:n]
        spec.draft_lengths.zero_()
        spec.draft_tokens.fill_(-1)
        if not n:
            return spec.draft_tokens[:0]
        positions = (original.seq_lens[:n] - values["num_rejected"][:n]).cpu().tolist()
        sampled = values["num_sampled"][:n].cpu().tolist()
        limit = spec.vllm_config.model_config.max_model_len
        rows = [i for i in range(n) if sampled[i] and positions[i] < limit]
        if not rows:
            return spec.draft_tokens[:n]
        initialized = replace(
            original,
            req_ids=[original.req_ids[i] for i in rows],
            idx_mapping=original.idx_mapping[rows],
            num_reqs=len(rows),
        )
        tables = spec.block_tables.gather_block_tables(
            initialized.idx_mapping, initialized.num_reqs
        )
        spec.state.begin(spec.model_state, initialized, tables, spec.kv_cache_config)
        spec.outer_positions = dict(zip(original.req_ids, positions, strict=True))
        anchors = values["last_sampled"][original.idx_mapping[rows], 0].clone()
        active_positions = [positions[i] for i in rows]
        canonical = None
        check_cases = set()
        if self.feasibility_quality:
            covered = {
                case for check in self.feasibility_checks for case in check["cases"]
            }
            check_cases.add("initial_proposal")
            if not original.has_prefill:
                check_cases.add(
                    "full_batch_decode"
                    if n == spec.max_num_reqs
                    else "compacted_decode"
                )
                if bool(values["num_rejected"][:n].any()):
                    check_cases.add("after_rejection")
            check_cases -= covered
        checked_batch = replace(original, idx_mapping=original.idx_mapping[:n])
        if check_cases:
            canonical = self._committed_batch(checked_batch)
        counts = [0] * n
        for offset in range(min(length, limit - max(active_positions))):
            batch, metadata, slots = make_batch(
                spec,
                original,
                rows,
                [p + offset for p in active_positions],
                anchors[:, None],
            )
            if (
                self.feasibility_probe
                and not original.has_prefill
                and len(rows) == spec.max_num_reqs
            ):
                self.feasibility_probe = False
                self._paired_probe(
                    spec, original, initialized, tables, rows, active_positions, anchors
                )
                batch, metadata, slots = make_batch(
                    spec, original, rows, active_positions, anchors[:, None]
                )
            predictions, _, _ = spec._verify(batch, metadata, slots)
            anchors = predictions.clone()
            spec.draft_tokens[rows, offset] = anchors
            for row in rows:
                counts[row] += 1
            if offset + 1 < length:
                spec.state.advance(
                    torch.zeros(len(rows), dtype=torch.int64, device=spec.device)
                )
        spec.draft_lengths.copy_(
            torch.tensor(counts, device=spec.device, dtype=torch.int32)
        )
        if canonical is not None:
            after = self._committed_batch(checked_batch)
            equal = all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for name in canonical
                for a, b in zip(canonical[name], after[name], strict=True)
            )
            self.feasibility_checks.append(
                {
                    "canonical_prefix_unchanged": equal,
                    "cases": sorted(check_cases),
                    "active_requests": n,
                }
            )
            assert equal, "Private drafting changed the committed prefix"
        spec.state.invalidate()
        return spec.draft_tokens[:n]

    @torch.inference_mode()
    def _paired_probe(
        self, spec, template, initialized, tables, rows, positions, anchors
    ):
        saved_state, saved_config = spec.state, spec.config
        saved_graphs = spec.preverify_graphs
        results = {}
        references = {}
        try:
            for variant in ("native_draft", "full", "v2"):
                spec.config = copy.copy(saved_config)
                self._select_state(spec, variant)
                state = spec.state
                state.begin(spec.model_state, initialized, tables, spec.kv_cache_config)
                batch, metadata, slots = make_batch(
                    spec, template, rows, positions, anchors[:, None]
                )
                before = state.snapshot()
                for _ in range(3):
                    spec._verify_eager(batch, metadata, slots)
                    state.restore(before)
                spec._verify_eager(batch, metadata, slots)
                eager_logits = spec.last_logits.detach().clone()
                state.restore(before)
                graph = torch.cuda.CUDAGraph()
                with (
                    graph_capture(spec.device) as capture,
                    torch.cuda.graph(graph, stream=capture.stream),
                ):
                    graph_output = spec._verify_eager(batch, metadata, slots)
                state.restore(before)
                graph.replay()
                torch.accelerator.synchronize()
                references[variant] = spec.last_logits.detach().clone()
                torch.testing.assert_close(
                    references[variant], eager_logits, atol=1e-3, rtol=1e-3
                )
                states = {
                    name: cache[1][1 :: 2 if variant == "native_draft" else 1].clone()
                    if variant == "native_draft"
                    else cache[1].clone()
                    for name, cache in state.caches.items()
                }
                if variant == "native_draft":
                    native_states = states
                elif variant == "full":
                    state_error = max(
                        float((states[name] - native_states[name]).abs().max())
                        for name in states
                    )
                    state_relative_l2 = max(
                        float(
                            (states[name] - native_states[name]).norm()
                            / native_states[name].norm().clamp_min(1e-12)
                        )
                        for name in states
                    )
                times = []
                for _ in range(20):
                    state.restore(before)
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                results[variant] = {
                    "full_forward_gpu_ms": times,
                    "batch": len(rows),
                    "width": 1,
                    "graph_eager_logits_close": True,
                }
                del graph, graph_output
            full = references["full"]
            native = references["native_draft"]
            results["forced_full_check"] = {
                "argmax_equal": bool(torch.equal(full.argmax(-1), native.argmax(-1))),
                "logit_max_abs": float((full - native).abs().max()),
                "state_max_abs": state_error,
                "state_relative_l2": state_relative_l2,
                "logits_close": bool(torch.allclose(full, native, atol=0.1, rtol=0.01)),
            }
        finally:
            spec.state, spec.config = saved_state, saved_config
            spec.preverify_graphs = saved_graphs
        self.feasibility_probe_result = results

    def feasibility_measure(self, audit=False, probe=False, quality=False):
        torch.accelerator.synchronize()
        self.feasibility_rows = []
        self.feasibility_events = []
        self.feasibility_audit = audit
        self.feasibility_probe = probe
        self.feasibility_quality = quality
        torch.accelerator.reset_peak_memory_stats()

    def feasibility_collect(self):
        torch.accelerator.synchronize()
        self.feasibility_audit = False
        return {
            "stages": [
                {"stage": name, "gpu_ms": a.elapsed_time(b)}
                for name, a, b in self.feasibility_events
            ],
            "steps": [
                {
                    "request_ids": ids,
                    "proposed": proposed.tolist(),
                    "sampled": sampled.cpu().tolist(),
                    "has_prefill": prefill,
                    "input_tokens": tokens,
                }
                for ids, proposed, sampled, prefill, tokens in self.feasibility_rows
            ],
            "paired_forward": self.feasibility_probe_result,
            "checks": self.feasibility_checks,
            **self.feasibility_info(),
        }

    def feasibility_info(self):
        runner = self.model_runner
        spec = runner.speculator
        return {
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "num_cache_blocks": runner.kv_cache_config.num_blocks,
            "private_bytes": sum(
                x.numel() * x.element_size()
                for pair in spec.state.caches.values()
                for x in pair
            )
            if hasattr(spec, "state")
            else 0,
        }

    def synchronize_feasibility(self):
        torch.accelerator.synchronize()
