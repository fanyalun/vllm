# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate throughput runs from per-cycle GPU timing and kernel profiling."""

import torch


class BatchWorker:
    def set_batch_case(self, case):
        spec = self.model_runner.speculator
        if spec is None or not spec.state.windowed:
            raise ValueError("Runtime cases require windowed private state")
        if case not in ("v1", "v2", "v3", "v4d", "v4q", "v4dq"):
            raise ValueError(case)
        spec.state.window_size = 1 if case == "v2" else 5
        spec.state.optimization = {
            "v4d": "cumulative_decay",
            "v4q": "multi_query",
            "v4dq": "combined",
        }.get(case, "none")
        spec.state.thresholds[1] = 0.0 if case == "v1" else 0.36328125
        spec.state.invalidate()

    def setup_batch_measurement(self):
        self.batch_audit = False
        self.batch_spans = []
        self.batch_outer = []
        self.batch_inner = []
        self.batch_quality = False
        self.batch_checks = []
        self.batch_cycle = 0
        self.batch_pending = {}
        self.batch_seen = []
        record_batch = self.model_runner.step_timing.record_batch

        def observe_batch(batch, *args, **kwargs):
            if self.batch_audit:
                self.batch_seen.append(
                    dict(
                        num_reqs=batch.num_reqs,
                        num_tokens=batch.num_tokens,
                        has_prefill=batch.has_prefill,
                    )
                )
            return record_batch(batch, *args, **kwargs)

        self.model_runner.step_timing.record_batch = observe_batch
        spec = self.model_runner.speculator
        original_record = spec.record_verification if spec is not None else None

        def record(logits, batch, sampled):
            if self.batch_audit:
                self.batch_outer.append(
                    (
                        list(batch.req_ids),
                        batch.num_draft_tokens_per_req.copy(),
                        sampled.clone(),
                        [self.batch_pending.pop(req, None) for req in batch.req_ids],
                    )
                )
            assert original_record is not None
            return original_record(logits, batch, sampled)

        if spec is not None:
            spec.record_verification = record

        def wrap(owner, name, label):
            original = getattr(owner, name)

            def measured(*args, **kwargs):
                if not self.batch_audit:
                    return original(*args, **kwargs)
                if label == "proposal":
                    self.batch_cycle += 1
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                if label == "proposal":
                    for req in args[0].req_ids:
                        previous = self.batch_cycle_starts.pop(req, None)
                        if previous is not None:
                            cycle, previous_start = previous
                            self.batch_cycles.append(
                                (req, cycle, previous_start, start)
                            )
                result = original(*args, **kwargs)
                end.record()
                n = getattr(args[0], "num_reqs", None) if args else None
                self.batch_spans.append((label, start, end, self.batch_cycle, n))
                if label == "proposal":
                    for row in spec.last_trace:
                        row = dict(row)
                        row.setdefault("request_id", args[0].req_ids[0])
                        row.setdefault("active_batch", 1)
                        row["cycle"] = self.batch_cycle
                        self.batch_inner.append(row)
                        self.batch_pending[row["request_id"]] = self.batch_cycle
                        self.batch_cycle_starts[row["request_id"]] = (
                            self.batch_cycle,
                            start,
                        )
                return result

            setattr(owner, name, measured)

        wrap(self.model_runner.cudagraph_manager, "run_fullgraph", "target_graph")
        wrap(self.model_runner.cudagraph_manager, "run_pw_graph", "target_piecewise")
        if spec is None:
            return
        wrap(spec, "propose", "proposal")
        wrap(spec, "_verify", "preverify")
        wrap(spec.small, "propose", "draft")
        wrap(spec.state, "begin", "initialization")
        wrap(spec.state, "advance", "conv_maintenance")
        verify = spec._verify

        def checked(batch, metadata, slots):
            key = (batch.num_reqs, tuple(batch.req_ids))
            if (
                not self.batch_quality
                or len(self.batch_checks) >= 8
                or key in self.batch_checked_keys
            ):
                return verify(batch, metadata, slots)
            self.batch_checked_keys.add(key)
            canonical = self._committed_batch(batch)
            before = spec.state.snapshot()
            result = verify(batch, metadata, slots)
            after = spec.state.snapshot()
            predictions = result[0].clone()
            logits, margins = spec.last_logits, spec.last_margins
            expected_logits = logits.clone()
            spec.state.restore(before)
            reference = spec._verify_eager(batch, metadata, slots)
            torch.testing.assert_close(reference[0], predictions, atol=0, rtol=0)
            torch.testing.assert_close(
                spec.last_logits, expected_logits, atol=1e-3, rtol=1e-3
            )
            for name, cache in spec.state.caches.items():
                for actual, expected in zip(cache, after[name], strict=True):
                    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
            spec.state.restore(after)
            spec.last_logits, spec.last_margins = logits, margins
            current = self._committed_batch(batch)
            for name in canonical:
                for original, actual in zip(
                    canonical[name], current[name], strict=True
                ):
                    assert torch.equal(
                        original.view(torch.uint8), actual.view(torch.uint8)
                    ), name
            self.batch_checks.append(
                dict(
                    request_ids=list(batch.req_ids),
                    active_batch=batch.num_reqs,
                    committed_prefix_bitwise=True,
                    graph_eager_equal=True,
                )
            )
            return result

        spec._verify = checked

    def _committed_batch(self, batch):
        from vllm.model_executor.layers.attention.attention import Attention

        spec = self.model_runner.speculator
        tables = spec.block_tables.gather_block_tables(
            batch.idx_mapping, batch.num_reqs
        )
        tables = [x.cpu() for x in tables]
        requests = batch.idx_mapping.cpu().tolist()
        positions = [
            getattr(spec, "outer_positions", {}).get(req, int(p))
            for req, p in zip(batch.req_ids, batch.num_computed_tokens_np, strict=True)
        ]
        model_state = spec.model_state
        bias = model_state.num_accepted_tokens_gpu.cpu() - 1
        attention = {
            x.layer_name: x for x in spec.model.modules() if isinstance(x, Attention)
        }
        result = {}
        for gid, group in enumerate(spec.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                pieces = []
                for row, request in enumerate(requests):
                    if name in spec.state.layers:
                        source = (
                            int(model_state._mamba_state_idx_gpu[request])
                            if model_state._align_mode
                            else 0
                        )
                        indices = [
                            int(tables[gid][row, source]),
                            int(tables[gid][row, source + bias[request]]),
                        ]
                        for x, index in zip(
                            spec.state.layers[name].kv_cache, indices, strict=True
                        ):
                            pieces.append(x[index].cpu().contiguous())
                    elif name in attention:
                        cache = attention[name].kv_cache
                        assert cache.ndim == 4
                        size = cache.shape[2]
                        for start in range(0, positions[row], size):
                            index = int(tables[gid][row, start // size])
                            pieces.append(
                                cache[index, :, : min(size, positions[row] - start)]
                                .cpu()
                                .contiguous()
                            )
                if pieces:
                    result[name] = pieces
        return result

    def begin_batch_measurement(self, audit=False, profile=False, quality=False):
        from vllm.utils import jit_monitor

        jit_monitor._mode = "warn" if audit or quality else "error"
        self.batch_audit = audit
        self.batch_spans = []
        self.batch_outer = []
        self.batch_inner = []
        self.batch_quality = quality
        self.batch_checks = []
        self.batch_checked_keys = set()
        self.batch_cycle = 0
        self.batch_pending = {}
        self.batch_cycle_starts = {}
        self.batch_cycles = []
        self.batch_seen = []
        spec = self.model_runner.speculator
        if spec is not None:
            spec.reset_policy_metrics()
        if profile:
            self.batch_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            self.batch_profiler.start()

    def end_batch_measurement(self, profile_path=None):
        from vllm.utils import jit_monitor

        jit_monitor._mode = "warn"
        torch.accelerator.synchronize()
        if profile_path:
            self.batch_profiler.stop()
            self.batch_profiler.export_chrome_trace(profile_path)
            del self.batch_profiler
        self.batch_audit = False
        spec = self.model_runner.speculator
        return {
            "policy": dict(spec.policy_metrics) if spec else {},
            "private_bytes": sum(
                x.numel() * x.element_size()
                for cache in spec.state.caches.values()
                for x in cache
            )
            if spec
            else 0,
            "stages": [
                {
                    "stage": label,
                    "ms": start.elapsed_time(end),
                    "cycle": cycle,
                    "active_batch": n,
                }
                for label, start, end, cycle, n in self.batch_spans
            ],
            "inner": self.batch_inner,
            "target_batches": self.batch_seen,
            "checks": self.batch_checks,
            "cycles": [
                {"request_id": req, "cycle": cycle, "ms": start.elapsed_time(end)}
                for req, cycle, start, end in self.batch_cycles
            ],
            "outer": [
                {
                    "request_ids": ids,
                    "scheduled": scheduled.tolist(),
                    "sampled": sampled.cpu().tolist(),
                    "proposal_cycles": cycles,
                }
                for ids, scheduled, sampled, cycles in self.batch_outer
            ],
            "graph_keys": [str(x) for x in spec.preverify_graphs] if spec else [],
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "cache": {
                "num_blocks": self.model_runner.kv_cache_config.num_blocks,
                "groups": [
                    dict(
                        layers=len(group.layer_names),
                        kind=type(group.kv_cache_spec).__name__,
                        page_bytes=group.kv_cache_spec.page_size_bytes,
                        speculative_blocks=getattr(
                            group.kv_cache_spec, "num_speculative_blocks", 0
                        ),
                    )
                    for group in self.model_runner.kv_cache_config.kv_cache_groups
                ],
            },
        }
