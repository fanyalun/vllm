# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime case switching with private states and graphs for every policy."""

import torch
from replay_tail_worker import ReplayTailWorker

from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class ThreeLevelWorker(ReplayTailWorker):
    def begin_replay_audit(self):
        super().begin_replay_audit()
        self.model_runner.speculator.reset_policy_metrics()

    def collect_replay_audit(self):
        result = super().collect_replay_audit()
        result["policy_metrics"] = dict(self.model_runner.speculator.policy_metrics)
        return result

    def export_three_level_kernels(self, directory):
        import json
        from pathlib import Path

        from vllm.model_executor.layers.mamba.gdn import replay_tail_update as impl

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for fn in (
            impl._replay_tail_update,
            impl._windowed_update,
            impl._advance_conv_many,
            impl._begin_private_states,
        ):
            for cache in fn.device_caches.values():
                for kernel in cache[0].values():
                    name = f"kernel_{len(rows)}.ptx"
                    (root / name).write_text(kernel.asm["ptx"])
                    rows.append(
                        dict(
                            file=name,
                            name=kernel.name,
                            registers=kernel.n_regs,
                            spills=kernel.n_spills,
                            shared_bytes=kernel.metadata.shared,
                            constants=str(
                                getattr(getattr(kernel, "src", None), "constants", {})
                            ),
                        )
                    )
        (root / "metadata.json").write_text(json.dumps(rows, indent=2))
        return len(rows)

    def set_three_level_jit_guard(self, strict):
        from vllm.utils import jit_monitor

        jit_monitor._mode = "error" if strict else "warn"

    def begin_three_profile(self):
        self._three_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
        )
        self._three_profiler.start()

    def end_three_profile(self, path):
        self._three_profiler.stop()
        self._three_profiler.export_chrome_trace(path)
        del self._three_profiler

    def begin_action_audit(self):
        self._replay_audit = False
        state = self.model_runner.speculator.state
        if not hasattr(state, "_action_counts_buffer"):
            state._action_counts_buffer = torch.zeros(
                8 if state.windowed else 3, dtype=torch.int64, device="cuda"
            )
        state.action_counts = state._action_counts_buffer
        state.action_counts.zero_()

    def collect_action_audit(self):
        state = self.model_runner.speculator.state
        result = dict(
            forward_action_counts=state.action_counts.tolist(),
            action_order=["full", "decay", "skip"],
            repair_included=False,
        )
        if state.windowed:
            values = state.action_counts.tolist()
            result.update(
                forward_action_counts=values[:3],
                head_window_counts=values[3:6],
                unchanged_heads=values[6],
                total_heads=values[7],
            )
        state.action_counts = None
        return result

    def begin_three_level_cost(self, path):
        from three_level_cost import measure

        spec = self.model_runner.speculator
        original = spec._verify
        self._three_cost_pending = True

        def verify(batch, *args):
            if self._three_cost_pending and batch.num_tokens == 5:
                measure(spec, batch, path)
                self._three_cost_pending = False
            return original(batch, *args)

        spec._verify = verify

    def capture_three_level_inputs(self, path):
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as impl

        original = impl.replay_tail_update
        values = []

        def capture(*args, **kwargs):
            if len(values) < 30:
                values.append(dict(values=[x.detach().cpu().clone() for x in args]))
                if len(values) == 30:
                    torch.save(values, path)
                    impl.replay_tail_update = original
            return original(*args, **kwargs)

        impl.replay_tail_update = capture

    def set_replay_case(self, case):
        spec = self.model_runner.speculator
        if not hasattr(self, "_three_cases"):
            super().set_replay_case("none")
            self._three_cases = {}
            self._three_checked = set()
            verify_original = spec._verify

            def checked_verify(batch, *args):
                key = self._replay_case, batch.num_tokens, spec.state.direction
                if key in self._three_checked:
                    return verify_original(batch, *args)
                state = spec.state
                if state.windowed and not getattr(
                    state, "initialization_checked", False
                ):
                    from vllm.model_executor.layers.mamba.mamba_utils import (
                        is_conv_state_dim_first,
                    )

                    request = int(batch.idx_mapping[0])
                    model_state = spec.model_state
                    bias = int(model_state.num_accepted_tokens_gpu[request]) - 1
                    source = (
                        int(model_state._mamba_state_idx_gpu[request])
                        if model_state._align_mode
                        else 0
                    )
                    tables = spec.block_tables.gather_block_tables(batch.idx_mapping, 1)
                    axis = 2 if is_conv_state_dim_first() else 1
                    for gid, group in enumerate(spec.kv_cache_config.kv_cache_groups):
                        for name in group.layer_names:
                            if name not in state.layers:
                                continue
                            conv, temporal = state.layers[name].kv_cache
                            private_conv, private_state = state.caches[name]
                            conv_idx = int(tables[gid][0, source])
                            ssm_idx = int(tables[gid][0, source + bias])
                            expected = temporal[ssm_idx : ssm_idx + 1]
                            assert torch.equal(private_state, expected)
                            positions = (
                                torch.arange(
                                    private_conv.shape[axis], device=conv.device
                                )
                                + bias
                            ).clamp(max=conv.shape[axis] - 1)
                            assert torch.equal(
                                private_conv,
                                conv[conv_idx : conv_idx + 1].index_select(
                                    axis, positions
                                ),
                            )
                    state.initialization_checked = True
                before = spec.state.snapshot()
                result = verify_original(batch, *args)
                after = spec.state.snapshot()
                tails = {name: x.clone() for name, x in spec.state.tails.items()}
                predictions = result[0].clone()
                logits, margins = spec.last_logits, spec.last_margins
                expected_logits = logits.clone()
                spec.state.restore(before)
                reference = spec._verify_eager(batch, *args)
                torch.testing.assert_close(reference[0], predictions, rtol=0, atol=0)
                torch.testing.assert_close(
                    spec.last_logits, expected_logits, rtol=1e-3, atol=1e-3
                )
                for name, cache in spec.state.caches.items():
                    for actual, expected in zip(cache, after[name], strict=True):
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-3, atol=1e-3
                        )
                for name, expected in tails.items():
                    torch.testing.assert_close(
                        spec.state.tails[name], expected, rtol=1e-3, atol=1e-3
                    )
                spec.state.restore(after)
                spec.last_logits, spec.last_margins = logits, margins
                self._three_checked.add(key)
                return result

            spec._verify = checked_verify
            original = spec.small.propose

            def draft(*args, _fn=original, **kwargs):
                if not self._replay_audit:
                    return _fn(*args, **kwargs)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                result = _fn(*args, **kwargs)
                end.record()
                self._replay_spans.append(("draft", start, end))
                return result

            spec.small.propose = draft
        update, tail, stop = case.split(":")
        mode = "none" if update == "none" else "replay_tail"
        windowed = update.startswith("windowed")
        policy = (
            "windowed_three_level"
            if windowed
            else ("three_level_p50" if update.startswith("three_level") else "exact")
        )
        if case not in self._three_cases:
            state = PreverifyState(
                spec.model,
                spec.depth + 1,
                spec.device,
                mode,
                policy,
                tail,
                window_size=1 if update == "windowed_token" else 5,
                tau_beta=0.0 if update == "windowed_full" else 0.36328125,
                optimization={
                    "windowed_decay": "cumulative_decay",
                    "windowed_query": "multi_query",
                    "windowed_combined": "combined",
                }.get(update, "none"),
            )
            if update == "three_level_base":
                state.execution_optimized = False
                state.kernel_tuned = False
            for name in ("begin", "advance"):
                original = getattr(state, name)

                def timed(*args, _fn=original, _name=name, **kwargs):
                    if not self._replay_audit:
                        return _fn(*args, **kwargs)
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    result = _fn(*args, **kwargs)
                    end.record()
                    self._replay_spans.append((_name, start, end))
                    return result

                setattr(state, name, timed)
            self._three_cases[case] = state, {}
        spec.state, spec.preverify_graphs = self._three_cases[case]
        spec.config.preverify_gdn_mode = mode
        spec.config.preverify_gdn_update_policy = policy
        spec.config.preverify_gdn_tail_policy = tail
        spec.config.hierarchical_stop_policy = stop
        self._replay_case = case
        self._replay_audit = False
        state = spec.state
        auxiliary = [state.state_indices, state.num_accepted]
        for attr in ("thresholds", "counts", "sequence_mask", "conv_pointers", "valid"):
            value = getattr(state, attr, None)
            if isinstance(value, torch.Tensor):
                auxiliary.append(value)
        auxiliary.extend(x for values in state.batch_constants.values() for x in values)
        auxiliary.extend(value[1] for value in state.begin_descriptors.values())
        return dict(
            case=case,
            outer_epoch=state.outer_epoch,
            request_slot_valid=state.initialized,
            initialization_checked=getattr(state, "initialization_checked", False),
            ssm_slots=[cache[1].shape[0] for cache in state.caches.values()],
            tail_buffers=len(state.tails),
            repair_buffers=len(state.repair_inputs),
            graphs=len(spec.preverify_graphs) + len(state.repair_graphs),
            auxiliary_bytes=sum(x.numel() * x.element_size() for x in auxiliary),
            worker_allocated_bytes=torch.accelerator.memory_allocated(),
            graph_eager_checked=[
                list(key[1:]) for key in sorted(self._three_checked) if key[0] == case
            ],
            private_bytes=sum(
                x.numel() * x.element_size()
                for cache in state.caches.values()
                for x in cache
            )
            + sum(x.numel() * x.element_size() for x in state.tails.values())
            + sum(
                x.numel() * x.element_size()
                for cache in state.repair_inputs.values()
                for x in cache
            ),
        )
