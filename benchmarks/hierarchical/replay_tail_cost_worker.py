# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-input performance experiment; no production configuration changes."""

import copy

import torch

from vllm.v1.worker.gpu.spec_decode.hierarchical import speculator as impl
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class ReplayTailCostWorker:
    def begin_reuse(self):
        if not hasattr(self, "_reuse_original"):
            spec = self.model_runner.speculator
            self._reuse_original = spec._verify

            def verify(batch, metadata, slots):
                if self._reuse_active and not self._reuse_rows:
                    self._measure(spec, batch)
                return self._reuse_original(batch, metadata, slots)

            spec._verify = verify
        self._reuse_rows = []
        self._reuse_active = True

    def collect_reuse(self):
        self._reuse_active = False
        return {
            "rows": self._reuse_rows,
            "checks": self._reuse_checks,
            "input_ids": self._reuse_ids,
            "position": self._reuse_position,
        }

    @torch.inference_mode()
    def _measure(self, spec, original_batch):
        tokens = original_batch.input_ids.clone()
        assert tokens.numel() == 5
        position = int(original_batch.positions[0])
        self._reuse_ids, self._reuse_position = tokens.tolist(), position
        original_state, original_config = spec.state, spec.config
        initial = original_state.snapshot()
        states, snapshots, graphs = {}, {}, {}
        inputs = {}
        flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        self._reuse_checks = []
        try:
            for case in ("baseline", "recurrent", "optimized"):
                mode = "none" if case == "baseline" else "replay_tail"
                state = PreverifyState(spec.model, 5, spec.device, mode)
                slot = 1 if mode == "none" else 0
                for name, (conv, ssm) in state.caches.items():
                    state._copy_conv(conv[slot : slot + 1], initial[name][0], 0)
                    ssm[slot : slot + 1].copy_(initial[name][1])
                if case == "recurrent":

                    def advance(accepted, state=state):
                        for conv, _ in state.caches.values():
                            state._copy_conv(conv, conv, accepted)

                    state.advance = advance
                spec.state = states[case] = state
                snapshots[case] = state.snapshot()
                spec.config = copy.copy(original_config)
                spec.config.preverify_gdn_mode = mode
                batch, metadata, slots = spec._batch(original_batch, position, tokens)
                if case == "baseline":
                    handles = []
                    for name, layer in state.layers.items():

                        def capture(module, args, name=name):
                            inputs[name] = args[0].clone()

                        handles.append(
                            layer.in_proj_qkvz.register_forward_pre_hook(capture)
                        )
                    spec._verify_eager(batch, metadata, slots)
                    for handle in handles:
                        handle.remove()
                    assert len(inputs) == len(state.layers) == 30
                    state.restore(snapshots[case])

                def isolated_gdn(
                    state=state, metadata=metadata, slots=slots, mode=mode
                ):
                    with (
                        state.activate(),
                        impl.use_workspace_lane(0),
                        impl.set_forward_context(
                            metadata,
                            spec.vllm_config,
                            num_tokens=5,
                            cudagraph_runtime_mode=impl.CUDAGraphMode.NONE,
                            slot_mapping=slots,
                            additional_forward_kwargs={"preverify_gdn_mode": mode},
                        ),
                    ):
                        return [
                            layer.forward_cuda(inputs[name])
                            for name, layer in state.layers.items()
                        ]

                for boundary in ("forward", "gdn", "advance", "combined"):

                    def execute(
                        boundary=boundary,
                        state=state,
                        batch=batch,
                        metadata=metadata,
                        slots=slots,
                        isolated_gdn=isolated_gdn,
                    ):
                        if boundary == "gdn":
                            return isolated_gdn()
                        output = None
                        if boundary in ("forward", "combined"):
                            output = spec._verify_eager(batch, metadata, slots)
                        if boundary in ("advance", "combined"):
                            state.advance(2)
                        return output

                    for _ in range(3):
                        state.restore(snapshots[case])
                        execute()
                    state.restore(snapshots[case])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = execute()
                    graphs[case, boundary] = graph, output
            reference = {}
            for repeat in range(35):
                order = ("baseline", "recurrent", "optimized")
                if repeat % 2:
                    order = order[::-1]
                for case in order:
                    for boundary in ("forward", "gdn", "advance", "combined"):
                        state = states[case]
                        state.restore(snapshots[case])
                        if boundary == "advance":
                            graphs[case, "forward"][0].replay()
                        flush.zero_()
                        graph, output = graphs[case, boundary]
                        start, end = [
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        ]
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        if boundary in ("forward", "combined"):
                            predictions = output[0].tolist()
                            reference.setdefault((case, boundary), predictions)
                            assert predictions == reference[case, boundary]
                        if repeat >= 5:
                            self._reuse_rows.append(
                                {
                                    "case": case,
                                    "boundary": boundary,
                                    "repeat": repeat - 5,
                                    "ms": start.elapsed_time(end),
                                }
                            )
            self._reuse_checks = {
                "gates": "per_token",
                "repeatable_predictions": True,
                "gdn_layers": len(inputs),
                "accepted_drafts_for_advance": 2,
            }
        finally:
            spec.config, spec.state = original_config, original_state
            original_state.restore(initial)
            spec._batch(original_batch, position, tokens)
