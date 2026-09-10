# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare exact and approximate GDN on the same five-token prefix."""

import copy
import os
from pathlib import Path

import torch
from forward_stage_worker import ForwardStageWorker

from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class GdnMeanStageWorker(ForwardStageWorker):
    @torch.inference_mode()
    def _measure_pair(self, spec, original_batch):
        tokens = original_batch.input_ids.clone()
        assert tokens.numel() == 5
        position = int(original_batch.positions[0].item())
        original_state, original_config = spec.state, spec.config
        initial = original_state.snapshot()
        graphs, states, snapshots = {}, {}, {}
        cases = [("none", 8), ("none", 4), ("ssm_mean", 4), ("input_mean", 4)]
        try:
            for mode, top_h in cases:
                key = mode, top_h
                state = PreverifyState(spec.model, 5, spec.device, mode)
                slot = 1 if mode == "none" else 0
                for name, (conv, ssm) in state.caches.items():
                    state._copy_conv(conv[slot : slot + 1], initial[name][0], 0)
                    ssm[slot : slot + 1].copy_(initial[name][1])
                spec.state = states[key] = state
                snapshots[key] = state.snapshot()
                spec.config = copy.copy(original_config)
                spec.config.preverify_gdn_mode = mode
                spec.config.moe_skip_top_h = top_h
                batch, metadata, slots = spec._batch(original_batch, position, tokens)
                for _ in range(3):
                    state.restore(snapshots[key])
                    spec._verify_eager(batch, metadata, slots)
                torch.accelerator.synchronize()
                for detailed in (False, True):
                    state.restore(snapshots[key])
                    spans = []
                    undo = self._instrument(spec, spans) if detailed else []
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph):
                            output = spec._verify_eager(batch, metadata, slots)
                    finally:
                        for obj, method, original in reversed(undo):
                            setattr(obj, method, original)
                    graphs[mode, top_h, detailed] = graph, output, spans
            reference = {}
            for repeat in range(23):
                for mode, top_h in cases if repeat % 2 == 0 else cases[::-1]:
                    for detailed in (False, True):
                        states[mode, top_h].restore(snapshots[mode, top_h])
                        graph, output, spans = graphs[mode, top_h, detailed]
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        predictions = output[0].tolist()
                        key = mode, top_h
                        reference.setdefault(key, predictions)
                        assert predictions == reference[key]
                        if repeat < 3:
                            continue
                        self._stage_rows.append(
                            {
                                "width": 5,
                                "position": position,
                                "input_ids": tokens.tolist(),
                                "mode": mode,
                                "top_h": top_h,
                                "detailed": detailed,
                                "repeat": repeat - 3,
                                "total_ms": start.elapsed_time(end),
                                "predictions": predictions,
                                "spans": [
                                    {
                                        "phase": phase,
                                        "name": name,
                                        "start_ms": start.elapsed_time(a),
                                        "end_ms": start.elapsed_time(b),
                                    }
                                    for phase, name, a, b in spans
                                ],
                            }
                        )
            self._measured_prefixes = getattr(self, "_measured_prefixes", 0) + 1
            if self._measured_prefixes == 3:
                root = os.environ["FORWARD_STAGE_OUTPUT"]
                try:
                    for mode, top_h in cases:
                        spec.state = states[mode, top_h]
                        spec.config = copy.copy(original_config)
                        spec.config.preverify_gdn_mode = mode
                        spec.config.moe_skip_top_h = top_h
                        batch, metadata, slots = spec._batch(
                            original_batch, position, tokens
                        )
                        path = Path(root) / f"{mode}_h{top_h}"
                        path.mkdir(parents=True, exist_ok=True)
                        os.environ["FORWARD_STAGE_OUTPUT"] = str(path)
                        self._kernel_trace(
                            spec,
                            batch,
                            metadata,
                            slots,
                            snapshots[mode, top_h],
                            {(top_h, False): graphs[mode, top_h, False]},
                            top_h,
                        )
                finally:
                    os.environ["FORWARD_STAGE_OUTPUT"] = root
                self._kernel_traced = True
        finally:
            spec.config = original_config
            spec.state = original_state
            spec.state.restore(initial)
            spec._batch(original_batch, position, tokens)
