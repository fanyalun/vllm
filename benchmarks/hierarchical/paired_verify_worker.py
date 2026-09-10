# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-input top-4/top-8 graph timing; diagnostic request latency is invalid."""

import torch
from cycle_worker import CycleWorker


class PairedVerifyWorker(CycleWorker):
    def begin_cycle_measurement(self):
        super().begin_cycle_measurement()
        self._paired_rows = []
        spec = self.model_runner.speculator
        original = spec._verify

        def verify(batch, metadata, slots):
            if self._cycle_active and not self._paired_rows:
                before = spec.state.snapshot()
                original_h = spec.config.moe_skip_top_h
                graphs = {}
                outputs = {}
                try:
                    for h in (4, 8):
                        spec.config.moe_skip_top_h = h
                        for _ in range(3):
                            spec.state.restore(before)
                            spec._verify_eager(batch, metadata, slots)
                        spec.state.restore(before)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            outputs[h] = spec._verify_eager(batch, metadata, slots)
                        graphs[h] = graph
                    for repeat in range(12):
                        for h in (4, 8) if repeat % 2 == 0 else (8, 4):
                            spec.state.restore(before)
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            graphs[h].replay()
                            end.record()
                            end.synchronize()
                            self._paired_rows.append(
                                {
                                    "repeat": repeat,
                                    "top_h": h,
                                    "width": batch.num_tokens,
                                    "positions": batch.positions.tolist(),
                                    "input_ids": batch.input_ids.tolist(),
                                    "stream_ms": start.elapsed_time(end),
                                }
                            )
                finally:
                    spec.config.moe_skip_top_h = original_h
                    spec.state.restore(before)
                # Rewrite the provisional attention suffix using normal P.
            return original(batch, metadata, slots)

        spec._verify = verify

    def collect_cycle_measurement(self):
        measured = super().collect_cycle_measurement()
        measured["paired_verify"] = self._paired_rows
        measured["diagnostic_only_latency_invalid"] = True
        return measured
