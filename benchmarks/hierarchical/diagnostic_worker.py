# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, warmup-excluded NVTX tracing of hierarchical decoding."""

import torch
from cycle_worker import CycleWorker


class DiagnosticWorker(CycleWorker):
    def begin_cycle_measurement(self):
        super().begin_cycle_measurement()
        if hasattr(self, "_diagnostic_installed"):
            return
        runner = self.model_runner
        spec = runner.speculator
        self._diagnostic_capture = False

        def wrap(obj, name, label):
            original = getattr(obj, name)

            def traced(*args, **kwargs):
                if not self._cycle_active:
                    return original(*args, **kwargs)
                if label == "target_execute":
                    if self._cycle_step == 1:
                        torch.cuda.profiler.start()
                        self._diagnostic_capture = True
                    elif self._cycle_step == 11 and self._diagnostic_capture:
                        torch.cuda.profiler.stop()
                        self._diagnostic_capture = False
                with torch.cuda.nvtx.range(label):
                    return original(*args, **kwargs)

            setattr(obj, name, traced)

        wrap(runner, "execute_model", "target_execute")
        wrap(runner, "sample", "target_sample")
        wrap(spec, "propose", "proposal")
        wrap(spec.small, "propose", "small_draft")
        wrap(spec, "_batch", "preverify_metadata")
        wrap(spec, "_verify", "preverify")
        for width, (graph, *_) in spec.preverify_graphs.items():
            wrap(graph, "replay", f"preverify_graph_{width}")
        wrap(spec.state, "begin", "state_begin")
        wrap(spec.state, "advance", "state_advance")
        self._diagnostic_installed = True

    def collect_cycle_measurement(self):
        if self._diagnostic_capture:
            torch.cuda.profiler.stop()
            self._diagnostic_capture = False
        return super().collect_cycle_measurement()
