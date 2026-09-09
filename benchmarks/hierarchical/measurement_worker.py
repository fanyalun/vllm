# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped measurement of a complete hierarchical outer proposal."""

import time

import torch


class MeasurementWorker:
    def begin_measurement(self, mode):
        if not hasattr(self, "_measurement_installed"):
            speculator = self.model_runner.speculator
            propose = speculator.propose
            verify = speculator.record_verification

            def measured_propose(batch, *args, **kwargs):
                if self._measurement_mode != "timing":
                    return propose(batch, *args, **kwargs)
                index = len(self._proposal_rows)
                start, end = self._event_pool[index]
                start.record()
                cpu_start = time.perf_counter_ns()
                output = propose(batch, *args, **kwargs)
                cpu_end = time.perf_counter_ns()
                end.record()
                self._proposal_rows.append(
                    {
                        "proposal_index": index,
                        "has_prefill": bool(batch.has_prefill),
                        "num_reqs": batch.num_reqs,
                        "draft_width": output.shape[1],
                        "actual_candidates": sum(
                            r["emitted"] for r in speculator.last_trace
                        ),
                        "inner_rounds": len(speculator.last_trace),
                        "cpu_submit_ms": (cpu_end - cpu_start) / 1e6,
                    }
                )
                return output

            def measured_verify(logits, batch, num_sampled):
                if self._measurement_mode == "acceptance":
                    self._accepted.append(num_sampled[: batch.num_reqs].clone())
                    self._scheduled.append(int(batch.num_draft_tokens_per_req[0]))
                return verify(logits, batch, num_sampled)

            speculator.propose = measured_propose
            speculator.record_verification = measured_verify
            self._measurement_installed = True
        if mode == "timing" and not hasattr(self, "_event_pool"):
            self._event_pool = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(512)
            ]
            for start, end in self._event_pool:
                start.record()
                end.record()
            torch.accelerator.synchronize()
        self._measurement_mode = mode
        self._proposal_rows = []
        self._accepted = []
        self._scheduled = []

    def collect_measurement(self):
        torch.accelerator.synchronize()
        rows = []
        if self._measurement_mode == "timing":
            for row, (start, end) in zip(self._proposal_rows, self._event_pool):
                rows.append({**row, "stream_elapsed_ms": start.elapsed_time(end)})
        accepted = torch.cat(self._accepted).cpu().tolist() if self._accepted else []
        self._measurement_mode = None
        return {
            "proposals": rows,
            "num_sampled": accepted,
            "scheduled": self._scheduled,
        }
