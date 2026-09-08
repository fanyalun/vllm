# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only worker extension for asynchronous proposal interval timing."""

import time

import torch


class DraftTimingWorker:
    def begin_draft_timing(self):
        if not hasattr(self, "_draft_timing_pool"):
            self._draft_timing_pool = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(512)
            ]
            for start, end in self._draft_timing_pool:
                start.record()
                end.record()
            torch.accelerator.synchronize()
            speculator = self.model_runner.speculator
            original = speculator.propose

            def timed_propose(input_batch, *args, **kwargs):
                index = len(self._draft_timing_pending)
                if index >= len(self._draft_timing_pool):
                    raise RuntimeError("Proposal event pool exhausted")
                metadata = {
                    "proposal_index": index,
                    "has_prefill": bool(input_batch.has_prefill),
                    "num_reqs": input_batch.num_reqs,
                    "target_tokens": input_batch.num_tokens,
                    "seq_len_upper_bound": int(
                        input_batch.seq_lens_cpu_upper_bound[: input_batch.num_reqs]
                        .max()
                        .item()
                    ),
                }
                start, end = self._draft_timing_pool[index]
                start.record()
                cpu_start = time.perf_counter_ns()
                output = original(input_batch, *args, **kwargs)
                cpu_end = time.perf_counter_ns()
                end.record()
                metadata["cpu_submit_ms"] = (cpu_end - cpu_start) / 1e6
                metadata["draft_width"] = output.shape[1]
                self._draft_timing_pending.append(metadata)
                return output

            speculator.propose = timed_propose
        self._draft_timing_pending = []
        return {"event_capacity": len(self._draft_timing_pool)}

    def collect_draft_timing(self):
        torch.accelerator.synchronize()
        rows = []
        for metadata, (start, end) in zip(
            self._draft_timing_pending, self._draft_timing_pool
        ):
            rows.append({**metadata, "stream_elapsed_ms": start.elapsed_time(end)})
        self._draft_timing_pending = []
        return rows
