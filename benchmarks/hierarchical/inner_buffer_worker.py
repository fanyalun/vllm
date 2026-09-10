# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma diagnostic: refresh only graph-owned sequence/query lengths."""

import torch
from cycle_worker import CycleWorker


class InnerBufferWorker(CycleWorker):
    def begin_cycle_measurement(self):
        super().begin_cycle_measurement()
        self._buffer_rows = []
        if hasattr(self, "_buffer_installed"):
            return
        spec = self.model_runner.speculator
        small = spec.small
        assert spec.preverify.model_family == "gemma4"
        assert small.advance_draft_positions is False
        original = small.propose

        def propose(batch, *args, **kwargs):
            if not self._cycle_active or len(spec.last_trace) == 0:
                return original(batch, *args, **kwargs)
            normal = original(batch, *args, **kwargs).clone()
            buffers = small.target_input_buffers
            seq = buffers.seq_lens[:1].clone()
            query = buffers.query_start_loc[:2].clone()
            try:
                buffers.seq_lens[:1].copy_(batch.seq_lens)
                buffers.query_start_loc[:2].copy_(batch.query_start_loc)
                refreshed = original(batch, *args, **kwargs).clone()
            finally:
                buffers.seq_lens[:1].copy_(seq)
                buffers.query_start_loc[:2].copy_(query)
            fresh = original(batch, *args, **{**kwargs, "is_profile": True})
            self._buffer_rows.append(
                {
                    "step": self._cycle_step,
                    "inner_round": len(spec.last_trace),
                    "normal_tokens": normal.tolist(),
                    "refreshed_graph_tokens": refreshed.tolist(),
                    "fresh_metadata_tokens": fresh.tolist(),
                    "normal_matches_fresh": torch.equal(normal, fresh),
                    "refreshed_matches_fresh": torch.equal(refreshed, fresh),
                    "captured_seq_lens": seq.tolist(),
                    "actual_seq_lens": batch.seq_lens.tolist(),
                    "captured_query_start": query.tolist(),
                    "actual_query_start": batch.query_start_loc.tolist(),
                }
            )
            return refreshed

        small.propose = propose
        self._buffer_installed = True

    def collect_cycle_measurement(self):
        measured = super().collect_cycle_measurement()
        measured["buffer_probe"] = self._buffer_rows
        measured["diagnostic_only_latency_invalid"] = True
        return measured
