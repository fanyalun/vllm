# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma Q-only probe: compare graph drafts against fresh-metadata drafts."""

import torch
from cycle_worker import CycleWorker


class InnerMetadataWorker(CycleWorker):
    def begin_cycle_measurement(self):
        super().begin_cycle_measurement()
        self._metadata_rows = []
        if hasattr(self, "_metadata_installed"):
            return
        spec = self.model_runner.speculator
        small = spec.small
        assert spec.preverify.model_family == "gemma4"
        assert small.advance_draft_positions is False
        original = small.propose

        def propose(batch, *args, **kwargs):
            if not self._cycle_active or len(spec.last_trace) == 0:
                return original(batch, *args, **kwargs)
            graph_tokens = original(batch, *args, **kwargs).clone()
            # Gemma's assistant is Q-only; neither call writes shared KV.
            eager_tokens = original(batch, *args, **{**kwargs, "is_profile": True})
            self._metadata_rows.append(
                {
                    "step": self._cycle_step,
                    "inner_round": len(spec.last_trace),
                    "graph_tokens": graph_tokens.tolist(),
                    "fresh_metadata_tokens": eager_tokens.tolist(),
                    "equal": torch.equal(graph_tokens, eager_tokens),
                    "captured_target_seq_lens": small.target_input_buffers.seq_lens[
                        :1
                    ].tolist(),
                    "actual_inner_seq_lens": batch.seq_lens.tolist(),
                    "captured_target_query_start": (
                        small.target_input_buffers.query_start_loc[:2].tolist()
                    ),
                    "actual_inner_query_start": batch.query_start_loc.tolist(),
                }
            )
            return eager_tokens

        small.propose = propose
        self._metadata_installed = True

    def collect_cycle_measurement(self):
        measured = super().collect_cycle_measurement()
        measured["metadata_probe"] = self._metadata_rows
        measured["diagnostic_only_latency_invalid"] = True
        return measured
