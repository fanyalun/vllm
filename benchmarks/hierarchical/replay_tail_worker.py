# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired replay-tail measurements and benchmark-only causal ablations."""

import torch

from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class ReplayTailWorker:
    def drain_replay_device(self):
        torch.accelerator.synchronize()

    def set_replay_case(self, case):
        spec = self.model_runner.speculator
        if not hasattr(self, "_replay_cases"):
            self._replay_cases = {}
            self._replay_audit = False
            self._replay_graph_checks = set()
            original_eager = spec._verify_eager
            original_verify = spec._verify
            original_propose = spec.propose
            original_record = spec.record_verification

            def verify(batch, *args):
                self._replay_width = batch.num_tokens
                width = batch.num_tokens
                if (
                    self._replay_case == "replay_tail"
                    and width not in self._replay_graph_checks
                ):
                    before = spec.state.snapshot()
                    result = original_verify(batch, *args)
                    after = spec.state.snapshot()
                    predictions = result[0].clone()
                    logits, margins = spec.last_logits, spec.last_margins
                    expected_logits = logits.clone()
                    spec.state.restore(before)
                    reference = original_eager(batch, *args)
                    torch.testing.assert_close(result[0], predictions, rtol=0, atol=0)
                    torch.testing.assert_close(
                        reference[0], predictions, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        spec.last_logits, expected_logits, rtol=1e-3, atol=1e-3
                    )
                    for name, cache in spec.state.caches.items():
                        for actual, expected in zip(cache, after[name], strict=True):
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-3, atol=1e-3
                            )
                    spec.state.restore(after)
                    spec.last_logits, spec.last_margins = logits, margins
                    self._replay_graph_checks.add(width)
                    return result
                if not self._replay_audit:
                    return original_verify(batch, *args)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                result = original_verify(batch, *args)
                end.record()
                self._replay_spans.append(("preverify", start, end))
                return result

            def propose(*args, **kwargs):
                if not self._replay_audit:
                    return original_propose(*args, **kwargs)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                result = original_propose(*args, **kwargs)
                end.record()
                self._replay_spans.append(("proposal", start, end))
                self._replay_pending = (start, [dict(r) for r in spec.last_trace])
                return result

            def record(logits, batch, sampled):
                if self._replay_audit and self._replay_pending is not None:
                    start, trace = self._replay_pending
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    self._replay_cycles.append(
                        (
                            start,
                            end,
                            trace,
                            int(batch.num_draft_tokens_per_req[0]),
                            sampled[:1].clone(),
                        )
                    )
                    self._replay_pending = None
                return original_record(logits, batch, sampled)

            spec._verify = verify
            spec.propose = propose
            spec.record_verification = record
        if case not in ("none", "replay_tail", "tail_only"):
            raise ValueError(case)
        self._replay_case = case
        mode = "replay_tail" if case == "replay_tail" else "none"
        if case not in self._replay_cases:
            state = PreverifyState(spec.model, spec.depth + 1, spec.device, mode)
            if case == "tail_only":

                def advance(accepted):
                    for conv, temporal in state.caches.values():
                        state._copy_conv(conv[1:2], conv[1:2], accepted)
                        temporal[1:2].copy_(
                            temporal[
                                self._replay_width : self._replay_width + 1
                            ].clone()
                        )

                state.advance = advance
            self._replay_cases[case] = state, {}
        spec.state, spec.preverify_graphs = self._replay_cases[case]
        spec.config.preverify_gdn_mode = mode
        self._replay_audit = False
        return {
            "case": case,
            "ssm_bytes": sum(
                s.numel() * s.element_size() for _, s in spec.state.caches.values()
            ),
            "conv_bytes": sum(
                c.numel() * c.element_size() for c, _ in spec.state.caches.values()
            ),
            "graphs": len(spec.preverify_graphs),
            "graph_eager_checked_widths": sorted(self._replay_graph_checks),
        }

    def begin_replay_audit(self):
        self._replay_spans = []
        self._replay_cycles = []
        self._replay_pending = None
        self._replay_audit = True

    def collect_replay_audit(self):
        self._replay_audit = False
        torch.accelerator.synchronize()
        return {
            "spans": [
                {"phase": phase, "ms": start.elapsed_time(end)}
                for phase, start, end in self._replay_spans
            ],
            "cycles": [
                {
                    "ms": start.elapsed_time(end),
                    "inner": trace,
                    "scheduled": scheduled,
                    "emitted": int(sampled.item()),
                }
                for start, end, trace, scheduled, sampled in self._replay_cycles
            ],
        }
