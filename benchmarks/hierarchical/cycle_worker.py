# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure proposal-to-next-verification cycles and nested execution spans."""

import time

import torch


def pair_cycles(spans):
    """Pair each proposal with the following Target sample, within one request."""
    proposals = {r["step"]: r for r in spans if r["phase"] == "proposal"}
    cycles = []
    for sample in spans:
        if sample["phase"] != "target_sample":
            continue
        proposal = proposals.get(sample["step"] - 1)
        if proposal is None:
            continue
        cycles.append(
            {
                "step": sample["step"],
                "proposal_step": proposal["step"],
                "scheduled": sample["scheduled"],
                "emitted": sample["emitted"],
                "accepted": sample["emitted"] - 1,
                "cycle_stream_ms": sample["end_ms"] - proposal["start_ms"],
                "cycle_wall_ms": sample["cpu_end_ms"] - proposal["cpu_start_ms"],
                "proposal_ms": proposal["stream_ms"],
            }
        )
    return cycles


class CycleWorker:
    def begin_cycle_measurement(self):
        if not hasattr(self, "_cycle_installed"):
            self._cycle_active = False
            runner = self.model_runner
            spec = runner.speculator

            def wrap(obj, name, phase):
                original = getattr(obj, name)

                def measured(*args, **kwargs):
                    if not self._cycle_active:
                        return original(*args, **kwargs)
                    if phase == "target_execute":
                        self._cycle_step += 1
                    index = len(self._cycle_rows)
                    start, end = self._cycle_events[index]
                    row = {"phase": phase, "step": self._cycle_step}
                    self._cycle_rows.append(row)
                    start.record()
                    row["cpu_start_ms"] = time.perf_counter_ns() / 1e6
                    result = original(*args, **kwargs)
                    row["cpu_end_ms"] = time.perf_counter_ns() / 1e6
                    end.record()
                    if phase == "target_sample":
                        batch = args[1]
                        row["scheduled"] = (
                            int(batch.num_draft_tokens_per_req[0])
                            if batch.num_draft_tokens_per_req is not None
                            else 0
                        )
                        row["count_tensor"] = result[1][:1].clone()
                        row["has_prefill"] = bool(batch.has_prefill)
                    if phase == "proposal" and hasattr(spec, "last_trace"):
                        row["inner_trace"] = [dict(r) for r in spec.last_trace]
                    return result

                setattr(obj, name, measured)

            wrap(runner, "execute_model", "target_execute")
            wrap(runner, "sample", "target_sample")
            wrap(spec, "propose", "proposal")
            if hasattr(spec, "small"):
                import vllm.v1.worker.gpu.spec_decode.hierarchical.speculator as impl

                wrap(spec.small, "propose", "small_draft")
                wrap(spec, "_batch", "preverify_metadata")
                wrap(spec, "_verify", "preverify")
                for width, (graph, *_) in spec.preverify_graphs.items():
                    wrap(graph, "replay", f"preverify_graph_{width}")
                wrap(spec.state, "begin", "state_begin")
                wrap(spec.state, "advance", "state_advance")
                wrap(impl, "accepted_prefix", "accept_prefix_sync")
            self._cycle_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(8192)
            ]
            self._cycle_origin = torch.cuda.Event(enable_timing=True)
            for start, end in self._cycle_events:
                start.record()
                end.record()
            self._cycle_origin.record()
            torch.accelerator.synchronize()
            self._cycle_installed = True
        self._cycle_rows = []
        self._cycle_step = -1
        self._cycle_origin.record()
        self._cycle_active = True

    def collect_cycle_measurement(self):
        self._cycle_active = False
        torch.accelerator.synchronize()
        for row, (start, end) in zip(self._cycle_rows, self._cycle_events):
            row["start_ms"] = self._cycle_origin.elapsed_time(start)
            row["end_ms"] = self._cycle_origin.elapsed_time(end)
            row["stream_ms"] = start.elapsed_time(end)
            row["cpu_ms"] = row["cpu_end_ms"] - row["cpu_start_ms"]
            if "count_tensor" in row:
                row["emitted"] = int(row.pop("count_tensor").item())
        result = {"spans": self._cycle_rows, "cycles": pair_cycles(self._cycle_rows)}
        state = getattr(self.model_runner.speculator, "state", None)
        if state is not None and hasattr(state, "caches"):
            result["private_gdn_state"] = {
                "mode": getattr(state, "mode", "none"),
                "layers": len(state.caches),
                "conv_bytes": sum(
                    c.numel() * c.element_size() for c, _ in state.caches.values()
                ),
                "ssm_bytes": sum(
                    s.numel() * s.element_size() for _, s in state.caches.values()
                ),
                "slots": sorted({s.shape[0] for _, s in state.caches.values()}),
            }
        return result
