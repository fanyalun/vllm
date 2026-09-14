# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in phase timing and CPU/CUDA traces for AR versus full-budget drafting."""

import time
from contextlib import nullcontext

import torch


class PerformancePathWorker:
    def begin_path_measurement(self, mode):
        if not hasattr(self, "_path_installed"):
            self._path_mode = None
            runner = self.model_runner

            def wrap(obj, name, phase):
                original = getattr(obj, name)

                def measured(*args, **kwargs):
                    if self._path_mode is None:
                        return original(*args, **kwargs)
                    if phase == "target_execute":
                        self._path_step += 1
                    row = {"phase": phase, "step": self._path_step}
                    if phase == "target_sample":
                        batch = args[1]
                        row.update(
                            has_prefill=bool(batch.has_prefill),
                            scheduled=int(batch.num_draft_tokens),
                            num_tokens=int(batch.num_tokens),
                        )
                    elif phase.endswith("fullgraph"):
                        row["num_tokens"] = int(args[0].num_tokens)
                    index = len(self._path_rows)
                    self._path_rows.append(row)
                    events = (
                        self._path_events[index]
                        if self._path_mode == "events"
                        else None
                    )
                    annotation = (
                        torch.profiler.record_function("path/" + phase)
                        if self._path_mode == "profile"
                        else nullcontext()
                    )
                    if events:
                        events[0].record()
                    row["cpu_start_ms"] = time.perf_counter_ns() / 1e6
                    with annotation:
                        result = original(*args, **kwargs)
                    row["cpu_end_ms"] = time.perf_counter_ns() / 1e6
                    if events:
                        events[1].record()
                    if phase == "target_sample":
                        row["count_tensor"] = result[1][:1].clone()
                    return result

                setattr(obj, name, measured)

            wrap(runner, "execute_model", "target_execute")
            wrap(runner, "sample", "target_sample")
            wrap(runner.cudagraph_manager, "run_fullgraph", "target_fullgraph")
            wrap(runner.cudagraph_manager, "run_pw_graph", "target_piecewise")
            if runner.speculator is not None:
                spec = runner.speculator
                wrap(spec, "propose", "proposal")
                wrap(spec.decode_cudagraph_manager, "run_fullgraph", "draft_fullgraph")
                wrap(spec, "_generate_draft", "draft_nonfull")
            self._path_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(8192)
            ]
            for start, end in self._path_events:
                start.record()
                end.record()
            self._path_origin = torch.cuda.Event(enable_timing=True)
            self._path_installed = True
        torch.accelerator.synchronize()
        self._path_rows = []
        self._path_step = -1
        self._path_mode = mode
        self._path_origin.record()
        if mode == "profile":
            self._path_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            )
            self._path_profiler.__enter__()

    def collect_path_measurement(self, trace_path=None):
        mode = self._path_mode
        self._path_mode = None
        torch.accelerator.synchronize()
        if mode == "profile":
            self._path_profiler.__exit__(None, None, None)
            self._path_profiler.export_chrome_trace(trace_path)
        for index, row in enumerate(self._path_rows):
            row["cpu_ms"] = row["cpu_end_ms"] - row["cpu_start_ms"]
            if mode == "events":
                start, end = self._path_events[index]
                row["start_ms"] = self._path_origin.elapsed_time(start)
                row["end_ms"] = self._path_origin.elapsed_time(end)
                row["stream_ms"] = start.elapsed_time(end)
            if "count_tensor" in row:
                row["emitted_before_terminal_clipping"] = int(
                    row.pop("count_tensor").item()
                )
        return {"mode": mode, "spans": self._path_rows}
