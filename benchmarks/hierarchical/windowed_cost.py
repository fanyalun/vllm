# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-input Pre-Verify timing and separate GDN module profiling."""

import statistics
from pathlib import Path

import torch


@torch.inference_mode()
def measure(spec, batch, metadata, slots, path):
    state = spec.state
    initial = state.snapshot()
    reference = spec._verify_eager(batch, metadata, slots)[0].clone()
    reference_logits = spec.last_logits.clone()
    expected = state.snapshot()

    def run():
        return spec._verify_eager(batch, metadata, slots)

    def check(predictions):
        assert torch.equal(predictions, reference)
        torch.testing.assert_close(
            spec.last_logits, reference_logits, rtol=1e-3, atol=1e-3
        )
        for name, cache in state.caches.items():
            for value, target in zip(cache, expected[name]):
                torch.testing.assert_close(value, target, rtol=1e-3, atol=1e-3)

    try:
        for _ in range(3):
            state.restore(initial)
            run()
        graph = torch.cuda.CUDAGraph()
        state.restore(initial)
        with torch.cuda.graph(graph):
            output = run()
        state.restore(initial)
        graph.replay()
        check(output[0])
        for _ in range(50):
            state.restore(initial)
            graph.replay()
        torch.accelerator.synchronize()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        samples = []
        for _ in range(200):
            state.restore(initial)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        state.restore(initial)
        torch.accelerator.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            profiled = run()[0]
            torch.accelerator.synchronize()
        check(profiled)
        profile.export_chrome_trace(str(Path(path)))
        return dict(
            samples_us=samples,
            median_us=statistics.median(samples),
            p95_us=sorted(samples)[189],
            warmup=50,
            repeat=200,
            profile_predictions_equal=True,
            profile=str(path),
            state_reset_outside_timer=True,
        )
    finally:
        state.restore(initial)
