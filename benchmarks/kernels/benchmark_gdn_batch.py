# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-input batched core latency using captured production strides.

The B1 capture is replicated into independent request states. This measures
batch scaling of the recurrent core, not online acceptance or request diversity.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    _windowed_update,
    windowed_replay_tail_update,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--profile-replays", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=False)
    assert len(saved) == 30
    rows = []
    configs = {
        "v1": (5, "none", 0.0),
        "v2": (1, "none", 0.36328125),
        "v3": (5, "none", 0.36328125),
        "v4d": (5, "cumulative_decay", 0.36328125),
        "v4q": (5, "multi_query", 0.36328125),
        "v4dq": (5, "combined", 0.36328125),
    }
    for batch in args.batches:
        inputs = []
        t = args.tokens
        for entry in saved:
            values = [x.cuda() for x in entry["values"]]
            tiled = []
            for x, stride in zip(values[:5], entry["strides"][:5], strict=True):
                one = x.repeat((4,) + (1,) * (x.ndim - 1))[:t]
                repeated = one.repeat((batch,) + (1,) * (x.ndim - 1))
                tiled.append(
                    torch.empty_strided(
                        repeated.shape, stride, device="cuda", dtype=x.dtype
                    ).copy_(repeated)
                )
            inputs.append((*tiled, *values[5:7], values[7].repeat(batch, 1, 1, 1)))
        initial = [x[-1].clone() for x in inputs]
        outputs = [torch.empty_like(x[2]) for x in inputs]
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * t
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        valid = torch.ones(batch, device="cuda", dtype=torch.int32)
        thresholds = torch.tensor([0.95, 0.36328125], device="cuda")
        for case, (window, optimization, beta) in configs.items():
            thresholds[1] = beta

            def restore(inputs=inputs, initial=initial):
                for values, source in zip(inputs, initial, strict=True):
                    values[-1].copy_(source)

            def run(
                inputs=inputs,
                outputs=outputs,
                starts=starts,
                slots=slots,
                valid=valid,
                thresholds=thresholds,
                window=window,
                optimization=optimization,
            ):
                for values, out in zip(inputs, outputs, strict=True):
                    windowed_replay_tail_update(
                        *values,
                        query_start_loc=starts,
                        state_slots=slots,
                        valid=valid,
                        thresholds=thresholds,
                        window_size=window,
                        optimization=optimization,
                        out=out,
                    )

            restore()
            run()
            expected = [
                (x[-1].clone(), y.clone()) for x, y in zip(inputs, outputs, strict=True)
            ]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            restore()
            graph.replay()
            for x, y, (state, output) in zip(inputs, outputs, expected, strict=True):
                torch.testing.assert_close(x[-1], state, atol=0, rtol=0)
                torch.testing.assert_close(y, output, atol=0, rtol=0)
            del expected
            for _ in range(args.warmup):
                restore()
                graph.replay()
            torch.accelerator.synchronize()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            samples = []
            for _ in range(args.repeat):
                restore()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            rows.append(
                dict(
                    batch=batch,
                    tokens_per_request=t,
                    case=case,
                    layers=30,
                    median_us=statistics.median(samples),
                    p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))],
                    amortized_request_us=statistics.median(samples) / batch,
                    samples_us=samples,
                )
            )
            if args.profile_replays:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    for _ in range(args.profile_replays):
                        restore()
                        graph.replay()
                    torch.accelerator.synchronize()
                trace = args.output / f"b{batch}_{case}_profile.json"
                profiler.export_chrome_trace(str(trace))
                events = sorted(
                    [
                        e
                        for e in json.loads(trace.read_text())["traceEvents"]
                        if e.get("cat") == "kernel" and "_windowed_update" in e["name"]
                    ],
                    key=lambda e: e["ts"],
                )
                assert len(events) == 30 * args.profile_replays
                rows[-1]["kernel_samples"] = [
                    dict(layer=i % 30, replay=i // 30, us=e["dur"])
                    for i, e in enumerate(events)
                ]
            (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
            print(batch, case, rows[-1]["median_us"], flush=True)
    kernels = []
    for cache in _windowed_update.device_caches.values():
        for kernel in cache[0].values():
            name = f"kernel_{len(kernels)}.ptx"
            (args.output / name).write_text(kernel.asm["ptx"])
            kernels.append(
                dict(
                    file=name,
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    constants=str(kernel.src.constants),
                )
            )
    (args.output / "kernels.json").write_text(json.dumps(kernels, indent=2))
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(),
                protocol=(
                    "replicated capture, independent states, "
                    "restore outside every timed sample"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
