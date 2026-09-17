# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-input windowed GDN graph timing; restore state outside every sample."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    _windowed_update,
    replay_tail_update,
    windowed_replay_tail_update,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 5, 6, 10, 16])
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=False)
    assert len(saved) == 30
    rows = []
    configs = {
        "v1": (5, "none", 0.0),
        "v2": (1, "none", 0.36328125),
        "v3": (5, "none", 0.36328125),
        "v4_d": (5, "cumulative_decay", 0.36328125),
        "v4_q": (5, "multi_query", 0.36328125),
        "v4_dq": (5, "combined", 0.36328125),
    }
    if args.cases:
        configs = {name: configs[name] for name in args.cases}
    for length in args.lengths:
        inputs = []
        for entry in saved:
            q, k, v, a, b, a_log, dt, state = [x.cuda() for x in entry["values"]]
            # Longer shapes repeat the captured T5 inputs and are shape probes.
            q, k, v, a, b = [
                x.repeat((4,) + (1,) * (x.ndim - 1))[:length] for x in (q, k, v, a, b)
            ]
            widths = [q[0].numel(), k[0].numel(), v[0].numel(), v[0].numel()]
            packed = torch.empty((length, sum(widths)), dtype=q.dtype, device="cuda")
            pq, pk, pv, _ = packed.split(widths, -1)
            pq, pk, pv = pq.view_as(q), pk.view_as(k), pv.view_as(v)
            pq.copy_(q)
            pk.copy_(k)
            pv.copy_(v)
            gates = torch.empty(length, 2 * a.shape[1], device="cuda", dtype=a.dtype)
            pb, pa = gates.chunk(2, -1)
            pa.copy_(a)
            pb.copy_(b)
            if "strides" in entry:
                pq, pk, pv, pa, pb = [
                    torch.empty_strided(
                        x.shape, strides, dtype=x.dtype, device=x.device
                    ).copy_(x)
                    for x, strides in zip((q, k, v, a, b), entry["strides"][:5])
                ]
            inputs.append((pq, pk, pv, pa, pb, a_log, dt, state))
        starts = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        valid = torch.ones(1, device="cuda", dtype=torch.int32)
        initial = [x[-1].clone() for x in inputs]
        out = [torch.empty(x[2].shape, dtype=x[2].dtype, device="cuda") for x in inputs]
        thresholds = torch.tensor([0.95, 0.36328125], device="cuda")
        for case, (window, optimization, beta) in configs.items():
            thresholds[1] = beta
            for value_tile, query_tile in ((16, 2), (8, 2), (16, 4)):
                warps = args.warps

                def restore(inputs=inputs, initial=initial):
                    for values, source in zip(inputs, initial):
                        values[-1].copy_(source)

                def run(
                    inputs=inputs,
                    out=out,
                    starts=starts,
                    valid=valid,
                    thresholds=thresholds,
                    window=window,
                    optimization=optimization,
                    value_tile=value_tile,
                    query_tile=query_tile,
                    warps=warps,
                ):
                    for values, output in zip(inputs, out):
                        windowed_replay_tail_update(
                            *values,
                            query_start_loc=starts,
                            valid=valid,
                            thresholds=thresholds,
                            window_size=window,
                            optimization=optimization,
                            out=output,
                            value_tile=value_tile,
                            query_tile=query_tile,
                            num_warps=warps,
                        )

                restore()
                run()
                eager = [
                    (values[-1].clone(), output.clone())
                    for values, output in zip(inputs, out)
                ]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                restore()
                graph.replay()
                errors = []
                for layer, (values, output, expected) in enumerate(
                    zip(inputs, out, eager)
                ):
                    torch.testing.assert_close(values[-1], expected[0], atol=0, rtol=0)
                    torch.testing.assert_close(output, expected[1], atol=0, rtol=0)
                    if case == "v1" and length <= 5:
                        reference = initial[layer].clone()
                        original = replay_tail_update(
                            *values[:-1],
                            reference,
                            value_tile=value_tile,
                            num_warps=warps,
                        )
                        errors.append(
                            dict(
                                layer=layer,
                                full_state_bitwise=torch.equal(reference, values[-1]),
                                full_output_bitwise=torch.equal(original, output),
                            )
                        )
                        passed = (
                            errors[-1]["full_state_bitwise"]
                            and errors[-1]["full_output_bitwise"]
                        )
                        if not passed:
                            difference = values[-1] - reference
                            failure = dict(
                                case=case,
                                tokens=length,
                                value_tile=value_tile,
                                query_tile=query_tile,
                                warps=warps,
                                layer=layer,
                                full_state_bitwise=errors[-1]["full_state_bitwise"],
                                full_output_bitwise=errors[-1]["full_output_bitwise"],
                                state_max_abs=difference.abs().max().item(),
                                state_relative_l2=(
                                    difference.norm()
                                    / reference.norm().clamp_min(1e-20)
                                ).item(),
                                output_max_abs=(output.float() - original.float())
                                .abs()
                                .max()
                                .item(),
                                input_sha256=hashlib.sha256(
                                    args.inputs.read_bytes()
                                ).hexdigest(),
                                reason="Forced Full bitwise mismatch; tuning rejected",
                            )
                            (args.output / "failure.json").write_text(
                                json.dumps(failure, indent=2)
                            )
                            raise AssertionError(failure)
                for _ in range(args.warmup):
                    restore()
                    graph.replay()
                torch.accelerator.synchronize()
                samples = []
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                for _ in range(args.repeat):
                    restore()
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                row = dict(
                    case=case,
                    tokens=length,
                    value_tile=value_tile,
                    query_tile=query_tile,
                    warps=warps,
                    samples_us=samples,
                    median_us=statistics.median(samples),
                    p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))],
                    layer_checks=errors,
                )
                rows.append(row)
                (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
                print(
                    case, length, value_tile, query_tile, row["median_us"], flush=True
                )
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
                warmup=args.warmup,
                repeat=args.repeat,
                timing="CUDA events, 30-layer graph; state reset outside timing",
                longer_inputs="T6/10/16 repeat captured T5 rows; shape probes only",
                captured_strides=all("strides" in entry for entry in saved),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
