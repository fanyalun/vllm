# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-input tile/warp screening with fixed-address CUDA graph replays."""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    _replay_tail_update,
    replay_tail_update,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--production-layout", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=False)
    thresholds = torch.tensor([0.98, 0.36328125], device="cuda")
    rows = []
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    for tokens in range(1, 6) if args.raw else (5,):
        inputs = []
        for entry in saved:
            if args.raw:
                values = [x.cuda() for x in entry["values"]]
                q, k, v, a, b, a_log, dt, state = values
                q, k, v, a, b = [x[:tokens] for x in (q, k, v, a, b)]
                if args.production_layout:
                    widths = [
                        q.shape[1] * q.shape[2],
                        k.shape[1] * k.shape[2],
                        v.shape[1] * v.shape[2],
                        v.shape[1] * v.shape[2],
                    ]
                    packed = torch.empty(
                        (tokens, sum(widths)), dtype=q.dtype, device=q.device
                    )
                    pq, pk, pv, _ = packed.split(widths, -1)
                    pq, pk, pv = pq.view_as(q), pk.view_as(k), pv.view_as(v)
                    pq.copy_(q)
                    pk.copy_(k)
                    pv.copy_(v)
                    gates = torch.empty(
                        (tokens, 2 * a.shape[1]), dtype=a.dtype, device=a.device
                    )
                    pb, pa = gates.chunk(2, -1)
                    pa.copy_(a)
                    pb.copy_(b)
                    q, k, v, a, b = pq, pk, pv, pa, pb
            else:
                q, k, v, a, b, state = [x.cuda() for x in entry["values"]]
                q, k, v, a, b = [x[0, :tokens].contiguous() for x in (q, k, v, a, b)]
                state = state.transpose(-1, -2).contiguous()
                a_log = dt = None
            inputs.append((q, k, v, a, b, a_log, dt, state))
        references = {}
        for conditional in (False, True):
            reference = []
            for values in inputs:
                tail = torch.empty_like(values[-1])
                out = replay_tail_update(
                    *values,
                    tail=tail,
                    thresholds=thresholds if conditional else None,
                    effective_gates=not args.raw,
                )
                reference.append((out.clone(), tail.clone()))
            references[conditional] = reference
        combinations = [(16, 4, False)] + [
            (tile, warps, True)
            for tile, warps in itertools.product((8, 16, 32), (1, 2, 4, 8))
        ]
        for tile, warps, conditional in combinations:
            reference = references[conditional]
            outputs = [
                (torch.empty_like(x[2]), torch.empty_like(x[-1])) for x in inputs
            ]

            def execute(
                inputs=inputs,
                outputs=outputs,
                tile=tile,
                warps=warps,
                conditional=conditional,
            ):
                for values, (out, tail) in zip(inputs, outputs, strict=True):
                    replay_tail_update(
                        *values,
                        tail=tail,
                        thresholds=thresholds if conditional else None,
                        out=out,
                        value_tile=tile,
                        num_warps=warps,
                        effective_gates=not args.raw,
                    )

            execute()
            errors = []
            for (out, tail), (expected, expected_tail) in zip(
                outputs, reference, strict=True
            ):
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(tail, expected_tail, rtol=1e-3, atol=1e-3)
                errors.append(
                    dict(
                        output=float((out.float() - expected.float()).abs().max()),
                        tail=float((tail - expected_tail).abs().max()),
                    )
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
            for _ in range(50):
                graph.replay()
            times = []
            for _ in range(200):
                flush.zero_()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            rows.append(
                dict(
                    conditional=conditional,
                    tokens=tokens,
                    tile=tile,
                    warps=warps,
                    median_ms=statistics.median(times),
                    samples_ms=times,
                    errors=errors,
                )
            )
            (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
            print(tokens, tile, warps, statistics.median(times), flush=True)
    metadata = []
    for cache in _replay_tail_update.device_caches.values():
        for kernel in cache[0].values():
            index = len(metadata)
            (args.output / f"kernel_{index}.ptx").write_text(kernel.asm["ptx"])
            metadata.append(
                dict(
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                    name=kernel.name,
                )
            )
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                layers=len(saved),
                raw_gates=args.raw,
                production_layout=args.production_layout,
                input_strides=[list(x.stride()) for x in inputs[0] if x is not None],
                gpu=torch.cuda.get_device_name(),
                warmup=50,
                repeats=200,
                boundary=(
                    "30 recurrence kernels, independent tails; CUDA events around graph"
                ),
                cold_l2_bytes=flush.numel(),
                compiled=metadata,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
