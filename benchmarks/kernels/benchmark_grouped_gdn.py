# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache projection tuning over three existing BF16 weight matrices."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.grouped_input import _linear
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    for n, k in ((12288, 2048), (64, 2048), (2048, 4096)):
        weights = [
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
            for _ in range(3)
        ]
        for tokens in range(1, 6):
            x = torch.randn(3, tokens, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(3, tokens, n, device="cuda", dtype=x.dtype)
            expected = torch.stack(
                [torch.nn.functional.linear(x[i], weights[i]) for i in range(3)]
            )
            functions = {
                "native": lambda x=x, weights=weights: [
                    torch.nn.functional.linear(x[i], weights[i]) for i in range(3)
                ]
            }
            for bn in (32, 64, 128):
                for bk in (64, 128):
                    functions[f"n{bn}_k{bk}"] = (
                        lambda bn=bn,
                        bk=bk,
                        x=x,
                        weights=weights,
                        out=out,
                        tokens=tokens,
                        n=n,
                        k=k: _linear[(triton.cdiv(n, bn), 3)](
                            x, *weights, out, tokens, n, k, bn, bk, num_warps=4
                        )
                    )
            graphs = {}
            for name, fn in functions.items():
                for _ in range(5):
                    fn()
                if name != "native":
                    torch.testing.assert_close(out, expected, atol=1e-3, rtol=8e-3)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = fn()
                graphs[name] = graph, result
            timings = {name: [] for name in functions}
            for repeat in range(35):
                order = list(graphs) if repeat % 2 == 0 else list(graphs)[::-1]
                for name in order:
                    flush.zero_()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    graphs[name][0].replay()
                    end.record()
                    end.synchronize()
                    if repeat >= 5:
                        timings[name].append(start.elapsed_time(end) * 1000)
            rows.append(
                dict(
                    tokens=tokens,
                    n=n,
                    k=k,
                    median_us={
                        name: statistics.median(values)
                        for name, values in timings.items()
                    },
                    samples_us=timings,
                    logical_weight_bytes=3 * n * k * 2,
                )
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in rows[-1].items()
                        if key != "samples_us"
                    }
                ),
                flush=True,
            )
    compiled = []
    for cache in _linear.device_caches.values():
        for kernel in cache[0].values():
            compiled.append(
                dict(
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                timing="CUDA events outside graph",
                cache="128 MiB explicit eviction before each timed replay",
                repeats=30,
                warmup=5,
                dtype="bfloat16",
                rows=rows,
                compiled=compiled,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
