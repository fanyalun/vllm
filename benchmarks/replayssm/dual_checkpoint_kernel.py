# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GDN ReplaySSM cycles under controlled acceptance trajectories."""

import argparse
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops.gdn_replayssm_dual_checkpoint import (
    commit_gdn_dual_checkpoint,
)
from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
)


def measure(dual, trajectory, batch, draft, cap):
    torch.manual_seed(0)
    t, h, hv, k, v = draft + 1, 16, 32, 128, 128
    logical = cap if dual else cap + t
    length = 1 << (logical - 1).bit_length()
    device = "cuda"
    s0 = torch.zeros(batch + 1, hv, v, k, device=device)
    s1 = torch.zeros_like(s0) if dual else None
    d = torch.zeros(batch + 1, hv, length, v, device=device, dtype=torch.float16)
    keys = torch.zeros(batch + 1, h, length, k, device=device, dtype=torch.float16)
    gates = torch.zeros(batch + 1, hv, length, device=device)
    wp, base, head, prev = [
        torch.zeros(batch + 1, device=device, dtype=torch.int32) for _ in range(4)
    ]
    flush = torch.zeros(batch + 1, device=device, dtype=torch.int8)
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    qsl = torch.arange(batch + 1, device=device, dtype=torch.int32) * t
    reset = torch.zeros(batch, device=device, dtype=torch.int8)
    accepted = {
        acc: torch.full((batch,), acc, device=device, dtype=torch.int32)
        for acc in (1, t)
    }
    qkv = (
        torch.randn(batch * t, 2 * h * k + hv * v, device=device, dtype=torch.bfloat16)
        * 0.2
    )
    a = torch.randn(batch * t, hv, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.full((hv,), -2.0, device=device)
    bias = torch.zeros(hv, device=device)
    out = torch.empty(batch * t, hv, v, device=device, dtype=torch.bfloat16)

    def step(acc):
        if dual:
            commit_gdn_dual_checkpoint(
                wp, base, flush, head, prev, accepted[acc], indices, qsl, reset, cap
            )
        else:
            commit_gdn_replayssm_spec(
                wp, base, flush, accepted[acc], indices, logical, t, length
            )
        gdn_replayssm_spec_decode(
            qkv,
            a,
            b,
            a_log,
            bias,
            s0,
            d,
            keys,
            gates,
            out,
            qsl,
            indices,
            wp,
            base,
            flush,
            logical,
            t,
            alternate_checkpoint=s1,
            head_slot=head if dual else None,
            hard_cap=cap if dual else None,
        )

    pattern = {"all": [t], "reject": [1], "mixed": [t, 1]}[trajectory]
    for i in range(32):
        step(pattern[i % len(pattern)])
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(32):
            step(pattern[i % len(pattern)])
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    begin, end = [torch.Event(enable_timing=True) for _ in range(2)]
    begin.record()
    for _ in range(20):
        graph.replay()
    end.record()
    end.synchronize()
    cycle_us = begin.elapsed_time(end) * 1000 / (20 * 32)
    history, flushed, promoted = [], 0, 0
    for i in range(64):
        old_head = head.clone()
        step(pattern[i % len(pattern)])
        history.extend(wp[1:].tolist())
        flushed += flush[1:].sum().item()
        promoted += (head[1:] != old_head[1:]).sum().item()
    buffers = [s0, d, keys, gates] + ([s1] if dual else [])
    return dict(
        mode="dual" if dual else "original",
        trajectory=trajectory,
        batch=batch,
        draft=draft,
        buffer_setting=cap,
        logical_cap=logical,
        physical_ring_len=length,
        cycle_us=cycle_us,
        mean_history=sum(history) / len(history),
        flush_rate=flushed / (64 * batch),
        promotion_rate=promoted / (64 * batch),
        state_and_ring_bytes_per_request=sum(
            x[0].numel() * x.element_size() for x in buffers
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--draft", type=int, default=4)
    parser.add_argument("--cap", type=int, default=16)
    args = parser.parse_args()
    results = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for trajectory in ("all", "reject", "mixed"):
        for dual in (False, True):
            row = measure(dual, trajectory, args.batch, args.draft, args.cap)
            results.append(row)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
