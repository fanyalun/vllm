# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small native versus windowed post-conv batch probe; no model loading."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    windowed_replay_tail_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 128])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 15, 29])
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--wait-for-idle", action="store_true")
    args = parser.parse_args()
    if args.wait_for_idle:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu not in ("0", "1"):
            parser.error("--wait-for-idle requires CUDA_VISIBLE_DEVICES=0 or 1")
        idle = 0
        while idle < 3:
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            idle = idle + 1 if not processes else 0
            print(
                f"Waiting for GPU {gpu}: processes={processes!r}, idle={idle}/3",
                flush=True,
            )
            if idle < 3:
                time.sleep(20)
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=False)
    torch.manual_seed(42)
    rows = []
    # Flush after restoring state so repeated B1 states do not stay in L2.
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for batch in args.batches:
        for layer in args.layers:
            probe(batch, layer, args, saved, flush, rows)
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                gpu=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                protocol=(
                    "CUDA events around one graph replay; state restore and "
                    "64MiB L2 flush outside timer; replicated captured QKV/BA/state, "
                    "synthetic Z and unit norm weights"
                ),
                boundary=(
                    "post-conv recurrence plus gated norm including windowed "
                    "gate reshape copy; _core excludes norm; no projections or Conv"
                ),
                args={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            ),
            indent=2,
        )
    )


@torch.inference_mode()
def probe(batch, layer, args, saved, flush, rows):
    q0, k0, v0, a0, b0, al, dt, s0 = [x.cuda() for x in saved[layer]["values"]]
    t, h, dk = q0.shape
    _, hv, dv = v0.shape
    n = batch * t
    packed = torch.cat([x.flatten(1) for x in (q0, k0, v0)], -1).repeat(batch, 1)
    q, k, v = packed.split([h * dk, h * dk, hv * dv], -1)
    q, k, v = q.view(n, h, dk), k.view(n, h, dk), v.view(n, hv, dv)
    ba = torch.cat([b0, a0], -1).repeat(batch, 1)
    b, a = ba.chunk(2, -1)
    initial = s0.repeat(batch, 1, 1, 1)
    state = initial.clone()
    native = s0.repeat(batch * t + 1, 1, 1, 1)
    indices = torch.arange(1, batch * t + 1, device="cuda", dtype=torch.int32).view(
        batch, t
    )
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * t
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    valid = torch.ones_like(slots)
    accepted = torch.ones_like(slots)
    thresholds = torch.tensor([0.95, 0.36328125], device="cuda")
    # The capture contains no Z/weight: use seeded synthetic norm inputs.
    z_packed = torch.randn(t, packed.shape[1] + hv * dv, device="cuda", dtype=v.dtype)
    z_packed = z_packed.repeat(batch, 1)
    z = z_packed[:, -hv * dv :].view(n, hv, dv)
    weight = torch.ones(dv, device="cuda", dtype=v.dtype)
    out = torch.empty_like(v)
    core = torch.empty_like(v)

    def reset():
        state.copy_(initial)
        native.view(-1, hv, dv, dk).copy_(s0.expand_as(native))

    def run(case, counts=None):
        if case == "v0":
            ops.fused_gdn_decode_post_conv_mtp(
                mixed_qkv=packed,
                a=a,
                b=b,
                A_log=al,
                dt_bias=dt,
                state_indices=indices,
                cu_seqlens=starts,
                num_accepted_tokens=accepted,
                state=native,
                output_gate=z,
                norm_weight=weight,
                out=out,
                scale=dk**-0.5,
                norm_eps=1e-6,
                output_gate_activation="silu",
            )
        else:
            windowed_replay_tail_update(
                q,
                k,
                v,
                a,
                b,
                al,
                dt,
                state,
                query_start_loc=starts,
                state_slots=slots,
                valid=valid,
                thresholds=thresholds,
                window_size=1 if case.startswith("v2") else 5,
                out=core,
                action_counts=counts,
            )
            if not case.endswith("_core"):
                layer_norm_fwd(
                    core.view(-1, dv),
                    weight,
                    None,
                    1e-6,
                    z=z.reshape(-1, dv),
                    out=out.view(-1, dv),
                    norm_before_gate=True,
                    is_rms_norm=True,
                    activation="silu",
                )

    reset()
    run("v0")
    exact_output = out.clone()
    exact_tail = native[indices[:, -1].long()].clone()
    thresholds[1] = 0
    reset()
    run("v2")
    relative_l2 = (
        out.float() - exact_output.float()
    ).norm() / exact_output.float().norm()
    assert relative_l2 < 1e-3, relative_l2.item()
    torch.testing.assert_close(state, exact_tail, atol=1e-3, rtol=1e-3)
    thresholds[1] = 0.36328125
    del exact_output, exact_tail
    cases = ["v0", "v2", "v3", "v2_core", "v3_core"]
    if layer % 2:
        cases.reverse()
    for case in cases:
        reset()
        run(case)
        result = core if case.endswith("_core") else out
        expected = result.clone()
        tail = native.clone() if case == "v0" else state.clone()
        # Replicated requests must remain independent and equal.
        torch.testing.assert_close(
            result.view(batch, t, hv, dv),
            result[:t].expand(batch, t, hv, dv),
            atol=0,
            rtol=0,
        )
        graph = torch.cuda.CUDAGraph()
        reset()
        with torch.cuda.graph(graph):
            run(case)
        reset()
        graph.replay()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            native if case == "v0" else state, tail, atol=0, rtol=0
        )
        del tail, expected
        for _ in range(args.warmup):
            reset()
            flush.zero_()
            graph.replay()
        torch.accelerator.synchronize()
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        samples = []
        for _ in range(args.repeat):
            reset()
            flush.zero_()
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000)
        counts = torch.zeros(8, dtype=torch.int64, device="cuda")
        if case != "v0":
            reset()
            run(case, counts)
        rows.append(
            dict(
                batch=batch,
                layer=layer,
                case=case,
                tokens=t,
                median_us=statistics.median(samples),
                p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))],
                samples_us=samples,
                actions=counts.tolist(),
                graph_equal=True,
                replicated_requests_equal=True,
                forced_full_relative_l2=relative_l2.item(),
            )
        )
        (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
        print(batch, layer, case, rows[-1]["median_us"], flush=True)
        del graph


if __name__ == "__main__":
    main()
