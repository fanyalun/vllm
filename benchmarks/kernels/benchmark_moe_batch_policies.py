# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small MoE policy latency probe using checkpoint weights and synthetic inputs."""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)


@torch.inference_mode()
def probe(args, layer, rows):
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.language_model.layers.{layer}.mlp."

    def load(suffix):
        name = prefix + suffix
        with safe_open(
            str(args.model / index[name]), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(name).cuda().contiguous()

    w1, w2 = load("experts.gate_up_proj"), load("experts.down_proj")
    gate = load("gate.weight")
    shared1 = torch.cat(
        [load("shared_expert.gate_proj.weight"), load("shared_expert.up_proj.weight")]
    )
    shared2, shared_gate = (
        load("shared_expert.down_proj.weight"),
        load("shared_expert_gate.weight"),
    )
    router = FusedTopKRouter(top_k=8, global_num_experts=256, renormalize=True)
    for batch in args.batches:
        torch.manual_seed(42 + layer)
        x = torch.randn(batch * 5, 2048, device="cuda", dtype=torch.bfloat16)
        cases = ["h8", "h4", "p0125"]
        if layer % 2:
            cases.reverse()
        for case in cases:
            measure(
                args,
                layer,
                batch,
                case,
                x,
                w1,
                w2,
                gate,
                shared1,
                shared2,
                shared_gate,
                router,
                rows,
            )


@torch.inference_mode()
def measure(
    args,
    layer,
    batch,
    case,
    x,
    w1,
    w2,
    gate,
    shared1,
    shared2,
    shared_gate,
    router,
    rows,
):
    settings = dict(
        routing_top_k=4 if case == "h4" else 8, routing_preserve_weights=True
    )
    if case == "p0125":
        settings["routing_min_weight"] = 0.125
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs=settings,
    )
    activation = torch.empty(x.shape[0], shared2.shape[1], device="cuda", dtype=x.dtype)

    def route():
        logits = F.linear(x, gate)
        weights, ids = router.select_experts(x, logits)
        return weights, ids

    def shared():
        torch.ops._C.silu_and_mul(activation, F.linear(x, shared1))
        return F.linear(activation, shared2) * F.linear(x, shared_gate).sigmoid()

    def run(include_shared):
        weights, ids = route()
        result = fused_experts(x, w1, w2, weights, ids)
        return result + shared() if include_shared else result

    with override_forward_context(ctx):
        weights, ids = route()
        selected = (ids >= 0).sum(-1)
        actual = fused_experts(x, w1, w2, weights, ids)
        # Check dispatch against all native assignments with discarded weights zeroed.
        native_ctx = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={},
        )
        with override_forward_context(native_ctx):
            native_weights, native_ids = route()
            ref_weights = torch.zeros_like(native_weights)
            for i in range(ids.shape[1]):
                ref_weights += torch.where(
                    native_ids == ids[:, i : i + 1], weights[:, i : i + 1], 0
                )
            reference = fused_experts(x, w1, w2, ref_weights, native_ids)
        torch.testing.assert_close(actual, reference, atol=0.01, rtol=0.02)
        if case == "h4":
            assert selected.eq(4).all()
        elif case == "h8":
            assert selected.eq(8).all()
        for include_shared in [False, True]:
            expected = run(include_shared).clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run(include_shared)
            graph.replay()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            for _ in range(30):
                flush.zero_()
                graph.replay()
            torch.accelerator.synchronize()
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            samples = []
            for _ in range(100):
                flush.zero_()
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000)
            rows.append(
                dict(
                    layer=layer,
                    batch=batch,
                    tokens=batch * 5,
                    case=case,
                    boundary="moe_with_shared"
                    if include_shared
                    else "routing_and_experts",
                    median_us=statistics.median(samples),
                    p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))],
                    samples_us=samples,
                    selected_mean=selected.float().mean().item(),
                    selected_min=selected.min().item(),
                    selected_max=selected.max().item(),
                    selected_total=selected.sum().item(),
                    unique_experts=ids[ids >= 0].unique().numel(),
                    graph_equal=True,
                    dispatch_reference_close=True,
                )
            )
            (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
            print(
                layer,
                batch,
                case,
                rows[-1]["boundary"],
                rows[-1]["median_us"],
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 19, 39])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for layer in args.layers:
        probe(args, layer, rows)
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                gpu=torch.cuda.get_device_name(),
                device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                model=str(args.model),
                layers=args.layers,
                batches=args.batches,
                protocol=(
                    "CUDA events per graph replay, 64MiB flush outside timer, "
                    "30 warmups/100 repeats; "
                    "real checkpoint weights, "
                    "independent synthetic BF16 hidden states; "
                    "routing uses preserve weights; functional shared expert "
                    "composition, not compiled full-model wrapper"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
