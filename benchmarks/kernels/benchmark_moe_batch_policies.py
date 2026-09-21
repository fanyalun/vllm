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

from benchmarks.kernels.moe_batch_policy_reference import batch_policy_reference
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.batch_expert_selection import (
    select_batch_experts,
)
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
        cases = list(args.policies)
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
    if case.startswith("batch_"):
        settings["routing_batch_policy"] = case
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

    def run(boundary):
        if boundary == "experts_only":
            return fused_experts(x, w1, w2, weights, ids)
        if boundary == "routing_only":
            return route()
        if boundary == "batch_filter_only":
            return select_batch_experts(native_weights, native_ids, native_logits, case)
        routed_weights, routed_ids = route()
        result = fused_experts(x, w1, w2, routed_weights, routed_ids)
        return result + shared() if boundary == "moe_with_shared" else result

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
            native_logits = F.linear(x, gate)
            ref_weights = torch.zeros_like(native_weights)
            for i in range(ids.shape[1]):
                ref_weights += torch.where(
                    native_ids == ids[:, i : i + 1], weights[:, i : i + 1], 0
                )
            reference = fused_experts(x, w1, w2, ref_weights, native_ids)
        torch.testing.assert_close(actual, reference, atol=0.01, rtol=0.02)
        expected_weights, expected_ids, counts = batch_policy_reference(
            native_weights,
            native_ids,
            native_logits,
            case if case.startswith("batch_") else "batch_top_half",
        )
        if case.startswith("batch_"):
            torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
            assert torch.equal(ids, expected_ids)
        counts["retained_unique"] = ids[ids >= 0].unique().numel()
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _prepare_expert_assignment,
            try_get_optimal_moe_config,
        )

        config = try_get_optimal_moe_config(
            w1.shape, w2.shape, ids.shape[1], None, x.shape[0]
        )
        _, expert_blocks, padded_count = _prepare_expert_assignment(
            ids, config, x.shape[0], ids.shape[1], 256, None
        )
        block_m = config["BLOCK_SIZE_M"]
        used_blocks = int(padded_count.item()) // block_m
        valid_blocks = int((expert_blocks[:used_blocks] >= 0).sum().item())
        if case == "h4":
            assert selected.eq(4).all()
        elif case == "h8":
            assert selected.eq(8).all()
        boundaries = [
            "routing_and_experts",
            "moe_with_shared",
            "experts_only",
            "routing_only",
        ]
        if case.startswith("batch_"):
            boundaries.append("batch_filter_only")
        for boundary in boundaries:
            expected = run(boundary)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run(boundary)
            graph.replay()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            for _ in range(0 if args.correctness_only else 30):
                flush.zero_()
                graph.replay()
            torch.accelerator.synchronize()
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            samples = []
            for _ in range(0 if args.correctness_only else 100):
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
                    boundary=boundary,
                    median_us=statistics.median(samples) if samples else None,
                    p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))]
                    if samples
                    else None,
                    samples_us=samples,
                    selected_mean=selected.float().mean().item(),
                    selected_min=selected.min().item(),
                    selected_max=selected.max().item(),
                    selected_total=selected.sum().item(),
                    unique_experts=ids[ids >= 0].unique().numel(),
                    **counts,
                    retained_weight_mass=weights.sum(-1).mean().item(),
                    block_size_m=block_m,
                    valid_expert_blocks=valid_blocks,
                    padded_token_slots=used_blocks * block_m,
                    valid_expert_token_slots=valid_blocks * block_m,
                    graph_equal=True,
                    dispatch_reference_close=True,
                )
            )
            (
                args.output
                / ("validation.json" if args.correctness_only else "timings.json")
            ).write_text(json.dumps(rows, indent=2))
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
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 19, 39])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["h8", "h4", "p0125", "batch_top_half", "batch_max_gap"],
        default=["h8", "h4", "p0125", "batch_top_half", "batch_max_gap"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for layer in args.layers:
        probe(args, layer, rows)
    (
        args.output
        / ("correctness_complete.json" if args.correctness_only else "complete.json")
    ).write_text(
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
                policies=args.policies,
                correctness_only=args.correctness_only,
                protocol=(
                    (
                        "Correctness only: no latency measurements; "
                        if args.correctness_only
                        else "CUDA events per graph replay, 64MiB flush outside timer, "
                        "30 warmups/100 repeats; "
                    )
                    + "real checkpoint weights, "
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
