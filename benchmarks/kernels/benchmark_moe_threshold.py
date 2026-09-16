# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI cold-L2 comparison of native, masked, and packed expert dispatch."""

import argparse
import json
import statistics
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe import threshold_assignment
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.weight_threshold import (
    threshold_experts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for model, experts, hidden, intermediate in (
        ("qwen36", 256, 2048, 512),
        ("gemma4", 128, 2816, 704),
    ):
        w1 = (
            torch.randn(
                experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
            )
            / 32
        )
        w2 = (
            torch.randn(
                experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
            )
            / 32
        )
        for tokens in (1, 4, 8, 16, 32, 64):
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            logits = torch.randn(tokens, experts, device="cuda") * 3
            scores, ids = logits.topk(8, dim=-1)
            ids = ids.to(torch.int32)
            weights = scores.softmax(-1)
            tw, ti = threshold_experts(weights, ids, logits, 0.125)
            reference = fused_experts(x, w1, w2, tw, ids)
            for mode in ("zero_weights", "masked", "packed", "aligned"):
                if mode == "masked" and tokens * 8 * 4 > experts:
                    continue
                ctx = ForwardContext(
                    no_compile_layers={},
                    attn_metadata={},
                    slot_mapping={},
                    additional_kwargs={"routing_min_weight": 0.125}
                    if mode in ("packed", "aligned")
                    else {},
                )
                selected_ids = ids if mode == "zero_weights" else ti

                def run(x, w1, w2, tw, selected_ids):
                    return fused_experts(x, w1, w2, tw, selected_ids)

                inputs = (x, w1, w2, tw, selected_ids)
                assignment = (
                    patch.object(
                        threshold_assignment,
                        "threshold_expert_assignment",
                        threshold_assignment.aligned_threshold_expert_assignment,
                    )
                    if mode == "aligned"
                    else nullcontext()
                )
                with assignment, override_forward_context(ctx):
                    actual = run(*inputs)
                    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
                    for _ in range(5):
                        run(*inputs)
                    us = (
                        statistics.median(
                            bench_gpu_time_with_cupti(
                                run,
                                use_cuda_graph=True,
                                cold_l2_cache=True,
                                dry_run_iters=5,
                                repeat_iters=20,
                                input_args=inputs,
                            )
                        )
                        * 1000
                    )
                rows.append(
                    dict(
                        model=model,
                        tokens=tokens,
                        mode=mode,
                        us=us,
                        mean_kept=(ti >= 0).sum(-1).float().mean().item(),
                    )
                )
                print(rows[-1], flush=True)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
