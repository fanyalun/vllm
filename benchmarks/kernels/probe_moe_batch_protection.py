# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired routing-count probe; real gate weights and synthetic hidden states."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from benchmarks.kernels.moe_batch_policy_reference import batch_policy_reference
from vllm.model_executor.layers.fused_moe.router.batch_expert_selection import (
    select_batch_experts,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    rows = []
    for layer in (0, 19, 39):
        name = f"model.language_model.layers.{layer}.mlp.gate.weight"
        with safe_open(str(args.model / index[name]), framework="pt") as f:
            gate = f.get_tensor(name).cuda().contiguous()
        for batch in (1, 64, 128):
            for seed in args.seeds:
                torch.manual_seed(seed + layer)
                x = torch.randn(batch * 5, 2048, device="cuda", dtype=torch.bfloat16)
                logits = F.linear(x, gate)
                weights, ids, _ = fused_topk(x, logits, 8, True)
                for policy in ("batch_top_half", "batch_max_gap"):
                    for protected_top_k in (2, 1):
                        w, i, counts = batch_policy_reference(
                            weights,
                            ids,
                            logits,
                            policy,
                            protected_top_k=protected_top_k,
                        )
                        if protected_top_k == 2:
                            actual_w, actual_i = select_batch_experts(
                                weights, ids, logits, policy
                            )
                            torch.testing.assert_close(w, actual_w, atol=0, rtol=0)
                            assert torch.equal(i, actual_i)
                        ranked = logits.gather(1, ids.long()).argsort(
                            dim=1, descending=True, stable=True
                        )
                        top1 = ids.gather(1, ranked[:, :1])
                        assert (i == top1).any(dim=1).all()
                        assert torch.equal(w[i >= 0], weights[i >= 0])
                        connections = int((i >= 0).sum())
                        native_connections = ids.numel()
                        rows.append(
                            dict(
                                layer=layer,
                                batch=batch,
                                tokens=batch * 5,
                                seed=seed,
                                protected_top_k=protected_top_k,
                                policy=policy,
                                **counts,
                                native_connections=native_connections,
                                retained_connections=connections,
                                expert_skip_fraction=(
                                    1
                                    - counts["retained_unique"]
                                    / counts["native_unique"]
                                ),
                                connection_skip_fraction=1
                                - connections / native_connections,
                                retained_weight_fraction=float(w.sum() / weights.sum()),
                            )
                        )
                print(f"layer={layer} batch={batch} seed={seed} complete", flush=True)
    sources = [Path(__file__), Path(batch_policy_reference.__code__.co_filename)]
    result = dict(
        model=str(args.model),
        input="synthetic BF16 normal hidden states",
        layers=[0, 19, 39],
        batches=[1, 64, 128],
        tokens_per_request=5,
        seeds=args.seeds,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        rows=rows,
    )
    path = args.output / "results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(
                cells=len(rows),
                top2_cuda_reference_checks=len(rows) // 2,
                results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope=(
                    "routing counts only; "
                    "no latency, generation, or acceptance measurement"
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
