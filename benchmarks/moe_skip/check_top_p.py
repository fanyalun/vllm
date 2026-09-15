# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU reference checks for top-p routing and invalid-expert GEMM handling."""

import json

import torch
from top_p_worker import truncate

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def main():
    torch.manual_seed(0)
    cases = 0
    for tokens in (1, 2, 3, 4):
        for uniform in (False, True):
            logits = torch.randn(tokens, 128, device="cuda")
            if uniform:
                logits.zero_()
            scores, ids = torch.topk(logits, 8, dim=-1)
            permutation = torch.randperm(8, device="cuda")
            scores, ids = scores[:, permutation], ids[:, permutation]
            ids = ids.to(torch.int32)
            q = scores.softmax(-1)
            scales = torch.rand_like(q) + 0.5
            weights = q * scales
            for p in (0.7, 0.8, 0.9, 1.0):
                counts = torch.zeros(8, dtype=torch.int64, device="cuda")
                w, chosen = truncate(weights, ids, logits, counts, p)
                order = scores.argsort(dim=-1, descending=True, stable=True)
                ordered = q.gather(1, order)
                ordered_keep = ordered.cumsum(-1) - ordered < p
                keep = torch.zeros_like(ordered_keep).scatter_(1, order, ordered_keep)
                if p == 1:
                    keep.fill_(True)
                ref = weights * keep / (q * keep).sum(-1, keepdim=True)
                torch.testing.assert_close(w, ref, rtol=1e-5, atol=1e-6)
                assert torch.equal(chosen, ids.masked_fill(~keep, -1))
                expected_counts = torch.bincount(keep.sum(-1) - 1, minlength=8)
                assert torch.equal(counts, expected_counts)
                if p == 1:
                    assert torch.equal(w, weights) and torch.equal(chosen, ids)
                x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
                w1 = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16) * 0.1
                w2 = torch.randn(128, 64, 32, device="cuda", dtype=torch.bfloat16) * 0.1
                reference = fused_experts(x, w1, w2, w, ids)
                actual = fused_experts(x, w1, w2, w, chosen)
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
                cases += 1
    for tokens in (5, 16):
        logits = torch.zeros(tokens, 128, device="cuda")
        weights = torch.ones(tokens, 8, device="cuda") / 8
        ids = torch.zeros(tokens, 8, dtype=torch.int32, device="cuda")
        counts = torch.zeros(8, dtype=torch.int64, device="cuda")
        try:
            truncate(weights, ids, logits, counts, 0.8)
        except ValueError as error:
            assert "naive MoE assignment" in str(error)
        else:
            raise AssertionError("Unsupported aligned-expert path was accepted")
    print(
        json.dumps(
            {"gpu_reference_cases": cases, "rejected_shapes": 2, "status": "passed"}
        )
    )


if __name__ == "__main__":
    main()
