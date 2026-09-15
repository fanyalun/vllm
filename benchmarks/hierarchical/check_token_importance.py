# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU references for last-row attention and block-level expert pruning."""

import json
from itertools import product

import torch
from token_importance_worker import last_attention, mask_assignment, select_pool

from vllm.model_executor.layers.fused_moe import fused_moe


def main():
    torch.manual_seed(15)
    attention_cases = expert_cases = 0
    for length, window, cap, dim in product(
        (19, 79), (-1, 11), (0.0, 3.0), (80, 256, 512)
    ):
        m, heads, kvheads, block = 5, 4, 2, 16
        query = torch.randn(m, heads, dim, device="cuda", dtype=torch.bfloat16)
        cache = torch.randn(
            8, kvheads, block, dim * 2, device="cuda", dtype=torch.bfloat16
        )
        table = torch.randperm(8, device="cuda").int()[None, :]
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        pos = torch.arange(length, device="cuda")
        keys = cache[table[0, pos // block].long(), :, pos % block, :dim]
        keys = keys.repeat_interleave(heads // kvheads, dim=1).float()
        scores = torch.einsum("hd,shd->hs", query[-1].float(), keys) * 0.2
        if cap:
            scores = cap * torch.tanh(scores / cap)
        if window >= 0:
            scores[:, pos < length - 1 - window] = -float("inf")
        reference = scores.softmax(-1)[:, -m:].mean(0)
        actual = last_attention(query, cache, table, lengths, window, 0.2, cap)
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-6)
        attention_cases += 1
    original = fused_moe._prepare_expert_assignment
    for m in (3, 4, 5, 17):
        for tied in (False, True):
            logits = torch.randn(m, 128, device="cuda")
            if tied:
                logits.zero_()
            values, ids = logits.topk(8)
            ids = ids.int()
            weights = values.softmax(-1) * (torch.rand_like(values) + 0.5)
            relevance = torch.rand(m, device="cuda")
            selected, pool, counters = select_pool(weights, ids, logits, relevance)
            native = values.softmax(-1)
            reference_scores = torch.zeros(128, device="cuda")
            active = set(ids[1:].flatten().tolist())
            for row in range(1, m - 1):
                for col in range(8):
                    reference_scores[ids[row, col]] += relevance[row] * native[row, col]
            ranking = sorted(active, key=lambda e: (-reference_scores[e].item(), e))
            selected_set = ranking[: (len(active) * 3 + 4) // 5]
            assert set(pool.nonzero().flatten().tolist()) == set(selected_set)
            assert counters[0].item() == len(active)
            assert counters[1].item() == len(selected_set)
            keep = pool[ids.long()]
            expected = (
                weights * keep / (native * keep).sum(-1, keepdim=True).clamp_min(1e-30)
            )
            torch.testing.assert_close(selected, expected)
            x = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
            w1 = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16) * 0.1
            w2 = torch.randn(128, 64, 32, device="cuda", dtype=torch.bfloat16) * 0.1
            reference = fused_moe.fused_experts(x, w1, w2, selected, ids)

            def assignment(*args, selected_pool=pool, **kwargs):
                sorted_ids, expert_ids, padded = original(*args, **kwargs)
                return sorted_ids, mask_assignment(expert_ids, selected_pool), padded

            fused_moe._prepare_expert_assignment = assignment
            try:
                actual = fused_moe.fused_experts(x, w1, w2, selected, ids)
            finally:
                fused_moe._prepare_expert_assignment = original
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
            expert_cases += 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_attention = last_attention(query, cache, table, lengths, window, 0.2, cap)
        graph_weights, graph_pool, _ = select_pool(weights, ids, logits, relevance)
    graph.replay()
    torch.testing.assert_close(
        graph_attention,
        last_attention(query, cache, table, lengths, window, 0.2, cap),
    )
    torch.testing.assert_close(graph_weights, selected)
    assert torch.equal(graph_pool, pool)
    print(
        json.dumps(
            {
                "attention_cases": attention_cases,
                "expert_cases": expert_cases,
                "status": "passed",
                "cuda_graph": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
