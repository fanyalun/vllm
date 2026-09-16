# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _threshold(
    W,
    IDS,
    G,
    OW,
    OI,
    K: tl.constexpr,
    E: tl.constexpr,
    P: tl.constexpr,
    RENORM: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    ids = tl.load(IDS + row * K + col, col < K, other=-1)
    valid = (col < K) & (ids >= 0) & (ids < E)
    scores = tl.load(G + row * E + ids, valid, other=-float("inf")).to(tl.float32)
    q = tl.where(valid, tl.exp(scores - tl.max(scores, 0)), 0.0)
    total = tl.sum(q, 0)
    q = q / tl.where(total > 0, total, 1.0)
    keep = valid & (q >= P)
    w = tl.load(W + row * K + col, col < K, other=0)
    if RENORM:
        mass = tl.sum(tl.where(keep, q, 0.0), 0)
        w = w / tl.where(mass > 0, mass, 1.0)
    tl.store(OW + row * K + col, tl.where(keep, w, 0), col < K)
    tl.store(OI + row * K + col, tl.where(keep, ids, -1), col < K)


def threshold_experts(weights, ids, logits, probability, renormalize=False):
    """Apply a native-gate threshold without sorting, counters, or CPU sync."""
    if not weights.is_cuda:
        raise ValueError("MoE-Skip threshold routing requires CUDA")
    out_weights, out_ids = torch.empty_like(weights), torch.empty_like(ids)
    if weights.shape[0]:
        _threshold[(weights.shape[0],)](
            weights,
            ids,
            logits,
            out_weights,
            out_ids,
            weights.shape[1],
            logits.shape[1],
            probability,
            renormalize,
            triton.next_power_of_2(weights.shape[1]),
            num_warps=4,
        )
    return out_weights, out_ids
