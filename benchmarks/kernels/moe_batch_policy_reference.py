# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def batch_policy_reference(
    weights, ids, logits, policy, is_padding=None, *, protected_top_k=2
):
    """Independent, untimed CPU oracle and diagnostics for batch routing."""
    if not 1 <= protected_top_k <= ids.shape[1]:
        raise ValueError("protected_top_k must be within the native top-k")
    device = weights.device
    w, routes, gate = weights.cpu(), ids.cpu(), logits.float().cpu()
    padding = (
        torch.zeros(routes.shape[0], dtype=torch.bool)
        if is_padding is None
        else is_padding.cpu()
    )
    scores = [0.0] * gate.shape[1]
    protected, active = set(), set()
    for row in range(routes.shape[0]):
        if padding[row]:
            continue
        selected = [
            (int(e), float(w[row, j]))
            for j, e in enumerate(routes[row])
            if 0 <= e < gate.shape[1]
        ]
        for expert, weight in selected:
            active.add(expert)
            scores[expert] += weight
        ordered = sorted(
            selected, key=lambda pair: (-float(gate[row, pair[0]]), pair[0])
        )
        protected.update(expert for expert, _ in ordered[:protected_top_k])
    candidates = sorted(active - protected, key=lambda e: (-scores[e], e))
    n = len(candidates)
    if policy == "batch_top_half":
        n = (n + 1) // 2
    elif policy == "batch_max_gap":
        if n > 1:
            gaps = [
                scores[a] - scores[b] for a, b in zip(candidates[:-1], candidates[1:])
            ]
            if max(gaps) > 0:
                n = gaps.index(max(gaps)) + 1
    else:
        raise ValueError(policy)
    retained = protected | set(candidates[:n])
    keep = torch.zeros_like(routes, dtype=torch.bool)
    for expert in retained:
        keep |= routes == expert
    keep &= ~padding[:, None]
    return (
        w.masked_fill(~keep, 0).to(device),
        routes.masked_fill(~keep, -1).to(device),
        dict(
            native_unique=len(active),
            protected_unique=len(protected),
            candidate_unique=len(candidates),
            retained_unique=len(retained),
        ),
    )
