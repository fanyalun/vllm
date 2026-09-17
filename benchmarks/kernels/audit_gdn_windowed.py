# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent FP32 recurrence and V4 rounding audits on captured layer inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    windowed_replay_tail_update,
)


def reference(values, window, cumulative=False, tau_beta=0.36328125):
    q, k, v, a, b, a_log, dt, initial = values
    state = initial[0].clone()
    q, k, v = q.float(), k.float(), v.float()
    q *= (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * q.shape[-1] ** -0.5
    k *= (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    q, k = [x.repeat_interleave(v.shape[1] // x.shape[1], 1) for x in (q, k)]
    x = a.float() + dt.float()
    g = -a_log.float().exp() * torch.where(x <= 20, (1 + x.exp()).log(), x)
    beta = b.float().sigmoid()
    outputs, actions = [], []
    for s in range(0, len(q), window):
        full = beta[s].bfloat16().float() >= tau_beta
        decay = ~full & (g[s].exp() <= 0.95)
        action = torch.where(full, 0, torch.where(decay, 1, 2))
        anchor, total = state.clone(), torch.zeros_like(g[s])
        for t in range(s, min(s + window, len(q))):
            decayed = state * g[t].exp()[:, None, None]
            delta = (v[t] - (decayed * k[t, :, None]).sum(-1)) * beta[t, :, None]
            updated = decayed + delta[:, :, None] * k[t, :, None]
            state = torch.where(full[:, None, None], updated, state)
            total += g[t]
            if cumulative:
                decayed = anchor * total.exp()[:, None, None]
            state = torch.where(decay[:, None, None], decayed, state)
            output = (state * q[t, :, None]).sum(-1)
            if cumulative:
                output = torch.where(
                    decay[:, None],
                    (anchor * q[t, :, None]).sum(-1) * total.exp()[:, None],
                    output,
                )
            outputs.append(output)
            actions.append(action)
    return torch.stack(outputs).to(values[2].dtype), state[None], torch.stack(actions)


def errors(actual, expected):
    delta = actual.float() - expected.float()
    return dict(
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / expected.float().norm().clamp_min(1e-20)).item(),
        passes=torch.allclose(actual, expected, atol=1e-3, rtol=1e-3),
        worst_head=delta.movedim(1, 0).flatten(1).abs().amax(1).argmax().item(),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    saved = torch.load(args.inputs, weights_only=False)
    configs = [("v1", 5, "none", 0.0), ("v2", 1, "none", 0.36328125)] + [
        (name, 5, opt, 0.36328125)
        for name, opt in (
            ("v3", "none"),
            ("v4_d", "cumulative_decay"),
            ("v4_q", "multi_query"),
            ("v4_dq", "combined"),
        )
    ]
    for length in (1, 5, 6, 10, 16):
        for layer, entry in enumerate(saved):
            values = [x.cuda() for x in entry["values"]]
            for i in range(5):
                values[i] = values[i].repeat((4,) + (1,) * (values[i].ndim - 1))[
                    :length
                ]
            starts = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            valid = torch.ones(1, device="cuda", dtype=torch.int32)
            initial = values[-1].clone()
            for case, window, optimization, tau_beta in configs:
                values[-1].copy_(initial)
                expected, expected_state, actions = reference(
                    values,
                    window,
                    optimization in ("cumulative_decay", "combined"),
                    tau_beta,
                )
                step_out, step_state, _ = reference(values, window, False, tau_beta)
                counts = torch.zeros(8, device="cuda", dtype=torch.int64)
                output = windowed_replay_tail_update(
                    *values,
                    query_start_loc=starts,
                    valid=valid,
                    thresholds=torch.tensor([0.95, tau_beta], device="cuda"),
                    window_size=window,
                    optimization=optimization,
                    action_counts=counts,
                )
                row = dict(
                    case=case,
                    layer=layer,
                    layer_name=entry.get("layer", str(layer)),
                    tokens=length,
                    output=errors(output, expected),
                    state=errors(values[-1], expected_state),
                    versus_stepwise_output=errors(output, step_out),
                    versus_stepwise_state=errors(values[-1], step_state),
                    action_match=counts[:3].tolist()
                    == torch.bincount(actions.flatten(), minlength=3).tolist(),
                )
                rows.append(row)
    result = dict(
        rows=rows,
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        complete=True,
        numerical_pass=all(
            r["output"]["passes"] and r["state"]["passes"] and r["action_match"]
            for r in rows
        ),
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
