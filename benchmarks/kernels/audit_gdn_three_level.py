# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit real raw gates against independent recurrence and archived replay-tail."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import replay_tail_update


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("archived_replay_tail", args.original)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    rows = []
    thresholds = torch.tensor([0.98, 0.36328125], device="cuda")
    for index, entry in enumerate(inputs):
        q0, k0, v0, a0, b0, al, dt, initial = [x.cuda() for x in entry["values"]]
        for tokens in range(1, 6):
            q, k, v, a, b = [x[:tokens] for x in (q0, k0, v0, a0, b0)]
            query = (
                q.float() * (q.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            )
            key = k.float() * (k.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            query = (
                query.repeat_interleave(v.shape[1] // q.shape[1], 1)
                * q.shape[-1] ** -0.5
            )
            key = key.repeat_interleave(v.shape[1] // k.shape[1], 1)
            alpha = (
                -al.float().exp() * torch.nn.functional.softplus(a.float() + dt.float())
            ).exp()
            beta = b.float().sigmoid()
            for action, settings in (
                ("full", [0.98, 0.0]),
                ("skip", [-1.0, 2.0]),
                ("decay", [2.0, 2.0]),
                ("mixed", [0.98, 0.36328125]),
            ):
                thresholds.copy_(torch.tensor(settings, device="cuda"))
                tail = torch.empty_like(initial)
                actual = replay_tail_update(
                    q,
                    k,
                    v,
                    a,
                    b,
                    al,
                    dt,
                    initial,
                    tail=tail,
                    thresholds=thresholds,
                    value_tile=8 if tokens < 3 else 16,
                    num_warps=4 if tokens < 3 else 8,
                )
                full = beta.bfloat16().float() >= thresholds[1]
                decay = full | (alpha <= thresholds[0])
                reference = initial.clone()
                outputs = []
                for t in range(tokens):
                    reference *= torch.where(decay[t], alpha[t], 1)[None, :, None, None]
                    delta = (
                        v[t].float() - (reference[0] * key[t, :, None, :]).sum(-1)
                    ) * beta[t, :, None]
                    reference += (delta * full[t, :, None])[None, :, :, None] * key[
                        t, None, :, None, :
                    ]
                    outputs.append((reference[0] * query[t, :, None, :]).sum(-1))
                expected = torch.stack(outputs).to(v.dtype)
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(tail, reference, rtol=1e-3, atol=1e-3)
                if action == "full":
                    old_state = initial.clone()
                    old_out = original.replay_tail_update(
                        q, k, v, a, b, al, dt, old_state
                    )
                    torch.testing.assert_close(actual, old_out, rtol=0, atol=0)
                    torch.testing.assert_close(tail, old_state, rtol=0, atol=0)
                if action == "skip":
                    torch.testing.assert_close(tail, initial, rtol=0, atol=0)
                delta = (tail - reference).abs()[0].amax((-1, -2))
                rows.append(
                    dict(
                        layer=index,
                        tokens=tokens,
                        action=action,
                        max_tail_error=float(delta.max()),
                        worst_head=int(delta.argmax()),
                        max_output_error=float(
                            (actual.float() - expected.float()).abs().max()
                        ),
                    )
                )
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                complete=True,
                rtol=1e-3,
                atol=1e-3,
                original_full_bitwise=True,
                skip_tail_bitwise=True,
            ),
            indent=2,
        )
    )
    print("PASS", len(rows), "real-input cases")


if __name__ == "__main__":
    main()
