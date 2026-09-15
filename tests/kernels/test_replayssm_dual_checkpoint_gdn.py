# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.gdn_replayssm_dual_checkpoint import (
    commit_gdn_dual_checkpoint,
    reset_gdn_dual_checkpoint,
)
from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)


def _oracle(state, qkv, a, b, a_log, bias, h, hv, k, v):
    states, outputs = [], []
    for x, ai, bi in zip(qkv.float(), a.float(), b.float()):
        q, key, value = x.split([h * k, h * k, hv * v])
        q = F.normalize(q.reshape(h, k), dim=-1, eps=1e-6)
        key = F.normalize(key.reshape(h, k), dim=-1, eps=1e-6)
        q = q.repeat_interleave(hv // h, 0) * k**-0.5
        key = key.repeat_interleave(hv // h, 0)
        gate = -a_log.exp() * F.softplus(ai + bias)
        state = state * gate.exp()[:, None, None]
        delta = bi.sigmoid()[:, None] * (
            value.reshape(hv, v) - torch.einsum("hvk,hk->hv", state, key)
        )
        state = state + delta[:, :, None] * key[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q))
        states.append(state)
    return torch.stack(outputs), states


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hard_cap,max_t", [(7, 7), (9, 7), (16, 7), (17, 17)])
def test_dual_checkpoint_trajectory(graph, dtype, hard_cap, max_t):
    torch.manual_seed(42)
    device = "cuda"
    h, hv, k, v, rows, slots = 2, 4, 128, 128, 3, 5
    length = 1 << (hard_cap - 1).bit_length()
    state0 = torch.randn(slots, hv, v, k, device=device) * 0.1
    state1 = torch.full_like(state0, float("nan"))
    cache_dtype = torch.float16 if dtype == torch.bfloat16 else dtype
    d = torch.zeros(slots, hv, length, v, device=device, dtype=cache_dtype)
    keys = torch.zeros(slots, h, length, k, device=device, dtype=cache_dtype)
    gates = torch.zeros(slots, hv, length, device=device)
    wp, base, head, prev = [
        torch.zeros(slots, device=device, dtype=torch.int32) for _ in range(4)
    ]
    flush = torch.zeros(slots, device=device, dtype=torch.int8)
    indices = torch.zeros(rows, device=device, dtype=torch.int32)
    qsl = torch.zeros(rows + 1, device=device, dtype=torch.int32)
    qkv = torch.zeros(rows * max_t, 2 * h * k + hv * v, device=device, dtype=dtype)
    a = torch.zeros(rows * max_t, hv, device=device, dtype=dtype)
    b = torch.zeros_like(a)
    a_log = torch.full((hv,), -2.0, device=device)
    bias = torch.zeros(hv, device=device)
    out = torch.empty(rows * max_t, hv, v, device=device, dtype=dtype)

    def forward():
        gdn_replayssm_spec_decode(
            qkv,
            a,
            b,
            a_log,
            bias,
            state0,
            d,
            keys,
            gates,
            out,
            qsl,
            indices,
            wp,
            base,
            flush,
            hard_cap,
            max_t,
            alternate_checkpoint=state1,
            head_slot=head,
            hard_cap=hard_cap,
            dot_precision="tf32" if dtype == torch.bfloat16 else "ieee",
        )

    forward()
    if graph:
        torch.accelerator.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            forward()
    ref = {blk: state0[blk].clone() for blk in (1, 2)}
    candidates: dict[int, list[torch.Tensor]] = {}
    previous_t: dict[int, int] = {}
    expected_h = {1: 0, 2: 0}
    expected_head = {1: 0, 2: 0}
    previous_flush = {1: False, 2: False}
    flush_count = promote_count = 0
    for step in range(48):
        order = [1, 2] if step % 2 else [2, 1]
        ts = [([5, 3, max_t, 1, 4, 2][(step + blk) % 6]) for blk in order]
        if 8 <= step <= 23:
            ts = [5, 5]
        elif 26 <= step <= 43:
            ts = [max_t, max_t]
        reset = step in (0, 25)
        accepted = []
        for blk in order:
            if reset:
                ref[blk] = torch.randn_like(ref[blk]) * 0.1
                state0[blk].copy_(ref[blk])
                expected_h[blk] = expected_head[blk] = 0
                accepted.append(1)
            else:
                t_prev = previous_t[blk]
                acc = [t_prev, 1, max(1, t_prev - 1), 1][(step + blk) % 4]
                if 9 <= step <= 24:
                    acc = 1
                elif 26 <= step <= 43:
                    acc = min(t_prev, max(1, step - 26))
                accepted.append(acc)
                ref[blk] = candidates[blk][acc - 1]
                if acc == t_prev:
                    expected_h[blk] = 0
                    expected_head[blk] = 1 - expected_head[blk]
                    promote_count += 1
                else:
                    expected_h[blk] = (
                        0 if previous_flush[blk] else expected_h[blk]
                    ) + acc
        indices.copy_(torch.tensor(order + [0], device=device))
        qsl.copy_(torch.tensor([0, ts[0], sum(ts), sum(ts)], device=device))
        if step == 25:
            reset_gdn_dual_checkpoint(
                wp,
                base,
                flush,
                head,
                prev,
                indices,
                torch.tensor([0, 1, 3, 3], device=device, dtype=torch.int32),
            )
        commit_gdn_dual_checkpoint(
            wp,
            base,
            flush,
            head,
            prev,
            torch.tensor(accepted + [0], device=device, dtype=torch.int32),
            indices,
            qsl,
            torch.tensor(
                [step == 0, step == 0, False], device=device, dtype=torch.int8
            ),
            hard_cap,
        )
        for blk, t in zip(order, ts):
            assert wp[blk].item() == expected_h[blk]
            assert head[blk].item() == expected_head[blk]
            should_flush = expected_h[blk] + t > hard_cap
            assert bool(flush[blk].item()) == should_flush
            previous_flush[blk] = should_flush
            flush_count += should_flush
        qkv.normal_(std=0.4)
        a.normal_()
        b.normal_()
        if graph:
            captured.replay()
        else:
            forward()
        offset = 0
        for blk, t in zip(order, ts):
            sl = slice(offset, offset + t)
            expected_out, states = _oracle(
                ref[blk], qkv[sl], a[sl], b[sl], a_log, bias, h, hv, k, v
            )
            torch.testing.assert_close(
                out[sl].float(), expected_out, atol=2e-3, rtol=4e-2
            )
            tail = state1 if expected_head[blk] == 0 else state0
            torch.testing.assert_close(tail[blk], states[-1], atol=3e-3, rtol=4e-2)
            candidates[blk] = states
            previous_t[blk] = t
            offset += t
        assert prev[0].item() == 0
        assert torch.isnan(state1[0]).all()
    assert flush_count > 0
    assert promote_count > 0
