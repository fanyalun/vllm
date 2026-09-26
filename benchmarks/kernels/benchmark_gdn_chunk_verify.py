# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound short-block parallel verification without claiming prefix recovery."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.triton_utils import tl, triton


@triton.jit
def _recover_prefix(
    INITIAL,
    KEYS,
    UPDATES,
    G,
    LENGTHS,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    nh = tl.program_id(0)
    n, h = nh // H, nh % H
    vs = tl.program_id(1) * 32 + tl.arange(0, 32)
    ks = tl.program_id(2) * 32 + tl.arange(0, 32)
    ts = tl.arange(0, BT)
    prefix = tl.load(LENGTHS + n).to(tl.int32)
    last = tl.load(
        G + (n * T + tl.maximum(prefix - 1, 0)) * H + h, mask=prefix > 0, other=0
    )
    included = ts < prefix
    gates = tl.load(G + (n * T + ts) * H + h, mask=ts < T, other=0)
    weights = tl.exp(tl.where(included, last - gates, 0)) * included
    values = (
        tl.load(
            UPDATES + ((n * T + ts[None, :]) * H + h) * D + vs[:, None],
            mask=(ts[None, :] < T) & (vs[:, None] < D),
            other=0,
        ).to(tl.float32)
        * weights[None, :]
    )
    keys = tl.load(
        KEYS + ((n * T + ts[:, None]) * H + h) * D + ks[None, :],
        mask=(ts[:, None] < T) & (ks[None, :] < D),
        other=0,
    ).to(tl.float32)
    offsets = nh.to(tl.int64) * D * D + vs[:, None] * D + ks[None, :]
    initial = tl.load(
        INITIAL + offsets, mask=(vs[:, None] < D) & (ks[None, :] < D), other=0
    )
    output = initial * tl.exp(last) + tl.dot(values, keys, input_precision="tf32x3")
    tl.store(OUT + offsets, output, mask=(vs[:, None] < D) & (ks[None, :] < D))


def recover_prefix_fused(initial, keys, updates, cumulative_g, prefix_lengths):
    batch, length, heads, dim = updates.shape
    output = torch.empty_like(initial)
    _recover_prefix[(batch * heads, triton.cdiv(dim, 32), triton.cdiv(dim, 32))](
        initial,
        keys,
        updates,
        cumulative_g,
        prefix_lengths,
        output,
        length,
        heads,
        dim,
        max(32, triton.next_power_of_2(length)),
        num_warps=4,
    )
    return output


def recover_prefix(initial, keys, updates, cumulative_g, prefix_lengths):
    batch, length, heads, _ = updates.shape
    rows = torch.arange(batch, device=updates.device)
    last = cumulative_g[rows, (prefix_lengths - 1).clamp_min(0)]
    last = torch.where(prefix_lengths[:, None] > 0, last, 0)
    included = (
        torch.arange(length, device=updates.device)[None] < prefix_lengths[:, None]
    )
    differences = torch.where(included[:, :, None], last[:, None] - cumulative_g, 0)
    weights = differences.exp() * included[:, :, None]
    scaled = updates.float() * weights[..., None]
    delta = scaled.permute(0, 2, 3, 1) @ keys.float().permute(0, 2, 1, 3)
    return initial * last.exp()[:, :, None, None] + delta


@torch.inference_mode()
def check_recovery_boundaries():
    initial = torch.randn(3, 2, 128, 128, device="cuda")
    keys, updates = [
        torch.randn(3, 17, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    prefixes = torch.tensor([0, 9, 17], device="cuda")
    for decay in (0.0, -1e-6, -5.0, -80.0):
        gates = torch.full((3, 17, 2), decay, device="cuda").cumsum(1)
        expected = recover_prefix(initial, keys, updates, gates, prefixes)
        actual = recover_prefix_fused(initial, keys, updates, gates, prefixes)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert torch.equal(actual[0], initial[0])


def chunk_intermediates(q, k, v, g, beta, initial, chunk_size):
    cumulative = chunk_local_cumsum(g, chunk_size=chunk_size)
    triangular = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=cumulative,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    triangular = solve_tril(A=triangular, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=triangular, g_cumsum=cumulative, cu_seqlens=None
    )
    h, updates, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=cumulative,
        initial_state=initial,
        output_final_state=False,
        chunk_size=chunk_size,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=updates,
        h=h,
        g=cumulative,
        scale=k.shape[-1] ** -0.5,
        chunk_size=chunk_size,
    )
    return cumulative, output, updates


@torch.inference_mode()
def probe(batch, length, heads, repeats):
    h, hv, dim = heads // 2, heads, 128
    q, k = [
        torch.randn(batch, length, h, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(batch, length, hv, dim, device="cuda", dtype=torch.bfloat16)
    a, b = [
        torch.randn(batch, length, hv, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    a_log, dt = [torch.randn(hv, device="cuda") for _ in range(2)]
    initial = torch.randn(batch, hv, dim, dim, device="cuda") * 0.05
    states = torch.empty(batch * length + 1, hv, dim, dim, device="cuda")
    indices = torch.arange(1, batch * length + 1, device="cuda").to(torch.int32)
    indices = indices.view(batch, length)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    prefix_lengths = torch.arange(batch, device="cuda") % (length + 1)
    flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def reset():
        states[indices[:, 0].long()] = initial

    def recurrent():
        return fused_sigmoid_gating_delta_rule_update(
            a_log,
            a.flatten(0, 1),
            b.flatten(0, 1),
            dt,
            q.flatten(0, 1)[None],
            k.flatten(0, 1)[None],
            v.flatten(0, 1)[None],
            initial_state=states,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            cu_seqlens=starts,
            use_qk_l2norm_in_kernel=True,
        )[0].view_as(v)

    def chunk():
        g = -a_log.exp() * F.softplus(a.float() + dt)
        beta = b.float().sigmoid()
        return chunk_gated_delta_rule(
            q.repeat_interleave(2, dim=2),
            k.repeat_interleave(2, dim=2),
            v,
            g,
            beta,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    def chunk_prefix(fused=False, short_tile=False):
        normalized_q = l2norm_fwd(q.repeat_interleave(2, dim=2))
        normalized_k = l2norm_fwd(k.repeat_interleave(2, dim=2))
        g = -a_log.exp() * F.softplus(a.float() + dt)
        beta = b.float().sigmoid()
        tile = max(16, triton.next_power_of_2(length)) if short_tile else 64
        cumulative_g, output, updates = chunk_intermediates(
            normalized_q, normalized_k, v, g, beta, initial, tile
        )
        recover = recover_prefix_fused if fused else recover_prefix
        state = recover(initial, normalized_k, updates, cumulative_g, prefix_lengths)
        return output, state

    reset()
    expected = recurrent().clone()
    expected_state = states[indices[:, -1].long()].clone()
    actual, final = chunk()
    errors = {}
    for name, x, y in (
        ("output", actual, expected),
        ("final_state", final, expected_state),
    ):
        errors[name] = {
            "max_abs": (x.float() - y.float()).abs().max().item(),
            "relative_l2": (
                (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)
            ).item(),
            "close": torch.allclose(x, y, atol=0.01, rtol=0.01),
        }
    if not all(x["close"] for x in errors.values()):
        return dict(batch=batch, length=length, errors=errors, timed=False)
    prefix_max_abs = 0.0
    for prefix in range(length + 1):
        prefix_lengths.fill_(prefix)
        output, recovered = chunk_prefix()
        reference = initial if prefix == 0 else states[indices[:, prefix - 1].long()]
        prefix_max_abs = max(prefix_max_abs, (recovered - reference).abs().max().item())
        torch.testing.assert_close(recovered, reference, atol=0.01, rtol=0.01)
        torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
        _, fused_state = chunk_prefix(fused=True)
        torch.testing.assert_close(fused_state, recovered, atol=1e-5, rtol=1e-4)
        short_output, short_state = chunk_prefix(fused=True, short_tile=True)
        torch.testing.assert_close(short_output, expected, atol=0.01, rtol=0.01)
        torch.testing.assert_close(short_state, reference, atol=0.01, rtol=0.01)
    prefix_lengths.copy_(torch.arange(batch, device="cuda") % (length + 1))
    _, mixed_state = chunk_prefix(fused=True)
    selected = indices.gather(1, (prefix_lengths - 1).clamp_min(0)[:, None])[:, 0]
    reference = torch.where(
        (prefix_lengths > 0)[:, None, None, None], states[selected.long()], initial
    )
    torch.testing.assert_close(mixed_state, reference, atol=0.01, rtol=0.01)
    errors["all_prefix_states"] = {"close": True, "max_abs": prefix_max_abs}
    graphs = {}
    for name, fn in (
        ("all_snapshots", recurrent),
        ("chunk_final_only", chunk),
        ("chunk_with_prefix_recovery", chunk_prefix),
        ("chunk_with_fused_recovery", lambda: chunk_prefix(fused=True)),
        ("short_chunk_with_fused_recovery", lambda: chunk_prefix(True, True)),
    ):
        for _ in range(5):
            reset()
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = fn()
        graphs[name] = graph, outputs
    times = {name: [] for name in graphs}
    for repeat in range(repeats):
        order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in order:
            reset()
            flush.zero_()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end))
    median = {name: statistics.median(values) for name, values in times.items()}
    return dict(
        batch=batch,
        length=length,
        errors=errors,
        timed=True,
        median_ms=median,
        samples_ms=times,
        chunk_speedup=median["all_snapshots"] / median["chunk_final_only"],
        chunk_prefix_speedup=(
            median["all_snapshots"] / median["chunk_with_prefix_recovery"]
        ),
        chunk_fused_prefix_speedup=(
            median["all_snapshots"] / median["chunk_with_fused_recovery"]
        ),
        short_chunk_fused_prefix_speedup=(
            median["all_snapshots"] / median["short_chunk_with_fused_recovery"]
        ),
        snapshot_state_bytes=(length + 1) * initial.numel() * 4,
        final_only_state_bytes=2 * initial.numel() * 4,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 17])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if (
        args.heads < 2
        or args.heads % 2
        or min(args.batches) < 1
        or min(args.lengths) < 1
        or max(args.lengths) > 64
        or args.repeats < 1
    ):
        parser.error("Expected even heads, positive batches/repeats and lengths 1..64")
    torch.manual_seed(42)
    check_recovery_boundaries()
    result = dict(
        scope="Synthetic post-conv GDN; prefix recovery is explicitly timed",
        timing="CUDA events, one graph replay, 64 MiB L2 flush outside timer",
        dtype="bfloat16",
        state_dtype="float32",
        heads=args.heads,
        gpu=torch.cuda.get_device_name(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recovery_boundary_checks=True,
        torch_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for length in args.lengths:
            row = probe(batch, length, args.heads, args.repeats)
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_ms"}),
                flush=True,
            )
    result["compiled_recovery"] = [
        {
            "registers": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
        }
        for cache in _recover_prefix.device_caches.values()
        for kernel in cache[0].values()
    ]
    result["complete"] = all(row["timed"] for row in result["rows"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
