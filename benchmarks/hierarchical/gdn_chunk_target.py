# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager-only target chunk verification with deferred accepted-state writeback."""

import inspect

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.kernels.benchmark_gdn_chunk_verify import (
    chunk_intermediates,
    recover_prefix_fused,
)
from benchmarks.kernels.gdn_parallel_fp32 import parallel_gdn_fp32_fused
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.triton_utils import triton


class ChunkTargetVerifier:
    def __init__(self, layers, width, check_enabled, check_batch, precision="bf16"):
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

        self.original = gdn.fused_sigmoid_gating_delta_rule_update
        self.signature = inspect.signature(self.original)
        self.canonical = {layer.kv_cache[1].data_ptr() for layer in layers.values()}
        self.width = width
        self.check_enabled = check_enabled
        self.check_batch = check_batch
        self.precision = precision
        self.pending = []
        self.calls = 0
        self.writebacks = 0
        self.checks = []
        self.checked = set()
        gdn.fused_sigmoid_gating_delta_rule_update = self.forward

    def forward(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        cache = p["initial_state"]
        if (
            cache is None
            or cache.data_ptr() not in self.canonical
            or p["num_accepted_tokens"] is None
            or p["is_kda"]
            or not p["use_qk_l2norm_in_kernel"]
            or not p["inplace_final_state"]
            or p["beta"] != 1.0
            or p["threshold"] != 20.0
            or p["scale"] is not None
            or p["k"].shape[-1] != p["v"].shape[-1]
        ):
            return self.original(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Chunk target prototype requires eager execution")
        starts = p["cu_seqlens"]
        n = len(starts) - 1
        indices = p["ssm_state_indices"][:n]
        previous = p["num_accepted_tokens"][:n].long() - 1
        source = indices.gather(1, previous[:, None])[:, 0].long()
        initial = cache[source]
        lengths = starts[1:] - starts[:-1]
        positions = torch.arange(self.width, device=cache.device)
        valid = positions[None] < lengths[:, None]
        offsets = (starts[:-1, None] + positions[None]).long()
        offsets = offsets.clamp_max(p["q"].shape[1] - 1)
        q, k, v = [p[name][0][offsets] for name in ("q", "k", "v")]
        ratio = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(ratio, dim=2)
        k = k.repeat_interleave(ratio, dim=2)
        a, b = [p[name].reshape(-1, v.shape[2])[offsets] for name in ("a", "b")]
        g = -p["A_log"].float().exp() * F.softplus(a.float() + p["dt_bias"])
        g = torch.where(valid[:, :, None], g, 0)
        beta = b.float().sigmoid() * valid[:, :, None]
        if self.precision == "fp32":
            output, k, updates, cumulative = parallel_gdn_fp32_fused(
                q, k, v, g, beta, initial
            )
        else:
            q, k = l2norm_fwd(q), l2norm_fwd(k)
            cumulative, output, updates = chunk_intermediates(
                q, k, v, g, beta, initial, max(16, triton.next_power_of_2(self.width))
            )
        output = output[valid][None]
        reference_states = None
        check = self.check_enabled() and not self.checked and n == self.check_batch
        if check:
            scratch = torch.empty(
                (n * self.width + 1, *cache.shape[1:]),
                dtype=cache.dtype,
                device=cache.device,
            )
            local_indices = torch.arange(
                1, n * self.width + 1, device=cache.device, dtype=torch.int32
            ).view(n, self.width)
            scratch[local_indices[:, 0].long()] = initial
            control = dict(p)
            control.update(
                initial_state=scratch,
                ssm_state_indices=local_indices,
                num_accepted_tokens=torch.ones_like(previous, dtype=torch.int32),
            )
            expected, _ = self.original(**control)
            torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
            reference_states = (
                scratch,
                local_indices,
                (output.float() - expected.float()).abs().max().item(),
            )
            self.checked.add(cache.data_ptr())
        self.pending.append(
            (cache, indices, initial, k, updates, cumulative, lengths, reference_states)
        )
        self.calls += 1
        return output, cache

    def finish(self, batch, sampled):
        if not self.pending:
            return
        proposed = batch.num_draft_tokens_per_req
        rows = np.flatnonzero(proposed[: batch.num_reqs] > 0).tolist()
        prefix = sampled[rows].long()
        for (
            cache,
            indices,
            initial,
            k,
            updates,
            cumulative,
            lengths,
            reference,
        ) in self.pending:
            if len(rows) != len(lengths):
                raise RuntimeError("Target chunk rows do not match sampled requests")
            prefix = torch.minimum(prefix, lengths.long())
            recovered = recover_prefix_fused(initial, k, updates, cumulative, prefix)
            column = (prefix - 1).clamp_min(0)
            destination = indices.gather(1, column[:, None])[:, 0].long()
            if reference is not None:
                scratch, local, output_error = reference
                expected = scratch[local.gather(1, column[:, None])[:, 0].long()]
                expected = torch.where(
                    (prefix > 0)[:, None, None, None], expected, initial
                )
                tolerance = 0.001 if self.precision == "fp32" else 0.02
                torch.testing.assert_close(
                    recovered, expected, atol=tolerance, rtol=tolerance
                )
                self.checks.append(
                    {
                        "rows": len(rows),
                        "state_close": True,
                        "output_max_abs": output_error,
                        "state_max_abs": (recovered - expected).abs().max().item(),
                        "prefix_lengths": prefix.cpu().tolist(),
                    }
                )
            recovered = torch.where(
                ((prefix > 0) & (destination > 0))[:, None, None, None],
                recovered,
                cache[destination],
            )
            cache.index_copy_(0, destination, recovered)
            self.writebacks += 1
        self.pending.clear()
