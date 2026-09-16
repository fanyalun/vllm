# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI comparison of causal GDN recurrence and private replay-tail updates."""

import argparse
import importlib.util
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    _replay_tail_update,
    replay_tail_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for tokens in range(1, 6):
        h, hv, dim = 16, 32, 128
        packed = torch.randn(
            tokens, (2 * h + hv) * dim, device="cuda", dtype=torch.bfloat16
        )
        q, k, v = packed.split([h * dim, h * dim, hv * dim], -1)
        q, k, v = (
            q.view(tokens, h, dim),
            k.view(tokens, h, dim),
            v.view(tokens, hv, dim),
        )
        a, b = torch.randn(tokens, 2 * hv, device="cuda", dtype=torch.bfloat16).split(
            hv, -1
        )
        full_a, full_b = a, b
        a_log, dt = [torch.randn(hv, device="cuda") for _ in range(2)]
        start = torch.randn(1, hv, dim, dim, device="cuda") * 0.05
        state = start.clone()
        baseline = start.repeat(tokens + 1, 1, 1, 1)
        indices = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32)[None]
        accepted = torch.ones(1, device="cuda", dtype=torch.int32)
        lengths = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

        exact = partial(
            fused_sigmoid_gating_delta_rule_update,
            a_log,
            full_a,
            full_b,
            dt,
            q[None],
            k[None],
            v[None],
            initial_state=baseline,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            cu_seqlens=lengths,
            use_qk_l2norm_in_kernel=True,
        )

        replay = partial(replay_tail_update, q, k, v, a, b, a_log, dt, state)

        expected = exact()[0][0].clone()
        actual = replay()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(state[0], baseline[-1], rtol=1e-3, atol=1e-3)
        timings = {"baseline": [], "replay_tail": []}
        for repeat in range(3):
            cases = [("baseline", exact), ("replay_tail", replay)]
            for name, fn in cases if repeat % 2 == 0 else cases[::-1]:
                state.copy_(start)
                baseline.copy_(start.expand_as(baseline))
                for _ in range(25):
                    fn()
                torch.accelerator.synchronize()
                timings[name].append(
                    statistics.median(
                        bench_gpu_time_with_cupti(
                            fn.func,
                            input_args=fn.args,
                            input_kwargs=fn.keywords,
                            use_cuda_graph=True,
                            cold_l2_cache=True,
                        )
                    )
                    * 1000
                )
        state_bytes = start.numel() * start.element_size()
        rows.append(
            {
                "tokens": tokens,
                "median_us": {
                    name: statistics.median(times) for name, times in timings.items()
                },
                "trials_us": timings,
                "state_store_bytes": {
                    "baseline": tokens * state_bytes,
                    "replay_tail": state_bytes,
                },
            }
        )
    compiled = []
    for cache in _replay_tail_update.device_caches.values():
        for kernel in cache[0].values():
            compiled.append(
                {
                    "registers": kernel.n_regs,
                    "spills": kernel.n_spills,
                    "shared_bytes": kernel.metadata.shared,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "timing_backend": (
                    "cupti"
                    if importlib.util.find_spec("cupti")
                    else "cuda_events_graph"
                ),
                "cold_l2_cache": True,
                "dtype": "bfloat16",
                "state_dtype": "float32",
                "heads": [16, 32],
                "head_dim": 128,
                "boundary": (
                    "Triton recurrence wrappers with packed inputs and per-token "
                    "gates; excludes gate projection and model CUDA fused GDN"
                ),
                "compiled": compiled,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
