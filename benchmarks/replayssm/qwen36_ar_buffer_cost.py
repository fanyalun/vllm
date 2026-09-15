# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare non-spec GDN update/writeback with ReplaySSM without state writeback."""

import argparse
import gc
import hashlib
import random
import statistics
import subprocess
from pathlib import Path

import torch
from qwen36_ar_no_state_write import fused_recurrent_gated_delta_rule_replayssm
from qwen36_flush_crossover import Inputs, capture, save
from qwen36_state_cost import state_store

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.triton_utils import triton


def benchmark(root):
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for filename in (
        "qwen36_ar_buffer_cost.py",
        "qwen36_flush_crossover.py",
        "qwen36_state_cost.py",
        "qwen36_ar_no_state_write.py",
    ):
        content = Path(__file__).with_name(filename).read_bytes()
        (root / (filename + ".txt")).write_bytes(content)
        sources[filename] = hashlib.sha256(content).hexdigest()
    save(
        root / "environment.json",
        dict(
            source_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            sources=sources,
            batch=[1, 4, 8, 16, 32],
            buffer_size=[4, 8, 16, 32],
            checkpoint_to_output_distance="h",
            cached_history="h-1",
            current_token_count=1,
            state_dtype="float32",
            cache_dk_dtype="float16",
            cache_g_dtype="float32",
            input_dtype="bfloat16",
            repeats=7,
            seed=0,
            kernel_configuration="production defaults; benchmark forces non-flush",
            full_state_writeback=False,
            history_cache_writes=True,
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv",
                ],
                text=True,
            ),
        ),
    )
    eviction = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
    clear = capture(
        lambda: state_store[(triton.cdiv(eviction.numel(), 1024),)](
            eviction, eviction.numel(), 1024, num_warps=4
        )
    )
    rows = []
    for batch in (1, 4, 8, 16, 32):
        x = Inputs(batch, 0, 0)
        previous = x.states[32]
        for h in (4, 8, 16, 32):
            checkpoint = x.states[33 - h]
            base = previous.clone()
            replay = checkpoint.clone()
            output_base = torch.empty(
                batch, 1, 32, 128, device="cuda", dtype=torch.bfloat16
            )
            output_replay = torch.empty_like(output_base)
            d = torch.zeros(batch + 1, 32, h, 128, device="cuda", dtype=torch.float16)
            k = torch.zeros(batch + 1, 16, h, 128, device="cuda", dtype=torch.float16)
            g = torch.zeros(batch + 1, 32, h, device="cuda")
            d[:, :, : h - 1] = x.history_d[:, :, 33 - h : 32]
            k[:, :, : h - 1] = x.history_k[:, :, 33 - h : 32]
            g[:, :, : h - 1] = x.history_g[:, :, 33 - h : 32]
            wp = torch.full((batch,), h - 1, device="cuda", dtype=torch.int32)
            common = dict(
                mixed_qkv=x.qkv,
                a=x.a,
                b=x.b,
                A_log=x.a_log,
                dt_bias=x.bias,
                scale=128**-0.5,
                ssm_state_indices=x.slots,
                use_qk_l2norm_in_kernel=True,
            )

            def baseline_call(common=common, base=base, out=output_base):
                fused_recurrent_gated_delta_rule_packed_decode(
                    **common, initial_state=base, out=out
                )

            def replay_call(
                common=common, replay=replay, out=output_replay, d=d, k=k, g=g, wp=wp
            ):
                fused_recurrent_gated_delta_rule_replayssm(
                    **common,
                    initial_state=replay,
                    out=out,
                    d_cache=d,
                    k_cache=k,
                    g_cache=g,
                    write_pos=wp,
                )

            graphs = {
                "baseline": capture(baseline_call),
                "replayssm": capture(replay_call),
            }
            base.copy_(previous)
            replay.copy_(checkpoint)
            baseline_call()
            replay_call()
            torch.testing.assert_close(
                output_replay, output_base, rtol=0.02, atol=0.002
            )
            torch.testing.assert_close(replay, checkpoint, rtol=0, atol=0)
            torch.testing.assert_close(replay[0], checkpoint[0], rtol=0, atol=0)
            output_error = (
                (output_replay.float() - output_base.float()).abs().max().item()
            )
            state_error = (replay - checkpoint).abs().max().item()
            gates = g[1:].float()
            decay = (gates.sum(-1, keepdim=True) - gates.cumsum(-1)).exp()
            recovered = checkpoint[1:] * gates.sum(-1).exp()[..., None, None]
            recovered += torch.einsum(
                "bhiv,bhik->bhvk",
                d[1:].float() * decay[..., None],
                k[1:].float().repeat_interleave(2, dim=1),
            )
            torch.testing.assert_close(recovered, base[1:], rtol=0.02, atol=0.002)
            recovered_error = (recovered - base[1:]).abs().max().item()
            for cache in ("warm", "evicted"):
                values = {name: [] for name in graphs}
                for repeat in range(7):
                    names = list(graphs)
                    random.Random(batch * 100 + h + repeat).shuffle(names)
                    for name in names:
                        graph = graphs[name]
                        dest, source = (
                            (base, previous)
                            if name == "baseline"
                            else (replay, checkpoint)
                        )
                        start = torch.Event(device="cuda", enable_timing=True)
                        end = torch.Event(device="cuda", enable_timing=True)
                        dest.copy_(source)
                        clear.replay()
                        if cache == "warm":
                            for _ in range(3):
                                graph.replay()
                            dest.copy_(source)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        values[name].append(start.elapsed_time(end) * 1000)
                rows.append(
                    dict(
                        batch=batch,
                        buffer_size=h,
                        history=h - 1,
                        distance_to_output=h,
                        cache=cache,
                        us=values,
                        correctness=True,
                        output_max_abs_error=output_error,
                        unchanged_checkpoint_max_abs_error=state_error,
                        reconstructed_state_max_abs_error=recovered_error,
                    )
                )
                save(root / "raw.json", rows)
                print(
                    batch,
                    h,
                    cache,
                    {k: round(statistics.median(v), 3) for k, v in values.items()},
                    flush=True,
                )
            del graphs, common, base, replay, d, k, g, wp, output_base, output_replay
        del x, previous, checkpoint
        gc.collect()
    save(
        root / "measurement_complete.json",
        dict(
            cells=20,
            cache_conditions=2,
            repetitions=7,
            timed_values=560,
            correctness=True,
            figures_validated=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    benchmark(Path(parser.parse_args().output))


if __name__ == "__main__":
    main()
