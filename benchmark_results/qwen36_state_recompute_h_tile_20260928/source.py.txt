# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare exact-h GDN recurrence with fixed-K16 reconstruction."""

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

import torch
from qwen36_flush_crossover import capture
from qwen36_state_cost import state_store
from qwen36_state_recompute_compute_only import (
    BATCHES,
    HISTORIES,
    REPEATS,
    recompute_no_state_io,
    timed,
)

from vllm.triton_utils import tl, triton


@triton.jit
def recompute_exact_h(
    d, k, g, checksum, full_out, H: tl.constexpr, STORE_FULL: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    ov = (tile // 4) * 32 + tl.arange(0, 32)
    ok = (tile % 4) * 32 + tl.arange(0, 32)
    state = (ov[:, None] * 128 + ok[None, :]).to(tl.float32) * 0.000001
    state = state + (batch * 32 + head) * 0.00001
    for step in tl.static_range(0, H):
        gate = tl.load(g + (batch * 32 + head) * H + step)
        delta = tl.load(d + ((batch * 32 + head) * H + step) * 128 + ov).to(tl.float32)
        key = tl.load(k + ((batch * 16 + head // 2) * H + step) * 128 + ok).to(
            tl.float32
        )
        state = tl.exp(gate) * state + delta[:, None] * key[None, :]
    if STORE_FULL:
        offset = (batch * 32 + head) * 128 * 128 + ov[:, None] * 128 + ok[None, :]
        tl.store(full_out + offset, state)
    else:
        value = tl.sum(tl.sum(state, 0), 0)
        tl.store(checksum + (batch * 32 + head) * 16 + tile, value)


@triton.jit
def recompute_h_tile(
    d, k, g, checksum, full_out, H: tl.constexpr, STORE_FULL: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    ov = (tile // 4) * 32 + tl.arange(0, 32)
    ok = (tile % 4) * 32 + tl.arange(0, 32)
    oh = tl.arange(0, H)
    gates = tl.load(g + (batch * 32 + head) * H + oh)
    prefix = tl.cumsum(gates, 0)
    total = tl.sum(gates, 0)
    decay = tl.exp(total - prefix)
    delta = tl.load(d + ((batch * 32 + head) * H + oh[None, :]) * 128 + ov[:, None]).to(
        tl.float32
    )
    keys = tl.load(
        k + ((batch * 16 + head // 2) * H + oh[:, None]) * 128 + ok[None, :]
    ).to(tl.float32)
    initial = (ov[:, None] * 128 + ok[None, :]).to(tl.float32) * 0.000001
    initial = initial + (batch * 32 + head) * 0.00001
    state = tl.exp(total) * initial
    for step in tl.static_range(0, H):
        dvec = tl.gather(delta, tl.full((32, 1), step, tl.int32), axis=1)
        kvec = tl.gather(keys, tl.full((1, 32), step, tl.int32), axis=0)
        weight = tl.gather(decay, tl.full((1,), step, tl.int32), axis=0)
        state += (dvec * weight[None, :]) * kvec
    if STORE_FULL:
        offset = (batch * 32 + head) * 128 * 128 + ov[:, None] * 128 + ok[None, :]
        tl.store(full_out + offset, state)
    else:
        value = tl.sum(tl.sum(state, 0), 0)
        tl.store(checksum + (batch * 32 + head) * 16 + tile, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    dependency = (
        Path(__file__).with_name("qwen36_state_recompute_compute_only.py").read_bytes()
    )
    (root / "source.py.txt").write_bytes(source)
    (root / "fixed16_source.py.txt").write_bytes(dependency)
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    eviction = torch.empty(256 * 1024 * 1024 // 4, device=device)
    eviction_graph = capture(
        lambda: state_store[(triton.cdiv(eviction.numel(), 1024),)](
            eviction, eviction.numel(), 1024, num_warps=4
        )
    )
    rows = []
    for batch in BATCHES:
        store_out = torch.empty((batch, 32, 128, 128), device=device)
        exact_out = torch.empty_like(store_out)
        tile_out = torch.empty_like(store_out)
        fixed_out = torch.empty_like(store_out)
        checksum = torch.empty((batch, 32, 16), device=device)
        count = store_out.numel()

        def store_call(out=store_out, count=count):
            state_store[(triton.cdiv(count, 1024),)](out, count, 1024, num_warps=4)

        store_graph = capture(store_call)
        for h in HISTORIES:
            d = (torch.randn((batch, 32, h, 128), device=device) * 0.01).half()
            k = (torch.randn((batch, 16, h, 128), device=device) * 0.1).half()
            g = -torch.rand((batch, 32, h), device=device) * 0.03

            def exact_call(
                store_full=False,
                batch=batch,
                d=d,
                k=k,
                g=g,
                checksum=checksum,
                out=exact_out,
                h=h,
            ):
                recompute_exact_h[(batch, 32, 16)](
                    d, k, g, checksum, out, h, store_full, num_warps=4
                )

            def fixed_call(
                store_full=False,
                batch=batch,
                d=d,
                k=k,
                g=g,
                checksum=checksum,
                out=fixed_out,
                h=h,
            ):
                recompute_no_state_io[(batch, 32, 16)](
                    d, k, g, checksum, out, h, store_full, num_warps=4
                )

            def tile_call(
                store_full=False,
                batch=batch,
                d=d,
                k=k,
                g=g,
                checksum=checksum,
                out=tile_out,
                h=h,
            ):
                recompute_h_tile[(batch, 32, 16)](
                    d, k, g, checksum, out, h, store_full, num_warps=4
                )

            exact_call(store_full=True)
            fixed_call(store_full=True)
            tile_call(store_full=True)
            gate = g.double()
            decay = (gate.sum(-1, keepdim=True) - gate.cumsum(-1)).exp()
            s0 = torch.arange(128 * 128, device=device, dtype=torch.float64)
            s0 = s0.reshape(1, 1, 128, 128) * 0.000001
            ids = torch.arange(batch * 32, device=device, dtype=torch.float64)
            s0 = s0 + ids.reshape(batch, 32, 1, 1) * 0.00001
            oracle = s0 * gate.sum(-1).exp()[..., None, None]
            oracle += torch.einsum(
                "bhiv,bhik->bhvk",
                d.double() * decay[..., None],
                k.double().repeat_interleave(2, dim=1),
            )
            for output in (exact_out, fixed_out, tile_out):
                torch.testing.assert_close(
                    output.double(), oracle, rtol=0.002, atol=0.0002
                )
            exact_error = (exact_out.double() - oracle).abs().max().item()
            fixed_error = (fixed_out.double() - oracle).abs().max().item()
            tile_error = (tile_out.double() - oracle).abs().max().item()
            graphs = {
                "store": store_graph,
                "fixed16": capture(fixed_call),
                "exact_h": capture(exact_call),
                "tile_h": capture(tile_call),
            }
            for cache in ("warm", "evicted"):
                values = {name: [] for name in graphs}
                for repeat in range(REPEATS):
                    names = list(graphs)
                    random.Random(batch * 1000 + h * 10 + repeat).shuffle(names)
                    for name in names:
                        values[name].append(
                            timed(graphs[name], eviction_graph, cache == "warm")
                        )
                rows.append(
                    dict(
                        batch=batch,
                        history=h,
                        cache=cache,
                        us=values,
                        state_bytes=count * 4,
                        exact_max_abs_error=exact_error,
                        fixed_max_abs_error=fixed_error,
                        tile_max_abs_error=tile_error,
                        correctness=True,
                    )
                )
                (root / "raw.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    batch,
                    h,
                    cache,
                    {
                        name: round(statistics.median(v), 3)
                        for name, v in values.items()
                    },
                    flush=True,
                )
    with (root / "summary.csv").open("w") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "batch",
                "history",
                "cache",
                "state_mib",
                "store_us",
                "fixed16_us",
                "exact_h_us",
                "tile_h_us",
                "exact_over_store",
                "tile_over_store",
            ),
        )
        writer.writeheader()
        for row in rows:
            medians = {
                name: statistics.median(samples) for name, samples in row["us"].items()
            }
            writer.writerow(
                dict(
                    batch=row["batch"],
                    history=row["history"],
                    cache=row["cache"],
                    state_mib=row["state_bytes"] / 1048576,
                    store_us=medians["store"],
                    fixed16_us=medians["fixed16"],
                    exact_h_us=medians["exact_h"],
                    tile_h_us=medians["tile_h"],
                    exact_over_store=medians["exact_h"] / medians["store"],
                    tile_over_store=medians["tile_h"] / medians["store"],
                )
            )
    environment = dict(
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=hashlib.sha256(source).hexdigest(),
        fixed16_source_sha256=hashlib.sha256(dependency).hexdigest(),
        torch=torch.__version__,
        triton=triton.__version__,
        batches=BATCHES,
        histories=HISTORIES,
        repeats=REPEATS,
        cuda_graph=True,
        state_shape_per_request=(32, 128, 128),
        state_dtype="float32",
        history_dk_dtype="float16",
        history_g_dtype="float32",
        exact_h_updates="H unrolled rank-one recurrence steps; no padding or dot",
        tile_h_width=(
            "32 x H and H x 32 operands; H SIMT outer products without padding"
        ),
        fixed16_dot="TF32x3 Tensor Core dot with 16-wide K tile",
        state_read_timed=False,
        full_state_result_write_timed=False,
        history_read_timed=True,
        checksum_write_timed=True,
        eviction_bytes=256 * 1024 * 1024,
    )
    (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (root / "measurement_complete.json").write_text(
        json.dumps(
            dict(points=len(rows), expected_points=24, correctness=True), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
