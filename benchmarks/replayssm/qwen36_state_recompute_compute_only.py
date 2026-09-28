# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure full-State stores and GDN reconstruction without State I/O."""

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

from vllm.triton_utils import tl, triton

BATCHES = (1, 8, 64)
HISTORIES = (1, 4, 8, 16)
REPEATS = 21


@triton.jit
def recompute_no_state_io(
    d,
    k,
    g,
    checksum,
    full_out,
    HISTORY: tl.constexpr,
    STORE_FULL: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    ov = (tile // 4) * 32 + tl.arange(0, 32)
    ok = (tile % 4) * 32 + tl.arange(0, 32)
    oh = tl.arange(0, 16)
    gates = tl.load(
        g + (batch * 32 + head) * HISTORY + oh,
        mask=oh < HISTORY,
        other=0.0,
    )
    prefix = tl.cumsum(gates, 0)
    total = tl.sum(gates, 0)
    decay = tl.where(oh < HISTORY, tl.exp(total - prefix), 0.0)
    delta = tl.load(
        d + ((batch * 32 + head) * HISTORY + oh[None, :]) * 128 + ov[:, None],
        mask=oh[None, :] < HISTORY,
        other=0.0,
    )
    keys = tl.load(
        k + ((batch * 16 + head // 2) * HISTORY + oh[:, None]) * 128 + ok[None, :],
        mask=oh[:, None] < HISTORY,
        other=0.0,
    )
    # Construct S_(t-h) in registers so the timed kernel does not read State.
    initial = (ov[:, None] * 128 + ok[None, :]).to(tl.float32) * 0.000001
    initial = initial + (batch * 32 + head) * 0.00001
    result = tl.dot(
        delta.to(tl.float32) * decay[None, :],
        keys.to(tl.float32),
        acc=tl.exp(total) * initial,
        input_precision="tf32x3",
    )
    if STORE_FULL:
        offset = (batch * 32 + head) * 128 * 128 + ov[:, None] * 128 + ok[None, :]
        tl.store(full_out + offset, result)
    else:
        value = tl.sum(tl.sum(result, 0), 0)
        tl.store(checksum + (batch * 32 + head) * 16 + tile, value)


def timed(graph, eviction_graph, warm):
    eviction_graph.replay()
    if warm:
        for _ in range(3):
            graph.replay()
    start = torch.Event(device="cuda", enable_timing=True)
    end = torch.Event(device="cuda", enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (root / "source.py.txt").write_bytes(source)
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
        out = torch.empty((batch, 32, 128, 128), device=device)
        checksum = torch.empty((batch, 32, 16), device=device)
        count = out.numel()

        def store_call(out=out, count=count):
            state_store[(triton.cdiv(count, 1024),)](out, count, 1024, num_warps=4)

        store_graph = capture(store_call)
        for h in HISTORIES:
            d = (torch.randn((batch, 32, h, 128), device=device) * 0.01).half()
            k = (torch.randn((batch, 16, h, 128), device=device) * 0.1).half()
            g = -torch.rand((batch, 32, h), device=device) * 0.03

            def replay_call(
                store_full=False,
                batch=batch,
                d=d,
                k=k,
                g=g,
                checksum=checksum,
                out=out,
                h=h,
            ):
                recompute_no_state_io[(batch, 32, 16)](
                    d, k, g, checksum, out, h, store_full, num_warps=4
                )

            replay_call(store_full=True)
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
            torch.testing.assert_close(out.double(), oracle, rtol=0.002, atol=0.0002)
            max_error = (out.double() - oracle).abs().max().item()
            replay_call()
            assert torch.isfinite(checksum).all().item()
            graphs = {"store": store_graph, "recompute": capture(replay_call)}
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
                        max_abs_error=max_error,
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
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "batch",
                "history",
                "cache",
                "state_mib",
                "store_us",
                "recompute_us",
                "recompute_over_store",
            ),
        )
        writer.writeheader()
        for row in rows:
            store = statistics.median(row["us"]["store"])
            replay = statistics.median(row["us"]["recompute"])
            writer.writerow(
                dict(
                    batch=row["batch"],
                    history=row["history"],
                    cache=row["cache"],
                    state_mib=row["state_bytes"] / 1048576,
                    store_us=store,
                    recompute_us=replay,
                    recompute_over_store=replay / store,
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
        dot_precision="tf32x3",
        history_tile=16,
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
