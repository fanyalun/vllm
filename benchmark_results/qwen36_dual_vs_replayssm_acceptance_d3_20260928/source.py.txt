# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native GDN ReplaySSM and dual-checkpoint acceptance transitions."""

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

import torch
from qwen36_flush_crossover import capture, time_graph

from vllm.model_executor.layers.fla.ops.gdn_replayssm_dual_checkpoint import (
    commit_gdn_dual_checkpoint,
)
from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    _launch_gdn_spec,
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
)
from vllm.model_executor.layers.mamba.ops.replayssm_config import (
    get_replayssm_config,
)

BATCHES = (1, 8, 64)
ACCEPTED_DRAFTS = (3, 2)
DRAFT = 3
WIDTH = DRAFT + 1
CAP = 16
REPEATS = 21
COMPONENTS = ("previous_decode", "commit", "next_decode", "verify_kernel", "pair")


class Inputs:
    def __init__(self, batch):
        torch.manual_seed(0)
        device = "cuda"
        self.initial = torch.randn(batch + 1, 32, 128, 128, device=device) * 0.01
        self.qkv = [
            (torch.randn(batch * WIDTH, 8192, device=device) * 0.2).bfloat16()
            for _ in range(2)
        ]
        self.a = [
            torch.randn(batch * WIDTH, 32, device=device).bfloat16() for _ in range(2)
        ]
        self.b = [
            torch.randn(batch * WIDTH, 32, device=device).bfloat16() for _ in range(2)
        ]
        self.a_log = torch.full((32,), -2.0, device=device)
        self.bias = torch.zeros(32, device=device)
        self.indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
        self.qsl = torch.arange(batch + 1, device=device, dtype=torch.int32) * WIDTH
        self.first_decode = torch.zeros(batch, device=device, dtype=torch.int8)


class Layer:
    def __init__(self, inputs, dual, accepted_drafts):
        self.inputs = inputs
        self.dual = dual
        self.accepted_drafts = accepted_drafts
        batch = inputs.indices.numel()
        self.logical_cap = CAP if dual else CAP + WIDTH
        self.length = 1 << (self.logical_cap - 1).bit_length()
        self.state0 = torch.empty_like(inputs.initial)
        self.state1 = torch.empty_like(inputs.initial) if dual else None
        self.d = torch.empty(
            batch + 1, 32, self.length, 128, device="cuda", dtype=torch.float16
        )
        self.k = torch.empty(
            batch + 1, 16, self.length, 128, device="cuda", dtype=torch.float16
        )
        self.g = torch.empty(batch + 1, 32, self.length, device="cuda")
        self.wp = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
        self.base = torch.zeros_like(self.wp)
        self.head = torch.zeros_like(self.wp)
        self.prev_len = torch.zeros_like(self.wp)
        self.flush = torch.zeros(batch + 1, device="cuda", dtype=torch.int8)
        self.accepted = torch.full(
            (batch,), accepted_drafts + 1, device="cuda", dtype=torch.int32
        )
        self.out = [
            torch.empty(batch * WIDTH, 32, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        self.graphs = {}

    def prepare(self):
        self.state0.copy_(self.inputs.initial)
        if self.state1 is not None:
            self.state1.zero_()
        self.d.zero_()
        self.k.zero_()
        self.g.zero_()
        self.wp.zero_()
        self.base.zero_()
        self.head.zero_()
        self.prev_len.fill_(WIDTH)
        self.flush.zero_()

    def decode(self, position):
        x = self.inputs
        gdn_replayssm_spec_decode(
            x.qkv[position],
            x.a[position],
            x.b[position],
            x.a_log,
            x.bias,
            self.state0,
            self.d,
            self.k,
            self.g,
            self.out[position],
            x.qsl,
            x.indices,
            self.wp,
            self.base,
            self.flush,
            self.logical_cap,
            WIDTH,
            alternate_checkpoint=self.state1,
            head_slot=self.head if self.dual else None,
            hard_cap=CAP if self.dual else None,
        )

    def commit(self):
        x = self.inputs
        if self.dual:
            commit_gdn_dual_checkpoint(
                self.wp,
                self.base,
                self.flush,
                self.head,
                self.prev_len,
                self.accepted,
                x.indices,
                x.qsl,
                x.first_decode,
                CAP,
            )
        else:
            commit_gdn_replayssm_spec(
                self.wp,
                self.base,
                self.flush,
                self.accepted,
                x.indices,
                self.logical_cap,
                WIDTH,
                self.length,
            )

    def verify_kernel(self):
        x = self.inputs
        block_v, warps, nk, stages = get_replayssm_config(
            "gdn_spec_verify", max_spec_len=WIDTH, head_k_dim=128
        )
        if self.dual:
            block_v, warps, nk = 64, 4, 2
        _launch_gdn_spec(
            mixed_qkv=x.qkv[1],
            a=x.a[1],
            b=x.b[1],
            A_log=x.a_log,
            dt_bias=x.bias,
            out=self.out[1],
            checkpoint_state=self.state0,
            d_cache=self.d,
            k_cache=self.k,
            g_cache=self.g,
            query_start_loc=x.qsl,
            ssm_state_indices=x.indices,
            write_pos=self.wp,
            cache_base=self.base,
            is_flush=self.flush,
            scale=128**-0.5,
            max_cache_len=self.logical_cap,
            max_spec_len=WIDTH,
            use_qk_l2norm_in_kernel=True,
            is_flush_kernel=False,
            block_v=block_v,
            num_warps=warps,
            num_stages=stages,
            nk=nk,
            null_block_id=0,
            dot_precision="tf32",
            alternate_checkpoint=self.state1,
            head_slot=self.head if self.dual else None,
        )

    def pair(self):
        self.decode(0)
        self.commit()
        self.decode(1)

    def capture_graphs(self):
        self.graphs["prepare"] = capture(self.prepare)
        functions = {
            "previous_decode": lambda: self.decode(0),
            "commit": self.commit,
            "next_decode": lambda: (self.commit(), self.decode(1)),
            "verify_kernel": self.verify_kernel,
            "pair": self.pair,
        }
        for name, function in functions.items():
            self.graphs[name] = capture_with_setup(
                function, lambda name=name: self.setup_for(name)
            )

    def setup_for(self, component):
        self.prepare()
        if component in ("commit", "next_decode", "verify_kernel"):
            self.decode(0)
        if component == "verify_kernel":
            self.commit()

    def replay_setup_for(self, component):
        self.graphs["prepare"].replay()
        if component in ("commit", "next_decode", "verify_kernel"):
            self.graphs["previous_decode"].replay()
        if component == "verify_kernel":
            self.graphs["commit"].replay()


def capture_with_setup(function, setup):
    for _ in range(3):
        setup()
        function()
    torch.accelerator.synchronize()
    setup()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def check_pair(original, dual, accepted_drafts):
    outputs = []
    metadata = []
    for layer in (original, dual):
        layer.prepare()
        layer.decode(0)
        first = layer.out[0].clone()
        if layer.dual:
            assert torch.isfinite(layer.state1[1:]).all().item()
        layer.commit()
        indices = layer.inputs.indices
        metadata.append(
            dict(
                write_pos=layer.wp[indices].tolist(),
                head_slot=layer.head[indices].tolist(),
                is_flush=layer.flush[indices].tolist(),
                previous_len=layer.prev_len[indices].tolist(),
            )
        )
        layer.decode(1)
        outputs.append((first, layer.out[1].clone()))
    expected_commit = accepted_drafts + 1
    assert set(metadata[0]["write_pos"]) == {expected_commit}
    assert set(metadata[1]["write_pos"]) == {
        0 if accepted_drafts == DRAFT else expected_commit
    }
    assert set(metadata[1]["head_slot"]) == {1 if accepted_drafts == DRAFT else 0}
    assert all(not any(m["is_flush"]) for m in metadata)
    errors = {}
    for position in (0, 1):
        left, right = outputs[0][position], outputs[1][position]
        torch.testing.assert_close(left, right, atol=0.01, rtol=0.04)
        errors[f"position_{position}_max_abs_error"] = (
            (left.float() - right.float()).abs().max().item()
        )
    for layer in (original, dual):
        layer.prepare()
        layer.decode(0)
        layer.commit()
        layer.verify_kernel()
        torch.testing.assert_close(layer.out[1], outputs[layer.dual][1], atol=0, rtol=0)
    return dict(original=metadata[0], dual=metadata[1], **errors)


def benchmark(root):
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (root / "source.py.txt").write_bytes(source)
    rows = []
    for batch in BATCHES:
        inputs = Inputs(batch)
        for accepted_drafts in ACCEPTED_DRAFTS:
            layers = {
                name: Layer(inputs, dual, accepted_drafts)
                for name, dual in (("replayssm", False), ("dual", True))
            }
            correctness = check_pair(
                layers["replayssm"], layers["dual"], accepted_drafts
            )
            for layer in layers.values():
                layer.capture_graphs()
            values = {
                name: {component: [] for component in COMPONENTS} for name in layers
            }
            for repeat in range(REPEATS):
                names = list(layers)
                random.Random(batch * 1000 + accepted_drafts * 100 + repeat).shuffle(
                    names
                )
                for name in names:
                    layer = layers[name]
                    for component in COMPONENTS:
                        layer.replay_setup_for(component)
                        values[name][component].append(
                            time_graph(layer.graphs[component])
                        )
            for name in layers:
                row = dict(
                    mode=name,
                    batch=batch,
                    draft=DRAFT,
                    accepted_drafts=accepted_drafts,
                    num_accepted_including_bonus=accepted_drafts + 1,
                    logical_cap=layers[name].logical_cap,
                    physical_ring_len=layers[name].length,
                    components_us=values[name],
                    correctness=correctness,
                )
                rows.append(row)
                (root / "raw.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    name,
                    batch,
                    accepted_drafts,
                    {
                        component: round(statistics.median(samples), 3)
                        for component, samples in values[name].items()
                    },
                    flush=True,
                )
    with (root / "summary.csv").open("w") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "batch",
                "accepted_drafts",
                "num_accepted_including_bonus",
                "mode",
                "previous_decode_us",
                "commit_us",
                "next_decode_us",
                "verify_kernel_us",
                "pair_us",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    batch=row["batch"],
                    accepted_drafts=row["accepted_drafts"],
                    num_accepted_including_bonus=row["num_accepted_including_bonus"],
                    mode=row["mode"],
                    **{
                        f"{component}_us": statistics.median(samples)
                        for component, samples in row["components_us"].items()
                    },
                )
            )
    files = (
        "vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py",
        "vllm/model_executor/layers/fla/ops/gdn_replayssm_dual_checkpoint.py",
        "vllm/model_executor/layers/mamba/ops/replayssm_config.py",
        "benchmarks/replayssm/qwen36_flush_crossover.py",
    )
    environment = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        torch=torch.__version__,
        batches=BATCHES,
        draft=DRAFT,
        accepted_drafts=ACCEPTED_DRAFTS,
        repeats=REPEATS,
        source_sha256=hashlib.sha256(source).hexdigest(),
        dependency_sha256={
            file: hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in files
        },
        cuda_graph=True,
        ssm_state_dtype="float32",
        qkv_dtype="bfloat16",
        history_dk_dtype="float16",
        original_logical_cap=CAP + WIDTH,
        dual_hard_cap=CAP,
        first_decode_reset=False,
        prior_window_executed=True,
        first_window_commit_excluded=True,
        no_flush_expected=True,
        timing_scope="one GDN layer, entire batch, GPU CUDA-event latency",
    )
    (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (root / "measurement_complete.json").write_text(
        json.dumps(
            dict(rows=len(rows), expected_rows=12, samples=1260, correctness=True),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    benchmark(Path(args.output))


if __name__ == "__main__":
    main()
