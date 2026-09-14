# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare baseline, history replay and parallel final-state GDN at h=16."""

import argparse
import hashlib
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from qwen36_flush_crossover import Inputs, Layer, capture, cursors, save, time_graph
from qwen36_parallel_last import parallel_last


class FinalLayer(Layer):
    def __init__(self, inputs):
        super().__init__(inputs)
        self.current = inputs.states[16].clone()
        self.final = torch.empty_like(self.current)

    def reset(self, history=16):
        super().reset(history)
        self.current.copy_(self.x.states[history])
        self.final.fill_(float("nan"))

    def parallel(self):
        return parallel_last(self.x, self.current, self.final, self.out)


def measure(args):
    x = Inputs(args.batch, args.draft, args.seed)
    layers = [FinalLayer(x) for _ in range(30)]
    cursor = cursors(args.batch, 16, False)
    methods = ("standard", "replay", "parallel_last")
    rows = []
    for count in (1, 30):
        active = layers[:count]

        def invoke(name, active=active):
            for layer in active:
                if name == "standard":
                    layer.standard()
                elif name == "replay":
                    layer.replay(*cursor)
                else:
                    layer.parallel()

        graphs = {name: capture(lambda name=name: invoke(name)) for name in methods}
        for layer in active:
            layer.reset()
        invoke("standard")
        references = [layer.state[x.indices[:, -1].long()].clone() for layer in active]
        # Baseline returns a separate output tensor; obtain the paired reference.
        output_refs = []
        for layer in active:
            layer.reset()
            output_refs.append(layer.standard().reshape_as(layer.out).clone())
        checks = {}
        for name in ("replay", "parallel_last"):
            for layer in active:
                layer.reset()
            invoke(name)
            output_error = 0.0
            state_error = 0.0
            for layer, output_ref, state_ref in zip(active, output_refs, references):
                torch.testing.assert_close(layer.out, output_ref, rtol=0.04, atol=0.01)
                output_error = max(
                    output_error,
                    (layer.out.float() - output_ref.float()).abs().max().item(),
                )
                if name == "parallel_last":
                    torch.testing.assert_close(
                        layer.final[1:], state_ref, rtol=0.04, atol=0.01
                    )
                    torch.testing.assert_close(
                        layer.current, x.states[16], rtol=0, atol=0
                    )
                    assert layer.final[0].isnan().all().item()
                    state_error = max(
                        state_error, (layer.final[1:] - state_ref).abs().max().item()
                    )
            checks[name] = dict(
                output_max_abs_error=output_error,
                final_state_max_abs_error=state_error
                if name == "parallel_last"
                else None,
            )
        values = {name: [] for name in methods}
        for repeat in range(21):
            names = list(methods)
            random.Random(repeat + args.seed * 1000).shuffle(names)
            for name in names:
                for layer in active:
                    layer.reset()
                values[name].append(time_graph(graphs[name]) / count)
        rows.append(
            dict(
                batch=args.batch,
                draft=args.draft,
                history=16,
                layers=count,
                seed=args.seed,
                us=values,
                checks=checks,
                correctness=True,
            )
        )
        print(count, {k: statistics.median(v) for k, v in values.items()}, flush=True)
    save(Path(args.output) / f"b{args.batch}_d{args.draft}_s{args.seed}.json", rows)


def queue(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    batches = (1, 8) if args.gpu == 0 else (4, 16)
    sources = [
        Path(__file__),
        Path(__file__).with_name("qwen36_parallel_last.py"),
        Path(__file__).with_name("qwen36_flush_crossover.py"),
    ]
    archive = root / "source"
    archive.mkdir(exist_ok=True)
    hashes = {}
    for path in sources:
        content = path.read_bytes()
        hashes[path.name] = hashlib.sha256(content).hexdigest()
        (archive / (path.name + ".txt")).write_bytes(content)
    for b in batches:
        for d in (4, 8, 16, 32):
            dest = root / f"b{b}_d{d}_s0.json"
            if dest.exists():
                continue
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(root),
                "--batch",
                str(b),
                "--draft",
                str(d),
            ]
            save(
                root / f"b{b}_d{d}_launch.json",
                dict(
                    command=cmd,
                    gpu=args.gpu,
                    sources=hashes,
                    source_head=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                ),
            )
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="8"
            )
            with (root / f"b{b}_d{d}.log").open("w") as log:
                subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            print("DONE", b, d, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--draft", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    (queue if args.queue else measure)(args)


if __name__ == "__main__":
    main()
