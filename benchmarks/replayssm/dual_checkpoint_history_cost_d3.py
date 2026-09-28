# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the production GDN verify kernel across committed-history lengths."""

import argparse
import hashlib
import json
import random
import statistics
from functools import partial
from pathlib import Path

from dual_checkpoint_acceptance_d3 import (
    BATCHES,
    WIDTH,
    Inputs,
    Layer,
    capture_with_setup,
)
from qwen36_flush_crossover import time_graph

HISTORY = (0, 4, 8, 12)
REPEATS = 21


def prepare(layer: Layer, history_windows: int) -> None:
    layer.prepare()
    for _ in range(history_windows):
        layer.decode(0)
        layer.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (root / "source.py.txt").write_bytes(source)
    rows = []
    for batch in BATCHES:
        inputs = Inputs(batch)
        cases = []
        for history in HISTORY:
            layer = Layer(inputs, dual=False, accepted_drafts=3)
            setup = partial(prepare, layer, history // WIDTH)
            setup()
            indices = inputs.indices
            assert set(layer.wp[indices].tolist()) == {history}
            assert not any(layer.flush[indices].tolist())
            graph = capture_with_setup(layer.verify_kernel, setup)
            cases.append(("replayssm", history, layer, setup, graph))
        dual = Layer(inputs, dual=True, accepted_drafts=3)
        dual_setup = partial(prepare, dual, 1)
        dual_setup()
        assert set(dual.wp[inputs.indices].tolist()) == {0}
        assert set(dual.head[inputs.indices].tolist()) == {1}
        dual_graph = capture_with_setup(dual.verify_kernel, dual_setup)
        cases.append(("dual", 0, dual, dual_setup, dual_graph))
        samples = {(mode, history): [] for mode, history, *_ in cases}
        for repeat in range(REPEATS):
            order = list(cases)
            random.Random(batch * 1000 + repeat).shuffle(order)
            for mode, history, _, setup, graph in order:
                setup()
                samples[(mode, history)].append(time_graph(graph))
        for mode, history, layer, _, _ in cases:
            item = samples[(mode, history)]
            row = dict(
                batch=batch,
                mode=mode,
                committed_history=history,
                draft=3,
                window_width=WIDTH,
                logical_cap=layer.logical_cap,
                physical_ring_len=layer.length,
                samples_us=item,
            )
            rows.append(row)
            (root / "raw.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(batch, mode, history, round(statistics.median(item), 3), flush=True)
    (root / "environment.json").write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256(source).hexdigest(),
                dependency_sha256={
                    path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in (
                        "benchmarks/replayssm/dual_checkpoint_acceptance_d3.py",
                        "vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py",
                    )
                },
                batches=BATCHES,
                draft=3,
                history=HISTORY,
                repeats=REPEATS,
                graph=True,
                timing_scope="direct GDN verify kernel; setup excluded",
            ),
            indent=2,
        )
        + "\n"
    )
    (root / "measurement_complete.json").write_text(
        json.dumps(dict(rows=len(rows), expected_rows=15, samples=315), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
