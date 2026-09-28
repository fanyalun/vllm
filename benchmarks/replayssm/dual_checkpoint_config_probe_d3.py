# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare ReplaySSM and dual GDN verify with identical launch tiles."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from dual_checkpoint_acceptance_d3 import (
    ACCEPTED_DRAFTS,
    BATCHES,
    WIDTH,
    Inputs,
    Layer,
    capture_with_setup,
)
from qwen36_flush_crossover import time_graph

from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    _launch_gdn_spec,
)

WARPS = (1, 4)
REPEATS = 21


def verify(layer: Layer, warps: int) -> None:
    x = layer.inputs
    _launch_gdn_spec(
        mixed_qkv=x.qkv[1],
        a=x.a[1],
        b=x.b[1],
        A_log=x.a_log,
        dt_bias=x.bias,
        out=layer.out[1],
        checkpoint_state=layer.state0,
        d_cache=layer.d,
        k_cache=layer.k,
        g_cache=layer.g,
        query_start_loc=x.qsl,
        ssm_state_indices=x.indices,
        write_pos=layer.wp,
        cache_base=layer.base,
        is_flush=layer.flush,
        scale=128**-0.5,
        max_cache_len=layer.logical_cap,
        max_spec_len=WIDTH,
        use_qk_l2norm_in_kernel=True,
        is_flush_kernel=False,
        block_v=64,
        num_warps=warps,
        num_stages=2,
        nk=2,
        null_block_id=0,
        dot_precision="tf32",
        alternate_checkpoint=layer.state1,
        head_slot=layer.head if layer.dual else None,
    )


def prepare_for_verify(layer: Layer) -> None:
    layer.prepare()
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
        for accepted_drafts in ACCEPTED_DRAFTS:
            for dual in (False, True):
                layer = Layer(inputs, dual, accepted_drafts)
                setup = partial(prepare_for_verify, layer)
                reference = None
                for warps in WARPS:
                    setup()
                    verify(layer, warps)
                    output = layer.out[1].clone()
                    if reference is None:
                        reference = output
                        max_error = 0.0
                    else:
                        torch.testing.assert_close(
                            output, reference, atol=0.01, rtol=0.04
                        )
                        max_error = (output.float() - reference.float()).abs().max()
                        max_error = max_error.item()
                    graph = capture_with_setup(partial(verify, layer, warps), setup)
                    samples = []
                    for _ in range(REPEATS):
                        setup()
                        samples.append(time_graph(graph))
                    row = dict(
                        batch=batch,
                        accepted_drafts=accepted_drafts,
                        mode="dual" if dual else "replayssm",
                        block_v=64,
                        num_warps=warps,
                        nk=2,
                        num_stages=2,
                        max_abs_warp_config_output_error=max_error,
                        samples_us=samples,
                    )
                    rows.append(row)
                    (root / "raw.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(
                        batch,
                        accepted_drafts,
                        row["mode"],
                        warps,
                        round(statistics.median(samples), 3),
                        flush=True,
                    )
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
                accepted_drafts=ACCEPTED_DRAFTS,
                warps=WARPS,
                repeats=REPEATS,
                timing_scope=(
                    "direct GDN verify kernel after previous verify and commit"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    (root / "measurement_complete.json").write_text(
        json.dumps(dict(rows=len(rows), expected_rows=24, samples=504), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
