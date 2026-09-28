# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check repeated D3 acceptance trajectories across GDN ReplaySSM modes."""

import argparse
import json
from pathlib import Path

import torch
from dual_checkpoint_acceptance_d3 import ACCEPTED_DRAFTS, BATCHES, Inputs, Layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in BATCHES:
        inputs = Inputs(batch)
        for accepted_drafts in ACCEPTED_DRAFTS:
            original = Layer(inputs, False, accepted_drafts)
            dual = Layer(inputs, True, accepted_drafts)
            for layer in (original, dual):
                layer.prepare()
                layer.decode(0)
            max_error = 0.0
            flush_counts = {"replayssm": 0, "dual": 0}
            for _ in range(32):
                for name, layer in (("replayssm", original), ("dual", dual)):
                    layer.commit()
                    flush_counts[name] += int(
                        layer.flush[layer.inputs.indices].sum().item()
                    )
                    layer.decode(1)
                left, right = original.out[1], dual.out[1]
                max_error = max(
                    max_error, (left.float() - right.float()).abs().max().item()
                )
                torch.testing.assert_close(left, right, atol=0.01, rtol=0.04)
            row = dict(
                batch=batch,
                accepted_drafts=accepted_drafts,
                num_accepted_including_bonus=accepted_drafts + 1,
                rounds=32,
                max_abs_output_error=max_error,
                flush_counts=flush_counts,
                correctness=True,
            )
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
