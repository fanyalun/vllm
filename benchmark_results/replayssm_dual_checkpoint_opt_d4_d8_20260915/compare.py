# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with PYTHONPATH=. using the repository virtual environment."""

import importlib.machinery
import importlib.util
import json
import statistics
import sys
from pathlib import Path

from benchmarks.replayssm import dual_checkpoint_kernel as bench

root = Path(__file__).resolve().parent
name = "vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode_before"
loader = importlib.machinery.SourceFileLoader(
    name, str(root / "spec_decode_before.py.txt")
)
spec = importlib.util.spec_from_loader(name, loader)
before = importlib.util.module_from_spec(spec)
sys.modules[name] = before
loader.exec_module(before)
after = bench.gdn_replayssm_spec_decode
rows = []
for draft in (4, 8):
    for batch in (1, 4):
        for trajectory in ("all", "reject", "mixed"):
            for repeat in range(3):
                for mode in (
                    ("before", "after") if repeat % 2 == 0 else ("after", "before")
                ):
                    bench.gdn_replayssm_spec_decode = (
                        before.gdn_replayssm_spec_decode if mode == "before" else after
                    )
                    row = bench.measure(True, trajectory, batch, draft, 16)
                    row.update(version=mode, repeat=repeat)
                    rows.append(row)
                    (root / "measurements.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
                for key in (
                    "mean_history",
                    "flush_rate",
                    "promotion_rate",
                    "state_and_ring_bytes_per_request",
                ):
                    assert rows[-1][key] == rows[-2][key], key
            times = {
                mode: statistics.median(
                    r["cycle_us"]
                    for r in rows
                    if r["draft"] == draft
                    and r["batch"] == batch
                    and r["trajectory"] == trajectory
                    and r["version"] == mode
                )
                for mode in ("before", "after")
            }
            print(
                draft,
                batch,
                trajectory,
                times,
                times["before"] / times["after"],
                flush=True,
            )
