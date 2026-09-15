# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare original and optimized dual checkpoints at buffer setting 16."""

import json
import statistics
from pathlib import Path

from benchmarks.replayssm.dual_checkpoint_kernel import measure

root = Path(__file__).resolve().parent
rows = []
summary = []
for draft in (4, 8):
    for batch in (1, 4):
        for trajectory in ("all", "reject", "mixed"):
            for repeat in range(3):
                for dual in (False, True) if repeat % 2 == 0 else (True, False):
                    row = measure(dual, trajectory, batch, draft, 16)
                    row["repeat"] = repeat
                    rows.append(row)
                    (root / "measurements.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
            cell = dict(draft=draft, batch=batch, trajectory=trajectory)
            for mode in ("original", "dual"):
                times = [
                    r["cycle_us"]
                    for r in rows
                    if (r["draft"], r["batch"], r["trajectory"], r["mode"])
                    == (draft, batch, trajectory, mode)
                ]
                cell[mode + "_us"] = statistics.median(times)
                cell[mode + "_min_us"] = min(times)
                cell[mode + "_max_us"] = max(times)
            cell["speedup"] = cell["original_us"] / cell["dual_us"]
            summary.append(cell)
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(cell), flush=True)
