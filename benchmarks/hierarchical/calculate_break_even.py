# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calculate measured-cycle break-even budgets relative to MTP.

Input JSON contains baseline and candidate lists of batch-cycle records.
Every record has wall_ms, returned_tokens, and preverify_ms (zero for MTP).
Optional candidate fields proposed and accepted are aggregate integer counts.
Nested proposal time must not be added again to wall_ms. Prefill and terminal
work must use the same accounting convention in both populations.
"""

import argparse
import json
import math
from pathlib import Path


def summarize(rows):
    if not rows:
        raise ValueError("A cycle population must not be empty")
    wall = preverify = 0.0
    tokens = 0
    for row in rows:
        cost, pv = float(row["wall_ms"]), float(row["preverify_ms"])
        count = row["returned_tokens"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("returned_tokens must be a nonnegative integer")
        if not math.isfinite(cost) or not math.isfinite(pv):
            raise ValueError("Cycle times must be finite")
        if not 0 <= pv <= cost or cost <= 0:
            raise ValueError("Require 0 <= preverify_ms <= wall_ms and wall_ms > 0")
        wall += cost
        preverify += pv
        tokens += count
    if tokens == 0:
        raise ValueError("A population must return at least one token")
    return dict(
        cycles=len(rows),
        wall_ms=wall,
        preverify_ms=preverify,
        returned_tokens=tokens,
        ms_per_token=wall / tokens,
        tokens_per_second=1000 * tokens / wall,
    )


def calculate(data, goal_speedup=1.0):
    if not math.isfinite(goal_speedup) or goal_speedup < 1:
        raise ValueError("goal_speedup must be finite and >= 1")
    baseline = summarize(data["baseline"])
    candidate = summarize(data["candidate"])
    if baseline["preverify_ms"] != 0:
        raise ValueError("The MTP baseline must have zero preverify_ms")
    q2 = baseline["ms_per_token"]
    non_pv = candidate["wall_ms"] - candidate["preverify_ms"]
    allowed = q2 * candidate["returned_tokens"] / goal_speedup
    budget = allowed - non_pv
    result = dict(
        baseline=baseline,
        candidate=candidate,
        measured_speedup=q2 / candidate["ms_per_token"],
        requested_speedup=goal_speedup,
        allowed_total_ms=allowed,
        non_preverify_ms=non_pv,
        preverify_budget_ms=budget,
        possible_by_preverify_only=budget > 0,
        preverify_speedup_strictly_greater_than=(
            candidate["preverify_ms"] / budget if budget > 0 else None
        ),
        zero_preverify_speedup_ceiling=(
            q2 * candidate["returned_tokens"] / non_pv if non_pv > 0 else None
        ),
        assumption="Fixed output distribution and unchanged non-Pre-Verify cost",
    )
    fields = ("proposed", "accepted")
    coverage = [all(k in row for k in fields) for row in data["candidate"]]
    if any(any(k in row for k in fields) for row in data["candidate"]):
        if not all(coverage):
            raise ValueError("Acceptance fields require complete cycle coverage")
        for row in data["candidate"]:
            if any(
                not isinstance(row[k], int) or isinstance(row[k], bool) for k in fields
            ):
                raise ValueError("Acceptance counts must be integers")
            if not 0 <= row["accepted"] <= row["proposed"]:
                raise ValueError("Require 0 <= accepted <= proposed")
        proposed = sum(row["proposed"] for row in data["candidate"])
        accepted = sum(row["accepted"] for row in data["candidate"])
        result["acceptance"] = dict(
            proposed=proposed,
            accepted=accepted,
            weighted_rate=accepted / proposed if proposed else None,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--goal-speedup", type=float, default=1.0)
    args = parser.parse_args()
    result = calculate(json.loads(args.input.read_text()), args.goal_speedup)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
