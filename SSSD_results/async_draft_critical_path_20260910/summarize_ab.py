# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
outputs = []
for variant in ("baseline", "stream_wait"):
    for repeat in range(1, 4):
        name = f"{variant}_async_r{repeat}"
        cell = root / name / "cells/performance_async_cache_graph_b1"
        if not (cell / "cell_complete.json").exists():
            continue
        result = json.loads((cell / "result.json").read_text())
        requests = json.loads((cell / "requests.json").read_text())
        tokens = [r["token_ids"] for r in requests]
        assert len(tokens) == 16 and all(len(t) == 256 for t in tokens)
        prompts = json.loads((root / name / "prompts.json").read_text())
        assert [p["prompt_index"] for p in prompts] == list(range(16))
        counts = result["metrics_delta"]

        def metric(s, counts=counts):
            return sum(v for k, v in counts.items() if k.split("{")[0] == s)

        rounds = metric("vllm:spec_decode_num_drafts_total")
        row = {
            "variant": variant,
            "repeat": repeat,
            "cell": str(cell.relative_to(root)),
            "tok_s": result["summary"]["completion_throughput_tok_s"],
            "seconds": result["summary"]["elapsed_seconds"],
            "rounds": rounds,
            "accepted": metric("vllm:spec_decode_num_accepted_tokens_total"),
            "hits": metric("vllm:async_draft_cache_hits_total"),
            "misses": metric("vllm:async_draft_cache_misses_total"),
            "output_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            "shutdown": json.loads((cell / "shutdown.json").read_text()),
        }
        row["ms_per_verify"] = row["seconds"] / rounds * 1000
        assert not row["shutdown"]["forced_kill"] and row["shutdown"]["exit_code"] == 0
        rows.append(row)
        outputs.append(tokens)
summary = {
    "scope": "Local response-transport A/B; not an AR/Sync correctness matrix",
    "expected_cells": 6,
    "completed_cells": len(rows),
    "rows": rows,
    "exact_same_outputs_all_cells": bool(outputs)
    and all(t == outputs[0] for t in outputs),
    "exact_same_counters_all_cells": bool(rows)
    and all(
        all(r[k] == rows[0][k] for k in ("rounds", "accepted", "hits", "misses"))
        for r in rows
    ),
    "same_prompts_all_cells": len({r["prompt_sha256"] for r in rows}) == 1,
    "measurement_complete": len(rows) == 6,
}
for variant in ("baseline", "stream_wait"):
    values = [r["tok_s"] for r in rows if r["variant"] == variant]
    if values:
        summary[variant + "_median_tok_s"] = statistics.median(values)
if all(k + "_median_tok_s" in summary for k in ("baseline", "stream_wait")):
    summary["median_improvement_percent"] = 100 * (
        summary["stream_wait_median_tok_s"] / summary["baseline_median_tok_s"] - 1
    )
(root / "ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
