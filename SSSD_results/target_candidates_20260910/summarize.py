# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


root = Path(__file__).resolve().parent
rows = []
reference = None
for name in (
    "local_r1",
    "baseline_r1",
    "local_r2",
    "metrics_off_r1",
    "local_r3",
    "baseline_r2",
    "metrics_off_r2",
    "local_r4",
):
    cell = root / name / "cells/performance_async_cache_graph_b1"
    assert (cell / "cell_complete.json").exists(), name
    result = json.loads((cell / "result.json").read_text())
    requests = json.loads((cell / "requests.json").read_text())
    tokens = [request["token_ids"] for request in requests]
    assert len(tokens) == 4 and all(len(row) == 256 for row in tokens)
    prompts = json.loads((root / name / "prompts.json").read_text())
    assert [prompt["prompt_index"] for prompt in prompts] == [0, 4, 8, 12]
    if reference is None:
        reference = (tokens, prompts)
    assert (tokens, prompts) == reference, name
    counts = result["metrics_delta"]

    def metric(key, counts=counts):
        return sum(value for name, value in counts.items() if name.split("{")[0] == key)

    shutdown = json.loads((cell / "shutdown.json").read_text())
    assert shutdown["exit_code"] == 0 and not shutdown["forced_kill"], name
    rounds = metric("vllm:spec_decode_num_drafts_total")
    seconds = result["summary"]["elapsed_seconds"]
    rows.append(
        {
            "name": name,
            "final_comparison": name in ("baseline_r2", "local_r4", "metrics_off_r2"),
            "tok_s": result["summary"]["completion_throughput_tok_s"],
            "tokens_per_gpu_second": result["summary"]["tokens_per_gpu_second"],
            "seconds": seconds,
            "ms_per_verify": seconds / rounds * 1000,
            "verify_rounds": rounds,
            "accepted_tokens": metric("vllm:spec_decode_num_accepted_tokens_total"),
            "cache_hits": metric("vllm:async_draft_cache_hits_total"),
            "cache_misses": metric("vllm:async_draft_cache_misses_total"),
            "local_hits": metric("vllm:async_draft_target_local_hits_total"),
            "async_metrics_exported": not name.startswith("metrics_off"),
            "output_sha256": digest(tokens),
            "prompt_sha256": digest(prompts),
            "shutdown": shutdown,
            "ipc_shutdown_warning": "Producer process has been terminated"
            in (cell / "server.log").read_text(),
        }
    )

assert len({row["verify_rounds"] for row in rows}) == 1
assert len({row["accepted_tokens"] for row in rows}) == 1
summary = {
    "scope": "Four-prompt transport experiment; not a full validation matrix",
    "all_output_tokens_equal": True,
    "standard_speculative_counters_equal": True,
    "rows": rows,
    "limitations": [
        "Early cells include instrumentation and shutdown fixes; "
        "use final_comparison rows for the final implementation.",
        "One final repeat per variant cannot establish sub-percent differences.",
        "Disabling Async metric export retains internal counters, clocks "
        "and all protocol synchronization.",
        "Target-local selection still reads the outcome on the CPU "
        "and retains the payload barrier.",
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
