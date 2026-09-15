# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit bounded confidence exports against weights and pre-change tokens."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(__file__).resolve().parent
repo = root.parents[1]
checkpoint = Path(
    "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark/model.safetensors"
)
with safe_open(checkpoint, framework="pt", device="cpu") as weights:
    expected = {
        name: hashlib.sha256(
            weights.get_tensor("confidence_head." + name)
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).hexdigest()
        for name in ("proj.weight", "proj.bias")
    }
results = []
for draft in (4, 8):
    smoke = json.loads((root / f"d{draft}_smoke.json").read_text())
    baseline = json.loads(
        (
            repo
            / "benchmark_results/replayssm_dual_checkpoint_opt_d4_d8_20260915"
            / f"dspark_d{draft}.json"
        ).read_text()
    )
    assert smoke["complete"]
    assert smoke["token_ids"] == smoke["warmup_token_ids"] == baseline["token_ids"]
    assert smoke["prompt_token_ids"] == baseline["prompt_token_ids"]
    assert [len(x) for x in smoke["token_ids"]] == [64, 64]
    traces = json.loads((root / f"d{draft}_confidence.json").read_text())
    assert len(traces) == 1
    trace = traces[0]
    assert trace["weight_sha256"] == expected
    rows = trace["rounds"]
    assert rows
    values = []
    requests = set()
    for row in rows:
        n = len(row["req_ids"])
        requests.update(row["req_ids"])
        assert (
            n
            == len(row["confidence"])
            == len(row["draft_tokens"])
            == len(row["positions"])
        )
        for confidence, tokens, positions in zip(
            row["confidence"], row["draft_tokens"], row["positions"]
        ):
            assert len(confidence) == len(tokens) == len(positions) == draft
            assert positions == list(range(positions[0], positions[0] + draft))
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in confidence)
            values.extend(confidence)
    assert len(requests) == 2
    results.append(
        dict(
            draft=draft,
            rounds=len(rows),
            requests=len(requests),
            confidence_count=len(values),
            confidence_min=min(values),
            confidence_max=max(values),
            first_confidence=rows[0]["confidence"],
            loaded_weights_match=True,
            repeated_tokens_match=True,
            baseline_tokens_match=True,
        )
    )
(root / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
