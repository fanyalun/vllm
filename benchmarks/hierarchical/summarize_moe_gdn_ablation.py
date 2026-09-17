# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MoE/GDN forward effects and independently measured online acceptance."""

import argparse
import json
import statistics
from pathlib import Path

from benchmarks.hierarchical.summarize_batch import profile, stats, summarize


def validate_dispatch(row):
    expected = 0 if row["gdn"] == "v0" else 30
    if row.get("windowed_kernel_launches") != expected:
        raise ValueError("Measured GDN kernel does not match the labeled variant")
    if expected and not sum(row.get("action_counts") or []):
        raise ValueError("Windowed GDN audit did not record any actions")


def paired_effects(windows, baseline, candidate):
    pairs = [(w[baseline], w[candidate]) for w in windows]
    before = sum(a["median_us"] for a, _ in pairs)
    after = sum(b["median_us"] for _, b in pairs)
    return dict(
        windows=len(pairs),
        saved_us=statistics.mean(a["median_us"] - b["median_us"] for a, b in pairs),
        saved_fraction=1 - after / before,
        speedup=before / after,
        paired_saved_us=[a["median_us"] - b["median_us"] for a, b in pairs],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    fixed = []
    for path in sorted((args.root / "fixed").glob("sample_*_boundary_*.json")):
        if "profile" in path.name:
            continue
        row = json.loads(path.read_text())
        assert row["committed_prefix_bitwise"] and row["target_repeat_bitwise"]
        for cell in row["rows"]:
            validate_dispatch(cell)
        fixed.append({f"{x['moe']}_{x['gdn']}": x for x in row["rows"]})
    cells = [f"{moe}_{v}" for moe in ("h8", "h4", "p0125") for v in ("v0", "v2", "v3")]
    assert fixed and all(set(w) == set(cells) for w in fixed)
    complete = json.loads((args.root / "fixed/complete.json").read_text())
    assert len(fixed) == complete["windows"]
    forward = {}
    for cell in cells:
        rows = [w[cell] for w in fixed]
        profiles = [profile(Path(r["profile"])) for r in rows]
        assert all(p["full_gdn_eager_gpu_busy_us"]["count"] == 30 for p in profiles)
        forward[cell] = dict(
            window_medians_us=stats([r["median_us"] for r in rows]),
            pooled_samples_us=stats([t for r in rows for t in r["samples_us"]]),
            fixed_mtp_accepted=stats([r["accepted"] for r in rows]),
            accepted_beyond_target=stats([r["accepted_beyond_target"] for r in rows]),
            rejected_target_accepted=stats(
                [r["rejected_target_accepted"] for r in rows]
            ),
            target_prefix_agreement=stats([r["target_prefix_agreement"] for r in rows]),
            selected_experts=stats(
                [n for r in rows for n in r["selected_experts_per_token_layer"]]
            ),
            profiled_gdn_modules_us=stats(
                [p["full_gdn_eager_gpu_busy_us"]["total"] for p in profiles]
            ),
            profiled_routed_expert_gemms_us=stats(
                [p["kernels"]["fused_moe_kernel"]["total"] for p in profiles]
            ),
        )
    effects = {}
    for baseline, candidate in [
        ("h8_v0", "h4_v0"),
        ("h8_v0", "h8_v2"),
        ("h8_v0", "h8_v3"),
        ("h4_v0", "h4_v2"),
        ("h4_v0", "h4_v3"),
        ("h8_v2", "h4_v2"),
        ("h8_v3", "h4_v3"),
        ("h8_v0", "h4_v2"),
        ("h8_v0", "h4_v3"),
        ("h8_v0", "p0125_v0"),
        ("p0125_v0", "p0125_v2"),
        ("p0125_v0", "p0125_v3"),
        ("h8_v2", "p0125_v2"),
        ("h8_v3", "p0125_v3"),
        ("h8_v0", "p0125_v2"),
        ("h8_v0", "p0125_v3"),
    ]:
        effects[f"{baseline}->{candidate}"] = paired_effects(fixed, baseline, candidate)
    online = {}
    baseline_tokens = json.loads(
        (args.root / "online/h8_native/timings.json").read_text()
    )[0]["tokens"]
    baseline_tokens = [t for group in baseline_tokens for t in group]
    for cell in cells:
        path = args.root / "online" / cell.replace("v0", "native")
        row = summarize(path)
        row["outer_acceptance_rate"] = (
            row["outer_accepted"]["total"] / row["outer_scheduled"]["total"]
        )
        tokens = json.loads((path / "timings.json").read_text())[0]["tokens"]
        tokens = [t for group in tokens for t in group]
        row["first_difference_from_h8_v0"] = [
            next((i for i, (a, b) in enumerate(zip(x, y, strict=True)) if a != b), None)
            for x, y in zip(baseline_tokens, tokens, strict=True)
        ]
        online[cell] = row
    (args.root / "summary.json").write_text(
        json.dumps(
            dict(
                fixed=forward,
                effects=effects,
                online=online,
                note=(
                    "Fixed windows are paired; "
                    "online trajectories differ across policies."
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
