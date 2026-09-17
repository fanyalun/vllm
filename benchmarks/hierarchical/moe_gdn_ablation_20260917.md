# Qwen Pre-Verify MoE and GDN ablation

This pilot crosses native top-8, fixed top-4, and p=0.125 routing with GDN V0,
V2 and V3, for nine cells. It changes benchmark code only. Shared experts remain
unchanged. Fixed top-4 uses no probability cutoff. The threshold strategy keeps
native top-8 candidates whose within-top-8 normalized gate probability is at
least 0.125. Both skipping strategies preserve retained native weights without
renormalizing them. A separate audit records actual retained expert counts.
The Draft remains MTP D4, with at most four hierarchical rounds and balanced
stopping. All runs use Qwen3.6-35B-A3B, one A100 80 GB PCIe, TP1, B1, greedy,
BF16 model/Conv and FP32 SSM. Prefix caching and async scheduling are disabled.

## What the comparisons measure

- h8/V0 is the complete-model computation reference, using native GDN and all
  eight normally selected routed experts, rather than all 256 available experts.
- h4/V0 isolates expert skipping with native recurrent state management.
- h8/V2 and h8/V3 isolate the GDN path changes with full expert computation.
- h4/V2 and h4/V3 combine the changes. Compare them with h4/V0 for the
  incremental GDN benefit, and with h8/V2 or h8/V3 for incremental MoE benefit.
- p0.125/V0, V2 and V3 repeat those comparisons with adaptive retained counts,
  without combining the threshold with fixed top-4.
- V0 keeps native candidate snapshots and restores the accepted state. V2/V3
  use one private slot and carry the full approximate tail after inner rejection.
  Consequently, online V0-to-V2/V3 differences include state-management semantics;
  they do not isolate conditional arithmetic alone. V2 uses window one, V3 five,
  both with alpha 0.95 and beta 0.36328125, with no V4 optimizations.

## Protocol

Fixed-input measurement uses four GSM8K prompts at saved AR prefix boundaries
32 and 96. At each prefix it captures the actual MTP anchor plus four candidates.
All nine cells receive identical tokens, positions and canonical initial states.
They use the same full-model forward wrapper, logits computation and selection,
so the complete-model reference and Pre-Verify have identical five-input widths.
It measures CUDA graph replay after 50 warmups, with 200 event samples per cell.
The private state is restored outside every timed sample. Cell order reverses
between windows. This is a warm-cache forward measurement; it excludes private
initialization, Conv advancement, Draft generation, scheduling and final Target
verification. The independent online stage audit records these additional costs.

Each cell has a separate eager CUPTI trace, graph/eager predictions and state
checks, and committed GDN/attention prefix byte checks. Complete-model logits
are checked before and after all nine cells. Fixed-window MTP acceptance is
conditional on these eight common windows; it is separate from online acceptance.
The shared model is constructed with the dynamic GDN dispatch enabled, then uses
native private state for prefix initialization. V0 takes that dispatch's exact
native branch. V2/V3 must show 30 windowed kernel launches and nonzero action
counts; the summarizer rejects mislabeled paths. An initial pass built with a
native-only model did not switch dispatch and was discarded in
`invalid_fixed_dispatch/`. Its timings are excluded from the results.

Online measurement uses the same first four prompts, 128 returned tokens each,
three repetitions after per-prompt warmup. Timing passes have no stage events or
profiler and reject new JIT. An independent audit covers all four prompts and
checks output equality. It records inner rounds, outer verification counts,
scheduled/accepted draft counts and stage timing. Outer acceptance subtracts the
correction/bonus token; inner acceptance excludes the anchor. Stages are nested:
proposal time must not be added to its Draft/Pre-Verify/maintenance components.
Acceptance uses the sampler's actual counters, including the final verification's
accepted suffix that may subsequently be truncated by the 128-token output cap.
It is not the number of accepted draft tokens ultimately returned to the user.
Online cells run in separate engines; repeated measurements are within each
cell, not interleaved across all nine engines. The six fixed-top-k online cells
ran on GPU 0. An unrelated job occupied GPU 0 during threshold startup, so the
three threshold online cells and all nine paired fixed-window cells run on
GPU 1, another A100 80 GB PCIe. Every model run remains TP1 on a single GPU.
Do not interpret cross-device online throughput differences as policy gains.
The forward comparisons use the same device for every cell.
The online wall timer uses the existing `LLM.generate` return boundary without
an additional post-return device drain. Its throughput is secondary and is not
used to attribute the two forward optimizations or certify complete device cost.
For V0, the stage named `conv_maintenance` also includes accepted SSM restoration;
for V2/V3 it advances Conv while retaining the approximate SSM tail.

## Paired forward and kernel results

All eight windows completed all nine cells. Forward is the mean of the eight
per-window medians; P95 pools 1,600 timed replays per cell. Times are milliseconds.

| MoE/GDN | Forward | P95 | Saved vs h8/V0 | Experts/token/layer | GDN GPU busy | Expert GEMM GPU busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h8_v0 | 11.823 | 12.237 | 0.00% | 8.000 | 2.723 | 5.588 |
| h8_v2 | 11.672 | 12.021 | 1.28% | 8.000 | 2.589 | 5.461 |
| h8_v3 | 11.622 | 11.973 | 1.70% | 8.000 | 2.564 | 5.440 |
| h4_v0 | 10.405 | 10.667 | 11.99% | 4.000 | 2.729 | 2.936 |
| h4_v2 | 10.245 | 10.472 | 13.34% | 4.000 | 2.600 | 2.832 |
| h4_v3 | 10.205 | 10.466 | 13.69% | 4.000 | 2.571 | 2.815 |
| p0125_v0 | 8.508 | 8.721 | 28.04% | 2.948 | 2.738 | 2.456 |
| p0125_v2 | 8.358 | 8.548 | 29.31% | 2.939 | 2.605 | 2.359 |
| p0125_v3 | 8.306 | 8.478 | 29.74% | 2.933 | 2.580 | 2.318 |

GDN GPU busy sums kernels within the 30 complete GDN module ranges; expert GEMM
GPU busy sums the routed-expert fused GEMMs across 40 MoE layers. These come from
separate eager CUPTI traces. Their boundaries differ and they must not be added
to explain the graph wall time exactly. Comparing a fused native GDN kernel
directly with the unfused windowed recurrent core would omit work on one side.

With native GDN, fixed h4 saves 1.418 ms (11.99%), while p=0.125 saves 3.315 ms
(28.04%). At h8, V2 saves 0.151 ms (1.28%) and V3 saves 0.201 ms (1.70%).
At h4, incremental V2/V3 savings are 0.160/0.200 ms (1.53%/1.92%); at p=0.125
they are 0.150/0.201 ms (1.76%/2.37%). V3 improves over V2 by only 0.041 to
0.052 ms per verification across these MoE policies. One h8 window regressed
for both V2 and V3; signed per-window differences are retained without filtering.

The GDN module busy time at h8 falls from 2.723 ms to 2.589/2.564 ms, or
4.93%/5.86%. Routed-expert GEMM busy time with V0 falls from 5.588 ms to
2.936 ms (h4) and 2.456 ms (p=0.125), or 47.47% and 56.04%. Thus expert
skipping supplies most of the measured forward reduction. GDN state lifecycle
savings are additional and are reported separately below.

All nine cells accept 26/32 MTP candidates across the eight fixed windows
(3.25 per window). This small fixed-prefix check cannot expose the online
acceptance losses accumulated over repeated rejection and carry trajectories.
The online counters below are required for that comparison.

Validation: 72 graph/eager and state checks passed, all committed prefix checks
passed, and the full-reference logits before/after each window were bitwise
equal. Every V2/V3 profile contains the expected 30 windowed recurrent launches
and its independent audit has nonzero actions. The measurement suite passed
34 tests; the six task files passed pre-commit. This does not establish strict
AR parity or a statistically powered end-to-end ranking.

## Online acceptance results

All nine cells passed within-cell repeat and audit output equality. Counts below
come from the independent four-prompt audit, not from summing timing repetitions.

| MoE | GDN | Inner accepted/round | Outer accepted/verify | Outer accepted/scheduled | Outer acceptance rate |
| --- | --- | ---: | ---: | ---: | ---: |
| h8 | V0 | 2.921 | 14.086 | 493/493 | 100.00% |
| h8 | V2 | 2.874 | 13.026 | 495/515 | 96.12% |
| h8 | V3 | 2.861 | 11.279 | 485/587 | 82.62% |
| h4 | V0 | 3.161 | 14.000 | 518/558 | 92.83% |
| h4 | V2 | 3.073 | 13.342 | 507/560 | 90.54% |
| h4 | V3 | 2.934 | 11.767 | 506/599 | 84.47% |
| p=0.125 | V0 | 3.007 | 13.231 | 516/574 | 89.90% |
| p=0.125 | V2 | 2.849 | 11.302 | 486/583 | 83.36% |
| p=0.125 | V3 | 2.934 | 11.814 | 508/610 | 83.28% |

Fixed h4 alone loses 7.17 percentage points of outer acceptance; threshold
routing alone loses 10.10 points. At h8, V2 loses 3.88 points, while V3 loses
17.38 points. These effects are not additive: trajectories, proposal lengths,
stopping decisions and the routed experts change across cells. In particular,
higher inner acceptance does not guarantee higher final Target acceptance.

The same-device h8 stage audit also separates state maintenance from forward
work: initialization averages 5.243 ms with V0 and 0.736 ms with V2; accepted
state advancement averages 2.774 ms and 0.079 ms respectively. This is a material
part of the GDN implementation change and must not be called pure recurrent
arithmetic savings. These are CPU-paced CUDA event spans from online execution,
not isolated kernel measurements.

The pilot measures acceptance, not GSM8K answer accuracy. Cross-cell output
sequences differ, including h4/V0 versus h8/V0, so no strict lossless claim is
made. First differing output token positions are retained in `summary.json`.

## Reproduction

Use the existing environment; no new dependencies are required.

```bash
export CUDA_VISIBLE_DEVICES=0
export PATH="$PWD/.venv/bin:$PATH"
for h in 8 4; do
  for case in native v2 v3; do
    .venv/bin/python benchmarks/hierarchical/run_batch.py \
      --batch 1 --case "$case" --top-h "$h" --samples 4 --tokens 128 \
      --repeats 3 --audit-all \
      --dataset benchmark_results/windowed_batch_20260917/prompts.jsonl \
      --output "benchmark_results/moe_gdn_ablation_reproduction/online/h${h}_${case}"
  done
done
for case in native v2 v3; do
  .venv/bin/python benchmarks/hierarchical/run_batch.py \
    --batch 1 --case "$case" --top-h 8 --min-weight 0.125 \
    --samples 4 --tokens 128 --repeats 3 --audit-all \
    --dataset benchmark_results/windowed_batch_20260917/prompts.jsonl \
    --output "benchmark_results/moe_gdn_ablation_reproduction/online/p0125_${case}"
done
.venv/bin/python benchmarks/hierarchical/run_moe_gdn_ablation.py \
  --dataset benchmark_results/windowed_batch_20260917/prompts.jsonl \
  --ar benchmark_results/windowed_gdn_20260917/ar/results.json \
  --samples 4 --boundaries 32 96 \
  --output benchmark_results/moe_gdn_ablation_reproduction/fixed
.venv/bin/python -m benchmarks.hierarchical.summarize_moe_gdn_ablation \
  benchmark_results/moe_gdn_ablation_reproduction
.venv/bin/python -m pytest -q tests/benchmarks/test_hierarchical_measurement.py
```

Large artifacts remain local under `benchmark_results/moe_gdn_ablation_20260917/`.
The manifest records source fingerprints and prompt identity; raw online tokens,
per-window latency samples, profiles and checks are retained. These artifacts
are not included in the code commit.
