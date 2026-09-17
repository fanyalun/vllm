# Windowed GDN evaluation on A100

The implementation is opt-in. Windowed carry reduces recurrent allocation and
work, but decreases fixed-prefix agreement. It is an approximate Pre-Verify
policy, including its forced-Full control; it is not a claim of lossless
generation. See [configuration and state contract](windowed_gdn.md).

## Reproducibility boundary

- Source baseline: `86be3421aafc971dbe95ea9280320f6b16908e46`.
- Local artifacts: `benchmark_results/windowed_gdn_20260917/`.
- Qwen3.6-35B-A3B, `/data1/fanya/Qwen/Qwen3.6-35B-A3B`, one A100 80 GB,
  TP1/B1, BF16 model/Conv, FP32 SSM, greedy, MTP D4, four rounds, top-h4,
  balanced stopping, no prefix caching or async scheduling.
- GSM8K first 16 prompts from
  `benchmark_results/three_level_p50_20260916/final_prompts.jsonl`, seed 42,
  exactly 256 returned tokens, EOS ignored. Prompt identity is checked.
- Source snapshots and SHA256 manifests identify measured working trees.
  Later additions to diagnostic scripts do not change production source.
  The interrupted `online/` run lacks a completion marker and is excluded;
  its timer preceded the common worker-side device drain.
- The 64-boundary quality run has its own source snapshot. The additional
  forced-rejection probe is recorded separately.

## Numerical and isolation evidence

The final four-suite regression run passed **363 tests** in 100.05 seconds;
all applicable pre-commit hooks passed. Both legacy and windowed
exception-binding tests passed. Tests cover T0/1/5/6/10/16, strided tensors,
alpha/beta boundaries, dynamic
graph gates and lengths, invalid slots, dirty state followed by Skip, and carry
with real Conv acceptance positions. Config tests reject unsupported roles and
nonzero CPU request temperatures.

`recurrent_correctness_final.json` contains 900 independent reference checks on
the captured 30-layer boundary-96 inputs. All pass `atol=rtol=1e-3` and action
classification checks. Maximum state absolute error is 8.5831e-6 (layer 0,
head 24, V1/T16); maximum output absolute error is 1.5259e-5 (layer 0, head 23,
V2/T16). The file also reports relative L2 and separate differences from
stepwise Decay. Forced Full is bitwise equal to the existing local sequential
replay recurrence at T1/T5 for all 30 layers and tested tiles. This is a
different check from equality to the native fused GDN backend.

All 64 real prefix boundaries pass committed GDN/attention isolation,
same-input native top-h4 predictions, and same-input full-top-8 Target logits
bitwise equality before/after the approximate calls. All 64 observed anchors
match AR. The attention comparison checks the committed BHNC prefix, and GDN
checks the active canonical checkpoint. Uncommitted physical suffix writes
are allowed and overwritten before later consumption.

## Fixed-prefix quality

L5/L16 are consecutive next-token matches against AR, capped at 5/16. L20 is
the continuous four-call D5 trace at boundary 96. These teacher-forced inputs
are separate from online rejection/correction trajectories and do not measure
GSM8K final-answer accuracy.

| Case | Mean L5 / 5 | Mean L16 / 16 | Perfect L16 / 64 | Mean L20 / 20 |
| --- | ---: | ---: | ---: | ---: |
| V0 native | 4.9063 | 14.0938 | 45 | 18.1875 |
| V1 forced Full carry | 4.9063 | 13.8750 | 43 | 18.1875 |
| V2 per-token | 4.7813 | 12.7656 | 37 | 17.5000 |
| V3 window 5 | 4.6875 | 11.3750 | 28 | 16.6250 |
| V4-D | 4.6875 | 11.3750 | 28 | 16.6250 |
| V4-Q | 4.6875 | 11.2969 | 27 | 16.0625 |
| V4-DQ | 4.6875 | 11.2969 | 27 | 16.0625 |

Native top-h4 already differs from AR at some positions. Windowed variants
add measurable quality loss; baseline disagreement does not excuse or explain
all new disagreements. Relative state L2 against the native path reaches
0.1519 for windowed variants across the fixed-prefix traces.

The current V1 output matches all 16 saved 256-token sequences from the
2026-09-16 `exact:carry:balanced` replay-tail baseline (source commit
`fd08718dcc9fbea17e4138e58b5623ab6ae520a2`, identical prompts/seed/length).
That historical baseline matches AR on only 3/16 sequences. This establishes
pre-existing hierarchical AR divergence and a full-carry output control; it
does not establish the root cause of every native/windowed difference.
`historical_divergence.json` records the first differing positions.

## Recurrent, GDN, and Pre-Verify costs

The core benchmark uses sample 0/boundary 96, all 30 real layers, packed stride
12288 for Q/K/V and stride 64 for gates. Every sample restores the same initial
state outside timing, then times one graph replay. There are 50 warmups and
200 raw samples per case. T6/10/16 repeat captured T5 inputs as shape probes.
The 90 rows cover five lengths, six variants, and three tile choices.

The observed native speculative backend is the CUDA
`gdn_decode_post_conv_mtp_kernel<float, 2, false>` with fused normalization,
called 30 times in the V0 profile. The private windowed path calls the Triton
`_windowed_update` 30 times and retains a separate full-head fused norm. Native
recurrence is therefore not isolated as a directly comparable pure-core row;
the complete GDN and Pre-Verify columns include each path's real fusion.

| Case | T5 core median / P95, us | Full Pre-Verify median / P95, ms | GDN GPU busy time, ms |
| --- | ---: | ---: | ---: |
| V0 | Native core not isolated | 10.762 / 10.943 | 2.735 |
| V1 | 459.78 / 461.82 | 10.641 / 10.824 | 2.584 |
| V2 | 454.66 / 456.70 | 10.567 / 10.715 | 2.593 |
| V3 | 441.34 / 443.39 | 10.529 / 10.678 | 2.571 |
| V4-D | 444.42 / 447.49 | 10.527 / 10.716 | 2.572 |
| V4-Q | 489.98 / 491.52 | 10.587 / 10.645 | 2.644 |
| V4-DQ | 465.92 / 467.97 | 10.580 / 10.610 | 2.632 |

Core times sum the 30 recurrent launches in one graph, with default BV16/BQ2,
four warps. Pre-Verify times cover the full top-h4 forward on the same T5
inputs. Full-top-8 Target costs 11.745 ms median in this fixed-input forward.
GDN busy time is the sum of correlated GPU kernel durations from a separate
eager profiler run: projections, Conv, recurrent update, normalization, and
output projection across 30 layers. It is not graph wall latency and is not
added to the Pre-Verify column. Each profiling run checks predictions, logits,
and private state against its unprofiled reference.

V3 improves core time only 1.042x over forced Full. Q tiling regresses this
shape; BV8/BQ2 Q-only kernels also have four spills for T5/6/10/16. Default
BV16/BQ2 has no spills and is retained. No 2x result was observed here.

An additional eight-warp sweep completed 18 T1 timing rows, then failed the
T5 forced-Full bitwise gate at layer 0/BV16/BQ2 and was stopped. A focused
reproduction saves `kernel_warps8_failure/failure.json`: state maximum error
9.5367e-7, relative L2 2.6643e-8, output maximum error 9.3132e-10. Although these
are small, the required bitwise gate failed; no tolerance was relaxed and
eight-warps is not recommended. The completed 90-row production-tile sweep
uses four warps. `kernel_warps8/` remains an explicitly rejected partial sweep,
not another complete performance matrix.

`kernel_final/kernel_15.ptx` is the default T5/window-5/no-optimization kernel
(72 registers, zero spills). Its dirty-state branch at lines 1469-1471 skips
the final FP32 state stores at 1475-1484 when every window is Skip. Full-only
K/V loads and delta arithmetic are reached through a separate action branch.
Earlier dirty windows preserve writeback even when the final window is Skip.
The kernel artifacts plus canary/bitwise tests establish control-flow and
state effects separately.

## Complete generation cost

All 640 uninstrumented runs completed: eight policies, 16 prompts, five
alternating-order repeats. Each generated exactly 256 tokens. Timing includes
the device drain and terminal proposal work. All five repetitions produce the
same per-policy token sequences. AR has 80 separately repeated drained runs.

| Case | Returned tokens/s | Ratio to AR | Identical to AR / 16 |
| --- | ---: | ---: | ---: |
| AR | 95.505 | 1.000x | 16 |
| V0 native | 129.020 | 1.351x | 4 |
| Old p50, alpha 0.98 | 163.088 | 1.708x | 7 |
| V1 forced Full carry | 164.775 | 1.725x | 3 |
| V2 per-token, alpha 0.95 | 161.087 | 1.687x | 7 |
| V3 window 5 | 152.276 | 1.594x | 2 |
| V4-D | 151.743 | 1.589x | 3 |
| V4-Q | 152.063 | 1.592x | 3 |
| V4-DQ | 152.477 | 1.597x | 5 |

These are throughput ratios for the recorded output behavior, not lossless
speedups. V3 is 18.0% faster than V0, but **7.6% slower than forced-Full carry**
and **6.6% slower than old p50**. The window approximation does not improve
the complete algorithm over those controls. None reaches 2x AR. V4's small
differences do not establish a useful additional benefit, especially given
the lower fixed-prefix agreement of Q tiling.

Each windowed policy owns 60 MiB of SSM and 3.75 MiB of Conv tensors, with no
tail or repair buffers. Old p50 uses 123.75 MiB including its second SSM buffer;
V0's snapshot path uses 424.6875 MiB. These are per-policy private tensor
allocations, excluding graph pools and small metadata; the benchmark retains
multiple policy instances to alternate them. All six windowed policies pass
the exact canonical SSM/Conv initialization check and finish proposals with
an invalid request slot.

The separate 16-prompt cycle audit records the following counts and costs.
Acceptance divides accepted draft candidates by proposed draft candidates;
the anchor is not counted as a fifth draft. Round counts include terminal
proposals. There are exactly 16 more proposals than Target-verified cycles,
one terminal proposal per request.

| Case | Proposals / verified cycles | Inner rounds | Inner acceptance | Init + Conv, ms/token | Pre-Verify, ms/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 | 338 / 322 | 1158 | 73.90% | 1.1776 | 3.0712 |
| Old p50 | 363 / 347 | 1232 | 71.41% | 0.0617 | 2.8756 |
| V1 | 363 / 347 | 1211 | 71.08% | 0.0611 | 2.8381 |
| V2 | 368 / 352 | 1244 | 71.78% | 0.0622 | 2.9104 |
| V3 | 395 / 379 | 1321 | 70.29% | 0.0665 | 3.0815 |
| V4-D | 393 / 377 | 1334 | 70.11% | 0.0664 | 3.1000 |
| V4-Q | 396 / 380 | 1319 | 69.67% | 0.0665 | 3.0941 |
| V4-DQ | 397 / 381 | 1313 | 70.22% | 0.0665 | 3.0776 |

Event intervals include host launch gaps. These audited costs are normalized
by 4096 returned tokens and are separate from formal throughput; proposal
events overlap their contained stages and must not be added to those stages.
V3 executes 9.1% more inner rounds and 9.2% more Target-verified cycles than
V1. Its small fixed-input kernel saving is outweighed by the complete workload.
Output trajectories differ between policies, so this is an observed complete
generation comparison, not an equal-output microbenchmark.

All 112 action-audit runs completed with identical tokens to the formal runs.
The final marker reports **880/880 rows**: 640 timing, 128 cycle audit, and
112 action audit. The separate full-generation profiler also passes its
32-token output parity check. `online_summary.json` contains each repeat's
throughput, first token divergences, integer counters, phase costs, and memory
metadata. Its assertions reconcile action totals against actual round counts.

| Case | Full / Decay / Skip token actions | Whole-call unchanged heads |
| --- | ---: | ---: |
| Old p50 | 51.739% / 10.955% / 37.306% | Not measured by legacy counter |
| V1 | 100% / 0% / 0% | 0% |
| V2 | 51.629% / 6.476% / 41.896% | 29.276% |
| V3 | 52.590% / 6.619% / 40.791% | 40.791% |
| V4-D | 52.615% / 6.697% / 40.688% | 40.688% |
| V4-Q | 52.519% / 6.661% / 40.820% | 40.820% |
| V4-DQ | 52.375% / 6.621% / 41.004% | 41.004% |

V3 records 666924 Full, 83935 Decay, and 517301 Skip head-window decisions;
each covers five actual input positions in this online run. Its 517301
unchanged heads avoid the final state store. V2 has a similar per-token Skip
rate, but only 29.3% of heads stay unchanged for the whole call. These real
work reductions are insufficient to offset the additional generation cycles.

The audit's Target-plus-between-stage interval rises from 1.5869 ms per
returned token for V1 to 1.7311 ms for V3. Terminal proposal cost is 0.1792 and
0.1787 ms per returned token respectively; these costs remain in final timing.
All acceptance positions 0 through 4 occur naturally for each windowed
policy; V3 records 184 pairs of consecutive rejecting rounds within verified
cycles. The separate forced-rejection probe verifies controlled trajectories.

The forced probe passes all six single-slot variants over four trajectories
each, **24 trajectories / 96 actual verification rounds**: all accepted,
two accepted, first rejected followed by acceptance, and four consecutive
rejections. Candidate construction uses real model predictions and an actual
mismatching candidate; acceptance is computed, not fabricated. Every round
preserves the full SSM tail bitwise, shifts Conv by the real acceptance count,
and consumes the previous correction at the next position. Committed-prefix
isolation and same-input Target logits bitwise equality still pass after the
probe. It runs without early stopping and is separate from the balanced
production controller's natural rejection traces.

## Decision

Implementation and the requested primary measurement coverage are complete.
Strict AR parity is **not** established. Keep the feature opt-in and retain
four-warps V3 as the independent windowed implementation; do not make it a
performance default over the measured old p50 or Full-carry controls. V4
variants are available for controlled experiments, but this workload gives
no reason to recommend them. The eight-warp candidate is explicitly rejected
by the numerical gate. No numerical contract was weakened to obtain a speedup.
