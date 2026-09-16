# Shared-input GDN results: 2026-09-16

## Conclusion

The implementation reduces fixed-window pre-verifier cost, but adding shared inputs to the
current replay-tail configuration does not improve final throughput in this pilot.
Keep `preverify_gdn_mode="replay_tail"` and `preverify_gdn_group_mode="none"` as the
recommended experimental configuration. The new group option defaults to `none`.

Qwen3.6-35B-A3B, unquantized BF16 weights and Conv state, FP32 SSM, A100 80GB,
TP1/PP1, B1, MoE top-h4, inner D4, at most four rounds, existing low_error policy.
MTP used GPU 0; DSpark used GPU 1. Each method used the same four prompts,
256 returned tokens and three paired timing trials. Auditing was separate from timing.

## Implementation

- Groups 3–9 share the first layer's pre-RMSNorm input; all layer weights and per-token gates remain independent.
- `projection` groups RMSNorm and Q/K/V/z/a/b projection; `full` also groups Conv, recurrence, gated normalization and output projection.
- Residuals and MoE remain sequential. The group path retains vLLM's FP32 normalization sum and BF16 residual storage.
- State policy `none` retains per-position checkpoints; `replay_tail` writes only the tail and deliberately reuses rejected tails until the next outer reset.
- The private execution plan references the existing Target parameters. Target and small Draft keep their normal entry points.
- Native normalization was fused in the private execution plan to avoid the eager multi-kernel overhead observed in the initial smoke implementation.

## Fixed-input GPU latency

Three real windows, three rounds, eight cases (including serial shared-input references),
five boundaries and 30 repeats: 10,800 rows. State restoration and explicit 128 MiB L2
eviction are outside timing. CUDA events enclose graph replay. Table values are means
of nine per-window/per-round medians. Combined is directly measured.

| State | Group | Forward ms | Advance ms | Combined ms | Change vs same state, no grouping |
| --- | --- | ---: | ---: | ---: | ---: |
| none | none | 8.3833 | 0.5810 | 8.9535 | -0.00% |
| none | projection | 8.2736 | 0.5772 | 8.8455 | -1.21% |
| none | full | 7.7930 | 0.5779 | 8.3622 | -6.60% |
| replay_tail | none | 8.0429 | 0.0665 | 8.0986 | -0.00% |
| replay_tail | projection | 7.9411 | 0.0665 | 8.0011 | -1.20% |
| replay_tail | full | 7.5773 | 0.0664 | 7.6325 | -5.76% |

Negative latency change means faster. Replay-tail plus full grouping is about
14.75% lower than the original per-position-state baseline.
Every paired combined measurement improves for projection and full grouping relative
to its corresponding ungrouped state policy. The serial shared-input references are
about 0.44–0.46 ms slower than the respective ungrouped models; sharing inputs alone
does not remove the projection work.

Isolated projection/GDN measurements cover only the seven selected groups with identical
shared anchors. At these boundaries, `none` deliberately uses the same serial shared-input
reference as `serial`. Its eager normalization/staging overhead is included, so those
isolated speedups must not replace the complete-model numbers above.

## Final generation throughput

Throughput is the median of three trials, each computed as total returned tokens / total time.
Changes compare against no grouping under the same state policy.

| State | Group | MTP tokens/s | MTP change | DSpark tokens/s | DSpark change |
| --- | --- | ---: | ---: | ---: | ---: |
| none | none | 98.54 | +0.00% | 89.44 | +0.00% |
| none | projection | 85.93 | -12.80% | 87.49 | -2.18% |
| none | full | 89.08 | -9.60% | 91.81 | +2.65% |
| replay_tail | none | 104.89 | +0.00% | 97.07 | +0.00% |
| replay_tail | projection | 94.05 | -10.34% | 88.48 | -8.85% |
| replay_tail | full | 101.95 | -2.81% | 89.07 | -8.24% |

Full grouping alone gives a small DSpark gain over the per-position-state baseline,
but remains slower than the existing replay-tail configuration. Both grouped modes
lose throughput when combined with replay-tail in this four-prompt experiment.

## Acceptance and loop counts

| Method | State:group | Outer accepted / scheduled | Inner accepted / proposed | Outer cycles | Inner rounds | AR equal |
| --- | --- | --- | --- | ---: | ---: | ---: |
| mtp | none:none | 928/1239 (74.90%) | 840/1596 (52.63%) | 130 | 399 | 1/4 |
| mtp | none:projection | 892/1372 (65.01%) | 909/1852 (49.08%) | 151 | 463 | 0/4 |
| mtp | none:full | 908/1444 (62.88%) | 981/1852 (52.97%) | 142 | 463 | 0/4 |
| mtp | replay_tail:none | 916/1301 (70.41%) | 888/1652 (53.75%) | 131 | 413 | 0/4 |
| mtp | replay_tail:projection | 894/1388 (64.41%) | 919/1876 (48.99%) | 153 | 469 | 0/4 |
| mtp | replay_tail:full | 915/1304 (70.17%) | 867/1748 (49.60%) | 141 | 437 | 0/4 |
| dspark | none:none | 904/1186 (76.22%) | 753/1732 (43.48%) | 138 | 433 | 0/4 |
| dspark | none:projection | 913/1256 (72.69%) | 811/1780 (45.56%) | 138 | 445 | 1/4 |
| dspark | none:full | 914/1317 (69.40%) | 877/1760 (49.83%) | 129 | 440 | 0/4 |
| dspark | replay_tail:none | 914/1239 (73.77%) | 794/1780 (44.61%) | 141 | 445 | 0/4 |
| dspark | replay_tail:projection | 891/1426 (62.48%) | 932/1976 (47.17%) | 154 | 494 | 0/4 |
| dspark | replay_tail:full | 873/1403 (62.22%) | 907/1984 (45.72%) | 162 | 496 | 0/4 |

For MTP, replay-tail plus full grouping increases inner rounds from 413 to 437
and outer cycles from 131 to 141. For DSpark they increase from 445 to 496 and
141 to 162. Faster individual forwards therefore do not yield fewer total GPU operations.
Acceptance ratios alone also miss the distribution of accepted lengths and stopping decisions.

AR was freshly rerun; prompt hashes and order matched across AR/MTP/DSpark.
The ungrouped baselines already fail strict AR equality. All 16 ungrouped outputs
(two methods × two state policies × four prompts) match the previous replay-tail report.
All new timing/audit outputs repeat their warmup tokens. Neither the kernel tolerances
nor this small generation matrix establish lossless acceleration or broad task accuracy.

## Original HF experiment reproduction

The original four-sample, 32-prediction experiment was rerun with the same model,
prompts, seed and horizons. All 60 windows, prompt IDs, boundary seeds, exact trajectories
and acceptance summaries match the supplied shared_group_acceptance_20260916_v2 artifacts.
Accepted/proposed tokens are D4: 128/128; D8: 125/128; D16: 117/128; D32: 117/128.
These overlapping windows contain only 128 distinct prediction positions. Identity-hook
controls have zero logit error and identical caches; that control is not a claim that
shared-input logits equal exact logits. Online top-h4, longer output and tail reuse
are additional conditions absent from that offline approximation experiment.

## Kernel validation and source audit

- Widths 1–5, both Conv layouts, both state policies, checkpoint canaries and repeated CUDA Graph replay pass.
- Identical-input recurrence uses atol=rtol=1e-3; BF16 GEMM reduction-order checks use atol=1e-3, rtol=8e-3.
- 216 tests passed in the complete related suite. A subsequent BF16-Conv startup guard passed the four-test grouped control suite, including its new rejection test.
- Cold-cache projection tuning selected output tiles 32 for BA and 64 otherwise, with reduction tiles 128 and four warps.
- At width 5: three native QKVZ projections 129.02 us vs grouped 109.57 us; output projections 70.66 us vs 53.25 us. These microbenchmarks exclude normalization and the rest of the model.
- Production group kernels have zero spills: recurrence 56 registers; grouped linear 48–52; Conv 32; gated normalization 20; input normalization 40; fused add/norm 44–48.
- Grouping does not change private state capacity: SSM 360 MiB for per-position state or 60 MiB for replay-tail; Conv 64.69 MiB or 3.75 MiB respectively. No scheduler-visible capacity gain is claimed.
- The only production change after the final measurements is a startup rejection for unsupported FP32 Conv state. Removing exactly that guard reproduces the measured source hashes; all hot paths match. Measured source snapshots and source_compatibility.json preserve this evidence.

## Artifacts and reproduction

Commands and semantics: [grouped_gdn.md](grouped_gdn.md).
Local root: `benchmark_results/grouped_gdn_20260916/`.

- `final/{mtp,dspark}`: six-case contracts, 96 rows per method, repeatability and completion markers.
- `final/{ar.json,summary.json,summary.csv}`: fresh AR and request-identity-checked summary.
- `cost_final/{contract.json,results.json,summary.json,paired.csv}`: final 10,800-row fixed-input matrix, graph/eager checks and compiler metadata.
- `kernel_tuning.json`: shape/tile sweep with cold-cache latency, logical bytes and spill counts.
- `hf_reproduction/reproduction_audit.json`: exact comparison with the supplied HF experiment.
- `baseline_compatibility.json`, `source_compatibility.json`, `tests_final.log`, `startup_guard_tests.log` and hook logs: validation evidence.

`smoke/`, `cost_probe/` and `cost/` are intermediate artifacts. The first cost probe
ran out of memory because it retained eight separate state snapshots; the harness now
shares two state pools and restores them outside timing. Final measurements completed.
Scripts and this compact report are published; raw measurements remain local.

AI assistance was used for implementation, tests and benchmark analysis.
