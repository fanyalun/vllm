# Hierarchical decoding failure diagnosis

Date: 2026-09-10. Runtime source: `afd237df26008cbea173cb50d35a277dd229f732`.
Only benchmark instrumentation and analysis changed in this investigation.

## Conclusion and priorities

The disappointing results combine a structural cost problem with implementation
problems. They do not establish that a correctly implemented hierarchy cannot
work. In particular, the previous Gemma MTP acceptance measurements are affected
by a confirmed stale-attention-metadata bug.

| Priority | Cause | Qwen | Gemma | Evidence |
| --- | --- | --- | --- | --- |
| 1, common performance limit | Four serial P backbone evaluations are too expensive relative to Target | Top-4 saves 26.6% at width 5, barely above the ideal N=4 break-even threshold | Saves only 14.0%, below that threshold | New same-input top-4/top-8 CUDA graph probe |
| 2, fix first before evaluating Gemma MTP | MTP first-step graph reads outer Target query/sequence lengths in later inner rounds | Shared autoregressive graph assumption needs a separate audit; impact not measured here | Confirmed: refreshing two buffers makes 357/357 proposals match fresh-metadata execution | New paired proposal and narrow buffer ablations |
| 3, major Qwen cost | Private GDN state operations and repeated metadata preparation/submission | 29.53/35.67 ms residual per MTP/DSpark loop | Only 6.38/6.58 ms residual | Existing complete-cycle spans plus new Qwen nsys trace |
| 4, secondary and model dependent | Final Target discards a suffix of the accumulated P candidates | Discards 29.1%/15.0% of MTP/DSpark candidates | Only 3.5%/3.2% in original runs | Recomputed integer counters and round-survival accounting |

Priority 1 ranks common performance impact. Priority 2 is the first engineering
action because acceptance conclusions require correct graph inputs. The prior
strict P sequential-equivalence failures also remain unresolved; this work does
not certify the decoder as lossless.

## Measurement contract

- Existing inputs: four original prompts per model, 512 generated tokens each,
  greedy sampling, B=1, TP=1, D=4, N=4, shared Target/P parameters, P top-4 versus
  Target top-8. No new parameter sweep.
- Historical complete-cycle evidence: `../cycle_profile_20260909/` and
  `../gemma4_d16_20260910/`. Their archived raw results are the source of the main
  latency and acceptance tables. `input_hashes.json` fingerprints live inputs.
- A complete cycle is proposal i through Target sampling i+1. Initial
  prefill-attached proposals, unverified terminal proposals, and post-limit async
  work are excluded using the original summarizer's accounting.
- Acceptance counts include full verified cycles, including the last cycle's
  possible generation-limit overshoot. These are cycle yields, not a claim that
  every sampled token was returned to the user.
- Final emitted length is accepted draft tokens plus one Target correction/bonus.
  P-to-T acceptance excludes that bonus. Inner acceptance excludes P bonuses.
- New model-cost probes: first original prompt, 512-token warmup, one 512-token
  diagnostic request, width 5 at the same prefix/input for h=4 and h=8; twelve
  alternating graph replays per h after warmup/capture. State restoration is
  outside the timed range. Both paths include model, logits and argmax. This is
  the full-top-k backbone in P's execution path, not the entire native Target
  runner, and not a broad prefix-distribution benchmark.
- New Gemma metadata probes: four original prompts, 512 tokens each, repeated
  same-input proposal calls. Their request latency is intentionally invalid as a
  performance metric. Warmup used the original implementation.
- New nsys capture: Qwen MTP, ten post-warmup Target steps, forty P graph replays.
  CUPTI node tracing adds overhead; use this trace for attribution and operation
  counts, not replacement throughput numbers. GPU attribution uses the launch
  correlation ID and innermost containing NVTX range on the launching thread.
  All 69,838 kernels have a matching CUDA runtime correlation.
- Two Gemma nsys attempts finished the diagnostic request but generated no
  report. Their logs are retained. No Gemma kernel-level nsys claim is made.
  Gemma conclusions use CUDA events, raw acceptance, paired h probes and code.

## 1. P is not a small draft model

`HierarchicalSpeculator._verify_eager` calls the same full model for every inner
round, with `routing_top_k=4`. It retains all layers, attention/GDN, dense
projections, normalization, logits computation, and any shared-expert work.
Reducing routed experts from eight to four does not halve the entire forward.
See `vllm/v1/worker/gpu/spec_decode/hierarchical/speculator.py:324` and
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:610`.

| Same-input width-5 graph | P top-4 ms | Full top-8 ms | P / full | Saving | Ideal 0.75 T - P, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 8.325 | 11.345 | 73.38% | 26.62% | +0.184 |
| Gemma | 9.566 | 11.119 | 86.03% | 13.97% | -1.227 |

`paired_verify.json` includes positions, repetition count and min/max times.
The Qwen prefix positions were 123..127; Gemma 131..135. These are bounded probes,
not claims about all contexts. Original complete-cycle P times are slightly
different because those samples cover many prefixes and intervening work.

The user's formula is a useful first filter, but assumes equal output yield and
equal Target verification cost. The actual comparison is

```text
hierarchy cost/token = [sum_r(S_r + P_r) + T(L+1) + H] / (A_T + 1)
native cost/token    = [S(D) + T(D+1) + H_native] / (A_native + 1)
L = sum_r(a_r + 1), not necessarily N*D
```

The outer Target processes L+1 positions, not the inner width 5. Qwen's tiny
0.184 ms ideal per-round margin disappears even before its substantial state
management and yield losses. Gemma has no positive margin in this same-width
idealization. Increasing N also changes prefix survival; it is not free
amortization.

The previous uninstrumented first-pass throughput was Qwen hierarchy MTP/DSpark
102.09/104.42 tok/s, versus native MTP D4 sync 183.22 and MoE-Skip D16 96.59.
Gemma hierarchy MTP/DSpark was 140.10/182.98, versus native MTP D16 401.58 and
MoE-Skip D16 126.72. Thus the hierarchy did accelerate autoregressive MoE-Skip,
but did not achieve the intended MTP-level cost. Native Gemma MTP used async
scheduling; hierarchy requires sync. Qwen's sync MTP control already shows that
the scheduling difference cannot explain the entire problem. Historical
MoE-Skip controls also used their original multimodal-input setting; see the
original reports for that configuration limitation.

## 2. Confirmed Gemma MTP graph metadata bug

The small autoregressive drafter captures its first-step graph using
`target_input_buffers` and `target_attn_groups`
(`autoregressive/speculator.py:214`). FULL graph execution then calls
`run_fullgraph` without consuming the new `attn_metadata` argument
(`autoregressive/speculator.py:348`). Hierarchical inner `_batch` prepares a
different buffer set and independent attention builders; later rounds pass this
new metadata to `small.propose`, but the captured graph still reads the outer
Target lengths. Gemma's Triton attention metadata directly references
`query_start_loc` and `seq_lens`.

For example, the first prompt's second inner round needed query offsets [0,5]
and sequence length 136. The graph read outer Target offsets [0,131] and length
131. This is a metadata/graph-lifetime violation, not evidence that MTP must be
incompatible with MoE-Skip hidden states.

| Gemma MTP | Inner round 1 | Round 2 | Round 3 | Round 4 | Candidates L/cycle | Final emitted/cycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original four-prompt profile | 86.89% | 28.65% | 21.35% | 25.54% | 10.50 | 11.14 |
| Fresh-metadata eager diagnostic | 83.88% | 86.98% | 87.40% | 88.02% | 17.85 | 17.06 |
| Refresh two graph buffers diagnostic | 88.51% | 89.41% | 86.71% | 91.67% | 18.25 | 18.40 |

These aggregate rows follow different generated trajectories after the first
changed proposal; the following within-call comparison establishes causality:

- Eager diagnostic: 387 later-inner-round same-input calls; original graph
  matched fresh-metadata drafting in only 73 cases.
- Narrow buffer diagnostic: 357 later-inner-round calls; original graph matched
  fresh execution in 88 cases. After temporarily refreshing **only** graph-owned
  `seq_lens` and `query_start_loc`, the graph matched fresh execution in **357/357**.
- Gemma's assistant is Q-only and does not write shared KV, allowing these
  repeated calls on the same input. The two outer buffers are restored after
  graph replay. Each diagnostic then chooses its documented proposal for P.
- The narrow graph diagnostic's outer acceptance was 95.31%, so restoring inner
  yield did not introduce a catastrophic final rejection penalty in this sample.

`gemma_metadata.json`, `gemma_buffers.json` and archived per-call token arrays
contain the evidence. Any latency fields in these diagnostic JSON files are
instrumented/repeated-work costs, not corrected end-to-end performance.

Production repair should give inner MTP graphs their own live attention metadata
and buffers, or explicitly refresh every relevant captured buffer with validated
lifetime/restore semantics. The two-buffer experiment is a causal probe, not a
general backend-safe fix. Audit Qwen separately; the shared graph assumption is
present there, but this work does not measure the corresponding Qwen effect.

## 3. Qwen has substantial avoidable host/state overhead

The following disjoint decomposition uses the existing full-cycle CUDA event
measurements. Residual includes state operations, metadata generation/refresh,
host-induced stream gaps and remaining glue; it is not all GPU compute or all
proven removable overhead.

| ms per outer loop | Qwen MTP | Qwen DSpark | Gemma MTP | Gemma DSpark |
| --- | ---: | ---: | ---: | ---: |
| Small drafter total | 14.08 | 7.52 | 13.12 | 6.97 |
| Four P graph replays | 33.63 | 33.82 | 40.65 | 40.64 |
| Outer Target execute + sample | 18.00 | 19.11 | 14.57 | 15.61 |
| Remaining stream time | 29.53 | 35.67 | 6.38 | 6.58 |
| Complete cycle | 95.24 | 96.12 | 74.71 | 69.81 |

Qwen MTP state begin alone is 5.20 ms, state advance 11.08 ms, metadata
construction 7.25 ms, and P wrapper beyond graph replay 4.17 ms. Together these
are 27.70 ms of its 29.53 ms residual. Gemma has no GDN layers and therefore no
comparable recurrent-state copying burden.

New Qwen nsys findings, normalized over ten captured outer steps:

- State begin: 334 kernels + 60 CUDA memcpy activities per outer step.
- State advance: 480 kernels + 360 memcpy activities per outer step, across four
  advances; about 548.5 MB of memcpy activity per outer step in this trace.
- Metadata construction: 104 kernels + 112 memcpy activities per outer step.
- P wrapper metadata refresh: 120 memcpy activities per outer step, totaling
  only 120 bytes. Recursive per-layer refresh can revisit aliased tiny tensors.
- Each width-5 P graph contains 1,033 kernels. The named `fused_moe_kernel`
  accounts for 29.55% of P's summed kernel duration in this trace. The remainder
  includes dense GEMMs, GDN/attention, routing, reductions and other kernels;
  this is not a complete semantic MoE-vs-non-MoE partition.

This points to many small submissions and repeated state/metadata handling,
not merely a large unavoidable bulk copy. `state.py:43` and `state.py:65` loop
over layers with indexing, arange, cloning and copy operations; `_batch` creates
small tensors/NumPy arrays every round; `refresh_graph_metadata` recursively
copies per-layer fields without alias deduplication.

Prioritize persistent private buffers, fused acceptance-aware state gather/copy,
cached metadata, alias deduplication and graph coverage for this glue. Preserve
canonical Target state and acceptance-dependent rollback. Do not delete state
management merely because a zero-copy counterfactual looks attractive.

`accepted_prefix().item()` waits about 8.4 ms on Qwen and 10.2 ms on Gemma on the
CPU, but only about 0.054 ms on the CUDA stream. Much of the CPU time waits for P
work already counted above. Adding both would double-count. Moving counts and
offsets onto the device could reduce serialization and submission gaps, but
cannot eliminate the data dependency between consecutive inner rounds.

## 4. Outer rejection wastes work on Qwen, much less on Gemma

| Original profile | Inner S accepted / proposed | Final T accepted / L | L | Final emitted | Fully unretained inner rounds | Rounds strictly after first rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen MTP | 55.14% | 70.93% | 12.82 | 10.09 | 29.95% | 25.00% |
| Qwen DSpark | 44.17% | 84.99% | 11.07 | 10.41 | 17.31% | 13.46% |
| Gemma MTP, affected by metadata bug | 40.61% | 96.55% | 10.50 | 11.14 | 7.16% | 5.14% |
| Gemma DSpark | 63.67% | 96.81% | 14.19 | 14.73 | 6.65% | 4.86% |

The percentages in the last two columns count rounds, not GPU milliseconds.
A round containing the first rejected token may still have a retained prefix;
the rounds after it are entirely discarded. A zero-retained round may also be
that first rejection round. Counters are in `diagnosis.json`, and every analyzed
cycle with its retained-per-round vector is in the four `*_cycles.json` files.

At fixed measured cost and fixed candidate trajectories, removing only outer
rejection changes ms/emitted-token from 9.44 to 6.89 for Qwen MTP, 9.24 to 7.97
for Qwen DSpark, 6.71 to 6.50 for Gemma MTP and 4.74 to 4.60 for Gemma DSpark.
These are accounting counterfactuals, not measured speedups. They show why final
rejection is material on Qwen and cannot explain Gemma's original slowdown.

## Bounds and next actions

Even deleting all residual time at the observed yields leaves Qwen MTP at
6.51 ms/token and Gemma MTP/DSpark at 6.14/4.29 ms/token. Corresponding native MTP
complete-cycle controls were Qwen D4 sync 4.37 ms/token and Gemma D16 async
1.84 ms/token. These estimates retain the original, bug-affected Gemma yield and
are not forecasts for the repaired implementation.

A stronger Gemma cost bound does not rely on the old low yield: with D4/N4 the
maximum is 20 P candidates plus one final Target token. At the observed component
costs, even zero residual overhead and all 21 emitted tokens gives approximately
3.25 ms/token for MTP hierarchy and 3.01 for DSpark hierarchy, versus 1.84 native
MTP D16. This holds the component costs fixed; it is an optimistic accounting
bound, not an implementation-independent impossibility theorem.

Recommended order:

1. Repair inner MTP graph metadata ownership, rerun per-step graph/fresh-input
   comparisons and P sequential/canonical-state correctness gates. Then rerun
   four-prompt uninstrumented performance. Do not retain the old Gemma MTP result
   as a clean architecture comparison.
2. On Qwen, fuse/preallocate private GDN state and metadata handling. Its measured
   residual is large enough to matter. Measure the disjoint cycle again after
   each change rather than summing CPU waits with GPU times.
3. Require a substantially faster P at the actual inner width, or change the
   amount of progress amortized per P call. Gemma top-4 currently fails the ideal
   N=4 cost threshold; Qwen barely clears it. More expert skipping or a smaller
   P must be evaluated against both S-to-P and P-to-T acceptance.
4. Tune/adapt N only after the above. Qwen suffix waste gives a reason to stop
   some loops early; original Gemma final survival gives little such reason.
   Choosing N from nominal D*N alone ignores actual candidate yield.

No runtime decoder fix or repaired-throughput claim is included in this report.
The benchmark-only narrow buffer ablation confirms one root cause and identifies
the production integration that should be changed next.
