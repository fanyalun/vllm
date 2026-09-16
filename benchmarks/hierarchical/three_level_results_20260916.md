# Three-Level p50 hierarchical preverification

## Result and measurement contract

Both frozen configurations improved final generation throughput in all five
paired trials on the held-out requests. This is approximate preverification;
the results do not establish lossless acceleration or improved model accuracy.

Qwen3.6-35B-A3B, one A100 80GB PCIe (GPU 0), TP1, B1, inner D4,
top-h4, at most four inner rounds, greedy sampling, seed 42, FP32 recurrent
state, existing BF16 model/convolution state. Each trial returns 256 tokens for
each of the original report's first 16 GSM8K requests. Timing includes the
complete synchronous `LLM.generate` call. Throughput is total returned tokens
divided by total elapsed time, including prefill; it is not an average of
request-level throughput ratios. Each table cell contains 20,480 timed tokens.

| Method | Original replay-tail, low_error | Replay-tail, matched stopping | Frozen Three-Level | ms/token | Gain vs original | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MTP | 132.172 tokens/s | 134.123 tokens/s, balanced | 164.586 tokens/s | 6.0759 | +24.52% | +19.52% to +29.87% |
| DSpark | 139.264 tokens/s | 144.372 tokens/s, aggressive | 163.261 tokens/s | 6.1252 | +17.23% | +11.49% to +23.35% |

Compared with matched stopping, gains are +22.71% for MTP (CI +18.34% to
+27.88%) and +13.08% for DSpark (CI +9.36% to +16.69%). The five gains against
original replay-tail are respectively:

- MTP: +24.27%, +24.27%, +23.95%, +25.80%, +24.32%.
- DSpark: +18.22%, +16.22%, +16.22%, +16.88%, +18.61%.

Case order alternates across trials. Both methods ran sequentially on GPU 0.
All cases and requests were warmed before timing. New JIT compilation is an
error during timing, and graph counts must remain unchanged. Stage, action,
and profiler audits run separately and must reproduce the corresponding output.
The confidence interval uses 10,000 paired prompt-cluster bootstrap resamples
of each prompt's five-trial mean time, seed 42. It describes these 16 prompts;
it is not evidence of generalization to arbitrary tasks or serving loads.

## Recommended opt-in configuration

The new defaults are `preverify_gdn_update_policy="exact"` and
`preverify_gdn_tail_policy="carry"`; existing configurations retain their path.
The following is the shared `speculative_config`:

```python
{
    "method": "hierarchical",
    "inner_method": "mtp",
    "inner_num_speculative_tokens": 4,
    "inner_num_rounds": 4,
    "moe_skip_top_h": 4,
    "draft_sample_method": "greedy",
    "hierarchical_stop_policy": "balanced",
    "preverify_gdn_mode": "replay_tail",
    "preverify_gdn_update_policy": "three_level_p50",
    "preverify_gdn_tail_policy": "repair_on_reject",
}
```

For DSpark, use `inner_method="dspark"`, `hierarchical_stop_policy="aggressive"`,
`preverify_gdn_tail_policy="carry"`, and set `model` to the DSpark speculator
checkpoint. The tested local checkpoint is
`/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark`; the target is
`/data1/fanya/Qwen/Qwen3.6-35B-A3B`.

Use `mamba_ssm_cache_dtype="float32"`, `async_scheduling=False`,
`enable_prefix_caching=False`, `max_num_seqs=1`, and TP1 as in the runner.
The feature requires ungrouped replay-tail hierarchical preverification.
It does not change the Target or independent Draft algorithm or use a second
Target parameter set.

## State and numerical semantics

Thresholds are fixed at alpha 0.98 and beta 0.36328125. Classification rounds
effective beta to BF16, matching the original effective-gate probe. Full updates
retain FP32 sigmoid beta and FP32 arithmetic. Beta equal to the threshold is
Full; below it, alpha greater than 0.98 is Skip, otherwise Decay-only.
The anchor can also be approximate. Runtime device threshold tensors permit
forced-action diagnostics without compiling a new kernel.

Full loads K/V, normalizes K, and performs the recurrence. Skip and Decay-only
use real control flow to bypass those operations; every position still computes
its Q readout. State stays in registers within a window, with one final tail
write. Two independent private FP32 buffers alternate without a full state copy
between windows. Actual compiled PTX contains branches around the K/V loads
and the Full update; production variants have no register spills.

`carry` retains the entire private window tail after rejection. With
`repair_on_reject`, a rejection followed by another inner window replays only
the consumed prefix, of length accepted drafts + 1 including the anchor.
The newly predicted correction is not consumed yet. Repair uses retained
post-convolution K/V and raw gates with the same approximation, without
projection or Q readout. It retains the window-start state and small inputs,
not full per-token states. No repair or advance is done after the final inner
round or a short terminal window. Private state is reset from the Target at
each outer proposal and is never committed into the Target cache.

Engineering work batches canonical-state initialization and convolution-cache
advances across layers, reuses metadata and count tensors, and avoids unnecessary
host reads. Accept-length reads needed for host control flow remain. Graphs
retain fixed addresses and distinguish buffer directions and window widths.

## Tuning and ablations

Four original replay-tail tuning prompts were checked against the 16 final
prompt hashes: zero overlap. The held-out prompts were not used for selection.
The complete eight-combination screen per method, before the final batched
initialization improvement, produced the following pooled tokens/s:

| Tail | Stop | MTP | DSpark |
| --- | --- | ---: | ---: |
| carry | none | 129.999 | 105.833 |
| carry | low_error | 129.645 | 107.228 |
| carry | balanced | 137.625 | 111.806 |
| carry | aggressive | 132.601 | 119.644 |
| repair_on_reject | none | 137.569 | 113.017 |
| repair_on_reject | low_error | 138.711 | 118.212 |
| repair_on_reject | balanced | 141.847 | 120.047 |
| repair_on_reject | aggressive | 133.919 | 120.292 |

After batched initialization, two-repeat tuning rechecked finalists and
unoptimized controls on the same four prompts:

| Case | MTP, balanced | DSpark, aggressive |
| --- | ---: | ---: |
| Original replay-tail, low_error | 111.22 | 104.90 |
| Replay-tail, matched stop | 115.19 | 111.78 |
| Three-Level repair, engineering optimizations disabled | 120.28 | 111.86 |
| Optimized Three-Level carry | 147.68 | 130.61 |
| Optimized Three-Level repair | 151.84 | 130.55 |

DSpark repair/balanced was also rechecked (128.98 tokens/s). MTP chose
repair/balanced; DSpark chose carry/aggressive because repair did not lower
final cost. The DSpark carry/repair tuning difference is small; it is not a
statistically established universal advantage of carry. Selection and source
hashes were frozen before the final matrix. These separate staged experiments
must not be combined into a claimed additive causal decomposition.

All 12 value-tile/warp combinations from {8,16,32} x {1,2,4,8} were screened.
The fastest local tile8/warps2 variant was rejected: complete forward testing
found 7 of 524,288 tail elements outside the unchanged 1e-3 tolerance, with
maximum absolute difference 0.002375. The retained choice is tile8/warps4 for
T1-2 and tile16/warps8 for T3-5. Complete forward logits/predictions matched
between engineering variants; tails met the existing tolerance.

With 30 real layers, 50 warmups and 200 cold-L2 graph timings, production-layout
raw-gate kernel totals (exact / selected conditional, milliseconds) were:
T1 0.17510/0.18637, T2 0.25805/0.25907, T3 0.33280/0.33075,
T4 0.40243/0.39834, T5 0.47104/0.46592. Actual Q/K/V row stride is 12,288
and raw-gate stride is 64. Compact CPU-captured tensors are a separate result.
The old effective-gate T5 probe was slower: 0.39219/0.43418 ms. This is not
evidence of a large recurrent-kernel acceleration.

Paired complete T5 preverify forward timings were 9.3102 ms exact, 9.3727 ms
unoptimized Three-Level, 9.3696 ms optimized carry, and 9.6348 ms repair.
Including state advance (and consumed-prefix-3 repair where applicable) gives
9.3757, 9.4372, 9.3870, and 9.8918 ms. Repair's cost is included in both these
measurements and final generation. Forward improvement alone was not the
selection criterion.

## Full-cycle audit and remaining cost

Separate audited milliseconds per returned token are shown below. Proposal
contains its sub-stages, so columns must not be summed. Target/sampling/scheduling
is a difference of nested intervals with the same start event. These audited
times are explanatory and do not replace event-free generation timing.

| Method/case | State begin | Draft | Preverify | Advance/repair | Proposal | Target/sampling/scheduling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP original | 0.4492 | 1.0709 | 3.2122 | 0.1864 | 5.5858 | 1.5594 |
| MTP selected | 0.0418 | 1.0259 | 2.8786 | 0.0284 | 4.1044 | 1.5604 |
| DSpark original | 0.4027 | 0.4989 | 3.0714 | 0.1769 | 5.1286 | 1.5932 |
| DSpark selected | 0.0447 | 0.4704 | 2.8356 | 0.0155 | 4.0586 | 1.6808 |

MTP original/selected have 336/339 verified outer cycles and 12.143/12.035
returned tokens per cycle; DSpark has 312/348 and 13.077/11.724. The original
prefill contributes another one token per request. Inner accepted/proposed
integer counts are 3488/4980 versus 3466/4800 for MTP and 3669/4716 versus
3440/4428 for DSpark. Proposal counters include the final unconsumed outer
proposal generated by the existing runner, whereas cycle records count
completed verification. Its cost remains in end-to-end timing.

Selected Full/Decay/Skip counts are 2,991,053 / 626,479 / 2,142,468 for MTP and
2,732,280 / 593,496 / 1,987,824 for DSpark. Counts cover forward decisions,
excluding repeated repair actions. Every count total was checked against
960 value-head/layer decisions per consumed forward position. Cycle records
preserve rejection positions, round/window counts, stopping reasons, and
request-limit clipping.

Separate 32-token profiler traces identify MoE and BF16 GEMM as the largest
remaining kernel groups. MTP fused-MoE interval union is 122.872 ms and the main
BF16 GEMM union 91.830 ms, versus 9.797 ms for replay-tail recurrence.
DSpark's corresponding first two groups are 85.093 and 35.756 ms. Group unions
can overlap across streams and are not additive. Host synchronization spans
include waiting for GPU work. Graph-stream launch gaps, CPU synchronization
counts, kernel distributions and PTX metadata are retained, not treated as
event-free throughput measurements. Projection/MoE fusion or shared-input
approximation was not introduced without evidence of a safe local win.

Private tensor storage is 66,846,720 bytes for exact replay-tail,
131,623,680 for MTP repair and 129,761,280 for DSpark carry, plus 7,717 bytes
of selected-path auxiliary tensors. These exclude allocator/graph overhead.
The warmed final cases use respectively 1 exact graph, 10 MTP selected graphs
(2 forward plus 8 repair), and 2 DSpark selected graphs. Final requests use T5;
T1-5 and alternate directions are covered by kernel/state tests. The actual T5
conditional variant uses 48 registers/thread, 32 bytes shared memory and no
spills; compiled PTX and metadata are retained.

## Correctness and reproducibility

- Targeted tests: 195 passed, 53 deselected. Coverage includes T1-5, all accepted
  prefix lengths, equality/both threshold sides, forced Full/Skip/Decay/mixed,
  alternating buffers, graph replay, repair consumption boundaries, exception
  restoration, Target isolation, both convolution layouts and aligned caches.
- A separate 600-case audit on 30 real layers found maximum output error
  0.0000076294 and tail error 0.0000054836 against an independent reference.
  All 150 forced-Full cases matched the archived pre-change kernel bitwise,
  including outputs and tails. Every forced-Skip tail matched its input bitwise
  while using independent storage. Per-case errors and worst heads are saved.
- Every final mode is internally reproducible over five repeats and agrees
  with its stage/action audit. Profiling also preserves output. AR itself was
  repeated twice. Full 256-token AR agreement is 4/16, 3/16, 7/16 for MTP
  original/matched-stop/selected, and 5/16, 3/16, 6/16 for DSpark. Selected
  equal token positions are 2781/4096 and 2718/4096. Position agreement after
  divergence is not a correctness or task-accuracy score.
- Normal `LLM` constructor configuration was checked separately for each
  selected method, with two 256-token repetitions of the first final prompt.
  Both repetitions and the corresponding benchmark tokens matched exactly.
  No benchmark worker case switching was used for this check.

## Artifacts and reproduction

Local artifact root: `benchmark_results/three_level_p50_20260916/`.
Large artifacts stay local and previous experiments are preserved.

- `baseline/`: fresh original none/replay-tail measurements.
- `tuning_optimized_safe/`: complete eight-combination screen for both methods.
- `tuning_final_validated/`: finalists, engineering controls and tail ablation.
- `prompt_split.json`, `tuning_prompts.jsonl`, `final_prompts.jsonl`: identities.
- `kernel_effective_paired/`, `kernel_raw/`, `kernel_production_layout/`:
  distinct gate/layout measurements, errors, PTX and compilation metadata.
- `fixed_forward_paired_validated.json`, `numerical_audit.json`:
  full-forward and independent numerical evidence.
- `final/freeze.json`: frozen configuration and source SHA256 values;
  `final/{mtp,dspark}/source_snapshot/`: exact measured sources.
- `final/{mtp,dspark}/results.json`: all timings, full output tokens/text, traces;
  `private_state.json`, `profile.json`, `profile_parity.json`, `kernels/`:
  memory, profiler controls and actual compiled paths.
- `report/`: pooled CSV/JSON/Markdown, raw request times, cycle CSV,
  trace summary and `coverage_complete.json` (480 timed request rows).
- `ar/`, `direct_api/`: AR controls and normal constructor-configuration checks.

The original final prompt token IDs came from the GSM8K samples in
`/home/fanya/GDN-Spec/benchmark/gdn_motivation/outputs/gate_skip_motivation_16x256_20260916/`,
alternating `shard_0`/`shard_1`, sample IDs 00000 through 00015. Decoding and
re-encoding preserved those token IDs. Old effective inputs are in the sibling
`gate_skip_next_16x256_20260916/probe_inputs.pt` file.

Run from the repository root, with the existing environment and local models:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  -k 'hierarchical or replay_tail or three_level' -x -q

CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/hierarchical/run_replay_tail.py \
  --inner-method mtp --three-level --seed 42 --samples 16 --max-tokens 256 \
  --repeats 5 --cases exact:carry:low_error exact:carry:balanced \
  three_level:repair_on_reject:balanced \
  --dataset benchmark_results/three_level_p50_20260916/final_prompts.jsonl \
  --action-audit --profile-output /tmp/three_level_mtp_reproduction/profile.json \
  --output /tmp/three_level_mtp_reproduction

# DSpark: change method to dspark, balanced to aggressive, and repair to carry.
.venv/bin/python benchmarks/hierarchical/summarize_three_level.py \
  --root benchmark_results/three_level_p50_20260916

CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/kernels/benchmark_gdn_three_level.py \
  --inputs benchmark_results/three_level_p50_20260916/raw_inputs.pt \
  --raw --production-layout --output /tmp/three_level_kernel_reproduction
```

For a full new run, prepare a fresh artifact root with the prompt files and
completed tuning controls, then use `run_three_level_final.py`; it refuses an
existing final directory. `check_three_level_api.py` checks the selected policy
through ordinary `LLM` construction without worker case switching.

Implementation, experiments and this report used OpenAI Codex assistance.
