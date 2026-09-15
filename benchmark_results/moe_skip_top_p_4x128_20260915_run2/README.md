# Top-p expert-budget pilot

Measured Qwen3.6 (GPU 0, D=8) and Gemma4 (GPU 1, D=4), each on four prompts, one per category, with 128 output tokens each. B=1, TP=1, greedy, seed=0, ignore_eos, CUDA graphs, no prefix cache. All configurations use the same model-specific prompts and settings. The two models run independently on two A100 80GB PCIe GPUs.

Top-p is applied inside native top-8, separately for every draft token and MoE layer. It keeps the shortest probability-mass prefix reaching p, renormalizes retained weights, and preserves Gemma's per-expert scale. Target routing is unchanged. Target and Draft share the same model instance and weights. Invalid expert IDs skip expert GEMMs; the routing tensor retains eight slots. Extra routing and histogram overhead are included in measured wall time. This is a benchmark worker extension, not a production configuration.

Time is the sum of llm.generate wall times, including prefill and API overhead, excluding model initialization and four full warmup requests. Acceptance length excludes the recovery/bonus token. ms/accepted divides the entire request time by actually emitted accepted draft tokens, clipping the final speculative step. ms/output uses all 512 output tokens. Mean h includes all generated draft token-layer invocations, including the final unused proposal. AR speedup uses the mean of start/end async controls.

Only four samples per configuration, no repeated speculative timing trials or confidence intervals. Do not extrapolate to the previous 16x512 matrix or interpret small timing differences as established gains.

## qwen36

AR (Async): 7.557 ms/output token; end/start time change: +1.47%.

| Method | Mean h | Accepted/step | ms/output | ms/accepted | AR speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| h2 | 2.000 | 5.057 | 10.661 | 12.843 | 0.709x |
| h4 | 4.000 | 6.386 | 9.162 | 10.637 | 0.825x |
| h6 | 6.000 | 7.574 | 8.427 | 9.588 | 0.897x |
| h8 | 8.000 | 7.850 | 8.824 | 9.996 | 0.856x |
| p07 | 4.827 | 6.735 | 9.324 | 10.752 | 0.810x |
| p08 | 5.861 | 7.172 | 9.037 | 10.351 | 0.836x |
| p09 | 7.023 | 7.672 | 8.930 | 10.161 | 0.846x |
| p1_control | 8.000 | 7.850 | 8.914 | 10.098 | 0.848x |

p=1 versus native h=8 exact output matches: 4/4 requests.

## gemma4

AR (Async): 7.953 ms/output token; end/start time change: -0.10%.

| Method | Mean h | Accepted/step | ms/output | ms/accepted | AR speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| h2 | 2.000 | 3.345 | 8.768 | 11.422 | 0.907x |
| h4 | 4.000 | 3.353 | 9.129 | 11.923 | 0.871x |
| h6 | 6.000 | 3.741 | 8.772 | 11.117 | 0.907x |
| h8 | 8.000 | 3.886 | 8.967 | 11.336 | 0.887x |
| p07 | 4.873 | 3.491 | 9.089 | 11.722 | 0.875x |
| p08 | 5.870 | 3.750 | 8.765 | 11.135 | 0.907x |
| p09 | 7.010 | 3.575 | 9.383 | 12.071 | 0.848x |
| p1_control | 8.000 | 3.952 | 8.933 | 11.237 | 0.890x |

p=1 versus native h=8 exact output matches: 0/4 requests.

## Validation and limitations

See audit.json for model/dataset/source identity, acceptance counter recomputation, routing invocation counts, warmup/JIT checks and per-cell output parity. GPU primitive validation is recorded in gpu_reference_check.log. It tests unsorted expert slots, tied logits, budget histograms, per-expert scaling, p=1 identity, and masked expert GEMMs against a zero-weight reference across 32 cases. Two unsupported aligned-assignment shapes must raise ValueError.

The eight-slot invalid-expert representation is restricted to naive MoE assignment (4 *tokens* 8 <= number of experts), which includes the measured B=1 draft path. An unsorted-slot test exposed incorrect results in the larger aligned-assignment path; it is now rejected. See unsupported_alignment_probe.log and post_measurement_guard.patch. The main timing matrix uses the frozen worker snapshots before this Python shape guard; the GPU kernel is unchanged. guard_validation contains additional p=0.8 model runs with the final guarded worker and guard_validation_audit.json records their checks. These runs do not enter the main timing table.

Gemma output reproducibility is unresolved: prior-attempt unmodified AR and h8 repeats each matched 0/4 requests against this run. These controls did not enable top-p. See gemma_repeat_control.json and prior_attempt_controls for copied results, exact source snapshots and provenance. This limits Gemma quality and fine-grained performance interpretation; no correctness-preserving serving claim is made.

The first attempt stopped on a benchmark extension import error. The corrected driver ran all 20 cells afresh under this directory. Only the explicitly archived native Gemma controls are reused for the reproducibility check; they do not enter the main timing table.

## Reproduction

```bash
.venv/bin/python benchmarks/moe_skip/check_top_p.py
.venv/bin/python benchmarks/moe_skip/run_top_p_probe.py --model qwen36 --gpu 0 --run-dir <fresh_directory>
.venv/bin/python benchmarks/moe_skip/run_top_p_probe.py --model gemma4 --gpu 1 --run-dir <fresh_directory>
.venv/bin/python benchmarks/moe_skip/analyze_top_p_probe.py <fresh_directory>
```

Each figure directory contains matching PNG, PDF and Markdown files. summary.csv holds aggregate metrics; layer_budgets.csv holds per-layer h=1..8 counts; output_parity.csv holds exact request-level comparisons. AI assistance was used for the implementation, experiments and analysis.
