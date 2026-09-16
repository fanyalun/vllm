# Default MoE-Skip threshold routing

MoE-Skip now defaults to retaining native top-k experts with normalized gate
probability at least 0.125. The same default applies to hierarchical
pre-verification. Target routing is unchanged, and drafts share the Target
model instance and weights.

```json
{"method": "moe_skip", "num_speculative_tokens": 16}
```

The explicit threshold option is `moe_skip_min_weight`, with CLI alias
`--moe-skip-min-weight`. Explicit `moe_skip_top_h: 3` or `4` selects fixed-h
routing instead. A threshold combined with top-h requires h to equal native k.
New `run_cell.py` and `run_performance.py` runs use the threshold default too;
resuming an old performance contract preserves its recorded fixed-h policy.
The default weight mode remains `preserve`; `renormalize` divides retained
weights by their retained normalized gate mass. Thresholds participate in the
compilation hash and do not carry into the hierarchical inner drafter.

Thresholding uses gate probabilities normalized within native top-k, before
Gemma per-expert scaling. Equality retains the expert. There is no top-1
fallback: native k=8 guarantees a maximum probability of at least 0.125.
Higher explicit thresholds may remove every routed expert; shared experts are
unaffected.

## Execution

The production router uses a single Triton threshold kernel, without the
benchmark monkeypatch, histogram atomics, sorting, or GPU-to-CPU synchronization.
Native weights and slot order are preserved. The Triton expert backend packs
active decode assignments and coalesces skipped slots into zero-output blocks.
Both expert GEMMs return before loading expert matrices for these zero blocks.
When `4 * tokens * native_k > num_experts`, batches use expert grouping with an
explicit skipped-slot bucket, so the reduction never reads unwritten scratch
slots. This fixes the limitation of the
experimental mask-only path, which was restricted to naive assignments.

Buffers retain native-k capacity because a uniform distribution can retain all
eight experts at p=0.125. Packed work counts are GPU tensors, so CUDA Graph replay
can change the number of retained experts without host synchronization. The
threshold path currently requires the modular `TritonExperts` backend and the
existing MoE-Skip TP1/PP1/DP1 configuration; other expert backends fail explicitly.

## Validation and reproduction

The retained implementation passed 94 targeted tests. Four prompts per model
match the legacy threshold prototype in both eager and CUDA Graph execution:
16/16 output comparisons and 16/16 full acceptance-counter comparisons.

| Model | Execution | Legacy tokens/s | Production tokens/s | Ratio |
| --- | --- | --- | --- | --- |
| Qwen3.6 | eager | 8.143 | 8.385 | 1.0298 |
| Qwen3.6 | CUDA Graph | 107.016 | 107.343 | 1.0030 |
| Gemma4 | eager | 10.470 | 10.509 | 1.0036 |
| Gemma4 | CUDA Graph | 86.052 | 86.317 | 1.0031 |

These are single small-sample trials; the roughly 0.3% graph differences are
within timing noise. The isolated one-token expert path improved from
25.088 to 23.962 microseconds for Qwen and 29.491 to 27.750 for Gemma.
No significant end-to-end graph speedup is established.

Eager and graph outputs are not identical (3/4 Qwen and 0/4 Gemma pairs match).
Both execution modes separately match the legacy implementation; this change
does not resolve that existing execution-mode discrepancy.

```bash
.venv/bin/python -m pytest \
  tests/config/test_moe_skip_config.py \
  tests/config/test_hierarchical_config.py \
  tests/model_executor/layers/fused_moe/test_routing_top_k.py \
  tests/v1/worker/gpu/spec_decode/test_moe_skip_trace.py -q
```

The GPU tests cover 1, 4, 5, 16, 32, 33, and 64 tokens, scaled weights,
uniform gates, threshold equality, complete pruning, both weight modes, and
CUDA Graph replay after changing the routing IDs to all skipped slots.

`run_threshold_performance.py` compares `--path default`, `legacy`, and `h4`.
It uses the same four prompts, D=16, 128 output tokens, greedy decoding, and
disabled prefix caching. Each prompt is warmed once before timing all four
requests; initialization and warmup are excluded. Add `--cuda-graphs` to test
the graph path. Detailed acceptance counters are enabled in all variants.

`benchmarks/kernels/benchmark_moe_threshold.py` checks output correctness before
timing the two expert GEMMs, activation, dispatch, and reduction together.
Routing selection itself is outside that microbenchmark. It requests CUPTI
timing with cold L2 and CUDA Graphs; FlashInfer falls back to CUDA events if
the CUPTI Python package is unavailable. Read the log for the actual timing
backend rather than assuming CUPTI was used.

Local evidence is in `benchmark_results/moe_skip_threshold_default_20260916/`.
An attempted Gemma router fusion passed isolated kernel checks but failed the
model CUDA Graph comparison, so it is excluded from the production path.
No independent AR equivalence is implied by agreement with the previous
threshold prototype. AI assistance was used for implementation and validation.
