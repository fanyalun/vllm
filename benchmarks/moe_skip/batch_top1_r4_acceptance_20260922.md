# Top-1 batch protection: four-round outer acceptance smoke

All eight cells completed, returning 4,096 tokens. In this four-prompt sample,
Top-1 protection with the half policy increased mean outer accepted length
from 12.175 to 12.550. With max-gap, it decreased from 9.857 to 8.873.
These are small-sample generation observations, not broad quality or speed claims.

## Contract and definitions

- Qwen3.6-35B-A3B, BF16, TP1, exact GDN, FP32 SSM storage, Triton MoE backend.
- Hierarchical MTP depth four, inner round limit four, outer capacity twenty.
- Four fixed prompts, B4, 128 greedy output tokens per prompt, ignore EOS,
  prefix caching disabled, CUDA Graphs, one excluded warmup batch per cell.
- GPU0: AR, native hierarchical, half Top-2, half Top-1.
- GPU1: AR, native hierarchical, max-gap Top-2, max-gap Top-1.
- Both A100 GPUs were shared with unrelated jobs. Every cell used
  `cpu_offload_gb=24` and `gpu_memory_utilization=0.70`; measured weight offload
  was about 24.57 GiB. Timing is retained in raw data but is not a speed result.

For request verify steps, let A be total accepted candidate tokens, D total
submitted candidates, and S total steps. Mean accepted length is A/S, mean
submitted length is D/S, and acceptance/submission is A/D. A excludes the Target
bonus/correction token. The existing `mean_acceptance_length` field is 1+A/S;
the new `mean_outer_accepted` field is A/S. Counts describe verification before
final output-length truncation, so A+S need not equal the returned token count.
The twenty-token capacity is an upper bound, not the measured submission length.

## Results

The two native hierarchical controls had identical acceptance counters and
are shown once. The AR controls matched each other on all four token sequences.

| Policy | Protected ranks | A | D | S | Mean accepted A/S | Mean submitted D/S | A/D | Strict AR parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Native Top-8 | All | 501 | 503 | 38 | 13.184 | 13.237 | 99.60% | 2/4 on each GPU |
| Half | Top-2 | 487 | 552 | 40 | 12.175 | 13.800 | 88.22% | 3/4 |
| Half | Top-1 | 502 | 540 | 40 | 12.550 | 13.500 | 92.96% | 2/4 |
| Max gap | Top-2 | 483 | 624 | 49 | 9.857 | 12.735 | 77.40% | 3/4 |
| Max gap | Top-1 | 488 | 681 | 55 | 8.873 | 12.382 | 71.66% | 3/4 |

Half Top-1 versus Top-2: mean accepted length +3.08%, A/D +4.74 percentage
points. Max-gap Top-1 versus Top-2: mean accepted length -9.99%, A/D -5.74
percentage points. A single four-prompt run does not establish that reducing
protection generally improves acceptance. These are independent generation
trajectories; candidates and batch composition can diverge between policies.

Strict AR parity failed even for native speculative controls. No lossless
decoding or broad model-quality claim is made. This B4 real-generation sample
must not be combined with the earlier B64/B128 synthetic routing-count pilot
as if both measured the same workload. Actual large-batch acceptance and
unloaded-GPU full-cycle throughput remain unmeasured.

## Validation and artifacts

The config and routing suites passed 182 tests before the final Target-isolation
test expansion. The expanded dispatch/graph/Target-isolation selection then
passed all 12 tests, and the expanded policy-hash test passed. Applicable
pre-commit hooks passed. The updated summarizer also audited the previous
14-cell matrix without changing its artifacts.

Both new four-cell matrices passed result hashes, prompt/seed pairing,
per-step versus histogram counters, output length, configuration checks, and
recomputed AR parity. Every recorded step satisfies 0 <= accepted <= submitted
<= 20. All recorded source hashes match the tested files. The run used base
commit `9b18ad27b0` plus the recorded working-tree changes.

Canonical local results:

- `benchmark_results/moe_batch_top1_r4_acceptance_20260922/offload24/acceptance_summary.json`
- `benchmark_results/moe_batch_top1_r4_acceptance_20260922/offload24/complete.json`
- `offload24/half/generation_summary.md` and `offload24/half/generation_audit.json`
- `offload24/max_gap/generation_summary.md` and `offload24/max_gap/generation_audit.json`

The parent experiment directory retains an earlier 16 GiB-offload attempt.
Its AR cells completed, but hierarchical initialization failed with no memory
available for cache blocks. Those cells are excluded from the final comparison.

## Reproduction

Run the following on the two GPUs concurrently. Each command executes its four
cells sequentially and writes a matrix completion marker only after all finish.

```bash
.venv/bin/python -m benchmarks.moe_skip.run_batch_policy_matrix \
  --device 0 --batch-size 4 --inner-rounds 4 --hierarchical-only --include-top1 \
  --policy-family half --gpu-memory-utilization 0.70 --cpu-offload-gb 24 \
  --output benchmark_results/top1_r4_reproduction/half/b4
.venv/bin/python -m benchmarks.moe_skip.run_batch_policy_matrix \
  --device 1 --batch-size 4 --inner-rounds 4 --hierarchical-only --include-top1 \
  --policy-family max_gap --gpu-memory-utilization 0.70 --cpu-offload-gb 24 \
  --output benchmark_results/top1_r4_reproduction/max_gap/b4
.venv/bin/python -m benchmarks.moe_skip.summarize_batch_policy_matrix \
  --root benchmark_results/top1_r4_reproduction/half --batches 4
.venv/bin/python -m benchmarks.moe_skip.summarize_batch_policy_matrix \
  --root benchmark_results/top1_r4_reproduction/max_gap --batches 4
```

Runtime opt-in values are `moe_skip_batch_policy="batch_top_half_top1"` and
`"batch_max_gap_top1"`. Existing policy names retain Top-2 protection, and final
Target routing continues to use its native experts. Code and this report are
versioned; raw generation artifacts remain local.
