# Dual-checkpoint D=4/8 optimization

## Scope and implementation

Only the current dual-checkpoint path is optimized; the benchmark-only
parallel-last implementation is unchanged. Device-side zero-history branches
skip history loads and matrix products after promotion or flush. Current-window
d/k/g writes, tail writes, acceptance bookkeeping and hard-cap flush semantics
remain active. The short tail reduction uses a matrix multiply padded to 16;
the verify solve retains its original width. Short windows use BV=64, NK=2,
four warps. Floating-point summation order changes, so tolerance correctness
does not imply bitwise token equivalence.

## Measurement contract

A100 80GB, BF16 inputs, FP32 checkpoints, FP16 d/k history; H=16 query heads,
HV=32 value heads, K=V=128. Hard cap W=16 and physical ring length 16 for both
versions. Only D=4/8 and batch=1/4 are measured. These are single-GDN-layer GPU
cycles, including commit, conditional flush launch and verify/tail, timed in
CUDA Graphs. They are not whole-model tokens/s or comparisons with parallel-last.

Each cell has three paired measurements with alternating before/after order.
Each measurement warms 32 cycles, captures 32 cycles, warms three graph replays,
and times 20 replays (640 cycles). Tables use the median of three measurements.
The source snapshot comes from the base commit recorded in manifest.json.
Final timing ran on GPU 0 without concurrent GPU 0 tests; the bounded model
smoke used GPU 1. An initial exploratory run overlapped the end of tests and
is retained separately as initial_measurements.json, excluded from final results.

All acceptance commits T=D+1 previous inputs; reject commits one input;
mixed alternates T and one. These synthetic trajectories are controlled
mechanism probes, not measured DSpark acceptance distributions.

## Results

| D | Batch | Trajectory | Before us | After us | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 1 | all | 30.86 | 14.84 | 2.079x |
| 4 | 1 | reject | 31.17 | 17.11 | 1.821x |
| 4 | 1 | mixed | 30.89 | 15.92 | 1.940x |
| 4 | 4 | all | 75.39 | 28.07 | 2.686x |
| 4 | 4 | reject | 79.43 | 32.81 | 2.421x |
| 4 | 4 | mixed | 76.10 | 30.28 | 2.513x |
| 8 | 1 | all | 26.59 | 23.15 | 1.149x |
| 8 | 1 | reject | 27.05 | 25.67 | 1.054x |
| 8 | 1 | mixed | 26.62 | 24.35 | 1.093x |
| 8 | 4 | all | 66.35 | 44.81 | 1.481x |
| 8 | 4 | reject | 67.20 | 51.28 | 1.310x |
| 8 | 4 | mixed | 66.48 | 47.97 | 1.386x |

D=4 improves across all tested trajectories. D=8 gains are smaller at batch 1,
especially continuous rejection. Three repeats are a small local performance
sample, not a serving-scale or cross-hardware conclusion. The final configuration
was selected using a bounded BV/warps/NK sweep recorded in tuning.txt. Its
columns are BV, warps, NK, D, batch, trajectory and cycle microseconds; these
were single exploratory probes, separate from the three-repeat final comparison.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
  benchmark_results/replayssm_dual_checkpoint_opt_d4_d8_20260915/compare.py

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/kernels/test_replayssm_dual_checkpoint_gdn.py \
  -k '5-5 or 16-5 or 9-9 or 16-9' -q

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/kernels/test_replayssm_flush_interval.py \
  -k 'None-4 or None-8 or 1-4 or 1-8 or flush_cursor_graph' -q

.venv/bin/python -m pytest \
  tests/config/test_replayssm_dual_checkpoint.py \
  tests/config/test_replayssm_flush_interval.py \
  tests/v1/worker/test_gdn_dual_checkpoint_metadata.py -q

# Repeat with --draft 8 and a separate output path.
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. .venv/bin/python \
  benchmarks/replayssm/dual_checkpoint_smoke.py --method dspark --dual \
  --draft 4 --output /tmp/dspark_d4.json
```

## Validation outcome

- 16 dual-checkpoint tests passed for T=5/9, W=T and W=16, FP32/BF16,
  eager/CUDA Graph. Each trajectory has 48 steps, variable lengths, full and
  partial acceptance, repeated rejection, ring wraparound, flush boundaries,
  request reuse and null padding. Outputs and candidate tails match the
  independent sequential oracle at the existing tolerances: output atol=2e-3,
  tail atol=3e-3, rtol=4e-2.
- 10 configuration/metadata tests and five original ReplaySSM D=4/8 regression
  tests passed. No D=16/32 performance or regression sweep was run in this task.
- All 36 paired timings have equal history, flush, promotion and memory statistics.
- Qwen3.6-35B-A3B + DSpark, V2 CUDA Graph, D=4 and D=8 each completed two
  64-token requests after an identical two-request warmup. Token IDs matched
  exactly between warmup and repeated requests for each D. Raw token IDs,
  logprobs, configuration, source hashes and counters are in dspark_d4.json and
  dspark_d8.json. These establish bounded integration/reuse behavior, not
  before/after token equivalence or whole-model throughput improvement.
- DSpark draft-token acceptance counters, including warmup, were 150/424 for
  D=4 and 152/864 for D=8. These are token acceptance rates, not probabilities
  of accepting the final position. Confidence-based length selection remains
  future work.
- Pre-commit checks passed; see precommit.log.
