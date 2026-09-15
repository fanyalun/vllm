# Optimized dual checkpoint versus original ReplaySSM

## Contract

Original means the current repository native ReplaySSM path with dual checkpoint
disabled and the default flush policy, not parallel-last or standard speculative
decoding. Both use buffer setting 16 and their respective native allocation rules:
original logical threshold 16+T and physical ring 32; dual hard cap 16 and physical
ring 16. This is an equal-configuration comparison, not equal cache capacity or
identical history lengths. Neither mode uses a custom flush interval.

A100 80GB, one GDN layer, BF16 inputs, FP32 state, FP16 d/k history,
H=16, HV=32, K=V=128. GPU 0 was otherwise idle. D=4/8, batch=1/4,
all/reject/mixed controlled acceptance. All commits T=D+1 previous inputs;
reject commits one; mixed alternates T and one. These are synthetic trajectories.
Three repeats per cell alternate original/dual order. Each measurement warms
32 cycles, captures 32, warms three graph replays, times 20 graph replays
(640 cycles), then probes 64 cycles for history/flush/promotion statistics.
Timing includes commit, conditional flush launch and verify; dual also writes
a candidate tail each cycle. This is not end-to-end serving throughput.

## Results

Medians of three repeats. Speedup = original / dual; below one means dual is slower.

| D | Batch | Trajectory | Original us | Dual us | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 1 | all | 18.88 | 14.84 | 1.273x |
| 4 | 1 | reject | 18.32 | 17.13 | 1.069x |
| 4 | 1 | mixed | 18.68 | 15.93 | 1.172x |
| 4 | 4 | all | 23.13 | 27.96 | 0.827x |
| 4 | 4 | reject | 22.47 | 32.75 | 0.686x |
| 4 | 4 | mixed | 22.90 | 30.19 | 0.758x |
| 8 | 1 | all | 35.12 | 23.14 | 1.518x |
| 8 | 1 | reject | 32.70 | 25.67 | 1.274x |
| 8 | 1 | mixed | 33.78 | 24.35 | 1.387x |
| 8 | 4 | all | 36.52 | 44.78 | 0.816x |
| 8 | 4 | reject | 34.24 | 51.21 | 0.669x |
| 8 | 4 | mixed | 35.24 | 47.92 | 0.735x |

Batch 1 improves in all tested cells; batch 4 regresses even with full acceptance.
Thus the previous dual-before/dual-after improvements do not establish superiority
over native ReplaySSM. Full acceptance still requires a candidate state write and
current-window d/k/g writes each cycle. The scaling difference is consistent with
extra state traffic and kernel occupancy costs, but no profiler-based causal
attribution was performed in this comparison. Confidence selection alone does not
remove the full-acceptance batch-4 deficit seen at this configuration.

Raw state plus ring allocation is 2,494,464 bytes per layer/request for original
and 4,392,960 for dual (about 76.1% more). This excludes shared metadata and
model-level page padding. No equal-memory or optimal-flush-interval sweep was run.
No production code changed in this comparison; preceding correctness and DSpark
smoke checks apply to the same production source fingerprints.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
  benchmark_results/replayssm_dual_vs_original_d4_d8_20260915/compare.py
```

See measurements.json for individual timings and state statistics, summary.csv
for medians and min/max ranges, and manifest.json for software and source identity.
