# Forced V2 paths: kernel-only pilot

A100 80 GB PCIe GPU 0, T5, BF16 inputs and FP32 state. Three captured layers
(indices 0, 15, 29), independent replicated request states, synthetic Z and unit
norm weights. No decoding, projections or Conv. Thirty warmups and 100 samples
per cell; state restore and 64 MiB L2 flush outside each timed graph replay.
The tested source is based on a4603457dc plus this benchmark-only change.

V2 uses window size one. Runtime thresholds force Full/Decay/Skip without
changing the captured gates or compiling out classification. Native V0 includes
candidate snapshots and gated norm. V2 includes one tail state, gated norm and
the required gate reshape copy. Full therefore isolates a whole implementation
change, not snapshot cost alone. All Skip retains Q readout and norm work.

Values below are the average of three layer medians, in microseconds per layer
for the entire batch. Speedups compare the same boundary to native V0.

| Batch | V0 | V2 Full | V2 Decay | V2 Skip |
| --- | ---: | ---: | ---: | ---: |
| 1 | 25.60 | 26.79 (0.955x) | 23.55 (1.087x) | 21.85 (1.172x) |
| 64 | 594.60 | 431.10 (1.379x) | 324.27 (1.834x) | 255.32 (2.329x) |
| 128 | 1199.45 | 847.19 (1.416x) | 635.22 (1.888x) | 488.62 (2.455x) |

At B128, even Full saves 29.37% against V0. Within the same V2 implementation,
Decay saves another 25.02% and Skip another 42.32% relative to Full. These
uniform-action extremes are not predictions for mixed-head real requests.

All 36 cells passed graph/eager and replicated-request equality. Forced Full
matches native output with relative L2 below 1e-3 and tail with atol=rtol=1e-3.
Skip state is bitwise unchanged; Decay state matches FP32 stepwise scaling.
Action counters confirm all requested head-token decisions and clean-head counts.
The PNG and one-page PDF were rendered and visually checked. Pre-commit passed.

Local evidence: `benchmark_results/gdn_forced_paths_20260918/run/` contains
raw samples, P95, counters and the completion marker. The sibling
`gdn_forced_paths/` contains the PNG/PDF/Markdown figure bundle. Figure (a) uses
a log latency axis; (b) shows speedup. Horizontal positions denote batch
categories. Large artifacts stay local.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_gdn_native_batch.py \
  --inputs benchmark_results/three_level_p50_20260916/raw_inputs.pt \
  --output benchmark_results/gdn_forced_paths_reproduction/run --forced-paths
.venv/bin/python benchmarks/kernels/plot_gdn_forced_paths.py \
  --input benchmark_results/gdn_forced_paths_reproduction/run \
  --output benchmark_results/gdn_forced_paths_reproduction/gdn_forced_paths
```
