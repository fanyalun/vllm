# Native versus windowed GDN at B64/B128

This is a kernel-only pilot, not hierarchical decoding or acceptance measurement.
The requested comparison is B64/B128; B1 provides a same-harness anchor.

## Protocol

- A100 80 GB PCIe, one GPU per run, BF16 QKV/BA and FP32 SSM.
- Three captured layers (indices 0, 15, 29), five tokens per request.
- Captured QKV, BA and initial state are replicated into independent requests.
  This measures batch scaling with correlated head actions, not request diversity.
- The capture lacks Z and norm weights: seeded synthetic Z and unit weights are
  used identically for each method. Packed QKV/BA and strided Z match the relevant
  layout; this is not a replay of every full-model intermediate.
- V0 is the native fused post-conv MTP kernel, including gated RMSNorm and all
  candidate state snapshots. V2/V3 use a single in-place state plus their existing
  gated norm and gate reshape copy. Thus the comparison includes snapshot versus
  single-tail writeback, not only conditional arithmetic.
- V2/V3 recurrent-core-only timings are also recorded. Those exclude norm and
  must not be presented as equivalent-work speedups against fused V0.
- Thirty warmups and 100 CUDA-event samples, one graph replay per sample. Restore
  initial states and flush 64 MiB outside every timed interval. CUDA events are
  used instead of repeated CUPTI calls to preserve the fixed-state replay contract.
- No projection, Conv, model loading, MoE, generation or acceptance measurement.
- Each cell checks replicated-request equality and graph/eager output and state
  equality. Forced-Full windowed output must have relative L2 below 1e-3 versus
  native V0; final state uses atol=rtol=1e-3. Approximate V2/V3 are not expected to
  equal V0. The native candidate snapshot and single-tail state contracts differ.

## Status

The three-layer B1/B64/B128 validation completed all 45 cells, including native
versus forced-Full checks. Both GPUs were occupied by unrelated jobs, so the
one-sample validation timings are excluded from performance conclusions.
`contended_validation3/` contains this correctness evidence; earlier validation
attempts remain preserved separately. Pre-commit passes for the benchmark.

The formal command below waits for three consecutive process-free checks on the
selected GPU before allocating CUDA inputs. No other process is terminated.
This is not an exclusive GPU reservation; new concurrent jobs invalidate timings.
Only a formal `complete.json` alongside all 45 timing rows establishes coverage.
There is no B64/B128 speedup conclusion yet.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_gdn_native_batch.py \
  --inputs benchmark_results/three_level_p50_20260916/raw_inputs.pt \
  --output benchmark_results/gdn_native_batch_20260918/formal \
  --batches 1 64 128 --layers 0 15 29 --warmup 30 --repeat 100 \
  --wait-for-idle
```

Results contain raw microsecond samples, median/P95, action counters and equality
checks. Each latency is one layer processing the whole batch, not per request or
an estimate of 30-layer model time. Keep large artifacts local.
