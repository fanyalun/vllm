# GDN drafting follow-up experiments

This continues [the initial feasibility study](gdn_feasibility_20260926.md).
The full-model draft still does not beat AR on Qwen3.6-35B-A3B after reducing
per-step host overhead. This is a result for the measured implementation and
hardware, not a general impossibility claim for GDN speculative decoding.

## Implemented changes

`--draft-block-graph` captures all K autoregressive draft steps in one CUDA
Graph. It keeps draft inputs, slot mappings, recurrent state advance and output
tokens on device between steps. Captured external tensors remain owned by the
graph cache. An initial diagnostic attempt exposed a freed zero-count tensor;
retaining that tensor fixed the illegal access, and subsequent paired checks
passed. The diagnostic run with `CUDA_LAUNCH_BLOCKING=1` is excluded from
performance comparisons.

`--batch-sharded-sampling` enables the existing distributed sampler through a
benchmark-only adapter for the outer multimodal model wrapper. Both the AR
baseline and speculative candidate receive the same option. The effective
worker flag is retained in initialization/audit artifacts; the startup warning
about the outer wrapper is not the final effective setting.

`--release-mtp-bootstrap` drops the unused MTP proposer after the full-model
adapter is installed. A paired B2 K7 smoke returns identical tokens and releases
only 33,879,040 allocated bytes per rank. It raises if a subsequent call needs
the removed MTP fallback. This is not a large memory saving.

`--probe` additionally measures a fixed-input oracle that reuses the exact GDN
layer outputs. It retains the remaining full-model work and checks identical
logits at the inspected boundary. It is a headroom probe, not a deployable draft
or a measurement of recurrent-kernel-only cost.

The summary pairs only matching models, sampling settings and diagnostic
environments, in addition to the original execution and prompt contract.
All new cells preserve copies of their driver and worker alongside source hashes.

## Paired full-model results

The hardware, prompts, FP32 SSM, native MoE, greedy sampling, 128 returned tokens
and TP2 settings match the initial study. Each formal result below has one
warmup, three uninstrumented repetitions and a separate correctness audit.

| Actual batch | Method | K | Tokens/s | Relative to paired AR | Acceptance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | AR, batch-sharded sampling | 0 | 1220.60 | 1.000 | N/A |
| 32 | V2, block graph, batch-sharded sampling | 7 | 1039.67 | 0.852 | 95.80% |
| 32 | V2, block graph, batch-sharded sampling | 3 | 1090.11 | 0.893 | 98.27% |
| 64 | AR, batch-sharded sampling | 0 | 1924.70 | 1.000 | N/A |
| 64 | V2, block graph, batch-sharded sampling | 7 | 1581.68 | 0.822 | 96.21% |

The block graph's candidates exactly match stepwise drafting, and recurrent
states pass `atol=rtol=1e-3`, on both ranks at full B32 and B64. Canonical target
prefix checks pass at initialization, after rejection, and after compaction.
Each cell is deterministic across warmup, repetitions and audit. Strict AR
token parity for K7 remains 13/32 and 32/64 requests respectively, matching the
initial study's numerical-parity boundary; these results do not certify
lossless AR equivalence. B32 K3 matches AR on 19/32 requests.

Without batch-sharded sampling, B32 block drafting reaches 1030.04 tokens/s,
versus the earlier 967.15 tokens/s with per-step graph dispatch. This is a
6.5% improvement in the speculative implementation, while remaining slower
than AR. The sampler change provides a smaller additional improvement and
also improves AR, which is why paired baselines are necessary.

Full-batch stage medians with block graphs and sharded sampling:

| Actual batch | Complete proposal, ms | Target verification, ms |
| ---: | ---: | ---: |
| 32 | 150.58 | 42.25 |
| 64 | 196.31 | 61.83 |

These are instrumented stage measurements, not replacement end-to-end metrics.
The table uses K7; B32 K3 has proposal 67.99 ms and target 32.68 ms.
Proposal contains the whole block and state maintenance; its inner stages
must not be added again.

## Headroom and capacity

At a B32 fixed-input boundary, the native complete draft forward takes about
18.35 ms. Reusing exact outputs for every GDN layer reduces it to about
14.88 ms, or 1.23x. Both ranks pass the same-logit check. Removing the whole GDN
branch saves about 19% at this boundary; accelerating only its recurrence
cannot be assumed to save that entire fraction.

The B64 K16 retry uses 41.5 GiB cache per rank, CPU-only capture snapshots,
block graphs and sharded sampling. It fails during prefill in the chunk GDN
temporary allocation: a 16 MiB allocation with about 10 MiB free on rank 0.
There is no completed throughput result. This is a capacity failure of this
configuration, even though sharded sampling addresses a different logits
all-gather allocation from the previous attempt.

## Parallel verification and accepted-prefix recovery prototype

`benchmarks/kernels/benchmark_gdn_chunk_verify.py` compares native recurrent
verification, the existing parallel chunk algorithm, and chunk verification
with accepted-prefix state recovery. It uses independent seeded synthetic
BF16 Q/K/V and gates, FP32 initial state, 8 Q/K heads and 16 value heads per
rank, dimension 128. This is a single-GPU post-convolution experiment, not a
full-model speedup measurement. A width of 17 corresponds to 16 candidates
plus the current target input token.

The recovery retains the chunk's normalized keys, corrected values and
cumulative log decay. For an accepted input prefix of length p within one
chunk, it computes:

```text
S_p = exp(G_p) * S_0
      + sum(j <= p) exp(G_p - G_j) * corrected_v_j * normalized_k_j^T
```

Zero-length recovery returns the initial state. This avoids replaying the
recurrence and avoids materializing every intermediate state. The initial
implementation used several PyTorch operations; a fused Triton implementation
performs the weighted matrix product and initial-state update together, using
FP32 accumulation and TF32x3 dot products.

| B | Verify width | Native all snapshots, ms | Chunk final only, ms | Chunk + PyTorch recovery, ms | Chunk + fused recovery, ms | Fused relative speed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.0256 | 0.0584 | 0.1034 | 0.0635 | 0.40x |
| 1 | 17 | 0.0471 | 0.0645 | 0.1147 | 0.0696 | 0.68x |
| 32 | 8 | 0.2084 | 0.1874 | 0.3574 | 0.2386 | 0.87x |
| 32 | 17 | 0.4081 | 0.2058 | 0.3983 | 0.2570 | 1.59x |
| 64 | 8 | 0.4157 | 0.3154 | 0.6072 | 0.4188 | 0.99x |
| 64 | 17 | 0.8238 | 0.3471 | 0.6912 | 0.4506 | 1.83x |

These medians use 30 alternating-order CUDA Graph measurements with a 64 MiB
L2 flush and state reset outside each timer. Gate conversion, Q/K normalization,
head expansion, chunk work and recovery are inside their respective graphs.
Compilation, random input generation and host copies are excluded. CUDA events
allow the recurrent state to be reset before every measured replay.

A further implementation uses a 16-position chunk for width 8 and a
32-position chunk for width 17, instead of padding both to 64. With fused
prefix recovery included, B64 width 17 falls to 0.3564 ms versus 0.8248 ms,
or 2.31x. B64 width 8 falls to 0.3072 ms versus 0.4183 ms, or 1.36x.
The reference here is the Triton recurrent all-snapshot kernel. The separate
CUDA fused MTP kernel used by production for supported short widths is not
benchmarked here, so width-8 ratios must not be substituted for its costs.

All-prefix comparisons also pass for the shorter chunks. Additional checks
cover mixed per-request prefix lengths, unchanged zero-prefix state and
log-decay inputs 0, -1e-6, -5 and -80. The recovery kernel uses 94 registers,
8 KiB shared memory and has zero spills in inspected compiled variants.

Every prefix from zero through full length passes a native-state comparison
with `atol=rtol=0.01`; fused recovery matches the PyTorch formula at
`atol=1e-5, rtol=1e-4`. Native versus chunk output relative L2 differences are
about 0.0043–0.0045, and final-state relative L2 differences about 0.0028–0.0031.
These numerical checks are not strict bitwise equivalence or model-output
quality evaluation. The chunk final-only column omits prefix recovery and
must not be presented as the usable verifier speedup.

This supplies a concrete next design: retain compact per-layer chunk
intermediates until target rejection sampling decides the accepted prefix,
then reconstruct only that prefix's state. Full-model integration must still
connect request/slot mapping, convolution state, graph buffer lifetimes,
rejection and compaction to canonical cache writeback. The current scheduler's
per-token state reservation also remains unchanged; the kernel prototype
does not by itself solve the observed B64 K16 allocation failure.

## Evidence and reproduction

Local evidence is under `benchmark_results/gdn_exploration_20260926/`.
Each completed model cell has `manifest.json`, `initialization.json`,
`warmup.json`, `timings.json`, `audit.json` and `complete.json`. Failed attempts
retain their logs and have no completion marker. Downloaded model assets are
outside the repository and are not published with benchmark code.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/home/fanya/vllm \
  .venv/bin/python benchmarks/hierarchical/run_gdn_feasibility.py \
  --output benchmark_results/my_gdn_block_b64_k7 \
  --tp 2 --batch 64 --length 7 --tokens 128 --repeats 3 \
  --kv-gib 28 --draft-block-graph --batch-sharded-sampling --quality

.venv/bin/python benchmarks/hierarchical/summarize_gdn_feasibility.py \
  benchmark_results/gdn_exploration_20260926

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/fanya/vllm \
  .venv/bin/python benchmarks/kernels/benchmark_gdn_chunk_verify.py \
  --output benchmark_results/my_chunk_verify.json \
  --batches 1 32 64 --lengths 8 17 --heads 16 --repeats 30
```

These runs use the same pre-existing dirty runtime as the original study:
runtime diff SHA-256
`e37ac0ae21723294e8733300b776155a4089caa26ee5f8438c50eb4feb6de407`.
Unrelated runtime edits are not part of the benchmark publication.

Validation: 41 measurement tests pass, including rejection of
cross-model, differently sharded and diagnostic-timing baseline comparisons.
Applicable pre-commit hooks pass for the six changed files. The final
short-chunk kernel matrix is retained as `chunk_short_prefix_verify_r2.json`
with all six cells complete, boundary checks and compiled-kernel resource data.
