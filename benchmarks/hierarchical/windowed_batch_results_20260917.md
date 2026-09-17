# Qwen hierarchical windowed GDN: real request batching

This change extends the existing compacting MTP controller to Qwen GDN. Each
request owns a private recurrent slot throughout an outer proposal. Compaction
changes the active slot indices, not the ownership of the underlying state.
The Target still performs final verification and owns all committed state.

The experiment uses one A100 80 GiB, TP1, Qwen3.6-35B-A3B, BF16 model/Conv,
FP32 SSM, greedy sampling, MTP D4 with at most four rounds, top-h4, and balanced
stopping. Prefix caching, asynchronous scheduling, grouping, LoRA, and
quantization are disabled. The user explicitly selected staying on one GPU:
unavailable large batches are capacity failures, not results from a smaller
active batch. No TP2 experiment is included.

## Implementation and validation contract

- `batched_state.py` allocates one SSM per request/layer for windowed policies,
  and disjoint native candidate slots for the exact recurrent path.
- Canonical initialization copies all layers and active requests in one Triton
  launch for the windowed path. Request identities are invalidated before each
  initialization; new epochs cannot reuse an old request's identity.
- Conv advancement uses each request's actual accepted count in a batched
  cross-layer launch. Windowed SSM carries the complete approximate tail;
  native private SSM restores its accepted position.
- The recurrent grid has an independent request axis, GPU query offsets, and
  private slot indices. Each request restarts its window phase at its own first
  query. Empty queries and invalid slots return before loading state.
- Private addresses remain fixed across graph replays. Graph keys include active
  request count and total width; capture/warmup restores private state.
- Qwen's Conv uses the existing variable-length batched operator. Normalization
  still spans the entire value head.
- The existing Gemma compaction path is retained. Batched legacy mean/grouped/
  `three_level_p50` configurations fail closed; they are not silently mapped to
  the new policy. Batched inner MTP calls retain fresh metadata via the existing
  `is_profile=True` path; their cost is included in the measurements.

Operator tests cover B1/4/8/16/32, different valid lengths, independent/permuted
slots, invalidation, all four optimization settings, and graph replay. State
tests cover B4/8/16/32, both Conv layouts, native/windowed policies, request
compaction, heterogeneous acceptance, unchanged Target buffers, and new epochs.
The benchmark separately checks graph/eager state and logits, committed GDN
and attention prefixes, and output parity under instrumentation. Attention
checks cover committed prefixes, not unused physical KV pool contents.

## Capacity boundary

The first automatic-budget B4 run actually reached only B2. Reducing prefill
capacity did not solve the conservative automatic cache allocation. The final
windowed B4 run explicitly reserves 7 GiB for the shared cache and records real
active B4. This memory setting is specific to this model/device/configuration.

The native B4 graph control failed during graph capture with CUDA OOM. It keeps
six private snapshots per request; the windowed path keeps one. Private memory
is 255 MiB for windowed B4, versus 1698.75 MiB for native B4.
The native B4 eager integration smoke completed four prompts of 32 tokens,
observed real active B4, and passed repeat/instrumentation output checks. Its
eager timing is not compared with the graph-based throughput table.

The padded cache allocation requires 64 pool blocks per request and 26.125 MiB
per pool block. Model loading reports 66.51 GiB. Including private windowed
state, the lower bounds below already exclude CUDA context, activations,
workspaces, and graphs:

| Requested B | Model + required cache + private state, GiB | Hierarchical status |
| --- | ---: | --- |
| 4 | 73.29 | Actual B4 measured with explicit cache budget |
| 8 | 80.07 | Capacity check failed; no B8 generation result |
| 16 | 93.63 | Capacity check failed; no B16 generation result |
| 32 | 120.75 | Capacity check failed; no B32 generation result |

The large-batch failures are memory-capacity checks, not fabricated OOM launches
or serial B1 measurements. Their recurrent kernels and multi-request state
contracts are nevertheless exercised at the requested batch sizes.

## Measurement boundaries

End-to-end timing uses the first 32 GSM8K test prompts, 128 returned tokens each,
seed 42, and two interleaved repetitions after warming every measured policy.
The first 16 prompts exactly match the previous experiment's prompt text and
token IDs. Throughput is total returned tokens divided by total elapsed time.
No stage events or profiler run in these timing passes; a strict JIT guard is
enabled after warmup. All proposal, Target, initialization, Conv maintenance,
scheduling, and terminal-proposal costs remain included.

Fine-grained audits are separate runs on the first complete request batch.
Thus the B4 acceptance/latency audit covers four prompts, while throughput
covers 32. Inner acceptance counts only the four candidates, excluding the
anchor and correction. Outer acceptance is `num_sampled - 1` for nonempty
scheduled draft windows. Traces retain request and proposal-cycle identities.
Proposal latency includes Draft/Pre-Verify/maintenance and must not be added
to its nested stages. Target graph and piecewise forward latencies are reported
separately from sampling and scheduling.

Core measurements use 30 real layer captures from sample 0, boundary 96,
replicated into independent request slots at the production strides. They are
batch-scaling probes, not diverse online request trajectories. Every timing
sample restores the same initial states outside the timed region: 50 warmups,
200 CUDA-event samples, one graph replay per sample. The measured graph contains
the 30 recurrent launches. A separate CUPTI profiler pass provides individual
kernel durations and preserves layer/replay identity.

## Complete B4 comparison

| Policy | Returned tokens/s | Inner accepted candidates/round | Outer accepted drafts/verify | Pre-Verify median, ms |
| --- | ---: | ---: | ---: | ---: |
| AR | 243.59 | N/A | N/A | N/A |
| V1, forced Full with carry | 239.32 | 3.396 | 13.886 | 15.263 |
| V2, window 1 | 242.62 | 3.273 | 13.432 | 15.158 |
| V3, window 5 | 230.88 | 3.122 | 11.349 | 14.572 |
| V4-D | 225.82 | 3.117 | 11.114 | 14.459 |
| V4-Q | 229.80 | 3.296 | 12.195 | 15.360 |
| V4-DQ | 227.01 | 3.119 | 11.581 | 15.220 |

The acceptance and Pre-Verify columns are from the four-prompt audit, not the
32-prompt throughput population. V3 reduces recurrent latency but does not
improve complete-generation throughput at B4. V2 is effectively tied with AR;
the small difference is not evidence of acceleration. V1 is a carry-state
control, not an online exact reference.

All measured windowed policies repeat their own outputs and preserve outputs
under instrumentation. They do not establish strict AR equivalence: whole
128-token sequence matches are V1 10/32, V2 9/32, V3 12/32, V4-D 8/32,
V4-Q 8/32, and V4-DQ 10/32 against the same-condition AR run. No lossless claim
is made, and this experiment does not expand into a general hierarchical
baseline repair.

## Large-batch kernels and AR reference

The following medians are milliseconds for all 30 recurrent launches together,
with 200 unprofiled fixed-input samples per cell. Individual V3 kernel latency
is from a separate CUPTI pass with 30 replays and 900 launches per cell.

| B | V1 | V2 | V3 | V4-D | V4-Q | V4-DQ | V3 individual kernel median/P95, us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.027 | 0.929 | 0.862 | 0.864 | 1.214 | 1.251 | 28.607 / 32.383 |
| 8 | 1.680 | 1.452 | 1.370 | 1.370 | 2.094 | 2.229 | 45.951 / 50.751 |
| 16 | 3.124 | 2.575 | 2.410 | 2.419 | 3.885 | 4.189 | 81.599 / 91.070 |
| 32 | 6.024 | 4.863 | 4.555 | 4.570 | 7.596 | 8.168 | 153.694 / 171.902 |

V3 reaches 1.323x core speedup over V1 at B32. V4-D adds no useful improvement
here. V4-Q compiles with 128 registers and six spills, versus V3's 72 registers
and zero spills. V4-DQ uses 168 registers and zero spills; its regression cannot
be attributed to spills. These versions are not recommended by this pilot.

An additional AR-only sweep uses one engine with capacity 32, the same 32
prompts and 128 tokens, two interleaved repetitions, and real active batches.
It is separate from the capacity-four AR control above.

| Actual AR B | Returned tokens/s | Output checks |
| --- | ---: | --- |
| 4 | 246.37 | Warmup/repeats/audit equal |
| 8 | 350.26 | Warmup/repeats/audit equal |
| 16 | 565.15 | Warmup/repeats/audit equal |
| 32 | 773.60 | Both timed repeats equal each other; differ from warmup |

AR B32 also fails the audit-versus-warmup check. Its timing is retained with a
failed output-stability gate. The AR run has no hierarchical private state;
the cause of this baseline discrepancy has not been established. The original
sweep retained timed tokens but not warmup tokens, so its first warmup
divergence cannot be reconstructed. The runner now saves warmup tokens.
A fresh B32-only run reproduced the failure: both timed repetitions match
warmup on 31/32 prompts; zero-based sample 24 differs at its first output token
(warmup ID 8160, timed ID 124305). `ar_b32_check/warmup_divergences.json` preserves
this evidence. This does not establish a cause or justify changing the original
sweep's failed check.

The separate profiled four-prompt V3 audit reports median/P95 milliseconds:
Target piecewise forward 40.900/42.231, initialization 1.768/1.777,
Draft 12.868/12.976, Pre-Verify 17.285/18.501, Conv maintenance 0.117/0.120,
and proposal 145.949/150.507. These instrumented values must not replace the
unprofiled throughput measurements. Matched per-request complete-cycle spans
have median/P95 193.169/198.488 ms; spans shared by concurrent requests are
duplicated observations and their sum is not GPU busy time.

All six policies passed committed-prefix byte checks and graph/eager checks
on eight distinct active request sets each, including B4 and compaction.
The four relevant pytest suites passed 401 tests (16 warnings). Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/config/test_hierarchical_config.py \
  tests/benchmarks/test_hierarchical_measurement.py
```

## Reproduction and artifacts

Run from the repository root using the existing environment. No new packages
or environment upgrades are required.

```bash
export CUDA_VISIBLE_DEVICES=0
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python benchmarks/hierarchical/run_batch.py \
  --batch 4 --case v3 --cases v1 v2 v3 v4d v4q v4dq \
  --samples 32 --tokens 128 --repeats 2 --kv-gib 7 \
  --dataset benchmark_results/windowed_batch_20260917/prompts.jsonl \
  --output benchmark_results/windowed_batch_reproduction
.venv/bin/python benchmarks/kernels/benchmark_gdn_batch.py \
  --inputs benchmark_results/windowed_gdn_20260917/quality/raw_inputs.pt \
  --output benchmark_results/windowed_batch_kernel_reproduction
# Run profiling separately from the 200-sample timing pass.
.venv/bin/python benchmarks/kernels/benchmark_gdn_batch.py \
  --inputs benchmark_results/windowed_gdn_20260917/quality/raw_inputs.pt \
  --profile-replays 30 --repeat 1 \
  --output benchmark_results/windowed_batch_kernel_profile
.venv/bin/python benchmarks/hierarchical/run_batch.py \
  --batch 32 --case ar --batches 4 8 16 32 \
  --samples 32 --tokens 128 --repeats 2 --kv-gib 7 --profile \
  --dataset benchmark_results/windowed_batch_20260917/prompts.jsonl \
  --output benchmark_results/windowed_batch_ar_reproduction
.venv/bin/python benchmarks/hierarchical/summarize_batch.py \
  benchmark_results/windowed_batch_20260917
```

Large local artifacts live under
`benchmark_results/windowed_batch_20260917/`: `online_b4/`, `kernel_final/`,
`ar_b4/`, smoke/quality runs, raw profiles, timing samples, `capacity.json`,
`summary.json`, source fingerprints/snapshots, and preserved logs.
They are intentionally excluded from the code commit. `complete.json` records
completion of an individual run; `full_batch_observed` is a separate coverage
condition. Early smoke markers predate this added coverage field and must be
interpreted using their actual active-batch traces.
