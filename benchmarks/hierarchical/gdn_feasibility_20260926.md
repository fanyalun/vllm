# GDN V2 drafting feasibility, 2026-09-26

## Finding

The measured V2 recurrent-kernel benefit did not translate into a useful
full-model autoregressive draft on this machine. Native MoE routing was retained.
With actual B32, V2 K16 reached 90.81% candidate acceptance but only 0.664x AR
throughput. With actual B64, V2 K7 reached 96.21% acceptance but 0.789x AR.
These are measurements of the current prototype, not a proof that every future
GDN approximation or parallel verifier must fail.

## Experiment contract

- Two A100 80 GB PCIe GPUs, TP2, cross-NUMA SYS topology, GPUs 0 and 1.
- Local Qwen3.6-35B-A3B, BF16 weights and convolution state, FP32 temporal state.
- First 32 or 64 distinct raw GSM8K questions; greedy decoding, seed 42,
  ignore EOS, 128 returned tokens per request. This is a decoding feasibility
  test, not a GSM8K answer-accuracy evaluation.
- One warmup generation, three uninstrumented repetitions, and a separate
  instrumented generation. Timing includes generation, verification, scheduling,
  state maintenance, output delivery, and a final collective device drain.
- Model Runner V2, CUDA Graphs, compilation mode 0, Triton MoE, no expert skip,
  no CPU offload, no prefix caching, synchronous scheduling.
- Common token budget 4096, maximum model length 512, initially 28 GiB cache
  per rank. Actual decode concurrency is audited rather than inferred from
  `max_num_seqs`.
- V2 uses alpha 0.95, beta 0.36328125, a per-token gate and private temporal
  state. It generates candidates autoregressively using the complete model.
- An MTP configuration bootstraps native speculative cache allocation; the V2,
  Full, and native-draft adapters replace MTP proposals entirely after startup.
  MTP weights remain allocated. The separately labeled MTP baseline uses real
  four-candidate MTP proposals.
- The TP adapter uses local head dimensions for the existing approximation
  projection function. Production TP restrictions and runtime files were not
  changed by this experiment.

All formal 128-token matrix cells used identical benchmark source hashes:
`94993814f5...` for the driver and `64b49641a3...` for the worker. The subsequent
validation revision adds coverage for cache isolation after rejection and
compaction, graph/eager logit checks, and accelerator API lint compliance.
The two revisions are preserved separately in the local artifacts.

## End-to-end results

| Requested B | Method | K | Actual peak decode B | Tokens/s | Relative to AR | Candidate acceptance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | AR | 0 | 32 | 1210.79 | 1.000 | N/A |
| 32 | V2 | 16 | 32 | 803.91 | 0.664 | 90.81% |
| 32 | Forced Full | 16 | 32 | 876.97 | 0.724 | 97.39% |
| 32 | V2 | 7 | 32 | 967.15 | 0.799 | 95.80% |
| 32 | MTP | 4 | 32 | 1272.45 | 1.051 | 75.59% |
| 64 | AR | 0 | 64 | 1894.83 | 1.000 | N/A |
| 64 | V2 | 7 | 64 | 1494.23 | 0.789 | 96.21% |
| 64 | V2, capacity limited | 16 | 43 | 895.68 | 0.473 | 90.39% |

Acceptance counts accepted candidates before the first rejection and excludes
the correction/bonus token. It is the ratio of summed integer counts over decode
rows. Terminal candidates can exceed the tokens ultimately returned to the
caller. Full-batch and shrinking-batch statistics are stored separately.
The last row measures a 64-request workload with restricted concurrency; it is
not a full B64 K16 measurement. All eight cells completed, and warmup, repetitions,
and instrumented generation returned the same token IDs within each cell.

The longer-output check uses 256 tokens per request with the same B32 prompts
and execution settings: AR reaches 1227.41 tokens/s, V2 K16 914.42 tokens/s,
or 0.745x, with 90.20% candidate acceptance. V2 has 15 full-B32 decode steps.
Longer output reduces the drain penalty but does not reverse the result.
Both ranks pass canonical-prefix checks at initial proposal, full batch,
after rejection and compaction. Native, Forced Full and V2 probes also pass
the separate eager-versus-graph logit comparison at `atol=rtol=1e-3`.
Strict AR parity is 10/32 requests for this longer V2 run.

## Why the kernel result did not carry over

The original kernel artifacts were found under
`benchmark_results/.sources/gdn_native_batch_20260918/formal/`. They measure
five input positions, recurrence and gated norm, with native intermediate state
snapshots versus a single approximate final state. Their aggregated V2 speedups
are 1.583x at B64 and 1.628x at B128. Forced Skip and Decay experiments have larger
local speedups. None includes the whole model or autoregressive draft generation.

The earlier 91.41% result was found in the GDN-Spec motivation artifacts:
16 GSM8K samples, four exact-prefix boundaries per sample, 64 windows, L16 mean
accepted 14.625. That is quality simulation at B1, not large-batch serving timing.
The present experiment independently observes roughly 90% acceptance at B32.

Paired single-position, full-model CUDA Graph probes use the same real tokens,
positions and starting state. They include projections, attention, MoE, output
projection, logits, and prediction selection; state initialization is outside
this probe's timer.

| Probe | Native full forward, ms | V2 full forward, ms | Speedup |
| --- | ---: | ---: | ---: |
| B32, K16-run boundary | 19.017 | 18.974 | 1.002x |
| B64, K7-run boundary | 23.466 | 23.216 | 1.011x |

These are warm-cache graph probes, not sums of isolated layer timings and not
end-to-end throughput. A five-position recurrent-kernel speedup is not the
one-position full-model saving required by autoregressive drafting.

## Break-even conditions

For a representative full-batch cycle, let A be accepted candidates, T_AR the
native AR step time, T_D a complete approximate draft step, T_V exact block
verification, and H the remaining cycle cost. The useful condition is:

```text
K * T_D + T_V + H < (E[A] + 1) * T_AR
```

For a desired speedup s, replace the right side by `(E[A] + 1) * T_AR / s`.
For K16 and mean acceptance 90%, the numerator is 15.4 AR steps, not 16 or 17.
Acceptance alone cannot determine viability.

Representative full-batch stage medians from the 128-token matrix give:

| Configuration | E[A], full batch | AR step, ms | Exact verify, ms | Complete proposal, ms |
| --- | ---: | ---: | ---: | ---: |
| B32 K16 | 14.478 | 22.576 | 70.683 | 367.859 |
| B32 K7 | 6.700 | 22.576 | 42.099 | 163.865 |
| B64 K7 | 6.726 | 27.789 | 61.814 | 208.540 |

At fixed acceptance and verification cost, B32 K16 needs its complete proposal
below about 278.7 ms for stage-level break-even, versus 367.9 ms now. B64 K7
needs below about 152.9 ms, versus 208.5 ms now. Remaining sampling/scheduling
costs tighten these budgets.

Holding the observed proposal-minus-paired-forward residual fixed, the estimated
draft-forward budgets are:

| Configuration | Current V2 step, ms | Break-even draft step, ms | Draft step for 1.2x, ms |
| --- | ---: | ---: | ---: |
| B32 K16 | 18.974 | 13.403 | 9.764 |
| B32 K7 | 18.251 | 13.661 | 9.522 |
| B64 K7 | 23.216 | 15.263 | 10.152 |

These are conditional stage-cost estimates, not promised end-to-end thresholds.
The residual includes state/metadata/control costs and differences between real
draft steps and the single-boundary probe. It is not a pure CPU-overhead profile.
The probe-containing proposal is excluded from these proposal medians. Actual
end-to-end performance also pays for prefill, drain, shrinking batches and clipped
terminal candidates. Current wall time would need to fall by 33.6% for B32 K16
or 21.1% for B64 K7 merely to match the measured AR throughput at fixed outputs.

## Parallel verification and capacity

Current K16 verification does not implement a fully parallel scan across all
GDN positions. K16 means a target input width of 17. The fused CUDA GDN path is
limited to width 8; the longer speculative path calls
`fused_sigmoid_gating_delta_rule_update`, whose kernel loops over positions and
writes each intermediate temporal state. Dense/MoE operations process the block
together, but this GDN recurrence remains sequential within a head.

Relevant code:

- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`:
  `MAX_FUSED_GDN_MTP_TOKENS`, `_can_use_fused_gdn_mtp_decode`, `_forward_core`.
- `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`:
  the `for i_t in range(0, T)` recurrence and intermediate state stores.
- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`:
  acceptance-aware state metadata on Model Runner V2.

With the 28 GiB cache setting, K16 has 2254 blocks and needs 52 blocks per
request in this model configuration: three GDN groups times 17, plus attention.
This explains the observed maximum of 43 requests. A full B64 needs roughly
41.34 GiB just for this cache allocation; model weights, private state, graph
buffers and temporary activations must fit in addition. Reducing private draft
state does not by itself reduce the target's scheduler-visible reservation.

An explicit 42 GiB-per-rank cache attempt reached startup but failed while
cloning private state for graph capture, with only 21.88 MiB free and a
64 MiB allocation requested. This is a prototype capture-memory limitation,
not proof of an irreducible B64 hardware limit. The optional
`--capture-state-on-cpu` stores only these capture/restore snapshots on CPU;
serving model weights and recurrent state remain GPU-resident.

The 41.5 GiB cache retry with CPU capture snapshots gets past that failure and
advertises capacity 64.25. It then fails in real target verification's logits
all-gather: 516 MiB requested with 343.88 MiB free on an approximately 79.25 GiB
device. It has no completed throughput result. This establishes a current
implementation capacity limitation, including target logits and graph buffers;
it does not establish a hardware-independent limit or measure a future
memory-efficient verifier.

A B32 K16 regression with CPU capture snapshots completes at 804.91 tokens/s
(one timing repetition). It matches the native-draft control on all 32 requests,
matches its own warmup/audit, and passes all cache-isolation scenarios on both
ranks. The snapshot-placement option changes capture-time memory, not the
steady-state conclusion.

A future exact parallel verifier must also restore the state at the accepted
prefix after rejection. Keeping only the block's final state is insufficient.
Either intermediate states, exact accepted-prefix replay, or an equivalent
recovery representation is needed. This future verifier is not implemented or
claimed as measured by this experiment.

## Correctness boundary

The 128-token formal V2 cells are deterministic across warmup, three repetitions
and audit. Initial private-draft cache-isolation checks pass on both ranks.
However, strict greedy AR parity fails: B32 V2 K16 matches 17/32 requests, B32 K7
13/32, and B64 K7 32/64. Forced Full K16 and native MTP each also match only
17/32 AR requests. Thus this is not evidence that approximation alone caused
the mismatch, and it is not sufficient to certify a lossless implementation.

Forced Full versus native paired probes agree on argmax at the inspected
boundary, but fail the stated logit tolerance `atol=0.1, rtol=0.01` and have
nonzero temporal-state differences. Numerical differences and rejection-state
semantics need to be distinguished using identical prefixes and shapes before
claiming strict equivalence. Random-sampling distribution equivalence and task
answer accuracy have not been established here.

The additional native-draft K16 control is especially useful: all 32 requests
match both the V2 K16 and Forced Full K16 outputs exactly, while the native
control itself matches only 17/32 AR outputs. This supports preservation of
the native block-verification result for this test. It does not establish
bitwise equality to a different AR execution path. Its full-batch,
after-rejection and compacted-batch canonical-prefix checks pass on both ranks.
The native-draft control uses an unoptimized private-state advance and only
one timing repetition; its 732.9 tokens/s is not an optimized AR comparison.

## Decision for further work

Prioritize a measured full-forward saving before extending the acceptance sweep.
At unchanged representative verification and residual costs, B32 K16 needs
about 1.42x acceleration of the entire draft forward, and B64 K7 about 1.54x,
to reach the estimated stage-level break-even budgets. A local kernel speedup
of 2x cannot imply either result. Under the simplified Amdahl model
`draft_ratio = 1 - p + p / kernel_speedup`, these budgets would require roughly
59% and 70% of the native forward to be in the accelerated GDN component.
Those percentages are conditional requirements, not measured GDN shares.

The next implementation targets are full-forward GDN integration, lower
per-candidate state/metadata/control costs, exact block verification with
accepted-prefix state recovery, and lower target cache/logits memory peaks.
Each needs a paired full-cycle check. Keeping acceptance near 90% is useful,
but increasing it alone did not rescue the Forced Full K16 control.

A practical go/no-go gate is a repeatable end-to-end gain of at least 1.1x on
the intended actual batch, with native-verifier output equivalence, explicit
AR numerical-parity characterization, rejection/compaction state checks and
sufficient memory headroom. The 1.1x margin is an engineering target, not an
experimentally discovered constant. This report does not recommend returning
to expert skipping on the basis of an unmeasured comparison.

## Reproduction and retained evidence

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/home/fanya/vllm \
  .venv/bin/python benchmarks/hierarchical/run_gdn_feasibility.py \
  --output benchmark_results/my_gdn_v2_b32_k16 \
  --tp 2 --batch 32 --length 16 --tokens 128 --repeats 3 \
  --kv-gib 28 --probe --quality

.venv/bin/python benchmarks/hierarchical/summarize_gdn_feasibility.py \
  benchmark_results/gdn_feasibility_20260926
```

The local result root is `benchmark_results/gdn_feasibility_20260926/`. Each
completed cell has a manifest, initialization data, returned token IDs, timing
repetitions, rank-specific audit, and completion marker. `summary.json` includes
acceptance histograms, survival probabilities, actual batches, matched AR
comparisons and cost estimates. Interrupted/failed probes remain visible and
are excluded from performance conclusions.

The runtime was a pre-existing dirty worktree at base commit
`b1b8d2746d72d809a091716cd0509e455935acad`; the runtime diff SHA-256 is
`e37ac0ae21723294e8733300b776155a4089caa26ee5f8438c50eb4feb6de407`.
Local source snapshots retain that patch, untracked runtime files and source
hashes. The published benchmark code does not publish the user's unrelated
runtime changes or large experiment artifacts.

Validation: all 38 tests in
`tests/benchmarks/test_hierarchical_measurement.py` pass; all applicable
pre-commit hooks for the five changed files pass. The study contains 12 primary
completed cells/controls, three completed exploratory smokes, and nine retained
incomplete attempts including the two capacity OOMs.
