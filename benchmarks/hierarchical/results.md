# Validation status — 2026-09-09

**Experimental implementation; strict equivalence gates failed.** The three-level
execution path runs, but the planned lossless validation and complete performance
matrix are not complete. No `RUN_COMPLETE` was created. Throughput in the raw
files is exploratory and must not be reported as a validated speedup.

The separate user-requested previous-configuration measurements are recorded in
[previous_config_20260909/results.md](previous_config_20260909/results.md).
They compare D=4/N=1,2,4,8 with historical performance and acceptance data;
they do not change the failed strict-equivalence status documented here.

## Implemented

- Shared Target/MoE-Skip parameters, top-k 8 versus top-h 4.
- MTP and DSpark inner drafting, D=4, N=4, maximum capacity 20.
- Inner prefix acceptance plus recovery/bonus, accepted-state continuation,
  Target refresh at outer boundaries, and actual-length scheduler propagation.
- Private GDN convolution/recurrent candidates with null block zero reserved.
- Eager execution and CUDA Graph execution, including variable Target widths,
  independent pre-verifier metadata builders, and refreshed graph metadata.
- Standard Target sampling with deterministic proposal semantics.
- Same-root sequential pre-verifier oracle and raw round traces.

## Tests

| Validation | Result |
| --- | --- |
| Configuration/state/routing/trace unit tests | 65 passed |
| Deterministic proposal GPU distribution tests | 6 passed |
| Python compilation and `git diff --check` | Passed |
| Full staged pre-commit checks, including Ruff and mypy 3.10 | Passed |
| mypy 3.12 on staged Python files | Passed |
| DSpark eager sequential-oracle smoke, 1 × 32 | Passed retained-prefix check; output matched eager AR |
| DSpark CUDA Graph smoke, 1 × 32 | Output matched eager AR; 36 pre-verifier replays recorded |
| Hierarchical MTP/DSpark graph greedy execution | Both completed 4 × 256 tokens |
| Hierarchical MTP/DSpark graph T=1, top-p=.95 execution | Both completed 4 × 256 tokens |
| Full greedy token parity | Failed |
| MTP retained-prefix sequential oracle | Failed |
| Full 6-method × 2-temperature × 3-repeat comparison | Not completed; correctness gates failed |

Unit commands:

```bash
.venv/bin/python -m pytest \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/config/test_moe_skip_config.py \
  tests/model_executor/layers/fused_moe/test_routing_top_k.py \
  tests/v1/worker/gpu/spec_decode/test_moe_skip_trace.py \
  tests/v1/worker/test_mamba_hybrid_model_state.py -q

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_rejection_sampler_utils.py \
  -k deterministic_cascade -q
```

The distribution tests use the actual GPU rejection sampler, point-mass drafts
(`draft_logits=None`), lengths 1/4/20, and a proposal outside the Target support.
These tests establish the sampler property for their fixed distributions. They
do not prove whole-model numerical equivalence.

## Failed gates and controls

First differing output token, **zero-based**, against the graph AR reference:

| Candidate | Sample 0 | Sample 1 | Sample 2 | Sample 3 |
| --- | ---: | ---: | ---: | ---: |
| Hierarchical MTP, graph | 4 | 134 | 187 | 2 |
| Hierarchical DSpark, graph | 4 | 190 | 206 | 68 |
| AR, eager control | 4 | 18 | 187 | 20 |

The eager hierarchical MTP run matched eager AR completely on samples 0 and 2,
and differed at positions 18 and 137 on samples 1 and 3. Ordinary MTP D=4 also
differed from eager AR on all four samples; ordinary MoE-Skip D=20 differed on
two. Thus AR mismatch alone does not isolate a hierarchical state-management
defect. It also does not excuse the failed gate.

The pre-verifier oracle failed at absolute position 172 for MTP:

```text
draft      = [16, 16, 198, 12237]
batch      = [16, 198, 1358, 2, 198]
sequential = [16, 25, 1358, 2, 198]
accepted   = 1
```

The recovery token differed, so the retained prefix check correctly failed.
First-layer hidden differences were about 1e-4 and grew across layers; maximum
final logit differences ranged from 2.33 to 5.41 in that batch. This supports
investigating numerical differences between execution shapes, but is not a
complete root-cause proof. The checkpoint already selects FP32 recurrent
state; explicitly selecting FP32 did not change this failure.

The existing batch-invariant mode cannot be used as a workaround:

```text
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```

Remaining correctness work includes isolating same-prefix Target logits and
cache contents across execution shapes, auditing the full Attention prefix on
GPU, and resolving the sequential pre-verifier mismatch. The current backend
does not provide a tested batch-invariant path for this GDN model. Broader
performance claims should wait for those checks.

## Evidence and interpretation

[evidence/manifest.json](evidence/manifest.json) records hardware, package
versions, baseline revision, and hashes of the raw artifacts. The evidence
directory contains token IDs, engine configurations, prompt hashes, raw timing,
acceptance metrics, and graph replay traces. [audit.json](evidence/audit.json)
and [failed_gates.json](evidence/failed_gates.json) preserve the failures;
[cells.csv](evidence/cells.csv) marks the greedy gate as failed.

Each experiment uses one A100 80GB PCIe with TP1 and B=1. Some exploratory cells
ran concurrently on the two GPUs of the host. They are single repetitions,
not an interleaved performance comparison. Engine initialization and warmup
are excluded from each recorded e2e sample time. E2e includes prompt work and
host orchestration; TTFT/decode fields are null when the engine did not collect
them. Round traces include the application warmup request (external index 0),
whereas output JSON rows contain measured requests only.

The evidence comes from the implementation during debugging. Later changes
added startup restrictions and chunked-prefill guards and received unit
coverage; the complete GPU matrix was not rerun after those guards. Earlier
failed attempts involving anchor positioning or null-block allocation remain
under the local `benchmark_results/hierarchical_20260909` tree and are not
used as validation for the corrected implementation.
