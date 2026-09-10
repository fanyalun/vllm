# Target device candidate publication experiment

This is a bounded EAGLE3 B=1 experiment, enabled with
`ASYNC_DRAFT_TARGET_CANDIDATES=1`. The default remains the existing remote
proposal path. It does not implement GPU-only outcome selection or asynchronous
vLLM scheduling.

## Protocol

The Draft process allocates a two-slot token table on the **Target GPU** and
shares it with the Target process. After fan-out construction, it copies the
candidate tokens into that table and publishes a small CPU header only after
the copy completes. The header includes generation, slot and full cache keys.
The allocation owner is the Draft process; its physical location is Target GPU 0.

For example, with generation 10, accepted depth 2 and recovery token 17, Target
can consume only a generation-9 entry matching its engine ID, request ID,
request epoch, depth 2 and token 17. A missing, stale, transient, prefilling or
forced-JIT case uses the original remote proposal path.

A local hit still sends the real Target payload to Draft. It copies the selected
local tokens into the existing Target proposal buffer and returns without
waiting for the Draft response. Before the next proposal or lifecycle command,
Target consumes and validates the delayed response, including confirmation
that Draft found the same outcome key. Hit counters are counted once.

Draft still performs its original cache lookup, cleanup, canonical refresh and
fan-out work. Provisional EAGLE3 KV is never promoted to canonical KV. Two slots
prevent the next publication from overwriting the current candidate source;
the existing payload barrier orders consumption before later slot reuse.

The experiment rejects other methods and `max_num_seqs != 1`, and rejects
logits tracing. This is not validation of continuous batching, cancellation,
preemption, stochastic sampling or other model families. Unit tests cover
identity/outcome rejection and deferred acknowledgement validation, while the
model smoke covers normal request completion, hit/miss and ring reuse.

## What is still on the foreground path

- Payload construction and its CUDA synchronization.
- Two scalar outcome reads to the CPU and a Python key lookup.
- One nonblocking publication-header check, plus the existing request send.
- Request state writeback, engine scheduling and next-forward preparation.

This first experiment removes the hit response wait and return peer copy; it
does not eliminate outcome-dependent host control. Candidate publication adds
background work, and the consumer still polls a Pipe header.

## Measurement contract

Llama-3.1-8B-Instruct with EAGLE3-LLaMA3.1-Instruct-8B, FP16, greedy, D=4,
F=3, B=1, CUDA Graph, max model length 512, Target GPU 0 and Draft GPU 1.
Each cell uses prompt indices 0, 4, 8 and 12, 256 output tokens each, with one
untimed 256-token warmup in a fresh server. E2E throughput includes request
prefill/HTTP overhead, excluding startup and warmup. Both variants use two GPUs.
No Nsight or token trace is enabled in throughput cells.

| Final small-sample cell | Tokens/s | Tokens/GPU/s | E2E ms/verify |
| --- | ---: | ---: | ---: |
| Remote baseline, `baseline_r2` | 181.270 | 90.635 | 14.2652 |
| Target candidates, `local_r4` | 181.865 | 90.933 | 14.2186 |
| Target candidates, Async export off, `metrics_off_r2` | 179.981 | 89.990 | 14.3674 |

The final local-buffer run is 0.33% above the final remote run. This is too small
to establish an improvement with this sample/repetition budget. Early runs were
around 183 tokens/s in both modes. Turning off metric export did not produce a
consistent speedup; its final run was 1.04% below local-buffer/export-on. That
does not imply that exporting metrics accelerates decoding.

All eight unprofiled cells produced the same four arrays of 256 output tokens,
396 request-level verification rounds and 634 accepted draft tokens. The
export-on cells recorded 259 hits and 141 misses, including bootstrap proposal
calls. Final local-buffer/export-on recorded 259 local hits: every cache hit
used the new path. Acceptance counters may include tokens clipped at the final
output-length boundary and should not replace the actual 1024-token output count.

The final local-buffer and export-off cells shut down without the additional
CUDA IPC producer warning seen in early reverse-ownership versions. The existing
Python semaphore-tracker warning also appears in remote baseline logs.

## Matched-setting Nsight diagnostic

The two final diagnostic profiles use the same four prompts and produce equal
output token arrays. These are separate from all throughput cells above.
`critical_path_summary.json` retains both summaries and trace fingerprints.
The analyzer identifies the parent local-hit/remote-miss marker so a delayed
child cache-hit range cannot misclassify the next parent proposal.

| Decode median | Remote response path | Target local hit |
| --- | ---: | ---: |
| Outcome ready to proposal ready | 1.231 ms | 0.665 ms |
| Proposal ready to next Target forward | 1.736 ms | 1.841 ms |
| Target GPU idle within the second interval | 1.702 ms | 1.807 ms |
| Target forward GPU span | 10.255 ms | 10.252 ms |

The first interval improves by 0.566 ms, approximately 46%, in the diagnostic
trace. The next-forward gap does not improve. Local outcome scalar reads take
0.065 ms median host time on hit calls, and the local token copy takes 0.048 ms
host time. The large remaining gap therefore cannot be explained by these
scalar reads alone. Request/engine preparation and submission remain exposed
after proposal readiness; this trace does not attribute the entire gap to one
specific scheduler function.

Misses still take the remote path. Their outcome-to-proposal median is 4.725 ms
without publication and 4.845 ms with publication; their next-forward gap is
0.296 and 0.342 ms respectively. The candidate publication/lookup protocol adds
work even when lookup misses. Profiled phase improvements must not be projected
directly into unprofiled throughput: the final unprofiled pair showed only
0.33%, without enough repeated evidence to establish a speedup.

`summary.json` records all raw runs, exact output hashes, speculative counters,
shutdown status and the final comparison rows. Early runs include incremental
instrumentation and shutdown fixes and are retained as exploratory evidence.
There is one final run per variant: sub-percent differences are not causal
performance claims. Full local per-cell artifacts retain commands, environment,
requests, metrics, logs and completion markers.

## Statistics overhead

`ASYNC_DRAFT_EXPORT_METRICS=0` skips returning the Async metric dictionary from
the worker. Internal counters, `perf_counter` calls, metric construction inside
Draft and all protocol synchronization remain enabled. This is an ablation of
per-step Async metric propagation/aggregation/export, not all instrumentation.
Ordinary speculative acceptance counters remain available for output auditing.

The proposal token trace explicitly calls `.cpu().tolist()` on previous tokens,
new proposals, sampled/rejected counts and recovery tokens. Logits tracing adds
top-k work and more CPU reads. Those modes can disturb the critical path and
must remain separate from uninstrumented throughput. Nsight measurements are
also diagnostic, not throughput results.

Several CUDA waits in the current runtime also protect payload readiness or
branch/KV resource reuse. They must not be removed merely because a timing
counter is recorded beside them.

## Reproduce the small experiment

Validation: 113 targeted tests passed, including the two-GPU IPC response/ring
test. Targeted mypy reports the same two pre-existing annotations as HEAD under
`--follow-imports skip`. Applicable pre-commit checks pass with the full mypy
hook and CUDA-specific API lint explicitly skipped; see `validation.json`.
Both GPUs returned to zero allocated process memory after the final tests.

Run from the repository root with an unused output directory:

```bash
ASYNC_DRAFT_TARGET_CANDIDATES=1 ASYNC_DRAFT_EXPORT_METRICS=1 \
  .venv/bin/python benchmarks/replayssm/profile_async_draft.py \
  --output-root /tmp/target_candidates_small --mode async_cache --port 54210
```

Set `ASYNC_DRAFT_TARGET_CANDIDATES=0` for the remote baseline, or
`ASYNC_DRAFT_EXPORT_METRICS=0` for the export ablation. Add `--profile` only
for a separate Nsight diagnostic. `summarize.py` recomputes the archived small
experiment and fails on incomplete cells, differing tokens or standard counters.
