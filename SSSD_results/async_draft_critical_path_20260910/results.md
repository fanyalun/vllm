# EAGLE3 Async critical path and response stream wait

The response transport change improves the median of three fresh-process runs
from **182.6009 to 186.4312 output tokens/s (+2.0976%)**. All six runs produce
identical outputs and acceptance/cache counters. This validates the local
transport optimization against the unchanged Async algorithm, not AR/Sync
correctness or an Async-over-Sync performance claim.

## Contract and results

Llama-3.1-8B-Instruct / EAGLE3-LLaMA3.1-Instruct-8B, FP16, greedy, D=4, F=3,
B=1, CUDA Graphs, max model length 512. Target GPU 0 and Draft GPU 1 are A100
80GB PCIe devices across NUMA. Each cell uses the same 16 prompt token arrays
(1,469 input tokens), exactly 256 output tokens per request, one untimed
256-token warmup, and a fresh server. Timing includes prefill and HTTP but
excludes startup/warmup. No dependencies were installed.

| Repeat | Host-wait baseline tok/s | Source-stream wait tok/s |
| --- | ---: | ---: |
| 1 | 182.0286 | 186.4937 |
| 2 | 182.9502 | 186.2134 |
| 3 | 182.6009 | 186.4312 |
| Median | 182.6009 | 186.4312 |

Execution order was baseline 1, stream 1, stream 2, baseline 2, baseline 3,
stream 3. Each run has 4,096 output tokens, 1,558 verify rounds, 2,547 accepted
Draft tokens, 1,037 cache hits and 537 misses (including 16 bootstrap misses).
All 16 output arrays match across all six runs. Median amortized time per verify
is 14.3976 ms before and 14.1018 ms after. Both modes use two GPUs, so median
tokens/GPU-second are 91.3004 and 93.2156.

`ab_summary.json` contains raw per-run measurements, output/prompt fingerprints,
counter equality, and shutdown records. `summarize_ab.py` recomputes this summary
from the original cell artifacts.

## What the matched-setting Nsight traces show

Profiles use prompt indices 0, 4, 8 and 12 from the same 16-prompt workload, with
256 output tokens each. CUDA graph node tracing is enabled; CUDA event tracing
is disabled. Profiled throughput is not included in the A/B results. The table
uses decode-only GPU correlations; request-boundary gaps are excluded.

| Median duration | Sync | Async cache hit | Async miss |
| --- | ---: | ---: | ---: |
| Target forward GPU span | 10.286 ms | 10.260 ms | 10.267 ms |
| Outcome GPU ready to proposal GPU ready | 2.628 ms | 1.125 ms | 4.579 ms |
| Proposal ready to next Target forward | 0.079 ms | 1.826 ms | 1.828 ms |
| GPU idle within that next-forward interval | 0.040 ms | 1.792 ms | 1.794 ms |

The idle calculation subtracts the union of recorded GPU 0 kernel and memcpy
intervals; nested ranges and overlapping operations are not double counted.
These are instrumented-run observations, not absolute unprofiled latency
estimates. Cross-mode output parity is not asserted by this diagnostic.

The cache-hit path does save Draft GPU work. However, the host-blocking proposal
protocol exposes the next iteration's scheduling/preparation/submission gap.
The 9.46 ms median `payload_synchronize` host range largely waits for already
queued Target work; it is not 9.46 ms of new communication. The conditioning
peer copy itself has a 0.0092 ms median GPU duration. Before the change,
`response_synchronize` has a 1.40 ms median CPU wait on misses, while it is
approximately 0.007 ms on hits.

Thus the main observed issue is loss of host/GPU pipelining plus the expensive
miss path, rather than copying the hidden-state bytes. Removing the response
CPU wait recovers part of the miss-path gap. It does not eliminate the earlier
payload synchronization or the per-proposal Pipe round trip.

The optimized trace confirms the mechanism on the same four requests (399
proposal calls, 258 hits and 141 misses, including four bootstrap calls):

| Median interval | Before | After |
| --- | ---: | ---: |
| Miss: proposal ready to next Target forward | 1.828 ms | 0.341 ms |
| Miss: GPU idle within that interval | 1.794 ms | 0.305 ms |
| Hit: proposal ready to next Target forward | 1.826 ms | 1.796 ms |
| Hit: GPU idle within that interval | 1.792 ms | 1.763 ms |

The new source-stream wait takes 0.037 ms median host time. It queues a device
dependency instead of blocking the host until the child completes its Draft
work. The remaining hit-path gap requires a different change to proposal
lookup/publication or scheduling; this patch does not implement Target-local
candidate lookup or remove the payload barrier.

## Implementation and validation

`AsyncDraftSpeculator._copy_response()` waits for the imported response event on
the **source GPU's current stream** before enqueuing the peer copy. The child
records the event before publishing the response header. The peer copy orders
the Target consumer, allowing its CPU thread to continue preparing work.
Generation validation, request epochs, canonical refresh, cache lookup, branch
construction and JIT fallback retain their existing semantics.

Opt-in NVTX scopes distinguish metadata/conditioning copies, payload wait,
request/response IPC, response wait, hit cleanup, and miss JIT. The analyzer
joins GPU work to launches by both process and correlation ID and retains raw
stage rows. Missing critical-path correlations fail the analysis.

Validation includes 92 existing Async unit tests, three analyzer tests, and a
real two-GPU IPC response test with eight generations over two reused slots.
Full-model A/B output and counter equality passed. The prior seven-way figure
did not pass universal AR/Sync output parity; this experiment does not waive
that limitation or establish correctness for other models/batch sizes.

Mypy reports 31 errors on both original HEAD and the final files; error messages
match exactly after ignoring line numbers (`mypy_comparison.json`). The existing
CUDA-API portability hook also rejects the original CUDA-specific runtime and
proxy files. The new IPC test necessarily uses CUDA Event: attempting the
recommended generic `torch.Event.ipc_handle()` on PyTorch 2.11.0+cu130 raises
`NotImplementedError: torch.Event ipc is not supported yet`. Other setup and
synchronization calls in the new test use `torch.accelerator`.
Mypy and the CUDA-API hook are explicitly skipped for commit after recording
these limitations; they are not reported as passing. Other applicable
pre-commit checks and final transport test results are recorded in the local
validation artifacts.

## Reproduction and local artifacts

From the repository, use `.venv/bin/python` with:

```bash
.venv/bin/python benchmarks/replayssm/profile_async_draft.py \
  --mode async_cache --all-prompts --output-root /tmp/eagle3_stream_wait
.venv/bin/python benchmarks/replayssm/profile_async_draft.py \
  --mode async_cache --profile --output-root /tmp/eagle3_stream_profile
/usr/local/cuda-12.9/bin/nsys export --type sqlite \
  --output /tmp/eagle3_stream_profile/timeline.sqlite \
  /tmp/eagle3_stream_profile/timeline.nsys-rep
.venv/bin/python benchmarks/replayssm/analyze_async_draft_profile.py \
  /tmp/eagle3_stream_profile/timeline.sqlite
```

The runner also accepts `--target`, `--draft` and `--prompts` overrides. Its default
prompt file is the existing September 9 16-prompt artifact. Use a new output
directory for each invocation. The before/after speculator snapshots are retained
locally; the baseline uses host `event.synchronize()` before the peer copy.

Full traces, SQLite exports, commands, manifests, request outputs, GPU samples,
and logs remain under this local result directory. The published summaries
are accompanied by the source and regression tests; profiler binaries are not
included in the commit.

The initial Sync sandbox attempt could not access NVML. A reference Sync trace
with CUDA event tracing enabled was retained but excluded from the matched
analysis. The original Async controller was mistakenly terminated after
warmup/profile start; its live server was recovered to collect the four requests
and then shut down normally. A duplicate startup failed for insufficient free
memory and contributed no measurements. Direct cross-device IPC event waiting
also failed a synthetic test; the source-stream implementation passed and was
used for all optimized model runs. These attempts are recorded in
`audit_progress.json` and are not counted as completed A/B cells.

AI assistance: OpenAI Codex performed the code change, local diagnostics, tests,
and report preparation. No upstream PR is opened by this work.
