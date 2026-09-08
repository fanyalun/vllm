# Gemma4 MTP asynchronous draft worker

Gemma4 MTP now has an independent Draft worker for the existing
`method=mtp, async_draft_device=1` interface. Initial support is text-only,
single-node TP1, eager, B=1, unquantized KV, and maximum context no longer than
the Target sliding window. Existing asynchronous scheduling, prefix caching,
LoRA and parallelism exclusions remain in force. Unsupported configurations
fail before readiness. CUDA Graphs and larger batches are not enabled by this
implementation.

The assistant loads its own checkpoint and retains its draft-dimensional LM
head. Only the Target input embedding is separately read through keyed
safetensors access and installed at the backbone hidden dimension. The child
does not load a second Target model. Shared-weight shapes and checksums are
reported at startup.

All assistant attention layers are Q-only. For each attention type, the proxy
selects the last non-shared Target KV layer, packs its logical pages, and copies
a snapshot into independent Draft-device ring storage. Three sliding assistant
layers reuse one snapshot, and the full-attention layer uses the other. Each
round copies the full current snapshot; incremental suffix transfer is not
implemented. Snapshot and attention buffers are bounded by the configured
maximum context, not by the number of cached branches.

Fresh JIT mirrors Sync's shifted token input, last accepted query index,
Target hidden state, and verification-window sequence-length semantics. All
steps within a proposal retain the same RoPE position and KV visibility. The
current Sync implementation's sequence length can include rejected verification
positions; this integration preserves that behavior for proposal parity instead
of silently changing it.

Cache hits return completed token-only branches. Background construction first
replays the returned token prefix against the newest real snapshot and Target
hidden state. It evaluates recovery candidates at depths 0 through D, excludes
the already-returned token at depths below D, and includes the all-accepted bonus
case. Each next-round branch uses a constant anticipated anchor position
`old_anchor + accepted_depth + 1`. Its feedback and old KV snapshot remain
provisional because the future Target states are unavailable. Branch state is
never promoted to canonical KV. Request epochs, generation checks and
release/reset messages invalidate stale outcomes. Construction completes before
the worker consumes the next command; CUDA events cover published tokens.

Broadcasted positions and block tables are materialized before calling kernels.
Profiling/dummy batches do not access real KV. With logits diagnostics enabled,
every branch build checks snapshot immutability. Performance runs disable that
diagnostic and CUDA launch blocking.

## Validation and reproduction

Local artifacts are under `SSSD_results/gemma4_mtp_async_20260908/`.
Correctness used four fixed Phase-B prompts, 256 greedy output tokens each,
D=6, F=3, bf16, Target GPU 0 and Draft GPU 1 on two A100 80GB PCIe GPUs.
Sync, forced-JIT, and cache-enabled Async produced identical final tokens for
all four requests. Forced-JIT matched Sync on all 266 paired real-prefix
proposals and all accepted outcomes. Cache-enabled correctness also checked
that Q-only branch forwards did not modify their KV snapshots. This is Sync
parity evidence; an AR comparison was not run in this task.

CPU validation: 98 tests passed in `test_async_draft.py` and
`test_async_gemma4.py`; the two existing Gemma configuration compatibility
tests also passed. Coverage includes independent embedding/head weights,
heterogeneous assistant dimensions, KV page boundaries and ownership, fixed
decode positions, full outcome-depth coverage, recovery exclusion, returned
prefix replay, configuration bounds, and existing IPC/lifecycle checks.
GPU abort/preemption, CUDA Graphs, larger batches, and contexts beyond the
sliding window have not been validated here.

The performance runner uses the same four tokenized prompts and excludes
startup plus 30 seconds of warmup. Run each mode with:

```bash
.venv/bin/python benchmarks/replayssm/gemma4_mtp_async_validation.py \
  --target /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it \
  --draft /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it-assistant \
  --prompts SSSD_results/gemma4_mtp_async_20260908/prompts.json \
  --output-root SSSD_results/gemma4_mtp_async_20260908 \
  --mode sync --performance
```

Use `--mode async_cache` for SSD. Omit `--performance` for correctness and use
`--mode async_jit` for forced-JIT. Completed cells are resumed rather than
overwritten. Use a fresh output root for independent repeats. Raw outputs,
commands, metrics, proposal traces, GPU samples and shutdown markers are saved
per cell. The quick comparison is one pair, not a repeated performance matrix.

## Measured short comparison

Measured on 2026-09-08 with the configuration above. Both modes completed
1,024 output tokens and matched on all four final token sequences.

| Metric | Sync | Async cache |
|---|---:|---:|
| Output tok/s | 50.6373 | 17.8445 |
| Elapsed seconds | 20.2223 | 57.3846 |
| GPUs | 1 | 2 |
| Tokens/GPU-second | 50.6373 | 8.9223 |
| Accepted drafts per verification round | 2.9275 | 2.5744 |
| Tokens per round including bonus | 3.9275 | 3.5744 |
| Verification rounds | 262 | 289 |
| Cache hit rate | N/A | 75.4325% |

Acceptance and hit statistics exclude the first bootstrap row of each request.
Verification acceptance can include tokens beyond the API output cutoff.
Async used 54.6733 seconds of paired previous-branch construction, of which
37.9521 seconds exceeded the measured overlap windows. Next-proposal waiting
totaled 39.5959 seconds. These observations identify exposed branch work as a
major cost, alongside reduced acceptance. The initial eager implementation
batches recovery candidates within each depth and processes depths serially.

Throughput changed by **-64.76%**. This implementation passes the measured
correctness checks but is a **no-go for replacing Sync for throughput** at
D=6/F=3/B=1. No fan-out sweep, cross-depth batching optimization, graph
optimization, or repeated performance matrix was run. Raw numbers and commands
are in `performance_summary.json`, `performance_summary.csv`, and the per-cell
artifacts under the local result root.
