# DSpark SSD branch-backbone validation

The asynchronous DSpark foreground now generates exactly D tokens using the
same D-wide query as Sync. Each provisional outcome branch reruns the backbone
with an explicit `[anchor, accepted draft prefix, recovery]` prefix. The last
prefix token seeds D Markov predictions. For `sample_from_anchor=False`, D masks
follow the recovery token instead.

Only complete KV pages before the old anchor are shared. The boundary page and
all writable query pages are private. A branch query completes before those
pages are released; the cache retains tokens and diagnostics, never canonical
KV. Requests and recovery candidates are batched within one accepted depth, and
temporary slots are reused at the next depth. Normal foreground CUDA graphs
retain width D; explicit-prefix branch forwards are eager.

Candidate evaluation covers depths 0 through D. The depth-zero normal JIT logits
are reused because the query inputs are identical. Other depths run separate
queries containing only their own accepted prefix. The returned draft token is
excluded at depths below D, and all-accepted bonus candidates are included.
Draft-to-target vocabulary mapping is preserved.

The standalone DSpark child installs the same IR kernel priorities and torch
wrapping policy as `WorkerBase`. Without this initialization, it falls back to
native RMSNorm and can diverge numerically from Sync. Markov diagnostics always
retain `argmax` sampling, including tied logits; enabling top-2 recording must
not select a different token through `topk` ordering.

## Current validation status

The first-stage configuration is Qwen3.6-35B-A3B with its DSpark checkpoint,
bfloat16, D=3, F=3, Target B=1, eager, and two A100 80GB PCIe GPUs. The internal
`ASYNC_DRAFT_DSPARK_FAN_OUT=3` override is explicit. Model defaults remain
unchanged. Four fixed prompts are taken from indices 0, 4, 8, and 12 of the
existing Phase-B 16-prompt manifest, with 128 generated tokens per prompt.

The strict correctness gate has failed in the existing AR-versus-Sync baseline:
manifest prompt 12 first diverges at zero-based output position 42. AR gives
tokens 13 and 440 equal log probabilities (-0.7649399042), while Sync prefers
440 (-0.7069147229) over 13 (-0.8319147229). This difference is retained as a
failure, without a numerical-tie waiver or a change to Target configuration.

After restoring the child's IR initialization, forced-JIT matches Sync at all
153 paired real prefixes, including every proposal and accepted outcome.
The earlier native-IR run and its first discrepancy are retained separately.
All four final outputs also match between Sync, forced-JIT, and the new cache
path. Each speculative mode matches AR on three of the four prompts.

Excluding bootstrap, Sync and forced-JIT accept 367 draft tokens over 149 rounds
(2.4631 per round). The cache path accepts 332 over 177 rounds (1.8757 per round),
with 152/177 proposal cache hits. These are four-prompt smoke diagnostics, not
a formal old-versus-new comparison. In the separate shadow run, 88/152 cached
proposals and 320/456 token positions match fresh JIT at the same real prefix.
The first mismatch includes both sides' top logits in the audit artifact.

CPU tests cover both anchor layouts, every depth, recovery exclusion and
vocabulary mapping, actual sampling offsets, mixed hit/miss ordering,
cross-page writes, canonical isolation, reclamation after success/failure, and
diagnostic-independent tie breaking. A single-GPU probe also runs the actual
Qwen DSpark backbone with synthetic Target features across all four depths and
checks canonical KV bytes and temporary allocation counts. This probe does not
establish end-to-end Target parity.

The formal 16-prompt, three-interleaved-repeat performance comparison remains
gated. No old-versus-new acceptance improvement, Sync throughput win, or CUDA
graph end-to-end validation is claimed from these smoke runs. Performance mode
in the runner requires a passed strict correctness audit.

## Artifacts and commands

Local artifact root:
`SSSD_results/dspark_branch_backbone_20260908/`.
It contains the pre-edit source archive, file hashes, dirty diff, per-cell
commands and environments, raw proposals and requests, GPU samples, preserved
failed attempts, correctness audit, and the separate shadow-JIT diagnostic.
The archive includes pre-existing uncommitted source. Only required Async
prerequisites are recorded in baseline commit `108c5a61b`; unrelated Target and
Gemma work remains outside these commits and is covered by the local snapshot.

Run an individual smoke mode (`ar`, `sync`, `async_jit`, or `async_cache`):

```bash
.venv/bin/python benchmarks/replayssm/dspark_branch_backbone_validation.py \
  --output-root SSSD_results/dspark_branch_backbone_20260908/smoke \
  --prompts SSSD_results/benchmarks/phase_b_throughput/phase_b_d3_b1_16x128_20260903T092945Z/models/qwen36_dspark/prompts.json \
  --mode async_cache
```

Use a separate output root and add `--shadow` for a cache/fresh-JIT diagnostic.
Shadow is forbidden in performance mode. Audit both runs with:

```bash
.venv/bin/python benchmarks/replayssm/audit_dspark_branch_backbone.py \
  SSSD_results/dspark_branch_backbone_20260908/smoke \
  --shadow-root SSSD_results/dspark_branch_backbone_20260908/shadow
```

The smoke audit excludes each request's bootstrap from acceptance statistics.
Trace acceptance and bonus-token counts describe verification rounds; the final
round can exceed the API output cutoff. They are not API completion throughput.
Branch row counts and batched forward counts are reported separately, alongside
context projection, candidate evaluation, branch backbone, Markov sampling,
complete branch-build time, and the next proposal wait.
