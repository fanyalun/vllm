# Hierarchical first-rejection distribution diagnostic

Compare Draft with Pre-Verify at each inner round's first rejection, then compare
Pre-Verify with Target at the outer cycle's first rejection. The benchmark keeps
the greedy sampling outputs and fixed four-round policy unchanged. It does not
implement an early-stop policy or measure a speedup.

## Protocol

- Gemma-4-26B-A4B-it and its assistant, TP1/B1, greedy, MTP D4, four inner rounds,
  outer capacity 20, max model length 1024.
- Reuse the four prompts from `gemma_round_decay_4x128_20260915`, with four warmup
  requests followed by four measured requests, each generating 128 tokens.
- Run uninstrumented and instrumented h4 controls. Analysis fails
  closed if output tokens or per-step acceptance/scheduled counters differ.
- Import the worker extension before graph capture. Capture Draft logits into
  persistent step-indexed buffers, and Pre-Verify logits into width-indexed
  buffers. Read these buffers after replay. Retain the original sampling result.
- At runtime assert that captured Draft and Pre-Verify logits reproduce their
  actual token choices, that emitted candidates match Pre-Verify Top-1, and that
  Target logits reproduce the recorded accepted prefix.
- Exclude proposals that never reach Target, and pair measured cycles with
  per-request accepted/scheduled counters.

```bash
.venv/bin/python benchmarks/hierarchical/run_disagreement.py \
  benchmark_results/gemma_disagreement_4x128_20260915 --gpu 1
.venv/bin/python benchmarks/hierarchical/analyze_disagreement.py \
  benchmark_results/gemma_disagreement_4x128_20260915
.venv/bin/python -m pytest tests/benchmarks/test_hierarchical_measurement.py -q
```

Use a fresh output directory. No dependencies are installed. Capture adds model
head work, synchronization, CPU analysis and I/O; timings are diagnostic only.

## Distribution metrics

Normalize the complete vocabulary with softmax at T=1, even though token
generation uses greedy decoding. These probabilities describe competition
between logits; they are not stochastic acceptance probabilities for this run.

Save both sides' Top-8 token IDs, logits and probabilities, entropy, opposing
Top-1 rank and logit gap, Top-1/Top-2 margin, Jensen-Shannon divergence in nats,
total variation, and both directional KL divergences. Distribution calculations
use float64 on the CPU. JS and TV compare full distributions rather than only
their top candidates. No raw difference between unnormalized logits across
models is treated as a distribution distance.

Predeclared descriptive groups:

- Near: each side's Top-1 is in the other's Top-2, and both opposing-Top-1 gaps
  are at most 1 nat.
- Far: either opposing-Top-1 rank exceeds 8 or its logit gap exceeds 2 nats.
- Middle: all other disagreements.

Also report near-group sensitivity at gaps 0.25, 0.5 and 1.0. These are descriptive
bins, not trained or validated predictor thresholds. Continuous metrics remain
available for alternate analyses.

## Outcomes and denominators

If an inner round starts at candidate offset `s` and accepts `a` Draft tokens,
its correction occupies `j = s + a`. Let outer acceptance be `A`, and scheduled
candidate length be `L`.

- Target reached correction: `j < L` and `A >= j`.
- Correction accepted: `A > j`, conditional on reaching it.
- Scheduled suffix after correction: `max(0, L - j - 1)`.
- Accepted suffix: `max(0, A - j - 1)`, conditional on reaching correction.
- Not reached is missing outcome, never a negative local label.

Report all inner first-rejection events and, separately, the earliest rejecting
round per outer cycle. Report empty suffixes separately from full suffix
acceptance. Preserve output-cap clipping and request/category identity.

The suffix is the actual Pre-Verify-corrected continuation. It is not the
discarded original Draft suffix. Outer-first-rejection comparisons are on the
candidate prefix up to that rejection. They do not prove later positions would
be wrong after Target correction. Shared token history also does not imply
identical hidden-state conditioning across the three models.

## Evidence limits

Four requests per method are a pilot, and cycles within a request are correlated.
Report source/model/dataset fingerprints, output control differences, and any
failed invariant. Existing Gemma AR/speculative parity limitations remain
unresolved. Descriptive association, an independently tested predictor, and a
measured early-stop speedup are separate results.

AI assistance was used.

## Confidence-only held-out validation

Use `--dataset <dataset.jsonl> --confidence-only` to test new h4 prompts with
Pre-Verify confidence capture. Each request still generates 128 tokens. A sibling
`hypotheses.json` is copied with the dataset when present.

This mode warms up the model, runs the uninstrumented control, then installs a
CPU-reading hook through RPC in the same engine. It reads existing logits after
graph replay without modifying Draft sampling or captured CUDA graphs. The
analysis requires exact output and per-step counter parity with that control.

```bash
.venv/bin/python benchmarks/hierarchical/run_disagreement.py <fresh_directory> \
  --gpu 1 --dataset <dataset.jsonl> --confidence-only
.venv/bin/python benchmarks/hierarchical/analyze_confidence.py \
  <pilot_directory> <fresh_directory>
```

The confidence analyzer checks disjoint prompt identities, fixed thresholds,
token-source rejection rates, correction margin groups, request-cluster bootstrap
intervals, and first-trigger stopping rules on the original trajectories. Offline
suffix cuts are not measured speedups or guarantees about shortened Target calls.
