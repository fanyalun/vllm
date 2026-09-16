# MoE-Skip routing weight smoke test

On 2026-09-16, preserving native top-8 weights improved acceptance for
Qwen3.6-35B-A3B, but slightly reduced aggregate acceptance for Gemma-4-26B-A4B-it.
This four-prompt smoke test does not establish a universal winner.

## Production weight modes

The production MoE-Skip implementation now defaults to preserving native top-k
weights. Set `moe_skip_weight_mode` to `preserve` or `renormalize` in the
speculative config. The same setting controls hierarchical MoE pre-verification;
it does not alter the inner drafter, full Target routing, or shared experts.

```json
{
  "method": "moe_skip",
  "num_speculative_tokens": 16,
  "moe_skip_top_h": 4,
  "moe_skip_weight_mode": "preserve"
}
```

Omitting `moe_skip_weight_mode` selects `preserve`. To recover the previous
behavior, use `"moe_skip_weight_mode": "renormalize"`. The CLI alias is
`--moe-skip-weight-mode preserve` or `--moe-skip-weight-mode renormalize`.

Preserve computes native top-k routing weights, selects the highest-gate top-h
experts, and gathers their weights unchanged. Equal gates retain native expert
order. Expert scaling is preserved, including Gemma's per-expert scale. The
expert kernels receive tensors with width h, so the skipped experts are removed
from dispatch. Renormalize uses the previous direct top-h routing path.
The mode is call-scoped and included in compilation hashes.

`run_weight_ablation.py` now uses this production config directly for fixed h;
it no longer installs the benchmark-only fixed-h router patch. The experimental
top-p path remains a separate worker extension, with weight mode selected by
`--mode`. Historical results below retain their explicitly recorded modes.

Production validation passed 151 targeted tests covering configuration, graph
hashes, CLI alias conflicts, independent and hierarchical routing/state behavior,
native weights and expert scales, tied gates, and invalid graph-capture IDs.
The four CUDA Graph cells (two models, two modes, four prompts each, D=16,
128 output tokens) completed. Eager preserve-mode runs matched the previous
experimental implementation's output tokens and detailed acceptance counters
exactly for all eight tested prompts across both models.

CUDA Graph completion is not strict output equivalence: the two graph weight
modes matched outputs on 4/4 Qwen prompts and 0/4 Gemma prompts; comparison to
the corresponding prior eager mode matched 3/4 Qwen prompts and 0/4 Gemma
prompts. The cause of these graph/eager output differences is not established
by this change. No AR equivalence or performance claim is made.
Actual-model validation covers standalone MoE-Skip; hierarchical integration is
covered by the targeted tests.

Validation artifacts, including the exact-source manifest, per-cell logs and
results, pytest/pre-commit output, and `validation.json`, are in
`benchmark_results/moe_skip_native_weight_modes_4x128_d16_cudagraph_20260916_validated/`.

## Contract

- Four fixed prompts per model: HumanEval, Alpaca, GSM8K, UltraFeedback.
- 128 generated tokens per prompt; greedy, seed 0, ignore EOS.
- Native top-k=8, retained top-h=4, speculative length D=4.
- B=1, TP=1, eager mode, prefix caching off, synchronous scheduling.
- Both methods share Target parameters; only draft routing changes.
- `renormalize`: existing routing normalized over the retained four experts.
- `preserve`: native top-8 routing, then gather the four highest-logit experts
  without changing their weights. Gemma expert scales remain intact.
- No latency comparison, independent AR run, or general losslessness claim.

Acceptance is total accepted draft tokens divided by total proposed draft tokens.
Mean acceptance length is `1 + accepted / verification_steps` and includes the
Target token. Counters include the final speculative cycle at the output limit.

## Results

| Model | Method | Accepted / proposed | Acceptance | Mean acceptance length |
| --- | --- | --- | --- | --- |
| Qwen3.6 | Renormalize | 402 / 460 | 87.39% | 4.496 |
| Qwen3.6 | Preserve | 408 / 432 | 94.44% | 4.778 |
| Gemma4 | Renormalize | 402 / 448 | 89.73% | 4.589 |
| Gemma4 | Preserve | 401 / 452 | 88.72% | 4.549 |

Preserving weights changes acceptance by +7.05 percentage points on Qwen3.6
and -1.02 percentage points on Gemma4.

| Model | Prompt category | Renormalize | Preserve |
| --- | --- | --- | --- |
| Qwen3.6 | HumanEval | 83.33% | 97.12% |
| Qwen3.6 | Alpaca | 82.50% | 91.07% |
| Qwen3.6 | GSM8K | 97.12% | 100.00% |
| Qwen3.6 | UltraFeedback | 87.93% | 90.18% |
| Gemma4 | HumanEval | 93.52% | 89.29% |
| Gemma4 | Alpaca | 89.29% | 98.08% |
| Gemma4 | GSM8K | 89.29% | 89.29% |
| Gemma4 | UltraFeedback | 87.07% | 79.84% |

All eight paired final outputs match token for token. Each of the 16 requests
emitted 128 tokens. Detailed per-cycle counters were validated with the existing
`validate_metrics` helper.

The actual fused and Gemma custom routers passed a synthetic GPU audit: same
expert sets, exactly preserved native weights, unchanged full-width Target
routing, and the expected retained-mass relationship between the two methods.
Pre-commit checks passed for both Python files.

## Reproduction and artifacts

The runner uses the local four-prompt datasets at
`benchmark_results/moe_skip_top_p_4x128_20260915/{model}/dataset.jsonl` and model
paths from `run_static_budget.py`. Those local datasets and checkpoints are
required and are not bundled with these scripts.

```bash
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/benchmarks/moe_skip:$PWD"
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
for mode in renormalize preserve; do
    .venv/bin/python benchmarks/moe_skip/run_weight_ablation.py \
        --model qwen36 --mode "$mode" \
        --output "benchmark_results/weight_ablation_repeat/qwen36/$mode"
done
```

Use `--model gemma4` and a corresponding output directory for Gemma4.
Original local artifacts are in
`benchmark_results/moe_skip_weight_ablation_4x128_20260916/`: `summary.csv`,
`paired_samples.csv`, router audit, dataset copies, source snapshots, manifest,
per-cell configs, logs, token IDs, detailed counters, and completion markers.

AI assistance was used to implement and run this benchmark and write this report.

## Expanded run: 16 prompts and D=16

The follow-up on 2026-09-16 uses four prompts from each of the same four
categories, D=16, and otherwise retains h=4, 128 output tokens, greedy sampling,
B=1, TP=1, and eager execution. All 64 requests completed, producing 8192 output
tokens. Both modes use identical prompts within each model.

| Model | Method | Accepted / proposed | Acceptance | Mean acceptance length |
| --- | --- | --- | --- | --- |
| Qwen3.6 | Renormalize | 1980 / 3104 | 63.79% | 11.206 |
| Qwen3.6 | Preserve | 2011 / 2464 | 81.62% | 14.058 |
| Gemma4 | Renormalize | 1939 / 3440 | 56.37% | 10.019 |
| Gemma4 | Preserve | 1937 / 3536 | 54.78% | 9.765 |

Preserving weights improves Qwen3.6 by 17.83 percentage points; it wins on
15 of 16 prompts and loses on one. Gemma4 decreases by 1.59 percentage points;
preserve wins on five prompts, loses on four, and ties on seven. Aggregates are
ratios of total counters, so prompt win counts do not determine the aggregate.
All 32 paired final outputs match token for token.

| Model | Domain (four prompts each) | Renormalize | Preserve | Change (pp) |
| --- | --- | --- | --- | --- |
| Qwen3.6 | HumanEval | 58.37% | 79.42% | +21.05 |
| Qwen3.6 | Alpaca | 61.75% | 77.08% | +15.33 |
| Qwen3.6 | GSM8K | 76.37% | 91.25% | +14.88 |
| Qwen3.6 | UltraFeedback | 61.25% | 79.81% | +18.56 |
| Gemma4 | HumanEval | 56.37% | 58.73% | +2.36 |
| Gemma4 | Alpaca | 65.16% | 63.67% | -1.49 |
| Gemma4 | GSM8K | 69.03% | 64.49% | -4.54 |
| Gemma4 | UltraFeedback | 42.50% | 39.81% | -2.69 |

The probability of accepting the entire 16-token draft prefix changes from
40.21% to 66.23% on Qwen3.6, and from 46.98% to 44.80% on Gemma4. These
position-wise rates count accepted prefixes over rounds proposing that position;
they are not conditional agreement given an accepted previous token.

This supports preserving weights for Qwen3.6 under the tested configuration.
Gemma4 shows mixed per-prompt effects and slightly favors renormalization overall.
No independent AR baseline or performance comparison was run. Sample count and
D changed together relative to the first smoke, so differences between the two
experiments cannot be attributed to D alone.

Reproduce using the environment exports above, with the expanded dataset and
additional runner arguments:

```bash
for mode in renormalize preserve; do
    .venv/bin/python benchmarks/moe_skip/run_weight_ablation.py \
        --model qwen36 --mode "$mode" --samples 16 --draft-length 16 \
        --dataset benchmark_results/moe_skip_static_budget_16x512_20260914/qwen36/dataset.jsonl \
        --output "benchmark_results/weight_ablation_d16_repeat/qwen36/$mode"
done
```

Use `gemma4` consistently for the model, dataset directory, and output directory
to reproduce that model. Existing run directories are protected from overwrite.
Local artifacts are in
`benchmark_results/moe_skip_weight_ablation_16x128_d16_20260916/`: frozen datasets
and source snapshots, manifest and source diff, logs, full token outputs and
per-cycle counters, `summary.csv`, `category_summary.csv`, `paired_samples.csv`,
`position_acceptance.csv`, corresponding JSON files, and completion markers.
The artifact-local `analyze.py` revalidates counters and regenerates the summaries.

## Top-p=0.7: same 16 prompts and D=16

The next run replaces fixed h=4 with the smallest native top-8 gate-probability
prefix reaching 0.7. The dataset fingerprints match the previous 16-prompt run.
All other parameters remain the same. Expert selection uses gate probabilities
before Gemma expert scaling. The preserve variant retains the original native
weights; the renormalize variant divides them by the retained gate mass. Both
variants retain Gemma's expert scales.

| Model | Method | Accepted / proposed | Acceptance | Mean length | Mean experts |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 | Renormalize | 1978 / 2848 | 69.45% | 12.112 | 4.869 |
| Qwen3.6 | Preserve | 2012 / 2352 | 85.54% | 14.687 | 4.871 |
| Gemma4 | Renormalize | 1939 / 3120 | 62.15% | 10.944 | 4.975 |
| Gemma4 | Preserve | 1935 / 3104 | 62.34% | 10.974 | 4.975 |

Preserving weights improves Qwen3.6 by 16.09 percentage points: 11 prompt wins,
two losses, three ties. Gemma4 improves by only 0.19 percentage points: five wins,
four losses, seven ties. The small Gemma difference does not establish a stable
advantage. Mean experts is weighted over actual draft token-layer routing events;
the two modes can have different intermediate draft states and routing counts.

| Model | Domain (four prompts each) | Renormalize | Preserve | Change (pp) |
| --- | --- | --- | --- | --- |
| Qwen3.6 | HumanEval | 59.08% | 86.02% | +26.94 |
| Qwen3.6 | Alpaca | 64.23% | 81.76% | +17.53 |
| Qwen3.6 | GSM8K | 84.29% | 92.28% | +7.99 |
| Qwen3.6 | UltraFeedback | 75.46% | 82.73% | +7.27 |
| Gemma4 | HumanEval | 64.49% | 68.32% | +3.83 |
| Gemma4 | Alpaca | 68.61% | 65.43% | -3.19 |
| Gemma4 | GSM8K | 75.15% | 75.46% | +0.30 |
| Gemma4 | UltraFeedback | 47.08% | 47.08% | +0.00 |

All 64 requests completed and emitted exactly 128 tokens each. All 32 pairs
match across weight modes, and all 64 outputs match the corresponding previous
h=4 outputs. Detailed per-cycle counters and completion markers passed the
artifact analysis checks. The expanded `check_top_p.py` passed 64 GPU reference
cases, including scaled weights, uniform gate ties, p=1 identity, identical
selected experts between modes, and fused-expert outputs with skipped IDs.
Two unsupported aligned-assignment shapes were rejected as expected.

With preserved weights, top-p raises acceptance over h=4 from 81.62% to 85.54%
for Qwen3.6, and from 54.78% to 62.34% for Gemma4, while selecting approximately
4.87 and 4.97 experts instead of four. This is a different compute budget.
The experimental implementation masks skipped IDs while retaining routing width
eight and supports the existing naive-assignment path. No performance comparison
or independent AR baseline was run.

Add `--expert-top-p 0.7` to the preceding D=16 reproduction command and use a new
output directory. The default without this flag remains fixed h=4. The original
top-p worker still defaults to renormalization when no weight mode is supplied.
Local artifacts are in
`benchmark_results/moe_skip_top_p07_weight_ablation_16x128_d16_20260916/`, including
the same summary files as above, per-layer `expert_budgets.csv`, comparisons to
the h=4 outputs, GPU check results, and frozen source and dataset copies.
