# Individual expert weight threshold smoke

This benchmark compares fixed h=4 with retaining every native top-8 expert whose
normalized gate probability is at least p, for p=0.1 and p=0.0625. This is an
individual weight threshold, not cumulative top-p selection.

The probabilities are normalized within the model's native top-8, before
Gemma's per-expert scaling. Retained experts use their original routing weights,
including expert scales. There is no post-pruning renormalization and no top-1
fallback. A normalized distribution over eight experts always has a maximum
probability of at least 0.125, so the requested thresholds naturally retain at
least one expert.

The smoke uses Qwen3.6-35B-A3B and Gemma-4-26B-A4B-it, with the same four prompts
per model as the preceding weight-mode smoke: one each from HumanEval, Alpaca,
GSM8K, and UltraFeedback. All methods use D=16, 128 output tokens, greedy,
seed=0, B=1/TP=1, eager execution, and prefix caching off.

Acceptance is total accepted draft tokens divided by total proposed draft tokens.
Mean acceptance length is `1 + accepted / verification_steps`, including the
Target token. These counters include the final speculative cycle at the output
limit. Expert counts cover actual draft token-layer routing events, including
draft work that may not subsequently be verified. Target and shared experts are
excluded. Average skipped experts is `8 - mean_retained_experts`; skip fraction
is that average divided by eight. Raw totals also depend on the number of draft
forwards, so compare averages alongside totals.

The adaptive benchmark masks skipped IDs while retaining routing width eight;
the h=4 control uses the production width-four path. The implementation supports
the existing naive expert-assignment path and rejects unsupported larger shapes.
This experiment measures acceptance and expert selection, not speedup.

## Results

| Model | Method | Accepted / proposed | Acceptance | Mean length | Mean kept | Mean skipped | Skip fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6 | h4 | 499 / 624 | 79.97% | 13.795 | 4.000 | 4.000 | 50.00% |
| Qwen3.6 | p=0.1 | 501 / 608 | 82.40% | 14.184 | 4.704 | 3.296 | 41.21% |
| Qwen3.6 | p=0.0625 | 507 / 592 | 85.64% | 14.703 | 7.387 | 0.613 | 7.66% |
| Gemma4 | h4 | 492 / 736 | 66.85% | 11.696 | 4.000 | 4.000 | 50.00% |
| Gemma4 | p=0.1 | 499 / 720 | 69.31% | 12.089 | 5.087 | 2.913 | 36.41% |
| Gemma4 | p=0.0625 | 506 / 640 | 79.06% | 13.650 | 7.497 | 0.503 | 6.29% |

At p=0.1, acceptance improves by 2.43 and 2.46 percentage points over h4,
with fewer skipped experts. At p=0.0625, most native experts are retained;
acceptance improves by 5.67 and 12.21 percentage points over h4.

Every adaptive output matches its paired h4 output (16/16 comparisons).
Both fresh h4 controls match the previous eager outputs and acceptance counters
(8/8 samples). No independent autoregressive baseline was run. Every observed
threshold routing event retains at least one expert without a fallback.

## Follow-up: h4, p=0.125, and h3

The follow-up comparison retains h4, p=0.125, and h3 only, using the same
four prompts, D=16, and 128 output tokens per model. h4 and p=0.125 are reused
from the preceding measurements; h3 is newly measured with `--top-h 3`.

| Model | Method | Acceptance | Mean length | Mean kept | Mean skipped |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 | h4 | 79.97% | 13.795 | 4.000 | 4.000 |
| Qwen3.6 | p=0.125 | 75.61% | 13.098 | 2.838 | 5.162 |
| Qwen3.6 | h3 | 71.95% | 12.512 | 3.000 | 5.000 |
| Gemma4 | h4 | 66.85% | 11.696 | 4.000 | 4.000 |
| Gemma4 | p=0.125 | 65.03% | 11.404 | 2.913 | 5.087 |
| Gemma4 | h3 | 57.29% | 10.167 | 3.000 | 5.000 |

All eight new h3 outputs match h4. The threshold retains slightly fewer experts
on average than h3 while improving acceptance in this smoke. This does not
establish a runtime advantage. The selected comparison and its provenance are
in `benchmark_results/moe_skip_h3_h4_threshold0125_4x128_d16_20260916/`.

## Reproduction

Use the existing local model paths from `run_static_budget.py` and the frozen
dataset in the artifact directory below. Model weights and prompts are not
bundled with the scripts.

```bash
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/benchmarks/moe_skip:$PWD"
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python benchmarks/moe_skip/run_weight_ablation.py \
    --model qwen36 --mode preserve --samples 4 --draft-length 16 \
    --track-expert-counts --expert-min-weight 0.1 \
    --dataset benchmark_results/moe_skip_weight_threshold_4x128_d16_20260916/qwen36/dataset.jsonl \
    --output benchmark_results/weight_threshold_repeat/qwen36/threshold_01
```

Repeat with `--expert-min-weight 0.0625` and a new output directory. Omit
`--expert-min-weight` for the h=4 control, retaining `--track-expert-counts`.
Use `gemma4` in the model and paths for that model. The existing cumulative
selection remains available through the mutually exclusive `--expert-top-p`.

GPU reference validation:

```bash
.venv/bin/python benchmarks/moe_skip/check_top_p.py
```

This passed 104 cases spanning both selection rules, scaled weights, ties,
threshold boundaries, and complete pruning without a fallback. Two unsupported
assignment shapes were rejected.

The local artifact root is
`benchmark_results/moe_skip_weight_threshold_4x128_d16_20260916/`. It contains
frozen datasets and source snapshots, source diff, launch and analysis scripts,
manifest, logs, token outputs, detailed acceptance counters, expert histograms,
summary CSV/JSON/Markdown, and completion markers.

AI assistance was used to implement, validate, and run this benchmark.
