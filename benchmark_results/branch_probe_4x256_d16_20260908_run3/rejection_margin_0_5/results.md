# First low-margin position versus actual first rejection

Offline reanalysis of the same four prompts per model, 256 output tokens,
B=1, D=16, top-h=4, greedy, eager. The new threshold is strictly margin < 0.5.
The preceding threshold-1 artifacts are preserved.

For each verified draft, predict at most one position: the first in-budget
position whose raw Top-1 minus Top-2 logit margin is below the threshold.
Ground truth is the actual first rejection, if any within the output budget.
Both positions are one-based within that draft. Missing predictions and
missing rejections are blank in `proposal_predictions.csv`.

An exact position match is TP. A prediction at a different position is FP,
and also FN if the draft has a real rejection. A draft with a real rejection
but no prediction is FN. Correctly predicting no position for a draft without
rejection does not contribute to precision or recall.

Precision = exact matches / drafts with a prediction.
Recall = exact matches / drafts with an actual first rejection.

| Metric | Qwen3.6 | Gemma4 |
| --- | ---: | ---: |
| Verified drafts | 108 | 75 |
| Drafts with a first margin < 0.5 position | 50 | 15 |
| Drafts with actual first rejection | 67 | 20 |
| Exact position matches (TP) | 20 | 6 |
| Wrong predictions (FP) | 30 | 9 |
| Missed rejection positions (FN) | 47 | 14 |
| Precision | 40.00% | 40.00% |
| Recall | 29.85% | 30.00% |

## Per-draft outcomes

| Outcome | Qwen3.6 | Gemma4 |
| --- | ---: | ---: |
| Exact position match | 20 | 6 |
| Prediction earlier than actual rejection | 11 | 5 |
| Prediction later than actual rejection | 18 | 1 |
| Prediction but no actual rejection | 1 | 3 |
| Actual rejection but no prediction | 18 | 8 |
| Neither prediction nor rejection | 40 | 52 |

All rows sum to the 108/75 verified drafts. Four further proposals per model,
one per request, were generated but never verified after the request finished;
they have no rejection ground truth and are excluded.

## Separate the threshold change from the selection-rule change

| Selection rule | Threshold | Qwen precision | Qwen recall | Gemma precision | Gemma recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mark every low-margin position | 1.0 | 20.56% | 55.22% | 24.59% | 75.00% |
| Mark every low-margin position | 0.5 | 27.47% | 37.31% | 31.43% | 55.00% |
| Mark only first low-margin position | 1.0 | 31.34% | 31.34% | 35.00% | 35.00% |
| Mark only first low-margin position | 0.5 | 40.00% | 29.85% | 40.00% | 30.00% |

At threshold 0.5, marking every low-margin position gives 91/35 predicted
positions and 25/11 exact first-rejection hits. Restricting to the earliest
low-margin position reduces predictions to 50/15 and hits to 20/6.

## Meaning of draft Top-2 hit

This is a SINGLE TOKEN comparison at the actual first rejection position j:
Target's emitted correction token equals draft's second candidate at j.
It does not compare whole sequences, count later matching positions, or measure
suffix acceptance. For example, Top-2 hit at j=3 says nothing about j=4...D.

Across all first rejections, the previous counts remain 51/67 (76.12%) for
Qwen and 12/20 (60%) for Gemma, independent of the margin threshold.
Conditional on the actual rejection's own margin being < 0.5, the single-token
Top-2 hit counts are 16/25 (64%) and 8/11 (72.73%). This latter condition does
NOT require that the position is also the FIRST low-margin position in its draft.

## Validation and reproduction

All reached draft tokens and rejection corrections were checked against the
emitted output. The per-draft outcome counts independently agree with the
position-level confusion matrix. Rerunning threshold 1 reproduces every prior
confusion-matrix value and the prior correction/event CSVs exactly.
There are only four prompts per model; correlated draft events are not
independent prompt samples.

```bash
.venv/bin/python benchmarks/moe_skip/analyze_rejection_margin.py \
  benchmark_results/branch_probe_4x256_d16_20260908_run3 \
  --threshold 0.5 \
  --output benchmark_results/branch_probe_4x256_d16_20260908_run3/rejection_margin_0_5
```

- `proposal_predictions.csv`: all 183 per-draft predicted/actual position pairs.
- `proposal_summary.csv`: per-draft outcomes and exact-position precision/recall.
- `detection.csv`: all-position and first-position confusion matrices.
- `correction_ranks.csv`: single-token rank statistics at actual first rejection.
- `audit.json`, `regression_audit.json`: source hashes and validation results.
- Sibling `rejection_margin_first_1/`: threshold-1 first-position comparison.
