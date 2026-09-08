# Margin detection and first-rejection correction ranks

Offline analysis of the completed Qwen3.6 and Gemma4 branch-probe traces.
Each model has four prompts, 256 output tokens, B=1, D=16, top-h=4,
greedy decoding and eager execution. No new model inference was run.

## Detect the actual first rejection before verification

Predict positive at every in-budget draft position whose raw logit margin is
strictly below 1. A true positive is the actual first rejection position:
`draft_position == accepted_draft_tokens + 1`. Positions after the first
rejection are negatives for this location-detection task, even if their local
draft and Target argmax disagree. They are not labeled accepted tokens.

Precision = TP / (TP + FP). Recall = TP / (TP + FN).

| Model | Positions | Margin < 1 | First rejections | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 | 1633 | 180 | 67 | 37 | 143 | 30 | 20.56% | 55.22% |
| Gemma4 | 1177 | 61 | 20 | 15 | 46 | 5 | 24.59% | 75.00% |

This scope matches preemptively marking positions throughout a draft. For two
other common definitions, the same traces give:

| Definition | Qwen precision | Qwen recall | Gemma precision | Gemma recall |
| --- | ---: | ---: | ---: | ---: |
| Only accepted prefix plus first rejection | 45.68% | 55.22% | 37.50% | 75.00% |
| All positions, local draft/Target Top-1 mismatch | 46.11% | 61.03% | 47.54% | 54.72% |

The first alternative excludes the unvisited suffix. The second includes local
mismatches on the original draft prefix even after an earlier rejection; those
are not additional actual rejection events.

## Correction ranks at actual first rejection

Top-1 means the actual draft greedy argmax. Top-2 means its highest-scoring
alternative; this explicitly handles tied logits and unstable `topk` ordering.
All correction tokens were checked against the emitted output token at the
corresponding position. The three categories are mutually exclusive.

| Model | First rejections | Draft Top-1 | Draft Top-2 | Neither |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 | 67 | 0 (0%) | 51 (76.12%) | 16 (23.88%) |
| Gemma4 | 20 | 0 (0%) | 12 (60.00%) | 8 (40.00%) |

Top-1 is necessarily zero at a genuine greedy rejection: matching the proposed
Top-1 would accept that position. This is not a result to extrapolate to random
sampling acceptance rules.

Conditional on first rejection AND margin < 1:

| Model | First rejections | Draft Top-1 | Draft Top-2 | Neither |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 | 37 | 0 (0%) | 26 (70.27%) | 11 (29.73%) |
| Gemma4 | 15 | 0 (0%) | 9 (60.00%) | 6 (40.00%) |

Consequently, only 26/180 (14.44%) and 9/61 (14.75%) of all low-margin
positions are both the actual first rejection and corrected to draft Top-2.
These are location/gating statistics, not suffix survival or speedup estimates.

## Boundary handling, validation and reproduction

Exclude trace rows beyond the 256-token request output budget. Include the final
draft position even when there is no remaining suffix: it can still be rejected.
This explains Qwen's 180 low-margin positions versus 171 branch pairs in the
earlier suffix probe, which required a nonempty remaining suffix.

Check contiguous verify steps and draft positions, agreement of acceptance
counts within each round, and correspondence of every reached draft position
with actual output. An accepted draft token must equal the Target argmax; the
first rejected token must differ. Audit trace hashes and excluded row counts.

Four prompts per model are a preliminary sample. Events within a prompt are
correlated; the 67/20 rejection events are not independent prompt samples.

```bash
.venv/bin/python benchmarks/moe_skip/analyze_rejection_margin.py \
  benchmark_results/branch_probe_4x256_d16_20260908_run3
```

`detection.csv` contains all three confusion matrices; `correction_ranks.csv`
contains both correction-rank conditionings; `first_rejections.csv` contains
all 87 actual events, margins, positions and token IDs; `audit.json` records
validation and source hashes.
