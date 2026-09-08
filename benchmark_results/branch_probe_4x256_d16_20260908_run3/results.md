# MoE-Skip low-margin branch sensitivity: 4 x 256

Both models completed four prompts and 1,024 normal output tokens. Qwen3.6
produced 171 eligible branch pairs and Gemma4 produced 61. Changing a low-margin
token to its highest-scoring alternative usually changed the remaining draft:
only 9.36% / 8.20% of the tested suffixes were entirely identical.

| Metric | Qwen3.6 | Gemma4 |
| --- | ---: | ---: |
| Normal proposal calls | 112 | 79 |
| Eligible branch pairs | 171 | 61 |
| Entire suffix identical | 16 (9.36%) | 5 (8.20%) |
| Partial aligned agreement | 30 (17.54%) | 11 (18.03%) |
| No aligned token agreement | 125 (73.10%) | 45 (73.77%) |
| Mean compared suffix length | 7.94 | 8.89 |
| Mean consecutive matching tokens after divergence | 0.784 | 0.770 |
| At least one consecutive matching token | 16.37% | 11.48% |
| Aligned agreement across all suffix tokens | 15.98% | 14.02% |
| Zero-margin branch pairs | 15 | 5 |

## Configuration and interpretation

- Each model: first four interleaved prompts, one each from HumanEval, Alpaca,
  GSM8K and UltraFeedback, 256 output tokens, B=1, TP=1, seed=0, greedy,
  ignore EOS, no chat template, no prefix caching.
- Qwen3.6 on GPU 1; Gemma4 on GPU 0. A100 80GB, eager execution.
- MoE-Skip retains top-h=4 experts, D=16. Trigger: raw Top-1 minus Top-2
  logit margin strictly below 1. Compare the remaining D-token draft suffix,
  capped at the normal request's 256-token budget.
- Both branches start from the same canonical Target prefix. The second branch
  forces the alternative at exactly one position and continues draft greedy.
  No recursive branching, no Target verification of alternative branches.
- All eligible low-margin positions are included, not only true first rejections
  or positions where Target selects Top-2. This is a sensitivity proxy, not an
  acceptance-rate estimate, and not an end-to-end performance experiment.
- Tokens are compared at identical offsets; this is not an edit-distance score.
  The forced branching token is excluded from every suffix statistic.
- Offset curves use only suffixes long enough to reach each offset. Denominators
  are in `offset_metrics.csv`; changing denominators can make curves rise.
- Four prompts per model are a smoke experiment. Branch pairs are correlated
  within prompts and proposals, and should not be treated as independent samples.

## Example

Gemma4, margin 0.625, 15-token suffix entirely identical:

```text
Top-1: stated.\n2.  **May:** The number of clips is not explicitly
Top-2: given.\n2.  **May:** The number of clips is not explicitly
```

Gemma4, margin 0.625, 15-token suffix with no aligned agreement:

```text
Top-1: # Check if any two numbers in the list are closer to each other than the
Top-2: for i in range(len(numbers)):\n        for j in range(
```

The first example supports testing selective suffix reuse. The aggregate does
not support treating `margin < 1` alone as a reliable suffix-stability signal.
Agreement cannot prove Target acceptance, and disagreement cannot prove the
original suffix has zero Target-acceptable tokens after a real correction.

## Validation and artifacts

Every event passed shared-prefix, forced-alternative and in-budget suffix
checks. Every proposal with a branch passed exact replay of its original full
draft before normal Target verification continued. Per-cell completion markers
and audits record 4 x 256 outputs. The existing local runtime was not edited;
source hashes are in `source_provenance.json`.

Independent probe-disabled controls matched all 8/8 full output sequences:
4/4 for Qwen3.6 and 4/4 for Gemma4, each exactly 256 tokens. The final aggregate
audit records their exact token comparison. Pre-commit
checks passed for all four benchmark source/documentation files.

- `branch_comparison.png` and `.pdf`: outcome and offset plots.
- `summary.csv` and `offset_metrics.csv`: aggregate values and denominators.
- `qwen36/result.json` and `gemma4/result.json`: every paired branch and output.
- `examples.json`: representative entire/partial/no-match examples.
- `*/samples.json`, `*/contract.json`: exact prompts and configuration.
- `qwen_trace/` and `gemma_trace/`: normal Target verification traces.
- `source_provenance.json`: runtime hashes and preserved failed-attempt paths.

See `benchmarks/moe_skip/branch_probe_readme.md` for reproducible commands and
the explicit tied-logit rule. The benchmark depends on the existing local
MoE-Skip runtime, which is not included in this benchmark-only publication.
