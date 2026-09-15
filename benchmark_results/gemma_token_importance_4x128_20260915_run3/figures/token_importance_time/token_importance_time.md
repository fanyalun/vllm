# token_importance_time

Source: ../../summary.csv; protocol and limitations: ../../README.md. Gemma4, A100 80GB PCIe, TP=1/B=1, greedy, 4x128; MTP D=4, four rounds, Target capacity=20. Aggregate wall-time ratios, no error bars. Full scoring/selection overhead is included. AR is the mean of two fresh async controls. Actual retained edge counts determine mean h; the expert-pool fraction is rounded up.

Reproduce with `.venv/bin/python benchmarks/hierarchical/analyze_token_importance.py benchmark_results/gemma_token_importance_4x128_20260915_run3`.
