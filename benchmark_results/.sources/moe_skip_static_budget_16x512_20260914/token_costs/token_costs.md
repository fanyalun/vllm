# Unit token costs

Source: ../summary.csv. Each cell is 16 fixed short multicategory prompts x 512 output tokens, B=1, TP1, greedy, CUDA Graph. Left: summed request wall time / actual emitted accepted draft tokens. Right: summed request wall time / all 8192 final output tokens. Both are milliseconds and include prefill, draft, target verification, state and API overhead. Final accepted-tail clipping is accounted for; the left denominator excludes correction/bonus tokens.

Lower is better. Color limits are shared across models within each column. Red outlines identify the minimum measured cost for that model/metric. Each cell ran once; no uncertainty estimate is implied. Strict AR output-parity failures remain in ../audit.json and do not support a lossless speedup claim. See ../readme.md for full contract.

Reproduce with benchmarks/moe_skip/analyze_static_budget.py and the --run-dir argument pointing to the parent of this directory.
