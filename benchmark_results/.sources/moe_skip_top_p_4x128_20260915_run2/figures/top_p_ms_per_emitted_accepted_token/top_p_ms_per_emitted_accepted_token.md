# ms / accepted token

Source: ../../summary.csv.

Qwen3.6 D=8; Gemma4 D=4; B=1; greedy; four prompts (one per category), 128 output tokens each; A100 80GB, TP=1. All requests are warmed up before measurement. Wall time includes prefill, decode, API and detailed speculative metrics. Top-p uses native top-8 probability mass with retained-weight renormalization, h=1..8. Dynamic routing and histogram overhead are included. p=1 is an implementation control. AR is the mean of start/end async measurements. No error bars; this is a small exploratory run. Accepted-token time divides the entire wall time by actually emitted accepted draft tokens after final-step clipping; AR has no such denominator. Mean h is weighted over measured draft token-layer invocations.

Reproduce: `.venv/bin/python benchmarks/moe_skip/analyze_top_p_probe.py benchmark_results/moe_skip_top_p_4x128_20260915_run2`.
