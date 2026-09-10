# Forward stage comparison

Source: ../stages.csv and ../totals.csv. B=1, TP=1, A100 80GB, BF16, top-8/full versus top-4/skip. Three fixed prefixes per model, 20 alternating replays per mode and prefix. Bars show the instrumented CUDA Graph interval partition in ms. Attention and GDN include their projections and internal normalization. Shared overlap is the union of shared/routed or shared/router overlap; inclusive layer timings are in ../layers.csv. Other includes residuals, argmax, unwrapped operators, graph/event overhead and gaps. No error bars; replays are not independent prompts. See ../audit.json for measurement boundaries.

Reproduce: `.venv/bin/python benchmarks/hierarchical/summarize_forward_stages.py benchmark_results/forward_stages_20260910`
