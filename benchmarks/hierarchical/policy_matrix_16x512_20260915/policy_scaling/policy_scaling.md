# h4 policy scaling

Source: `benchmark_results/gemma_h4_policy_16x512_20260915_run3/summary.json`, audited 20-cell matrix.

Gemma-4-26B-A4B-it + assistant; h4, D4, at most four inner rounds; 16 raw prompts, 512 output tokens each, TP1, greedy, ignore_eos, synchronous scheduling, prefix caching disabled. Panel (a) includes prefill and decode in generate wall time, excluding startup and warmup. Panel (b) divides accepted candidate tokens by verified candidates, excluding Target bonus; AR has no candidate rate. One warm measurement per cell; no error bars or significance claim. Output equivalence is reported separately in output_comparisons.csv.

The acceptance axis spans 85–97% to resolve differences.

Reproduce with `.venv/bin/python -m benchmarks.hierarchical.plot_policy_matrix <run_directory> <output_directory>`.
