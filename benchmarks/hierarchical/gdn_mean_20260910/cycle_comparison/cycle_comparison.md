# Mean GDN comparison

Source: `benchmarks/hierarchical/gdn_mean_20260910/summary.csv`. Qwen3.6-35B-A3B, TP=1/B=1, inner D=4/N=4, top-8 Target and top-4 Pre-Verify, four fixed prompts, 256 output tokens each, greedy sampling. MTP modes share GPU 1; DSpark modes share GPU 0. Compare modes within each inner method.

(a) uses the second uninstrumented pass after all prompts were warmed. (b-d) use instrumented proposal-to-next-Target cycles, excluding the prefill-associated cycle and proposals with no subsequent verification. (c) clips the final cycle to tokens actually returned under the output budget. (d) is accepted/proposed at Target, before output-budget clipping. Bars aggregate four prompts; no confidence intervals or significance claim. AR output equivalence is not certified; see results.md.

Reproduce with `benchmarks/hierarchical/plot_gdn_mean.py --input benchmarks/hierarchical/gdn_mean_20260910/summary.csv --output benchmarks/hierarchical/gdn_mean_20260910/cycle_comparison` using `.venv/bin/python`.
