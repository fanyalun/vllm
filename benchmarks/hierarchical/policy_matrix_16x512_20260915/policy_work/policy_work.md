# h4 stopping and executed work

Source: `benchmark_results/gemma_h4_policy_16x512_20260915_run3/summary.json`, audited 20-cell matrix.

Same Gemma h4, D4, four-round, 16×512 protocol as policy_scaling. Panel (a) sums remaining request rounds omitted when a policy fires; it is neither wall-time saving nor a counterfactual measurement. Panel (b) counts executed batch inner rounds, one Pre-Verify per call; each call may serve multiple requests. These units differ. Counts cover all 16 measured requests, excluding warmup. One measurement per cell; no error bars. Output trajectories differ, so cross-policy differences do not isolate a causal stopping effect.

Reproduce with `.venv/bin/python -m benchmarks.hierarchical.plot_policy_matrix <run_directory> <output_directory>`.
