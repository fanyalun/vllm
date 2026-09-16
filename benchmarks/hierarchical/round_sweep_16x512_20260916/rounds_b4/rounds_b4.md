# Round limit comparison, batch size 4

Source: ../summary.json. Gemma h4, MTP D4, 16 identical raw prompts, 512 output tokens/request, TP1, greedy, ignore_eos, synchronous. Round limits: [4, 6]. R4 reuses the historical 2026-09-15 baseline; larger round limits are new runs. One warmed measurement per configuration; no error bars.

(a) Total output throughput including prefill, excluding loading/warmup. (b) Accepted candidate tokens / submitted candidate tokens. (c) Accepted candidates per nonempty request verification, no bonus. (d) Fraction of nonempty request verifications accepting every submitted candidate; proposal lengths can differ. Final verification is included. Output equivalence remains a separate check.

Reproduce: `.venv/bin/python -m benchmarks.hierarchical.plot_round_sweep <report_directory>`.
