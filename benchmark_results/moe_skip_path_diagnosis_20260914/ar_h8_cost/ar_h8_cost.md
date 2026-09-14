# AR and full-budget draft output cost

Source: ../costs.csv and per-cell e2e.json. Lower is better. Panels show Qwen3.6 and Gemma4 on one A100 80GB, TP1/B1, greedy, two archived prompts x 512 outputs x three repetitions per mode. Bars are summed uninstrumented request wall time divided by actual output count; initialization and warmup are excluded. No uncertainty interval is claimed. Modes were run sequentially, not interleaved. The controlled AR comparison is exact for all six Qwen requests and three of six Gemma requests. This is not a new 16-prompt matrix.

Reproduce with `.venv/bin/python benchmarks/moe_skip/report_performance_paths.py --output <run-directory>`.
