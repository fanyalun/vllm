# gemma4: static expert budget

Source: ../summary.csv and per-cell result.json.

16 identical ordered prompts (4 per category), 512 output tokens, B=1, TP1, greedy, seed 0, CUDA Graph, prefix cache off. AR scheduling: Async; speculative scheduling remains synchronous. Two warmup requests excluded. Each budget cell runs once. Detailed acceptance collection is included in wall time.

(a) Total output tokens divided by summed request wall time, including prefill/decode/API. (b) Accepted draft tokens / drafted tokens. (c) 1 + accepted draft tokens / speculative steps. (d) Throughput versus acceptance length; endpoint labels are h. AR uses mean elapsed time of its two controls. External async controls were measured afterward, not bracketing the original matrix. Gray bands span the two AR endpoint measurements, not confidence intervals. AR repeat output agreement is reported in audit.json. (e) Summed request wall time / summed actually emitted accepted draft tokens in ms, excluding correction/bonus and accepted tokens discarded by the final output cap from the denominator. (f) Summed request wall time / 8192 final output tokens in ms. Both costs include draft, verification, state, prefill and API overhead; they are amortized costs, not timestamps of individual accepted tokens, and are ratios of totals, not means of ratios. No error bars; no repeated-cell uncertainty estimate.

Acceptance length is a counter convention, not exact final yield: output limits may truncate the final step. First-nonaccepted positions in position_acceptance.csv use pre-clipping counters. Consult ../audit.json and ../output_consistency.csv for strict greedy parity; differing outputs prevent a lossless speedup claim.

Reproduce: `.venv/bin/python benchmarks/moe_skip/analyze_static_budget.py --run-dir /home/fanya/vllm/benchmark_results/moe_skip_static_budget_16x512_20260914 --models qwen36 gemma4`
