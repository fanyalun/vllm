# Previous experiment comparison

Qwen3.6, one A100 80GB PCIe per cell, B=1, TP1, greedy, CUDA Graph. New hierarchical inner D=4; N=1/2/4/8. Horizontal axis is nominal budget D*N, not materialized candidate count. Each inner round can append a recovery/bonus token.

E2E and CUDA-event drafting use 16 fixed prompts x 512 output tokens. Acceptance uses the sample count in contract.json and includes the final Target bonus/recovery. Drafting time is the call-weighted decode-only mean around the full N-round propose(), including pre-verification and CPU-induced stream idle gaps, excluding Target verification. One warmup request is excluded from the two timing passes; acceptance has no extra warmup. No error bars: one run per cell.

Historical lines come from the 2026-09-08 E2E/timing experiments and the 128x512 acceptance summary. The available Qwen3.6 128x512 acceptance summary has no DSpark baseline; no value was invented. New samples are not paired on identical generated prefixes across methods. Strict numerical equivalence gates failed. New runs disable async scheduling and multimedia inputs as required by the hierarchical implementation.

Sources: [summary.csv](../../summary.csv), [comparison.csv](../../comparison.csv), original prompts in `../../samples_16.jsonl` and `../../samples_128.jsonl`. Reproduce: `.venv/bin/python benchmarks/hierarchical/summarize_previous.py --run-dir /home/fanya/vllm/benchmark_results/hierarchical_previous_config_20260909`.
