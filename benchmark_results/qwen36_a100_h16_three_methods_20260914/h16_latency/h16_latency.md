# Fixed history: latency

One A100 80GB per experiment; Qwen3.6 GDN dimensions, synthetic inputs. Thirty independent layer buffers, total latency divided by 30. D=4/8/16/32; verify width T=D+1; batch=1/4/8/16; history h=16. Replay uses R=64, ring capacity=128, history tile=64, and no flush. Baseline writes every candidate state; parallel-last writes only the final state, with no history-ring writes. FP32 state, BF16 inputs, FP16 replay d/k ring.

Bars: median latency or ratio of median baseline/method latency over 21 paired rounds. Whiskers: min/max latency or paired speedup, not confidence intervals. Latency uses a log axis. Speedup is kernel-only, not end-to-end. Reset, compilation, warmup and GPU prelude are outside timing. Final candidate state is scratch; no acceptance or scheduler integration is measured.

Source: ../b*_d*_s0.json and ../summary.csv. Reproduce: `.venv/bin/python benchmarks/replayssm/plot_qwen36_fixed_history.py --output benchmark_results/qwen36_a100_h16_three_methods_20260914`.
