# GDN history distance, B=16

Qwen3.6 dimensions on one A100; BF16 input, FP32 state, FP16 d/k ring. Thirty independent layer buffers rotate. Curves show medians of 21 repetitions; shaded bands show min/max, not confidence intervals. State restoration and GPU timing prelude are outside timed regions. The dotted line marks the default flush trigger. No-flush measurements beyond it are controlled counterfactuals. Ordinary attention and MoE are excluded. These measurements are not DRAM byte counters.

Source: kernels/distance_b*_s0.json. Regenerate with `analyze_qwen36_flush_study.py --mode figures --output <root>`.
