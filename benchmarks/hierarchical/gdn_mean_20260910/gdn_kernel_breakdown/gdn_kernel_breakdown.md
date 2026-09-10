# GDN kernel work

Source: `benchmarks/hierarchical/gdn_mean_20260910/gdn_components.csv`. Same five-token candidate block and initial state, Qwen3.6, TP=1/B=1, top-4 Pre-Verify, BF16 activations and FP32 SSM. One CUDA graph trace of the third fixed-prefix probe after all 20-replay timing passes. All kernels match the annotated eager invocation by name and order. No per-layer CUDA events are present in the traced graph.

The baseline recurrence kernel includes normalization; mean modes use a separate normalization kernel included in Other GDN work, along with pooling, casts, copies and initialization. Projections include GEMV reduction kernels. Bars sum active GDN kernel durations over 30 layers. This is an eager-captured diagnostic, not the compiled end-to-end graph timing. It does not include state reset or metadata CPU overhead.
