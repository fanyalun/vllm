# MTP comparison

Historical 16 prompts x 512 output tokens; h4 D4, R4/R6. Sources: policy_matrix_16x512_20260915 and round_sweep_16x512_20260916 summary.json. Left: all three policies normalized to native MTP D4. Right: default low_error batch-level Target and Pre-Verify calls, normalized to MTP Target calls. Call counts are NOT time fractions; Target counters include prefill. No error bars: one valid E2E run per cell, different outputs; strict AR equivalence not established. Reproduce with benchmarks/hierarchical/analyze_verify_width.py.
