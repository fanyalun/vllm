# Compiled forward comparison

Source: ../compiled_stages.csv and ../compiled_trace_audit.json. Each bar partitions one warmed CUDA graph trace at the first prompt prefix. Actual kernel durations are measured with CUPTI; attribution transfers unique names and exact names in stream order from the annotated reference. Gemma dense MLP four-kernel blocks are asserted against reference names; named RMS kernels include fused residual operations. Unmatched fused kernels remain in Other; attribution is approximate. Shared overlap is counted once. Graph gaps remain in Other. Unprofiled totals use 3 prefixes x 20 replays; see ../compiled_totals.csv. B=1, TP=1, BF16, A100 80GB, top-8 versus top-4. Both methods use identical private preverify metadata; no scheduling, state restore, metadata building or proposal generation is timed.

Reproduce: `.venv/bin/python benchmarks/hierarchical/report_compiled_forward.py benchmark_results/forward_stages_20260910`
