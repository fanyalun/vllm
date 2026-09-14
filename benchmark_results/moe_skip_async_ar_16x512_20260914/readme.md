# Async AR controls for the static budget figures

These controls replace the plotted synchronous AR baseline in
`../moe_skip_static_budget_16x512_20260914`. The 32 speculative budget cells
were not remeasured. The old synchronous AR raw files remain archived.

Each model has two independent AR engine runs (`ar_start`, `ar_end`), each
using the original ordered 16 prompts, exactly 512 outputs per prompt,
temperature 0, seed 0, B1/TP1, CUDA graphs, prefix caching off, and two excluded
512-token warmups. AR uses asynchronous scheduling. Directory names retain
the existing schema; these repeats occurred after the old matrix, rather than
bracketing it. Qwen3.6 runs on GPU 0 and Gemma4 on GPU 1, both A100 80GB PCIe.

The plotted reference is 8192 divided by the mean elapsed time of the two
runs. Bands span the two measured AR values; they are not confidence intervals.
The same mean elapsed time is the numerator of `speedup_vs_ar`. Request timing
includes prefill, decode and API overhead and excludes initialization/warmup.

Commands, configuration, prompt hashes, measured runner snapshots, runtime
logs, full output IDs and completion markers are retained per model/run.
The static figure analyzer validates model identity, dataset fingerprint,
common settings, output counts and post-warmup JIT before publishing its audit.

```bash
.venv/bin/python benchmarks/moe_skip/run_static_budget.py --model qwen36 --gpu 0 --run-dir benchmark_results/moe_skip_async_ar_16x512_20260914 --dataset benchmark_results/moe_skip_static_budget_16x512_20260914/qwen36/dataset.jsonl --async-ar-only
.venv/bin/python benchmarks/moe_skip/run_static_budget.py --model gemma4 --gpu 1 --run-dir benchmark_results/moe_skip_async_ar_16x512_20260914 --dataset benchmark_results/moe_skip_static_budget_16x512_20260914/gemma4/dataset.jsonl --async-ar-only
.venv/bin/python benchmarks/moe_skip/analyze_static_budget.py --run-dir benchmark_results/moe_skip_static_budget_16x512_20260914 --ar-baseline-dir benchmark_results/moe_skip_async_ar_16x512_20260914
```

Use fresh output paths for a new measurement. Existing directories require
`--resume` and an unchanged runner fingerprint. `ar_baseline.json` in the
static result folder persists the selected reference for plot regeneration.
