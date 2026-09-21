# Qwen3.6 long-context acceptance pilot

Qwen3.6-35B-A3B; MoE-Skip top-h=4 versus native MTP; D=4/8/16/32.
Input lengths are exactly 16,384 and 32,768 tokens. Each cell has four requests
and 256 generated tokens per request; B=1, TP=1, A100 80GB, CUDA graphs,
prefix caching disabled, temperature=1, top_p=0.95, ignore_eos=True.
Seeds are 20260911 + sample index. The methods run on separate GPUs;
this experiment does not measure throughput.

Four disjoint concatenated C4 blocks are supplied as raw token continuations,
without a chat template. Each 16K input is the suffix of its paired 32K input.
This is a context-length pilot, not a long-document QA quality evaluation.
The same input tokens and request seeds are used across methods and widths.
Stochastic methods may consume random numbers differently and need not produce
identical token trajectories. No distributional correctness claim is made.

Mean acceptance length = 1 + sum(accepted draft tokens) / sum(spec verify steps).
This is weighted by verification steps, with the conventional +1 recovery/bonus
term; it is not the exact emitted-token yield at the generation boundary.
The CSV also reports accepted draft tokens per step and the actual output token
count divided by speculative steps (which includes any non-spec output).
No confidence intervals are estimated from this four-sample pilot.
Panels (a)/(b) show 16K/32K respectively. Raw data: ../acceptance_summary.csv,
../acceptance_by_request.csv, ../cells/*/result.json; checks: ../audit.json.

AI assistance was used to prepare the benchmark and report.

Reproduce from the repository root:

```bash
.venv/bin/python benchmarks/moe_skip/run_long_context_acceptance.py --run-dir RUN_DIR
export MPLCONFIGDIR=/tmp/moe_skip_mpl
.venv/bin/python benchmarks/moe_skip/plot_long_context_acceptance.py --run-dir RUN_DIR
```
