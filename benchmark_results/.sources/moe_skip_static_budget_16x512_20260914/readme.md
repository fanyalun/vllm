# MoE-Skip static expert budgets

This run measures standalone shared-weight MoE-Skip drafting on Qwen3.6 and
Gemma4. It does not measure hierarchical Pre-Verify.

The current figures and speedup tables use **AR (Async)**, measured separately
on the same 16 ordered prompts with 512 outputs and two excluded warmups.
`ar_baseline.json` selects the new control directory. The original synchronous
AR files below remain archived; they are no longer the plotted reference.
Only AR was remeasured; all 32 speculative cells retain their original data.
The two new AR repeats do not bracket the earlier matrix in time. New control
placement is Qwen3.6 on GPU 0 and Gemma4 on GPU 1, both A100 80GB PCIe.

## Original measurement contract

- Models: Qwen3.6-35B-A3B on GPU 1 and Gemma4-26B-A4B-it on GPU 0.
- Hardware: two NVIDIA A100 80GB PCIe GPUs; one independent TP1 model per GPU.
- Fixed expert budgets h = 2, 4, 6, 8; draft widths D = 4, 8, 16, 32.
- Each cell: 16 fixed prompts, four each from HumanEval, Alpaca, GSM8K and
  UltraFeedback; exactly 512 output tokens.
- Prompts reuse the first 16 entries of each model's existing multicategory
  manifest. Their source hashes and token counts are checked.
- Input lengths: Qwen3.6 7–157 tokens (mean 63.8125); Gemma4 7–153 tokens
  (mean 64.625). These measurements concern short inputs.
- B=1 sequential requests, greedy, seed 0, ignore_eos=True, CUDA Graph,
  prefix caching off, asynchronous scheduling off for both AR and MoE-Skip.
- Two 512-token warmup requests are excluded before every cell. Logs retain
  runtime JIT warnings for subsequent timing audits.
- Each budget cell runs once, in a reproducibly shuffled order. Fresh AR
  controls run before and after the budget sweep. Two model sweeps can run
  concurrently on separate GPUs and share host resources.
- Existing Python environment is reused. No dependencies are installed.

## Metrics

Let T be the sum of measured llm.generate wall times in seconds, A the sum
of accepted draft-token counters, P the sum of proposed draft-token counters,
and R the number of speculative verification steps over the 16 requests.
Let E be accepted draft tokens actually returned after final-output clipping.

| Metric | Definition |
| --- | --- |
| Throughput | 8192 / T output tokens/s |
| Cost per actual accepted draft token | 1000 T / E milliseconds |
| Cost per verified accepted draft token | 1000 T / A milliseconds |
| Cost per final output token | 1000 T / 8192 milliseconds |
| Draft acceptance rate | A / P |
| Accepted draft tokens per step | A / R |
| Conventional acceptance length | 1 + A / R |
| Output tokens per speculative step | 8192 / R |

Time includes prefill, drafting, verification, state maintenance and offline
API return. Initialization, warmup, JSON writing and external service/network
latency are excluded. Detailed per-request acceptance statistics are enabled
and their overhead is included. These are amortized ratios of totals, not
averages of request ratios or timestamps of individual token acceptance.

Accepted draft counters exclude correction/bonus tokens. The scheduler records
A before truncating the final output step. For this sequential, single-prefill,
non-preempted run, the analyzer reconstructs E from one initial prefill output
and the ordered per-step accepted counts. Every nonfinal step contributes
accepted + 1 outputs. The final step contributes accepted tokens first, then
its correction/bonus token, subject to the remaining output allowance. The
analyzer fails if these bounds disagree with the 512-token output. Both A and
E, as well as A - E, are retained. The main cost plot uses E.

The acceptance-length convention is not an exact emitted yield. Position
tables use raw verification counters before output clipping. AR has no
accepted-draft denominator.

## Validation and interpretation

Every result retains all 512 output token IDs, prompt identities, request wall
times and detailed integer acceptance traces. The analyzer checks coverage,
counter arithmetic and exact greedy output agreement against the fresh AR
start control. Measurement completeness and strict output parity are separate
audit fields. A parity failure does not become a passing correctness gate
because coverage is complete; such results cannot establish lossless speedup.

The curves are descriptive measurements with one execution per cell and no
repeated-cell error bars. AR start/end drift is reported separately. They do
not isolate drafting forward latency from the full request time.

## Reproduction

```bash
.venv/bin/python benchmarks/moe_skip/run_static_budget.py \
  --model qwen36 --gpu 1 \
  --dataset benchmark_results/moe_skip_static_budget_16x512_20260914/qwen36/dataset.jsonl \
  --run-dir benchmark_results/moe_skip_static_budget_reproduction
.venv/bin/python benchmarks/moe_skip/run_static_budget.py \
  --model gemma4 --gpu 0 \
  --dataset benchmark_results/moe_skip_static_budget_16x512_20260914/gemma4/dataset.jsonl \
  --run-dir benchmark_results/moe_skip_static_budget_reproduction
.venv/bin/python benchmarks/moe_skip/analyze_static_budget.py \
  --run-dir benchmark_results/moe_skip_static_budget_reproduction
```

Use a fresh directory for a new experiment. Use --resume only for unchanged
inputs and the original model/GPU mapping. Per-cell command.json and run.log
record child execution; contract.json records settings and fingerprints.
To regenerate the archived plots directly, pass this archived experiment
directory to analyze_static_budget.py instead of the reproduction directory.

The measurement runner is archived as each model's runner_snapshot.py and its
hash is recorded in contract.json. After measurements completed, the published
runner gained an optional --dataset argument so the shipped 16-prompt manifests
can be reused without the historical 128-prompt source directories. The later
AR refresh adds --async-ar-only and a per-cell async_scheduling flag, defaulting
to False for original configs. Each run retains its own worker snapshot.
Resume requires the exact original runner fingerprint.

AI assistance was used to prepare the runner, analysis and report.
