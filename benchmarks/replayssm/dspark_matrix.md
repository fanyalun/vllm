# DSpark confidence-prefix comparison

The V2 DSpark proposer can return a variable verification length with
`dspark_confidence_threshold`. For example:

```json
{
  "method": "dspark",
  "model": "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
  "num_speculative_tokens": 8,
  "dspark_confidence_threshold": 0.8
}
```

All eight candidates are generated. The scheduler receives the longest
contiguous prefix whose individual confidence scores are finite and at least
the threshold. A failing first position produces zero drafts. Target executes
only the scheduled prefix plus its bonus/recovery input. The feature requires a
loaded confidence head and synchronous scheduling; explicitly enabling async
scheduling raises an error. Without the threshold, DSpark retains fixed lengths.

The length tensor has a persistent GPU address for CUDA Graph replay. Its copy
to the synchronous scheduler uses the existing draft-output copy stream. In
adaptive baseline SD, zero-draft post-prefill rows still use the speculative
SSM state-slot lookup. Original ReplaySSM and dual-checkpoint both receive V2
prefill lengths for their existing first-decode history reset on request reuse.

## Matrix

The benchmark has 52 cells: four AR cells and 48 speculative cells. Speculative
methods are baseline SD, original ReplaySSM, and dual-checkpoint; policies are
D4, D8, and max-eight prefixes at p=0.8 and p=0.6. All use batch limits 1, 4, 8,
and 16, the same sixteen prompts, and 256 output tokens per prompt.

The buffer parameter is 16. Original ReplaySSM retains its native flush
threshold of 16 + D + 1 and a 32-slot ring for this matrix. Dual-checkpoint
uses the hard threshold of 16 and a 16-slot ring. The total KV budget is the
same 10 GiB for every method.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/replayssm/dspark_matrix_run.py \
  --output benchmark_results/dspark_qwen36_e2e_16x256_cuda1_small_warmup_20260915 --gpus 1
PYTHONPATH=. .venv/bin/python benchmarks/replayssm/dspark_matrix_report.py \
  --output benchmark_results/dspark_qwen36_e2e_16x256_cuda1_small_warmup_20260915
```

Each process warms one cohort for 32 output tokens per request and the remaining
prompts for one token to cover prefill shapes, then measures the full workload
once on CUDA device 1. Supplementary two-token warmups for shrinking cohorts
are recorded separately when used. The scheduler
is paused while each multi-request cohort is enqueued, then released. The
actual number of newly admitted requests is recorded and must match the batch
limit for every request. This prevents background processing from splitting a
cohort that the client is still submitting. The pause/release overhead is
included in wall time for all methods.

The scheduler
records integer histograms of actual verified and accepted draft lengths.
Full acceptance conditions on nonempty draft windows. Separate zero-draft
fractions and effective decode lengths prevent empty windows from inflating
the apparent success rate. Counts precede final output-length truncation;
throughput counts only emitted tokens. Inspect the final report and parity
audit before interpreting throughput as a correctness-qualified speedup.

`dspark_matrix_logprob_audit.py` runs the same configuration with top-five
logprobs exported. Its timings are diagnostic and must not be mixed into the
performance matrix.

The September 15 small-warmup run was stopped after batch 8 at the user's
request: 39 configurations were measured and batch 16 was cancelled.
`scope_override.json` records this change without rewriting the original
manifest. The report honors that explicit scope; it does not silently accept
missing cells. See the run directory's `run_notes.md` for warmup amendments,
excluded attempts, and launch overrides.
