# Complete verification cycles and hierarchical decoding cost

This is a focused performance diagnosis of the existing experimental implementation.
It does not repair or supersede the failed strict numerical-equivalence gates in
[the implementation results](../results.md). No inference runtime code was changed.

## Measurement contract

The first four prompts from the previous 16-prompt manifest are reused, with 512
generated tokens each, greedy sampling, seed 0, B=1, TP1, Qwen3.6-35B-A3B, CUDA
Graphs, max model length 1024, max batched tokens 4096, memory utilization 0.95,
prefix caching off, and image/video limits zero. The shared Target uses top-k=8;
the hierarchical pre-verifier uses top-h=4. Inner D=4, N=4. Each cell has a full
512-token warmup before measurement. This is a four-prompt diagnostic, not a
replacement for the previous 16/128-prompt experiment matrix.

`cycle_stream_ms` starts at proposal i and ends after Target verification and
sampling of that proposal in worker step i+1. It includes the full N*D inner loop,
Target execution, rejection sampling, and intervening stream idle/scheduling time.
It is **not** proposal i plus the preceding Target verification. The initial
proposal attached to prompt prefill and the terminal proposal with no subsequent
verification are excluded from the steady-cycle summary. Acceptance is paired
with the same final Target verification, including its recovery/bonus token and
before max-token truncation. Extra asynchronous in-flight verification batches
after the 512-token output limit are excluded and counted separately (three in
the MTP async control). All 512 output token IDs are retained per request.

CUDA events measure elapsed time on the execution stream, including gaps caused
by host submission. They do not measure only active GPU kernels. Nested spans
must not be added twice. CPU spans overlap GPU spans and must not be added to
them. Events are collected after the request, with no added per-span synchronize.

The old `propose()`-only timing remains a component named `proposal`; historical
files are preserved because they contain no direct Target-cycle timing. Main
timing comparisons should use `cycle_stream_ms` and `cycle_ms_per_emitted`.
Dividing the cycle by N is useful for diagnosis, but is not the headline whole-loop
metric and does not normalize for final accepted token yield.

## MTP diagnosis

The synchronous native MTP control and hierarchical MTP both produce identical
token sequences between their uninstrumented and profiled passes on all four
prompts. Their outputs need not match each other or AR.

| Measured quantity | Native MTP D=4 | Hierarchical MTP D=4, N=4 |
| --- | ---: | ---: |
| Steady verification cycles | 549 | 202 |
| Whole verification cycle, ms | 16.168 | 95.238 |
| Final emitted tokens per cycle | 3.698 | 10.094 |
| Cycle ms per final emitted token | 4.372 | 9.435 |
| Cycle ms divided by inner rounds | 16.168 | 23.810 |
| Final emitted tokens divided by inner rounds | 3.698 | 2.524 |
| Uninstrumented end-to-end tokens/s | 183.222 | 102.093 |
| Profiled end-to-end tokens/s | 217.919 | 101.099 |

The instrumented native control is faster than its uninstrumented pass despite
identical tokens. This is a measurement/pass effect that this experiment does not
isolate; do not apply its component timings as an exact decomposition of the
uninstrumented historical throughput. Cost and token yield above come from the
same profiled pass. The profiled throughput ratio (2.16x) agrees with the measured
cost-per-token ratio (2.16x).

The hierarchical MTP proposal takes 76.683 ms per outer loop. Its disjoint
components are approximately:

| Component | ms per N=4 loop |
| --- | ---: |
| Four small MTP proposals | 14.081 |
| Four pre-verifier CUDA graph replays | 33.631 |
| Copy canonical Target GDN state into private state | 5.197 |
| Advance private GDN state after four inner acceptances | 11.077 |
| Build pre-verifier metadata four times | 7.251 |
| Pre-verifier wrapper outside graph replay | 4.172 |
| Remaining proposal work and gaps | 1.274 |
| Subsequent Target execution and sampling | 18.000 |
| Remaining cycle work and gaps | 0.556 |

The state and metadata categories alone consume about **27.70 ms per outer loop**.
The model has 30 GDN layers. `PreverifyState.begin()` and `advance()` use Python
loops with small tensor allocations, gathers and copies per layer. `_batch()`
rebuilds tensors and attention metadata each inner round. `_verify()` recursively
refreshes graph metadata and slot mappings before replay; the shared GDN metadata
object is visited through each layer entry. These costs occur outside the captured
pre-verifier graph.

`accepted_prefix()` reads a GPU reduction using `.item()` each inner round. Its
mean CPU span is 8.403 ms but its stream span is only 0.054 ms: the CPU primarily
waits for the already-accounted-for pre-verifier graph (8.408 ms). Adding that CPU
wait to the graph duration would double count. This synchronization does constrain
submission of the next inner round.

## Why the forward-count model overpredicts speedup

Let S(D) be small drafting, P(D+1) pre-verification, T(w) full Target verification,
H the implementation overhead, and A the final emitted tokens per cycle. The
comparison is:

```text
native cost/token = [S(D) + T(D+1) + H_native] / A_native
hierarchical cost/token = [N*(S(D) + P(D+1)) + T(L+1) + H_hier] / A_hier
```

`(1-1/N)*T-P` assumes equal token progress, equal Target costs at different widths,
equal small-draft costs, and negligible added overhead. None is guaranteed here.

1. **MoE-Skip saves less model time than the formula needs.** Native MTP Target
   execution plus sampling is 12.824 ms at width five; the pre-verifier graph is
   8.408 ms at width five. Even treating that entire difference as available
   savings is favorable to the hierarchy, because the native span includes input
   preparation that the pre-verifier graph span excludes. Reducing routed experts
   from eight to four does not remove half the attention, GDN, dense projections,
   normalization, or output-head work.
2. **The outer Target is wider.** It receives a mean 12.822 candidates, rather
   than four, and executes plus samples in 18.000 ms. Its amortized cost is
   4.500 ms/inner round. The optimistic model-only hierarchical total is already
   `3.520 + 8.408 + 18.000/4 = 16.428 ms/inner round`, versus the measured native
   complete cycle of 16.168 ms. These are workload averages, not a controlled
   same-prefix top-k microbenchmark.
3. **The final yield is lower.** Hierarchical MTP produces 10.094 tokens per outer
   verification, only 68.2% of `4 * 3.698`. Nominal D*N=16 is a candidate budget;
   inner rejections and final Target rejection reduce real progress. Here the
   pre-verifier emits about 12.822 candidate tokens and the final Target emits
   about 10.094 tokens including its correction. One cannot substitute 16 into
   the throughput numerator.
4. **The loop has substantial added implementation cost.** State/metadata work
   adds 27.70 ms, about 6.92 ms/inner round, before the remaining bookkeeping.

Even an optimistic calculation removing all non-model hierarchical overhead while
holding token yield fixed gives about 65.7 ms/outer loop, or 6.51 ms/token. That is
still above native MTP's measured 4.37 ms/token. Removing the measured 27.70 ms of
state/metadata cost alone would give roughly 67.5 ms/loop, an optimistic 1.41x
improvement over this hierarchy, not enough to catch native MTP. These are bounds
from measured components, not implemented optimization results.

## DSpark cross-check

Hierarchical DSpark has a 96.121 ms cycle, 10.405 final tokens/cycle, and a
76.369 ms proposal loop. Its four small proposals total 7.521 ms, but pre-verifier
metadata spans total 12.536 ms and state handling totals 16.652 ms. The four
pre-verifier graph replays cost 33.816 ms. End-to-end throughput is 104.419 tokens/s
uninstrumented and 103.628 profiled; token IDs match across those passes on 4/4
requests.

A shorter small-draft GPU span exposes more host preparation time as stream idle
time in the following metadata span. Do not interpret the larger DSpark metadata
event interval alone as evidence of more metadata GPU computation. Both methods
remain dominated by the same pre-verifier and state/metadata submission path.

## Engineering implications

- Capture or fuse private GDN state initialization/advance and metadata staging;
  preallocate small buffers, avoid repeated copies of aliased graph metadata, and
  keep inner acceptance/position updates on device where possible. Validate state
  rollback and numerical behavior before claiming an optimized implementation.
- Optimize net accepted tokens per unit time, not N alone. Improving the P-to-T
  prefix survival and reducing P cost are necessary to beat the small-draft
  baseline under these measurements, even after removing substantial overhead.
- Keep async-scheduling controls separate. The main native controls explicitly
  disable async scheduling to match the hierarchy, so async scheduling cannot be
  the sole explanation for the observed gap.

## Additional controls and the same-budget comparison

| Case | Uninstrumented tokens/s | Cycle ms | Final tokens/cycle | ms/final token |
| --- | ---: | ---: | ---: | ---: |
| Native MTP D=4, synchronous | 183.222 | 16.168 | 3.698 | 4.372 |
| Native MTP D=4, asynchronous | 205.281 | 15.024 | 3.882 | 3.871 |
| Hierarchical MTP D=4, N=4 | 102.093 | 95.238 | 10.094 | 9.435 |
| Native DSpark D=4, synchronous | 136.155 | 17.750 | 2.839 | 6.253 |
| Hierarchical DSpark D=4, N=4 | 104.419 | 96.121 | 10.405 | 9.238 |
| Native MoE-Skip D=16, historical input mode | 96.593 | 109.393 | 10.886 | 10.049 |

The native D=4 controls test the per-inner-round amortization argument. MoE-Skip
D=16 instead matches the hierarchy's nominal total budget of 4*4. Hierarchical
MTP's cycle is about 13% shorter, but MoE-Skip emits about 8% more tokens per
cycle. Consequently, hierarchical MTP improves uninstrumented throughput by only
5.7%; hierarchical DSpark improves it by 8.1%. This is consistent with the small
gains in the prior larger experiment.

The asynchronous MTP uninstrumented pass is 12.0% faster than the synchronous
one in this run. This single sequential control is not an isolated estimate of
pure scheduling overhead: different acceptance paths, warm state and measurement
order can contribute. Both synchronous controls already outperform their
hierarchical counterparts, so asynchronous scheduling is not the primary cause.

All six cases have identical output token IDs between their uninstrumented and
profiled passes on 4/4 requests. This does **not** prove identical draft tokens or
acceptance trajectories. The later three cases additionally repeat an
uninstrumented pass after profiling: DSpark sync 151.307, MTP async 242.726, and
MoE-Skip D=16 97.298 tokens/s. Those output IDs also match on 4/4 requests. The
changes across passes make the difference between component profiling and a
rigorous repeated throughput comparison explicit.

MoE-Skip D=16 with image/video limits zero failed during initialization with
`AttributeError: 'NoneType' object has no attribute 'size'`. Disabling compilation
cache reuse did not resolve it. Its standalone drafter enters the shared compiled
model with `inputs_embeds`, whereas the disabled-multimodal Target path enters
with token IDs. The successful control restores the historical multimodal input
configuration (no explicit limits); all actual prompts remain text-only. This
control therefore has a documented input-path difference from the hierarchy.
The MTP async control initially failed when loading compiled code; a fresh
compilation with the same zero image/video limits succeeded. Successful MTP async
and MoE-Skip controls used `VLLM_DISABLE_COMPILE_CACHE=1`; compilation and warmup
were excluded from every timing. No measured successful case logged a post-warmup
inference JIT warning.

## Reproduction and artifacts

`run_cycle_profile.py` records per-request token IDs, whole-cycle pairs and nested
CPU/CUDA-event spans. `summarize_cycles.py` validates emitted/scheduled counts and
produces `summary.csv` and `phases.csv`. Successful raw cells and logs are archived
alongside these tables. Failed startup/probe attempts are kept separately in the
local run directory and are not counted as completed measurements.

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$PWD/benchmarks/hierarchical:$PWD" PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python benchmarks/hierarchical/run_cycle_profile.py \
  --method mtp --rounds 4 --output /path/to/new_output

.venv/bin/python -m pytest tests/benchmarks/test_hierarchical_measurement.py -q
```

AI assistance was used to implement the measurement instrumentation and analyze
the results. These are single-run, four-prompt diagnostic measurements on two
A100 80GB PCIe GPUs, not a statistical or lossless-correctness certification.
