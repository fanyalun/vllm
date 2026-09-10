# Gemma4: hierarchical D=4, N=4 versus native D=16

The four-cell performance measurement is complete. Hierarchical decoding improves
on MoE-Skip D=16 in this small sample, especially with DSpark, but remains slower
than native Gemma4 MTP D=16. **Strict numerical equivalence is not certified:** the
MTP pre-verifier sequential oracle failed; the DSpark oracle passed only its
bounded one-prompt/64-token check. See the correctness section below.

## Configuration and scope

- Target: the previous `gemma-4-26B-A4B-it` checkpoint; MTP assistant:
  `gemma-4-26B-A4B-it-assistant`; DSpark: `gemma4-26b-a4b-dspark`.
- Same first four prompts from the previous Gemma4 16-prompt manifest, each with
  512 output tokens. Greedy sampling, ignore EOS, seed 0, B=1, TP1, max model
  length 1024, max batched tokens 4096, GPU memory utilization 0.95, prefix cache
  disabled, CUDA Graphs, V2 runner, two A100 80GB PCIe GPUs.
- Shared Target/pre-verifier weights: Target top-k=8, pre-verifier top-h=4.
  Hierarchical inner D=4 and N=4 only. The total nominal budget is 16; capacity
  is 20 because each inner round can also emit a recovery/bonus token.
- Native D=16 controls: MTP and MoE-Skip. The historical DSpark experiment had
  D=4/8, so no historical DSpark D=16 result is invented or substituted.
- Every cell runs one full 512-token warmup, four uninstrumented requests,
  four profiled requests, then four further uninstrumented requests. Four cells
  total 48 measured requests and 24,576 output tokens, excluding warmup.
- MTP cases use GPU0; DSpark hierarchy and MoE-Skip use GPU1. Native MTP retains
  historical asynchronous scheduling; hierarchical methods require synchronous
  scheduling. Native controls retain historical multimodal input initialization;
  hierarchical methods set image/video limits to zero. All measured prompts are
  text-only. These required differences are not silently treated as identical
  runtime configurations.
- Compile-cache reuse is disabled in all cells to avoid reusing incompatible
  compiled entry signatures. Compilation, graph capture and warmup are excluded
  from timing. All successful performance logs have no post-warmup inference JIT
  warnings.

## Results

| Method | First uninstrumented tokens/s | Repeated uninstrumented tokens/s | Complete cycle ms | Final tokens/cycle | ms/final token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native MTP D=16 | 401.58 | 493.24 | 26.95 | 14.68 | 1.84 |
| Native MoE-Skip D=16 | 126.72 | 128.22 | 117.54 | 15.96 | 7.36 |
| Hierarchical MTP D=4, N=4 | 140.10 | 142.40 | 74.71 | 11.14 | 6.71 |
| Hierarchical DSpark D=4, N=4 | 182.98 | 200.30 | 69.81 | 14.73 | 4.74 |

Whole-cycle and acceptance columns are paired within the profiling pass;
throughput columns are separate uninstrumented passes. Profiling throughput was
497.04 / 129.76 / 141.31 / 198.51 tokens/s in table order. Output token IDs match
between all three passes for every cell (16/16 request pairs for each comparison).
Identical final outputs do not guarantee identical draft acceptance trajectories.
The native MTP and hierarchical DSpark throughput changes across passes are
material; both results are shown rather than selecting the faster one.

Compared with the freshly rerun MoE-Skip D=16 control:

- Hierarchical MTP improves initial throughput by **10.6%**, or **11.1%** in the
  repeated pass. Its cycle is 36.4% shorter but final token yield falls to 69.8%
  of the control, offsetting most of the time saving.
- Hierarchical DSpark improves initial throughput by **44.4%**, or **56.2%** in
  the repeated pass. Its cycle is 40.6% shorter while retaining 92.3% of the
  control's final token yield.
- Native MTP D=16 remains substantially faster than either hierarchy. Its
  asynchronous scheduling differs, so this is the requested historical-setting
  comparison, not an isolated experiment attributing the full gap to hierarchy.

The archived original first-four-prompt results were MTP D=16 274.65 tokens/s and
MoE-Skip D=16 117.61 tokens/s; their original full-16-prompt means were 358.51 and
109.89 respectively. These are historical reference values only. The speedup
claims above use the new four-prompt controls, not old full-dataset averages.
The historical/new MTP difference is not isolated as a code or scheduling gain.

## What changed relative to the Qwen diagnosis

| Hierarchical component | MTP, ms/full loop | DSpark, ms/full loop |
| --- | ---: | ---: |
| Small model proposals, four calls | 13.12 | 6.97 |
| Pre-verifier graph replay, four calls | 40.65 | 40.64 |
| Pre-verifier metadata preparation | 1.56 | 1.67 |
| Pre-verifier wrapper outside graph replay | 3.04 | 3.10 |
| Recurrent state begin/advance wrappers | 0.06 | 0.06 |
| Complete proposal | 59.67 | 53.71 |
| Subsequent Target execution and sampling | 14.57 | 15.61 |
| Complete verification cycle | 74.71 | 69.81 |

Gemma4 has no GDN recurrent state. It avoids the approximately 16 ms of GDN
initialization/advancement measured per Qwen N=4 loop, and its metadata stream
interval is also much smaller. The near-zero state spans here are measurement
and no-op wrapper cost, not hidden recurrent copies. CUDA-event spans include
stream idle time and are not pure GPU kernel utilization measurements.

Pre-verifier graph replay still costs about 10.16 ms per inner round. Once the
GDN overhead is removed, these four forwards account for most of the proposal
time. DSpark helps both by reducing small-draft cost and by materializing a longer
candidate sequence: mean scheduled candidates are 14.19 for DSpark versus 10.50
for MTP. Final accepted tokens including recovery/bonus are 14.73 and 11.14.

## Implementation and correctness boundaries

The previous hierarchy rejected Gemma4 at configuration and model-state setup.
This change permits the existing MoE-Skip Gemma4 architecture, selects
`DefaultModelState`, skips recurrent initialization when no GDN layers exist,
and avoids constructing GDN metadata for attention-only models. Qwen keeps its
hybrid state path and RecoverSSM rejection. The pre-verifier still shares the
Target model instance and parameters; it does not load another Target checkpoint.
Gemma4 assistant MTP retains its existing read-only sharing of Target attention KV.

Attention suffix ownership remains restricted to B=1, no prefix sharing and
causal text decoding. Pre-verifier writes occupy the reserved uncommitted suffix;
the subsequent Target verification rewrites that suffix. No broader batching or
multimodal support is claimed.

The separate sequential oracle uses the first prompt, 64 output tokens, N=4/D=4,
and `VLLM_HIERARCHICAL_CHECK_PREVERIFY=1`. This disables pre-verifier graph replay
and compares the retained pre-verifier prefix against sequential evaluation:

- **MTP: failed during warmup**, at absolute position 149. Four draft tokens
  matched, but the recovery/bonus differed: batch token 1202 versus sequential
  token 659. The maximum logit difference at that row was 7.1328125. No completed
  request or correctness marker is claimed for this failed run.
- **DSpark: passed the bounded check**, including a 64-token warmup and one
  measured 64-token request. This does not establish full 512-token model
  equivalence, graph/eager equivalence, or a general lossless guarantee.

All four performance methods also differ from the archived AR output on all
four prompts. The original native MTP/MoE-Skip results already differed from
that AR reference. The historical comparison is recorded, but it is not a new
same-runtime AR correctness control. These are measurements of an experimental
implementation; do not describe the MTP adapter as lossless or production-ready.

## Artifacts and reproduction

`summary.csv` and `phases.csv` contain machine-readable timing and counters.
`raw_evidence.tar.gz` includes successful performance configs/results/logs and
both correctness probes, including the failed MTP log. `historical_reference.json`
preserves the original first-four outputs and source hashes.

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_DISABLE_COMPILE_CACHE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$PWD/benchmarks/hierarchical:$PWD" PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python benchmarks/hierarchical/run_cycle_profile.py \
  --model /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it \
  --draft-model /home/fanya/data1/fanya/models/gemma4-26b-a4b-dspark \
  --dataset benchmark_results/moe_skip_e2e_16x512_b1_20260908_gemma4/gemma4_16.jsonl \
  --method dspark --rounds 4 --output /path/to/new_output
```

Full commands and environment flags for all six runs are recorded in `audit.json`.
AI assistance was used for the adapter, measurement driver changes and analysis.
These are single-run small-sample measurements, with no confidence intervals.
