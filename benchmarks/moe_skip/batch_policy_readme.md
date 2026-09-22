# Batch-wide MoE-Skip policies

`moe_skip_batch_policy` adds two opt-in Qwen3.6 routing policies to shared-weight
MoE-Skip drafts and hierarchical pre-verification. The final Target forward
retains native routing. The default threshold policy is unchanged.

## Semantics

Each MoE layer aggregates its own current forward's non-padding token rows.
For each expert, sum the original normalized native Top-K routing weights over
all selected token/expert connections. The protected set is the union of every
token's Top-2 experts, selected by gate logits with ascending expert-ID ties.
All original connections to a protected expert survive, including that expert's
lower-ranked connections on other tokens. No new connections are introduced.

The opt-in `batch_top_half_top1` and `batch_max_gap_top1` variants protect the
per-token Top-1 union instead. Their remaining selection and weight semantics
are identical. Existing policy names continue to protect Top-2. The policy
name participates in the compilation hash and is cleared in the inner drafter.

Only active experts outside that protected union are candidates:

- `batch_top_half`: keep `ceil(candidate_count / 2)` by descending aggregate
  weight; break equal-weight ties by ascending expert ID.
- `batch_max_gap`: sort aggregate weights, take the largest adjacent difference,
  and retain the prefix before that gap. Take the first maximum when gaps tie.
  Keep all candidates when there are fewer than two or every score is equal.

Retained connections preserve their original weights without renormalization.
Discarded connections have weight zero and expert ID `-1`, using the existing
invalid-expert assignment path to skip actual expert work and initialize zeros.
Shared experts are unchanged. Padding contributes neither scores nor protection.
FP32 aggregation uses fixed chunks and reduction order without floating atomics.
Expert counts stay on the device, and output shapes remain native `[M, K]` for
CUDA Graph replay. Different effective batch compositions can change routing.

The Top-2 union is a hard lower bound on retained experts. If it covers the
native active set, neither policy removes any expert. For the half policy,
retained expert count is `|protected| + ceil((|active| - |protected|) / 2)`.

## Configuration

Supported: Qwen3.6 MoE, unquantized BF16, CUDA TritonExperts, TP1/PP1/DP1, no EP.
The new option excludes explicit `moe_skip_top_h` and `moe_skip_min_weight` and
requires `moe_skip_weight_mode="preserve"`. It suppresses the implicit threshold
default. Choose a Triton MoE backend through `kernel_config`.

Direct Draft configuration:

```json
{
  "method": "moe_skip",
  "num_speculative_tokens": 4,
  "moe_skip_batch_policy": "batch_max_gap"
}
```

Hierarchical configuration with exact GDN and an MTP inner depth of four:

```json
{
  "method": "hierarchical",
  "inner_method": "mtp",
  "inner_num_speculative_tokens": 4,
  "inner_num_rounds": 1,
  "moe_skip_batch_policy": "batch_max_gap"
}
```

The corresponding CLI flag is `--moe-skip-batch-policy`. Policy configuration is
hashed and scoped to the Draft/Pre-Verify call; the inner MTP config clears it.
Text-only shared Draft calls pass token IDs, matching the Target AOT signature.
Draft input preparation slices padded sequence lengths and rejection counts to
the current active request count when requests finish within a graph bucket.
Padded scratch-state indices use Mamba's null block (`0`), which both Conv1D
and recurrent GDN kernels recognize, including after the active batch shrinks.

## Reproduction

Run from the repository root, with an idle GPU and the existing `.venv`.
The microbenchmark uses real layer 0/19/39 weights and matched synthetic inputs:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.kernels.benchmark_moe_batch_policies \
  --output benchmark_results/moe_batch_selection_reproduction/latency
```

The default compares native Top-8, Top-4, p=0.125, and both new policies at
B1/B64/B128 with five token rows per request. Timing uses 30 warmups, 100 graph
replays, CUDA events, and a 64 MiB L2 flush outside each timed replay, preserving
the earlier pilot's timing protocol. It records routed GEMMs/assignment,
routing-inclusive, shared-inclusive, routing-only and batch-filter-only times.
Allocation/compilation and diagnostic CPU transfers are outside graph replay.
Reported padded slots and valid blocks use the selected Triton assignment config.
No device-specific tuned MoE config is available for this checkpoint shape.

For correctness only, add `--correctness-only`; this produces `validation.json`
and `correctness_complete.json`, not latency samples or a timing completion file.

Each generation command runs seven cells in separate processes: AR and both
Draft/Pre-Verify methods with native Top-8 or either new policy. All use four
fixed prompts, 128 greedy output tokens, exact GDN, one excluded warmup batch,
and the same Triton backend. Direct Draft uses D=4; hierarchical uses one MTP
round of depth four, with an outer capacity of five including the bonus token.

```bash
.venv/bin/python -m benchmarks.moe_skip.run_batch_policy_matrix \
  --device 0 --batch-size 1 --output benchmark_results/moe_batch_selection_reproduction/b1
.venv/bin/python -m benchmarks.moe_skip.run_batch_policy_matrix \
  --device 0 --batch-size 4 --output benchmark_results/moe_batch_selection_reproduction/b4
.venv/bin/python -m benchmarks.moe_skip.summarize_batch_policy_matrix \
  --root benchmark_results/moe_batch_selection_reproduction
```

Each complete cell contains token IDs, prompt hashes, seeds, integer acceptance
counters, batch wall-clock elapsed time, returned-token throughput and AR parity.
The batch timer is counted once, not once per request. `matrix_complete.json`
requires every cell. Acceptance length is `1 + accepted_draft_tokens / steps`;
an AR mismatch remains a reported mismatch, not a claim of lossless decoding.
The four-prompt smoke is integration evidence, not a broad quality evaluation.

For a four-round outer acceptance comparison, use `--inner-rounds 4
--hierarchical-only --include-top1`. The inner MTP depth remains four, and
outer candidate capacity becomes twenty. `mean_outer_accepted` excludes the
Target bonus/correction token; `mean_outer_submitted` counts candidates actually
submitted for verification. Both divide by the number of request verify steps.
Their ratio equals total accepted candidates divided by submitted candidates.
It is not an average of per-step acceptance ratios or returned output length.

`--policy-family half` or `--policy-family max_gap` restricts each matrix to AR,
native hierarchical, and the selected family's Top-2/Top-1 pair. This permits
paired acceptance runs on separate GPUs with a native control on each device.
For shared GPUs, `--cpu-offload-gb` and `--gpu-memory-utilization` are applied
identically to every cell and stored in the manifest. Such runs do not establish
unloaded-GPU throughput. Summarize a B4-only matrix with
`--batches 4` in `summarize_batch_policy_matrix`.

For BS32, pass `--batch-size 32 --num-samples 32` and an explicit `--dataset`.
The first 32 samples of `benchmarks/hierarchical/previous_config_20260909/samples_128.jsonl`
contain eight distinct prompts from each of four categories. Generation remains
128 tokens per prompt. `--routing-counts` adds a second, instrumented generation
pass for each hierarchical batch policy. It requires exact token and per-step
acceptance parity with that cell's uninstrumented pass before marking completion.

The benchmark worker rebuilds only private Pre-Verify graphs. Inside each layer,
integer scatter counts record native and retained active experts and connections,
excluding padding. After each real `_verify` call, its latest layer counts are
added once; internal graph warmups/capture and the excluded request warmup are
not accumulated. Final Target and inner MTP routing are outside this scope.
Whole-expert skip fraction is one minus summed retained expert invocations
divided by summed native expert invocations across layers and calls. Connection
skip fraction uses token-expert assignments instead. These are weighted ratios,
not the union of experts across an entire generation or an unweighted mean of
per-call percentages. Raw counts include per-layer and active-request breakdowns
so full BS32 can be distinguished from the shrinking tail.

`--eager` avoids private graph-capture state snapshots when shared-GPU memory
is constrained. It is supported by both the cell and matrix runners and by the
routing counter. `--ssm-dtype bfloat16` explicitly selects BF16 SSM storage;
`auto` follows the model configuration and can resolve to FP32. Record this
precision change when comparing against earlier FP32 runs. A fixed
`--kv-cache-memory-bytes` allocation must accommodate the requested concurrency;
setting `--batch-size 32` alone does not prove that 32 requests ran together.

## Tests

```bash
.venv/bin/python -m pytest \
  tests/config/test_moe_skip_config.py \
  tests/config/test_hierarchical_config.py \
  tests/model_executor/layers/fused_moe/test_routing_top_k.py \
  tests/v1/worker/gpu/spec_decode/test_moe_skip_trace.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/v1/worker/test_mamba_hybrid_model_state.py -q
```

Tests cover routing reference agreement, empty batches, equal scores, unsorted
slots, partial/all padding, changed inputs across graph replays, both assignment
paths, final-Target route isolation, configuration conflicts and shared compiled
input compatibility. The real-weight benchmark also checks dispatch against
native assignments with the same deleted weights and checks graph/eager equality.
