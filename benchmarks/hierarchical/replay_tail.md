# Replay-tail GDN pre-verification

`preverify_gdn_mode="replay_tail"` keeps one private recurrent state per GDN
layer. Each outer proposal starts from the Target's accepted state. Each inner
verification reads the previous inner window's complete tail and overwrites it
with the new complete tail, including rejected draft positions. The next outer
proposal resets it from Target again.

## Semantics

- Only the first input token's hidden state is projected through `in_proj_ba`.
  Each head's resulting decay and beta are shared across the current window.
- Q/K/V/Z, causal Conv, Q/K normalization, and the output projection remain
  position-specific. The first input is the anchor, followed by up to four drafts.
- The block kernel retains all causal delta dependencies. It solves a small
  triangular system and produces every output without storing intermediate full
  recurrent states. It writes the last actual input's state, not an unprocessed
  correction or bonus token's state.
- Each program owns complete state rows for its value tile. All initial-state
  projections finish before any in-place store. Key tiling limits register
  pressure; the tail pass rereads the initial state and overwrites each element
  once. There is no cross-program state dependency or saved replay history.
- Conv retains acceptance-aware history selection. Attention positions and
  candidate compaction follow the existing accepted-length bookkeeping.
- Target cache bindings are restored after the private forward. The Target's
  checkpoint provides the outer head; there is no redundant private head copy.

This is an experimental approximation. The default remains `none`. Initial
support is Qwen3.6 MoE, one request, TP=1, CUDA, unquantized weights, FP32 recurrent
state, inner draft length four, and actual window lengths one through five.
The existing hierarchical restrictions on LoRA, prefix caching, and asynchronous
scheduling continue to apply.

## Run

Set `preverify_gdn_mode` alongside the existing hierarchical configuration:

```json
{
  "method": "hierarchical",
  "inner_method": "mtp",
  "inner_num_speculative_tokens": 4,
  "inner_num_rounds": 4,
  "moe_skip_top_h": 4,
  "draft_sample_method": "greedy",
  "preverify_gdn_mode": "replay_tail"
}
```

The benchmark retains the current `low_error` stopping policy, so four is the
maximum number of inner rounds, not a promise that every proposal runs four.

```bash
.venv/bin/python benchmarks/hierarchical/run_cell.py \
  --method hierarchical --inner-method mtp --gdn-mode replay_tail \
  --num-samples 1 --max-tokens 32 --output /tmp/replay_tail_smoke.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/hierarchical/run_replay_tail.py \
  --inner-method mtp --output benchmark_results/replay_tail/mtp
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/hierarchical/run_replay_tail.py \
  --inner-method dspark --output benchmark_results/replay_tail/dspark
.venv/bin/python benchmarks/hierarchical/run_cell.py --method ar \
  --num-samples 4 --max-tokens 256 --output benchmark_results/replay_tail/ar.json
.venv/bin/python benchmarks/hierarchical/summarize_replay_tail.py \
  benchmark_results/replay_tail
```

Each method uses the same four prompts and 256 output tokens. It warms every
case and prompt, then runs three alternating baseline/replay-tail pairs.
Timing runs have auditing disabled. Separate instrumented passes collect
proposal-to-next-Target cycles, pre-verifier time, and integer acceptance counts.
The two diagnostic ablations use existing per-position recurrence and are not
kernel performance comparisons:

| Case | Gates | State after inner rejection |
| --- | --- | --- |
| `none` | Per token | Accepted position |
| `replay_tail` | First token | Complete tail |
| `gates_only` | First token | Accepted position |
| `tail_only` | Per token | Complete tail |

Gate projection overrides exist only during the diagnostic private forward and
are restored in `finally`. They never change a Target forward.

The completion marker certifies measurement coverage and repeatable outputs,
not AR equivalence. The summary independently verifies the full cell matrix,
clips returned-token counts at the requested output limit, and reports exact AR
token comparisons separately. Kernel/state equivalence uses a sequential
fixed-gate reference, not the original per-token-gate model.

## Kernel validation

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  tests/benchmarks/test_hierarchical_measurement.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_gdn_replay_tail.py \
  --output benchmark_results/replay_tail/kernel.json
```

The microbenchmark checks all outputs and the tail against the existing
recurrence with matching fixed gates before timing. It uses cold-L2 CUDA graph
timing through FlashInfer, records the actual timing backend and compiler spill
counts, and excludes gate projection from both paths. When the CUPTI Python
package is unavailable, FlashInfer falls back to CUDA events with rotating input
buffers. Byte counts describe logical full-state writes, not measured DRAM
transactions. Private cache savings do not change scheduler-visible Target cache
capacity.
