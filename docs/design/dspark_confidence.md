# Qwen DSpark confidence output

Qwen DSpark checkpoints with `enable_confidence_head=true` now load and evaluate
their existing confidence head. No new serving flag is needed. The head is an
unquantized, replicated linear projection followed by an FP32 sigmoid. When
`confidence_head_with_markov=true`, its input concatenates the backbone hidden
state and the previous token's Markov embedding, in that order. The first draft
position uses the anchor token's embedding; later positions use the preceding
sampled draft token in target vocabulary.

This follows the training model's
[head definition](https://github.com/vllm-project/speculators/blob/main/src/speculators/models/dspark/model_definitions.py)
and [feature construction](https://github.com/vllm-project/speculators/blob/main/src/speculators/models/dspark/core.py).
Missing projection weights or bias fail model loading when the head is enabled.
Checkpoints without an enabled head retain their existing drafting behavior.

## Runtime access

After `DSparkSpeculator.propose(...)` returns:

```python
confidence = speculator.get_draft_confidence(input_batch.num_reqs)
```

The result is an FP32 GPU view shaped `[num_reqs, num_speculative_tokens]`,
aligned with the returned draft-token rows and columns. It is `None` for models
without a head. Read only the active rows; padding rows can contain stale values.
Clone the result before the next proposal if it must be retained. The buffer
has a fixed address for CUDA Graph replay. Computation remains on the GPU;
normal drafting adds no device-to-host synchronization for confidence output.

The values are per-position acceptance predictions, not LM token logprobs or a
calibrated guarantee that the whole prefix will be accepted. This change does
not truncate drafts, change rejection sampling, or promote ReplaySSM checkpoints
based on predictions. Confidence calibration and adaptive verification remain
separate work. The standard HTTP response and `RequestOutput` are unchanged;
confidence is available through the speculator and the debug export below.

The upstream [training loss](https://github.com/vllm-project/speculators/blob/main/src/speculators/models/dspark/metrics.py)
uses the distribution overlap `sum_v min(p_v, q_v)` as the soft confidence
target. That is rejection-sampling acceptance, not a direct greedy token-match
label. Calibrate against the actual sampler before interpreting scores as
greedy acceptance probabilities or using cumulative products for draft length.

## Bounded per-round export

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
  PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. .venv/bin/python \
  benchmarks/replayssm/dual_checkpoint_smoke.py \
  --method dspark --dual --draft 4 \
  --output /tmp/dspark_d4_smoke.json \
  --confidence-output /tmp/dspark_d4_confidence.json
```

Use `--draft 8` for eight drafts. This runs two 64-token requests after a
matching warmup. The confidence JSON contains one entry per worker, with:

- `weight_sha256`: hashes of the loaded projection weight and bias.
- `rounds[].req_ids`: active request IDs in proposal row order.
- `rounds[].draft_tokens`: proposed tokens in target vocabulary.
- `rounds[].positions`: their predicted absolute positions.
- `rounds[].confidence`: corresponding per-position probabilities.
- `rounds[].previous_num_rejected` and `previous_num_sampled`: counters from
  target verification immediately before this proposal. They describe the
  previous proposal for that request, not the confidence values in this row.

The diagnostic recorder starts after warmup, retains GPU copies, and transfers
results after generation. It fails beyond 256 rounds and removes its wrapper
when read. This instrumentation is for alignment/debugging, not throughput
measurement; its clones and retained tensors add overhead.

## Tests

```bash
.venv/bin/python -m pytest \
  tests/kernels/test_dspark_confidence.py \
  tests/transformers_utils/test_speculators_dspark.py -q
```

Tests cover projection numerics, Markov-free input, enabled/disabled weight
loading, missing weights, target-vocabulary predecessor alignment, D=4/8,
CUDA Graph replay with changed inputs, and untouched padding rows.
