# Qwen3.6 DSpark confidence integration validation

## Change

Load the checkpoint's enabled confidence projection and bias, evaluate the
training feature layout (backbone hidden followed by previous-token Markov
embedding), and export FP32 sigmoid probabilities in a persistent GPU buffer.
The model loader rejects missing enabled-head weights. The head is unquantized
and replicated. No adaptive draft length or acceptance change is implemented.

Normal inference keeps confidence on the GPU. An opt-in worker extension records
bounded per-round request IDs, target-vocabulary drafts, predicted positions,
confidence, and preceding verification counters. It uses named RPC calls.
The trace is diagnostic instrumentation and is not a throughput benchmark.
See [usage](../../docs/design/dspark_confidence.md) for API and semantics.

## Validation

- 16 tests passed: head numerics, Markov-free head, enabled/disabled/missing
  weight loading, predecessor token alignment including vocabulary remapping,
  D=4/8 eager and CUDA Graph, changed inputs across replay and padding rows.
- Qwen3.6-35B-A3B + local DSpark, TP1, A100, V2 CUDA Graph: two 64-token requests
  per D after identical warmup. Both D=4 and D=8 completed.
- Both loaded confidence weight hashes exactly match model.safetensors.
- All 128 generated token IDs per D match the warmup and the saved pre-confidence
  optimized-dual baseline, with matching prompt token IDs. This is a bounded
  greedy equivalence check, not proof across all inputs or sampling modes.
- Every recorded confidence is finite and in [0, 1], with matching request,
  token and position dimensions and consecutive positions within each proposal.
- Pre-commit checks passed. No whole-model throughput comparison was performed.

| D | Proposal rounds | Requests | Exported scores | Min | Max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 36 | 2 | 224 | 0.200558 | 0.980281 |
| 8 | 39 | 2 | 456 | 0.143188 | 0.964321 |

Score counts include rejected and unused proposed tokens; they are not final
output token counts. First exported D=4 vector:
`[0.919643, 0.286169, 0.218669, 0.218669]`.
First exported D=8 vector:
`[0.810479, 0.236516, 0.205615, 0.210766, 0.255344, 0.180107, 0.359364, 0.551091]`.

The upstream confidence loss targets distribution overlap, not greedy token
match labels. Calibration and full-prefix acceptance prediction are not validated
by this small integration check. See the training URLs and hashes in manifest.json.

## Audit and reproduction

```bash
.venv/bin/python -m pytest tests/kernels/test_dspark_confidence.py \
  tests/transformers_utils/test_speculators_dspark.py -q

# Repeat for --draft 8 with separate output paths.
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
  PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. .venv/bin/python \
  benchmarks/replayssm/dual_checkpoint_smoke.py --method dspark --dual \
  --draft 4 --output /tmp/d4_smoke.json \
  --confidence-output /tmp/d4_confidence.json

.venv/bin/python benchmark_results/dspark_confidence_20260915/audit.py
```

The audit checks these saved exports against the local checkpoint and
`replayssm_dual_checkpoint_opt_d4_d8_20260915/dspark_d{4,8}.json` baseline tokens.
The initial D=4 callable-RPC serialization failure and D=8 GPU1 memory-gate
failure are retained separately. Final exports use named RPC and sequential
runs on GPU0; other processes were not interrupted.
