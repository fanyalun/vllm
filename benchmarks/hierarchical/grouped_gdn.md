# Shared-input GDN groups in hierarchical pre-verification

`preverify_gdn_group_mode` is an opt-in approximation for Qwen3.6 MoE. Groups
3 through 9 share the first layer's pre-RMSNorm input. Each group contains
three consecutive GDN layers; the current 40-layer model uses zero-based
indices 8–10, 12–14, 16–18, 20–22, 24–26, 28–30 and 32–34.

Every layer retains its own normalization, Q/K/V/z/a/b weights, per-token gates,
Conv history, recurrent state, output gate and output projection. The real
residual and MoE chain remains sequential. The implementation never skips a
complete decoder layer. It preserves vLLM's FP32 residual sum for normalization
and BF16 residual storage; HF and vLLM need not be bitwise equivalent.

## Configuration

```json
{
  "method": "hierarchical",
  "inner_method": "mtp",
  "inner_num_speculative_tokens": 4,
  "inner_num_rounds": 4,
  "moe_skip_top_h": 4,
  "preverify_gdn_mode": "replay_tail",
  "preverify_gdn_group_mode": "full"
}
```

| Group mode | Behavior |
| --- | --- |
| `none` | Existing model path; default |
| `projection` | Compute three independent normalizations and input projections together; run each GDN core at its original layer |
| `full` | Also group Conv, recurrence, output normalization and output projection; consume the three outputs along the sequential residual/MoE chain |

The group option is independent of `preverify_gdn_mode`. State mode `none`
retains all speculative checkpoints and selects the accepted state. State mode
`replay_tail` writes only the final checkpoint and deliberately retains rejected
tails within an outer cycle. Both start each outer proposal from Target's
accepted state. `ssm_mean` and `input_mean` cannot be combined with grouping.

Initial support is CUDA BF16, unquantized Qwen3.6 MoE, BF16 Conv state, FP32 recurrent state,
TP1/PP1, one request, inner D4 and actual window widths 1–5. LoRA, sequence
parallelism, layer scaling and incompatible GDN layouts are rejected. Existing
hierarchical restrictions on prefix caching and asynchronous scheduling apply.

The hierarchical speculator calls a private execution plan over the existing
model instance. Target and Draft retain their normal entry points. No model
parameters are copied or concatenated. Temporary projections and outputs have
one-forward lifetimes; CUDA Graph owns captured allocations and reuses them on
replay. No Python hook or cross-forward intermediate cache is used in production.

## Measurement

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=benchmarks/hierarchical:. \
  .venv/bin/python benchmarks/hierarchical/run_replay_tail.py \
  --grouped-gdn --inner-method mtp --output benchmark_results/grouped_gdn/mtp
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=benchmarks/hierarchical:. \
  .venv/bin/python benchmarks/hierarchical/run_replay_tail.py \
  --grouped-gdn --inner-method dspark --output benchmark_results/grouped_gdn/dspark
.venv/bin/python benchmarks/hierarchical/run_cell.py \
  --method ar --device 0 --num-samples 4 --max-tokens 256 \
  --output benchmark_results/grouped_gdn/ar.json
.venv/bin/python benchmarks/hierarchical/summarize_replay_tail.py \
  benchmark_results/grouped_gdn
```

The online matrix has six cells: two state policies times three group modes.
Each uses the same four prompts, 256 returned tokens, three timing trials and
the existing `low_error` stopping policy. Warmup and auditing are separate from
timing. Prompt hashes and order must match AR, MTP and DSpark before token
comparisons. Acceptance uses integer totals; throughput uses total returned
tokens divided by total elapsed time in each trial. Completion does not imply
AR equality or broadly preserved task accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=benchmarks/hierarchical:. \
  .venv/bin/python benchmarks/hierarchical/run_replay_tail_cost.py \
  --grouped-gdn --output benchmark_results/grouped_gdn/cost
.venv/bin/python benchmarks/hierarchical/summarize_replay_tail_cost.py \
  benchmark_results/grouped_gdn/cost
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_grouped_gdn.py \
  --output benchmark_results/grouped_gdn/kernel_tuning.json
```

The fixed-input matrix additionally includes a benchmark-only `serial` mode
for each state policy. It shares inputs but uses independent native projections.
Three real windows, three rounds, eight cases, five boundaries and 30 repeats
produce 10,800 timing rows. Initial state restoration and explicit 128 MiB L2
eviction are outside each event pair. Graph/eager state checks and repeated
predictions must pass, and instrumented eight-token control generations must
match the uninstrumented control.

The `projection` and `gdn` timing boundaries cover the seven selected groups
using identical captured shared anchors. At these isolated boundaries, `none`
uses the same serial shared-input reference as `serial`; these numbers measure
execution cost, not unmodified model semantics. The `forward` and `combined`
boundaries run the actual complete model variant. `combined` directly measures
forward plus acceptance-aware state advancement; it is not a sum of separately
timed stages. The isolated serial reference includes eager normalization and
staging, so its speedup must not substitute for complete-forward improvement.

Kernel tuning compares the actual projection dimensions at widths 1–5 with
the three native linear calls. It reports cold-cache latency, logical weight
bytes and compiler spills. The chosen implementation uses the original weight
pointers; it reduces launch overhead without eliminating weight traffic.

## Validation

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  tests/benchmarks/test_hierarchical_measurement.py -q
```

Tests check distinct layer weights, FP32 normalization sums, Conv history,
every requested checkpoint, canaries, all widths, both Conv layouts, repeated
CUDA Graph replay and the sequential residual/MoE chain. Identical-input
recurrence retains atol=rtol=1e-3. BF16 GEMM reduction-order differences permit
adjacent BF16 rounding (atol=1e-3, rtol=8e-3); this is not bitwise equivalence.

AI assistance was used for implementation, tests and measurement tooling.
