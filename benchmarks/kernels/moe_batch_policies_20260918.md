# MoE policy kernel latency pilot

Single A100 80GB PCIe GPU0, BF16, Qwen3.6-35B-A3B checkpoint layers 0/19/39.
B1/B64/B128 with five tokens per request. Inputs are independent seeded Gaussian
hidden states, not captured prompts; all policies see identical inputs per cell.
Native means top-8 out of 256, not evaluating all 256 experts per token. Top-4
and p=0.125 preserve native weights without renormalizing retained weights.

The router uses the existing FusedTopKRouter and actual checkpoint gate weights.
Experts use the existing Triton fused_experts, including its threshold assignment
path. The benchmark reports routing/projection plus routed-expert work, and a
second boundary adding the unchanged shared expert and final sum. The latter is
an isolated functional composition, not the compiled model wrapper; graph
capture removes host launch gaps but does not reproduce all compiler fusion or
production scheduling. A device-specific tuned MoE config is absent, so this
uses existing default launch configuration. No kernels were optimized here.

Thirty warmups and 100 samples per cell. Explicit 64MiB L2 flush before each
CUDA-event-timed graph replay. The initial CUPTI helper attempt fell back to
CUDA events without L2 flush and was discarded; no dependency was installed.

Values are the mean of three layer medians, in microseconds per layer for the
whole batch. Speedup uses the same shared-inclusive boundary.

| B | Policy | Routing + experts | With shared | Speedup | Experts/token |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | top-8 | 196.95 | 226.65 | 1.000x | 8.000 |
| 1 | top-4 | 151.89 | 182.27 | 1.243x | 4.000 |
| 1 | p=0.125 | 110.25 | 139.26 | 1.627x | 2.933 |
| 64 | top-8 | 1075.03 | 1113.09 | 1.000x | 8.000 |
| 64 | top-4 | 980.65 | 1019.05 | 1.092x | 4.000 |
| 64 | p=0.125 | 908.29 | 944.98 | 1.178x | 3.040 |
| 128 | top-8 | 1405.61 | 1458.18 | 1.000x | 8.000 |
| 128 | top-4 | 1339.73 | 1395.37 | 1.045x | 4.000 |
| 128 | p=0.125 | 1284.44 | 1337.00 | 1.091x | 3.029 |

At B128, the whole batch touches an average of 252.33/239.33/231.00 distinct
experts for top-8/top-4/threshold. This helps explain why reduced per-token
assignments need not proportionally reduce memory traffic or padded GEMM work.
It is not a hardware-counter proof of a specific bottleneck. Results are
synthetic-input scaling evidence, not quality, acceptance or E2E performance.

All 54 timing cells passed graph/eager equality. Each of the 27 policy cells
passed a native-assignment reference with discarded weights zeroed, using
atol=0.01 and rtol=0.02. The real weights are unchanged across policies.
Pre-commit passed; PNG and the one-page PDF were visually checked.

Artifacts: `benchmark_results/moe_batch_policies_20260918/run/` contains
raw samples, P95, retained/unique expert counts and completion marker;
`moe_batch_policies/` contains the PNG/PDF/Markdown figure bundle. Large files
stay local. Discarded attempts are separate and excluded from aggregation.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_moe_batch_policies.py \
  --output benchmark_results/moe_batch_reproduction/run
.venv/bin/python benchmarks/kernels/plot_moe_batch_policies.py \
  --input benchmark_results/moe_batch_reproduction/run \
  --output benchmark_results/moe_batch_reproduction/moe_batch_policies
```
