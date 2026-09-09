# Hierarchical decoding: previous-configuration measurements

The measurement matrix is complete. **Strict numerical-equivalence gates failed in the implementation validation and remain failed.** These are performance and acceptance measurements of that experimental implementation.

## Configuration and measurement boundaries

Qwen3.6-35B-A3B, TP1, B=1, greedy, seed=0, ignore_eos=True, 512 output tokens/request, max_model_len=1024, GPU memory utilization=0.95. Target top-k=8 and shared-weight MoE-Skip pre-verifier top-h=4. MTP/DSpark inner D=4 and N=1/2/4/8. CUDA Graph execution.

Nominal budget D*N=4/8/16/32 aligns the old D. Materialized candidates also contain each inner round's recovery/bonus token, so capacity is N*(D+1)=5/10/20/40. Actual candidate counts are reported separately.

- E2E: the original 16 prompts, one full 512-token warmup, GPU1, max_num_batched_tokens=4096, no measurement instrumentation. 8192 / sum(request wall time); includes prefill, decode, and offline API.
- Drafting time: the same 16 prompts and warmup on GPU1. CUDA events enclose one complete outer propose(), including all N small-draft/pre-verifier rounds and state handling. Target verification is excluded. Decode-only call-weighted mean, with prefill, median and P90 in summary.csv. Events are collected after each request. CPU submission time is separate and overlaps the GPU stream interval; do not add the two.
- Acceptance: 128 original prompts, no extra warmup, max_num_batched_tokens=1024, on A100 80GB PCIe device(s) [0, 1]. Mean length = 1 + sum(final-Target accepted drafts) / Target verify calls, including the final Target recovery/bonus. Worker counts are cross-checked against every scheduler per-step metric. Counts are before the final max_tokens truncation, matching the old metric.

## Results

| Inner model | N | D*N | Tokens/s | vs AR | Acceptance length | Mean scheduled candidates | Full loop ms | CPU submit ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hierarchical_mtp | 1 | 4 | 111.205 | 0.839x | 4.4695 | 3.7156 | 22.440 | 32.687 |
| hierarchical_mtp | 2 | 8 | 110.236 | 0.832x | 6.8977 | 6.8945 | 40.193 | 52.775 |
| hierarchical_mtp | 4 | 16 | 102.203 | 0.771x | 10.5972 | 13.2472 | 75.972 | 90.881 |
| hierarchical_mtp | 8 | 32 | 77.179 | 0.582x | 14.1865 | 25.1230 | 147.194 | 166.098 |
| hierarchical_dspark | 1 | 4 | 99.142 | 0.748x | 4.1600 | 3.3778 | 22.500 | 32.174 |
| hierarchical_dspark | 2 | 8 | 102.851 | 0.776x | 6.5993 | 6.4733 | 39.301 | 51.628 |
| hierarchical_dspark | 4 | 16 | 97.590 | 0.737x | 10.3147 | 12.7418 | 74.206 | 89.060 |
| hierarchical_dspark | 8 | 32 | 75.510 | 0.570x | 14.3898 | 24.6334 | 145.096 | 164.126 |

Ratios against historical MoE-Skip at the same nominal budget:

| Inner model | N | Throughput ratio | Acceptance-length ratio | Full-loop-time ratio (lower is faster) |
| --- | ---: | ---: | ---: | ---: |
| hierarchical_mtp | 1 | 0.966x | 0.996x | 0.963x |
| hierarchical_mtp | 2 | 0.981x | 0.961x | 0.867x |
| hierarchical_mtp | 4 | 1.107x | 0.974x | 0.818x |
| hierarchical_mtp | 8 | 1.122x | 0.969x | 0.792x |
| hierarchical_dspark | 1 | 0.861x | 0.927x | 0.965x |
| hierarchical_dspark | 2 | 0.915x | 0.919x | 0.847x |
| hierarchical_dspark | 4 | 1.057x | 0.948x | 0.799x |
| hierarchical_dspark | 8 | 1.098x | 0.983x | 0.780x |

Best measured hierarchical cell: hierarchical_mtp, N=1, 111.205 tokens/s. Historical AR: 132.500 tokens/s.

## Comparability and limitations

Historical baselines are snapshots in baselines/ with source paths/hashes in audit.json. They were not rerun contemporaneously. One measurement per cell; no confidence intervals or same-prefix pairing. Acceptance and performance queues ran concurrently on separate GPUs. Per-cell device indices are in audit.json. Historical Qwen3.6 and Gemma4 performance queues also ran on separate GPUs concurrently.

The implementation requires disabled multimodal inputs, disabled async scheduling, and disabled prefix caching. Historical MTP/DSpark could use automatic async scheduling; old acceptance used the automatic prefix caching default and additional logit tracing. These restrictions prevent an exact match of all resolved runtime settings. Old acceptance MTP used GPU0 and MoE-Skip GPU1. All three experimental passes preserve their respective historical sample and scheduler-token-budget settings.

Historical DSpark E2E/timing exists only at D=4/8. The available Qwen3.6 128x512 acceptance summary has no DSpark baseline; no missing points are interpolated or inferred from throughput.

The instrumented timing pass and uninstrumented E2E pass do not always follow identical token trajectories. Exact-request counts against each other and the historical AR reference are in summary.csv; zero-based first differences against AR are in audit.json. These measurements cannot isolate component costs by subtracting values from different passes.

Audited 24 cells, 1280 measured requests, 655360 output tokens. Excluded startup/warmup work and the interrupted acceptance pilot are outside these totals. Token IDs, raw loop events, final verification counts and logs are retained.

## Reproduction

```bash
.venv/bin/python benchmarks/hierarchical/compare_previous.py --run-dir benchmark_results/hierarchical_previous_config_new --phases e2e timing --cuda-device 1
.venv/bin/python benchmarks/hierarchical/compare_previous.py --run-dir benchmark_results/hierarchical_previous_config_new --phases acceptance --cuda-device 0 --resume
.venv/bin/python benchmarks/hierarchical/summarize_previous.py --run-dir benchmark_results/hierarchical_previous_config_new
```
