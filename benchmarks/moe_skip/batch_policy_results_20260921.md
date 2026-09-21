# Qwen3.6 batch-wide MoE-Skip validation

## Microbenchmark

Single A100 80GB PCIe, BF16, actual Qwen3.6-35B-A3B layers 0/19/39, five token
rows per request and matched seeded synthetic inputs. Values below are means of
the three layer medians, including routing, expert assignment/GEMMs, shared
expert and final sum. Each cell uses 30 warmups and 100 CUDA-event samples of
graph replay, with 64 MiB L2 flushing outside the timed region. Default Triton
launch configs are used; this is an isolated MoE composition, not serving.

| B | Policy | Shared-inclusive latency (us) | Speedup vs Top-8 | Unique experts |
| --- | --- | ---: | ---: | ---: |
| 1 | Native Top-8 | 226.65 | 1.000x | 36.33 |
| 1 | Top-4 | 180.57 | 1.255x | 19.33 |
| 1 | p=0.125 | 139.95 | 1.620x | 14.67 |
| 1 | Batch top half | 188.07 | 1.205x | 23.33 |
| 1 | Batch max gap | 167.94 | 1.350x | 18.67 |
| 64 | Native Top-8 | 1112.58 | 1.000x | 243.67 |
| 64 | Top-4 | 1018.71 | 1.092x | 217.00 |
| 64 | p=0.125 | 943.27 | 1.179x | 204.67 |
| 64 | Batch top half | 990.21 | 1.124x | 209.67 |
| 64 | Batch max gap | 862.21 | 1.290x | 177.67 |
| 128 | Native Top-8 | 1454.76 | 1.000x | 252.33 |
| 128 | Top-4 | 1391.27 | 1.046x | 239.33 |
| 128 | p=0.125 | 1335.30 | 1.089x | 231.00 |
| 128 | Batch top half | 1383.77 | 1.051x | 231.67 |
| 128 | Batch max gap | 1289.56 | 1.128x | 214.67 |

The max-gap policy improves the large-batch isolated MoE result over the old
per-token threshold in this sample. The Top-2 union still limits the effect:
it averages 175.33 experts at B64 and 210.67 at B128. At B128 the half policy
retains 7.905 connections/token and max-gap retains 7.686, versus 3.029 for the
old threshold; batch policies reduce whole experts while preserving many
connections to protected experts. This is not equivalent to reducing each
token to two experts. Kernel traffic bottlenecks are not established by these
counts alone.

All 45 policy/layer/batch combinations passed the native-assignment zero-weight
reference. All 198 recorded execution boundaries passed graph/eager equality.
New policies also matched an independent CPU selection reference. A separate
real-weight correctness-only run completed the same 198 checks on September 20;
it contains no timing measurements.

Raw data and completion markers:

- `benchmark_results/moe_batch_selection_20260921/latency/timings.json`
- `benchmark_results/moe_batch_selection_20260921/latency/complete.json`
- `benchmark_results/moe_batch_selection_20260920/correctness_host/validation.json`
- `benchmark_results/moe_batch_selection_20260920/correctness_host/correctness_complete.json`

These results were obtained from the task working tree based on `8e9bb904`.
Large artifacts remain local. Reproduction and exact routing semantics are in
[batch_policy_readme.md](batch_policy_readme.md).

## Full-model validation

Both seven-cell matrices completed: **14 cells, 7,168 returned tokens**. Each
cell uses four fixed prompts, 128 greedy tokens each, exact GDN, BF16 weights,
FP32 SSM storage, CUDA Graphs and a separate excluded warmup batch. B1 uses GPU1
and B4 uses GPU0, each with its own same-device AR and native-route controls.
Direct MoE-Skip uses D=4; hierarchical uses MTP depth four and one inner round,
with an outer capacity of five. Throughput is returned tokens divided by total
batch wall-clock time, including the complete generation cycle.

| B | Method | Policy | tok/s | vs method native | vs AR | Accept length | AR parity |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | AR | Native | 74.02 | 1.000x | 1.000x | - | Reference |
| 1 | MoE-Skip | Native | 85.54 | 1.000x | 1.156x | 4.971 | 2/4 |
| 1 | MoE-Skip | Half | 84.09 | 0.983x | 1.136x | 4.804 | 2/4 |
| 1 | MoE-Skip | Max gap | 83.86 | 0.980x | 1.133x | 4.778 | 2/4 |
| 1 | Hierarchical | Native | 74.88 | 1.000x | 1.012x | 4.924 | 2/4 |
| 1 | Hierarchical | Half | 75.59 | 1.009x | 1.021x | 4.832 | 2/4 |
| 1 | Hierarchical | Max gap | 77.10 | 1.030x | 1.042x | 4.544 | 2/4 |
| 4 | AR | Native | 244.70 | 1.000x | 1.000x | - | Reference |
| 4 | MoE-Skip | Native | 253.28 | 1.000x | 1.035x | 4.915 | 3/4 |
| 4 | MoE-Skip | Half | 269.36 | 1.063x | 1.101x | 4.813 | 4/4 |
| 4 | MoE-Skip | Max gap | 273.63 | 1.080x | 1.118x | 4.549 | 4/4 |
| 4 | Hierarchical | Native | 205.67 | 1.000x | 0.841x | 4.981 | 3/4 |
| 4 | Hierarchical | Half | 218.37 | 1.062x | 0.892x | 4.933 | 4/4 |
| 4 | Hierarchical | Max gap | 224.15 | 1.090x | 0.916x | 4.734 | 1/4 |

Acceptance length is `1 + accepted / steps`. The audit recomputes integer
acceptance counters, token parity and throughput, checks prompt hashes/seeds
and result hashes, and verifies the fixed experiment settings. No cell has
post-warmup JIT warnings. The matrix contains one measured repeat per cell;
small differences have no confidence interval and are not robust speed claims.

Both policies improve the B4 direct-Draft sample over native routing, while
B1 direct Draft is slightly slower. Hierarchical improves over its own native
control at B4 but remains slower than AR. Native speculative controls already
fail strict AR parity; the source of those mismatches was not resolved here.
In particular, hierarchical max-gap matches only 1/4 B4 prompts. Even the 4/4
cells are a small smoke, not proof of lossless decoding or a broad quality eval.
There is no B64/B128 full-model throughput measurement in this experiment.

Completed local artifacts:

- `benchmark_results/moe_batch_selection_20260921/b1/matrix_complete.json`
- `benchmark_results/moe_batch_selection_20260921/b4/matrix_complete.json`
- `benchmark_results/moe_batch_selection_20260921/generation_summary.json`
- `benchmark_results/moe_batch_selection_20260921/generation_summary.md`
- `benchmark_results/moe_batch_selection_20260921/generation_audit.json`

Each cell has a JSON result and log. Source hashes and resume histories remain
in each `request.json`; the cells were run while integration fixes were being
completed, rather than all from a single committed snapshot. All B4 speculative
cells used the final null-block fix, including rerun native and half direct
controls. B1 has no padded request row and its earlier complete cells were
retained. Failed attempts and pre-fix B4 controls remain in separate historical
files/directories and are excluded from the final matrix.

## Regression validation

The six suites listed in `batch_policy_readme.md` passed: **319 tests**. All
applicable pre-commit hooks and the explicit Python 3.12 mypy hook passed.
Configuration tests require both local GPUs to remain visible, since existing
negative tests construct TP2 configurations; execution used device order `1,0`
so GPU tests ran on the idle second card.

Full-model testing exposed and fixed three existing shared-Draft integration
problems: an embeddings-only call incompatible with the Target's compiled
token-ID signature, unsliced sequence-length buffers when requests finish, and
`-1` padded scratch indices where the Mamba kernels require null block `0`.
A compact CUDA core located the last issue in `_causal_conv1d_update_kernel`
at padded request index 3 of a four-request graph. The regression checks that
three active requests retain their scratch blocks and the fourth uses block 0.
Hierarchical configuration cloning also clears the resolved native top-k
before revalidating a batch policy, avoiding a false explicit-option conflict.
