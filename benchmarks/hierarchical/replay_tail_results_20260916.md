# Recurrent replay-tail results: 2026-09-16

## Implementation and scope

Qwen3.6-35B-A3B, unquantized BF16, FP32 SSM, A100 80GB, TP1, B1, MoE top-h 4.
All positions now use their own gates. The private kernel holds recurrence in registers
and writes only the complete tail. Value tile 16 with four warps was selected from
the 16/32 by 4/8 sweep. Output normalization explicitly uses the fused CUDA path
inside the eager custom-op body. Conv history advances in place without index tensors.
Target state stays canonical; rejected SSM tails are deliberately reused until the
next outer proposal. The default mode remains `none`.

## Fixed-input GPU latency

Three real windows (positions 123, 7, 38), one anchor plus four drafts, three rounds,
30 measured alternating replays after five untimed replays per point. Each case restores
identical initial state outside timing. Events surround graphs; 128 MiB L2 eviction
precedes every sample. There are 3,240 timing rows. Isolated GDN uses identical hidden
inputs at every layer. Advancement accepts two drafts. Table values average the nine
per-window/per-round medians. Combined is directly measured, not a sum of stages.

| Variant | Forward ms | 30 GDN layers ms | Advance ms | Combined ms | Combined reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8.3941 | 2.8650 | 0.5779 | 8.9533 | 0.00% |
| recurrent | 8.0469 | 2.5165 | 0.3822 | 8.4217 | 5.94% |
| optimized | 8.0480 | 2.5169 | 0.0655 | 8.1084 | 9.44% |

`recurrent` includes the new recurrence and fused normalization but retains the old
Conv advancement; `optimized` adds the new Conv shift. Across all nine paired points,
the final combined reduction ranges from 9.09% to 9.92%.
Forward excludes draft generation, rejection sampling, Target validation and CPU scheduling.

## Full generation

Four fixed prompts, 256 returned tokens each, three alternating timing trials,
inner D4 and up to four rounds with the existing low_error policy. Auditing is disabled
during timed trials. MTP ran on GPU 1 and DSpark on GPU 0, with an independent model
on each GPU. Throughput is the median of per-trial total returned tokens / total time.

| Method | Baseline tokens/s | Replay-tail tokens/s | Throughput change |
| --- | ---: | ---: | ---: |
| mtp | 98.67 | 105.02 | +6.43% |
| dspark | 91.87 | 98.44 | +7.15% |

### Acceptance and output checks

| Method | Case | Outer accepted / scheduled | Inner accepted / proposed | Exact AR matches |
| --- | --- | ---: | ---: | ---: |
| mtp | none | 928/1239 (74.90%) | 840/1596 (52.63%) | 1/4 |
| mtp | replay_tail | 916/1301 (70.41%) | 888/1652 (53.75%) | 0/4 |
| mtp | tail_only | 923/1277 (72.28%) | 862/1660 (51.93%) | 0/4 |
| dspark | none | 904/1186 (76.22%) | 753/1732 (43.48%) | 0/4 |
| dspark | replay_tail | 914/1239 (73.77%) | 794/1780 (44.61%) | 0/4 |
| dspark | tail_only | 908/1214 (74.79%) | 790/1696 (46.58%) | 1/4 |

AR was rerun for this experiment. The summary verified prompt hashes and ordering
across AR, MTP, and DSpark before comparing tokens. Baseline already fails strict AR
equivalence. Removing gate sharing does not establish output-quality recovery: final
outer acceptance remains below baseline. These four prompts are a performance and
semantics pilot, not a broad accuracy evaluation or lossless-acceleration claim.

`tail_only` is the original per-position checkpoint implementation with tail reuse
after rejection. It is not a performance proxy for the optimized kernel. Its 256-token
outputs match the new implementation on only 1/4 prompts for each inner method.
Kernel checks establish numerical tolerance, not bitwise equality; small arithmetic
differences can change later greedy decisions. No strict native-tail parity is claimed.

Private SSM allocation is 360 MiB -> 60 MiB; private Conv allocation is 64.69 MiB
-> 3.75 MiB. These allocations do not change scheduler-visible Target capacity.

## Kernel evidence and validation

The explicit-eviction tuning experiment compares packed BF16 inputs with per-token
gates. At width 5: native Triton wrapper 35.84 us, old parallel per-token-gate kernel
27.65 us, selected recurrent kernel 17.41 us (representative third-trial medians).
The native Triton wrapper includes input staging and is distinct from the model's
fused CUDA GDN baseline; kernel speedups must not replace model-level measurements.
The selected kernel uses 56 registers at width 1 and 64 at widths 2–5, with zero spills.

The standard kernel benchmark also completed with its FlashInfer CUDA-event fallback
(CUPTI unavailable); it uses a different replay/cache protocol, so its absolute timings
are recorded separately in kernel.json and not mixed with the explicit-eviction sweep.

- 177 tests passed: hierarchical configuration/state, GDN kernels, benchmark auditing.
- Per-token gates, packed BF16/FP32 inputs, extreme gates, repeated tail overwrite,
  canaries and CUDA Graph checks passed at atol=rtol=1e-3 without loosening tolerances.
- In-place Conv shifts matched the reference for accepts 0–4 and both storage layouts.
- Fused output normalization matched the per-token reference for SiLU/sigmoid and
  BF16/FP32 norm weights; this concerns the output gate, not the recurrent alpha/beta.
- Fixed-input instrumentation preserved all nine 8-token generation controls.
- E2E outputs repeated across warmup, timing and audit; each method completed 36 rows.
- AR identity regressions cover differing prompts, ordering, duplicate indices and
  missing identities. The correct request set still summarizes successfully.

## Artifacts and reproduction

Local root: `benchmark_results/replay_tail_recurrent_20260916/`.
Final data: `cost_final/`, `final/{mtp,dspark,ar.json,summary.json,summary.csv}`,
`tuning.json`, `kernel.json`, `tests_release.log`, `hooks_final.log` and `manifest.json`.
The earlier `cost/` and `mtp/` directories are intermediate implementation measurements
and are not used for the final tables.

Run commands are documented in [replay_tail.md](replay_tail.md). Use this artifact root
for the output paths, GPU 1 for MTP/AR, and GPU 0 for DSpark/fixed-input cost. Run
`summarize_replay_tail_cost.py` on `cost_final/` and `summarize_replay_tail.py` on `final/`.
Runtime source hashes in both E2E contracts match the final source files. Scripts and
compact reports are published; raw measurements remain local.

AI assistance was used for implementation, tests, and benchmark analysis.
