# Replay-tail GDN evaluation, 2026-09-15

The implementation passes its fixed-gate recurrence and state-management tests.
Private recurrent storage falls from 360 MiB to 60 MiB. End-to-end results are
mixed: MTP slows by 2.26%, while DSpark improves by 5.19% on this four-prompt pilot.
The default remains `preverify_gdn_mode="none"`.

## Contract and coverage

- Qwen3.6-35B-A3B, BF16 weights/activations, FP32 recurrent state, TP=1, B=1.
- Four existing prompts, 256 returned tokens each, greedy sampling, top-h=4,
  inner draft length four, at most four inner rounds, existing `low_error` policy.
- MTP ran on A100 80GB PCIe GPU 0; DSpark ran on GPU 1. Some work overlapped on
  the same host. Compare paired cases within each method, not absolute speeds
  between GPUs or against a different benchmark configuration.
- All four cases/prompts were warmed. Three alternating baseline/replay-tail
  timing pairs per method contain 48 uninstrumented rows total. Separate audit
  passes contain 32 rows across the baseline, full mode, and two ablations.
- Both methods have complete 40-row markers. Output tokens repeat across warmup,
  timing, and audit within each case. No new graph was captured during timing.
- Full-model replay/eager checks passed at the actual width five. Kernel tests
  also cover widths one through five, repeated in-place updates, extreme gates,
  and canary state slots. The related test suites report **147 passed**.

Commands and case semantics are in [replay_tail.md](replay_tail.md). Local raw
artifacts, source fingerprints, outputs, counters, and completion markers are
under `/home/fanya/vllm/benchmark_results/replay_tail_20260915/`. They are kept
separate from the source commit.

## Throughput and useful work

Throughput is `1024 / summed wall seconds` for each four-prompt trial; the table
uses the median of three trials. Returned tokens per cycle clip the final cycle
at the output limit and exclude the initial token before the first proposal.
Outer acceptance uses raw accepted-draft/scheduled-draft integer counters,
including the final verification before output truncation.

| Draft | Mode | Tokens/s | Returned tokens/cycle | Outer accepted/scheduled |
| --- | --- | ---: | ---: | ---: |
| MTP | Baseline | 96.22 | 7.846 | 928/1239 |
| MTP | Replay-tail | 94.05 | 7.445 | 896/1250 |
| DSpark | Baseline | 91.49 | 7.391 | 904/1186 |
| DSpark | Replay-tail | 96.24 | 7.846 | 907/1250 |

MTP baseline trials were 97.02, 95.89, 96.22 tokens/s; replay-tail trials were
94.46, 94.05, 93.75. DSpark baseline trials were 91.66, 91.33, 91.49;
replay-tail trials were 95.94, 96.24, 96.56.

The ablations help separate the two approximations. These are audited useful-work
counts, not timing comparisons of alternative production kernels:

| Draft | Mode | Inner accepted/proposed | Returned tokens/cycle |
| --- | --- | ---: | ---: |
| MTP | Baseline | 840/1596 | 7.846 |
| MTP | Gates only | 894/1688 | 7.907 |
| MTP | Tail only | 862/1660 | 7.612 |
| MTP | Replay-tail | 823/1708 | 7.445 |
| DSpark | Baseline | 753/1732 | 7.391 |
| DSpark | Gates only | 815/1616 | 7.786 |
| DSpark | Tail only | 790/1696 | 7.500 |
| DSpark | Replay-tail | 827/1692 | 7.846 |

The combination has different effects with the two drafters; these counts do not
establish a general accuracy or acceptance improvement. Audited pre-verifier
forward time also increased: MTP 9.49 to 10.08 ms, DSpark 9.57 to 10.19 ms.
These spans use events and different generated candidate streams, so they are
diagnostics rather than matched-input forward microbenchmarks.

## Storage and kernel timing

For 30 GDN layers, private SSM allocation is 360 to 60 MiB (83.33% reduction).
Private Conv allocation is 64.69 to 3.75 MiB, retaining acceptance-aware history.
No scheduler-visible Target cache capacity change is claimed.

At width five, logical full-state writes fall from 10 to 2 MiB per layer, or
300 to 60 MiB across the 30 layers. The final kernel reports zero register spills
for every tested width. It does reread the initial state in the tail pass.

| Actual width | Existing recurrence, microseconds | Replay-tail, microseconds |
| --- | ---: | ---: |
| 1 | 5.53 | 19.97 |
| 2 | 10.44 | 20.68 |
| 3 | 12.90 | 21.20 |
| 4 | 15.46 | 21.20 |
| 5 | 18.02 | 21.91 |

These are medians of three cold-L2 CUDA-graph event trials on GPU 0, with matching
fixed gates and no gate projection in either path. CUPTI Python was unavailable;
FlashInfer used its event fallback with rotating inputs. Logical bytes are not
measured DRAM transactions. The block calculation is currently slower than the
existing recurrence at these small widths, despite lower state-write volume.

## AR comparison remains a failed gate

The original hierarchical baseline already differs from AR. The new mode also
does not reproduce AR on every prompt. The following entries are common-prefix
lengths in output tokens; 256 means the entire requested output matched:

| Draft/mode | Prompt 0 | Prompt 1 | Prompt 2 | Prompt 3 |
| --- | ---: | ---: | ---: | ---: |
| MTP baseline | 4 | 175 | 256 | 2 |
| MTP replay-tail | 256 | 190 | 256 | 2 |
| DSpark baseline | 4 | 134 | 206 | 2 |
| DSpark replay-tail | 4 | 175 | 206 | 2 |

This pilot is not an AR-equivalence certification or a model-quality evaluation.
The implementation's tested correctness claim is narrower: causal fixed-gate
outputs/tail, acceptance-aware Conv advancement, private-state isolation, outer
reset, and graph/eager consistency. The AR discrepancy is retained explicitly
in `summary.json` as `all_ar_equal=false`.

AI assistance was used to implement, test, and analyze this experimental change.
