# Gemma hierarchical MTP metadata repair

Date: 2026-09-10. Base commit: `5a71a9b50520949acfff98ed2fbc3cb3e700c07f`.

## Change

Gemma's Q-only MTP first-step CUDA graph captures the outer Target's Triton
attention sequence lengths and query offsets. Later hierarchical rounds use a
different input batch, so replay previously consumed stale lengths.

The runtime now refreshes `seq_lens` and `query_start_loc` around the second and
subsequent inner MTP proposals, then restores the canonical Target buffers in a
`finally` block. Save buffers are preallocated during attention setup. Their
addresses remain stable and the hot path does not clone them. The existing B=1,
synchronous, text-only hierarchy restrictions make this temporary use serialized.
The repair is enabled only for Gemma hierarchical MTP; DSpark and Qwen do not
refresh these buffers. CUDA graphs remain enabled.

## Small-sample results

Four original prompts (human_eval, alpaca, gsm8k, ultra_feedback), 512 output tokens
each, seed 0, greedy, TP=1, B=1, max model length 1024, GPU memory utilization
0.95, prefix cache disabled. Hierarchy uses D=4, N=4 and shared Target/P weights
with P top-4. Each performance case has a full 512-token warmup followed by
uninstrumented, profiled and uninstrumented-repeat passes.

| Metric | Previous hierarchy MTP | Repaired hierarchy MTP | Change |
| --- | ---: | ---: | ---: |
| First uninstrumented throughput, tok/s | 140.10 | 215.88 | +54.09% |
| Repeated uninstrumented throughput, tok/s | 142.40 | 223.02 | +56.61% |
| Candidates per complete loop | 10.50 | 18.25 | +73.88% |
| Final emitted tokens per loop, including Target bonus | 11.14 | 18.40 | +65.21% |
| Complete loop time, ms | 74.71 | 77.00 | +3.06% |
| Complete loop ms per emitted token | 6.71 | 4.19 | -37.62% |

The previous numbers are archived measurements from
`../gemma4_d16_20260910/`, not an interleaved before/after run. Acceptance and
cycle time come from the same profiled pass. First/repeated throughput comes
from passes without diagnostic duplicate proposals. All four requests have
identical output token IDs across the three repaired performance passes; no
post-warmup JIT compilation was logged.

| Inner round | Previous acceptance | Repaired acceptance |
| --- | ---: | ---: |
| 1 | 86.89% | 88.51% |
| 2 | 28.65% | 89.41% |
| 3 | 21.35% | 86.71% |
| 4 | 25.54% | 91.67% |

The repair recovers the later-round proposal yield. Complete-loop time grows
slightly as the final Target verifies a longer candidate sequence. Four P graph
replays still take 40.86 ms per loop, the small drafter takes 13.13 ms, and final
Target execute/sample takes 16.28 ms. This repair does not remove the P backbone
cost identified in the preceding diagnosis.

The freshly rerun native MTP D16 control reaches 419.49 tok/s on the first pass
and 428.45 tok/s on the repeat, with 27.23 ms per cycle and 12.77 emitted tokens
per cycle. Repaired hierarchy reaches about 52.1% of native MTP's repeated
throughput. Native uses its historical async/default-multimodal configuration;
hierarchy uses sync/text-only configuration. This is the previous native-control
setup, not a claim of otherwise identical execution modes. Both native measured
passes match their profiled output token IDs on all four prompts.

The final Target accepts 95.31% of the repaired hierarchy's materialized draft
candidates, excluding its own correction/bonus token. The cycle accounting keeps
the whole final verification, which can sample beyond the output-token limit;
returned request lengths remain exactly 512. It excludes initial
prefill-attached proposals and terminal proposals without verification.

## Validation and remaining limitation

- Four-prompt, 512-token same-input graph/fresh-metadata probe: **357/357** later
  inner-round proposals match. Captured Target lengths equal current inner-batch
  lengths during every checked proposal. This probe repeats drafting and is
  excluded from throughput claims.
- **48 targeted tests pass**, including normal and exceptional buffer
  restoration, captured-address stability, and existing hierarchy/state tests.
- Scoped pre-commit and mypy 3.12 pass.
- The independent strict P batch-versus-sequential oracle **still fails during
  the 64-token warmup**, at position 138. Four draft tokens match, but the bonus
  prediction is 531 in batch execution versus 1082 sequentially. Maximum logit
  differences by row are `[0.3125, 5.984375, 1.23828125, 1.8671875, 1.125]`.
  No completed oracle request is counted as passing. The original implementation
  also failed this gate. The MTP graph metadata repair is validated, but this
  report does not certify the complete hierarchy as lossless.

`summary.csv`, `phases.csv`, `acceptance.json`, `metadata_audit.json` and raw
config/result/log archives contain the measured evidence. The separate native
MTP D16 rerun uses its historical async/default-multimodal settings; see
`comparison.json` for its final measured values and configuration caveat.
