# Top-1 versus Top-2 batch protection: routing-count pilot

Top-1 protection increased whole-expert skipping in all 54 paired comparisons
in this pilot. This is a routing-count experiment, with no expert GEMMs,
generation, acceptance, or latency measurements. Production routing remains
Top-2; the Top-1 option exists only in the independent benchmark reference.

## Contract

- Qwen3.6-35B-A3B real BF16 gate weights, layers 0/19/39, native normalized Top-8.
- Matched synthetic BF16 normal hidden states; seeds 42/43/44 plus layer index.
- B1/B64/B128, five token rows per request, matching the earlier MoE pilot.
- Each pair shares exactly the same native routing weights, IDs, and logits.
- Protect the union of every row's Top-1 or Top-2 experts. All connections to
  protected experts survive. Rank other active experts by aggregate weight.
- Half retains ceil(candidates / 2); max-gap keeps the prefix before the largest
  adjacent score gap. Original connection weights are preserved.
- One A100 80GB; other jobs were active. No timing conclusions are drawn.

Percentages below are arithmetic means of nine layer/seed cells per batch and
policy. Whole-expert skip fraction is `(native active - retained) / native active`,
not the fraction of all 256 experts. Connection skip fraction uses native
`tokens * 8` assignments as its denominator.

## Results

| B | Policy | Expert skip Top-2 | Expert skip Top-1 | Increase (pp) | Connection skip Top-2 | Connection skip Top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Half | 35.89% | 42.70% | 6.81 | 33.61% | 40.00% |
| 1 | Max gap | 55.63% | 80.25% | 24.61 | 51.67% | 75.28% |
| 64 | Half | 13.42% | 23.29% | 9.87 | 2.67% | 6.55% |
| 64 | Max gap | 25.53% | 45.91% | 20.38 | 9.54% | 25.18% |
| 128 | Half | 8.54% | 16.14% | 7.60 | 1.15% | 3.29% |
| 128 | Max gap | 15.97% | 30.66% | 14.69 | 3.93% | 12.18% |

| B | Native active | Protected Top-2 | Protected Top-1 | Half retained Top-2 / Top-1 | Max-gap retained Top-2 / Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.44 | 9.89 | 5.00 | 24.00 / 21.44 | 16.78 / 7.33 |
| 64 | 245.11 | 179.22 | 131.22 | 212.44 / 188.22 | 183.00 / 133.00 |
| 128 | 253.22 | 209.78 | 171.44 | 231.67 / 212.44 | 212.89 / 175.78 |

Top-1 materially reduces the protected union, although that union still grows
with batch size. Max-gap offers the larger reduction in this sample. Retained
routing-weight mass for max-gap drops from 91.54% to 76.94% at B64, and from
96.52% to 88.87% at B128. At B1 it drops from 53.55% to 30.24%. Routing mass is
not an output-error or acceptance metric; full-model quality remains untested.
Max-gap recomputes its split on a changed candidate set, so monotonic improvement
is an observation here, not a general guarantee.

These are synthetic inputs to real routers, not hidden states captured from
requests. The earlier published table used only seed 42, so this three-seed
aggregate intentionally differs. Five rows per request follow the earlier
pre-verification-shaped microbenchmark; a single direct-Draft step may have a
different number of rows. Neither B64 nor B128 denotes measured serving here.

## Validation and reproduction

108 count cells completed; all 54 Top-2 controls exactly matched the current
CUDA selector's IDs and weights. Every cell preserved each token's Top-1 and
the original retained connection weights. Four reference tests passed, covering
both protection depths and both policies with unsorted native slots. Applicable
pre-commit hooks passed.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m benchmarks.kernels.probe_moe_batch_protection \
  --output benchmark_results/moe_batch_top1_probe_reproduction --seeds 42 43 44
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/model_executor/layers/fused_moe/test_routing_top_k.py -k batch_reference -q
```

Final local artifacts are under
`benchmark_results/moe_batch_top1_probe_20260922/final/`: `results.json` holds
integer counts, fractions, source hashes and the base Git revision;
`complete.json` records coverage and the result SHA-256. The parent directory
contains the initial run before formatting was finalized. Only `final/` is the
canonical result. Raw artifacts remain local; this report and the probe are
versioned.
