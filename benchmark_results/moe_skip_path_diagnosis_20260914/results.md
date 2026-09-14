# Why full-budget MoE-Skip appeared faster than AR

The apparent h=8 speedup is against synchronous AR, which leaves substantial
gaps between GPU work. Asynchronous AR removes most of this disadvantage and
beats h=8/D=8 on both models in this diagnostic. Full-budget drafting is not
reducing model work: it pays for full draft forwards plus Target verification.

## Uninstrumented request measurements

One A100 80GB (GPU 1), TP1/B1, greedy, prefix caching off, CUDA graphs enabled,
max model length 1024 and batched-token budget 4096. The first two prompts of
the archived static-budget dataset were each warmed up with 512 outputs. Each
mode then ran those two prompts three times, with exactly 512 output tokens per
request. There are 36 measured requests and 18,432 outputs across six cells.
This is a small reproduction, not a replacement 16-prompt budget matrix.

| Model | Sync AR, ms/output | h=8/D=8, ms/output | Async AR, ms/output | h=8 speedup over sync AR | h=8 latency overhead over async AR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 | 10.5120 | 7.8363 | 6.9019 | 1.3415x | 13.54% |
| Gemma4 | 10.7866 | 8.3712 | 7.5702 | 1.2885x | 10.58% |

Only the AR scheduling flag changes between the two AR configurations. Their
six Qwen outputs are all exactly equal. Gemma has three of six exact matches,
so Qwen provides the stronger control for identical generated work. Each of
the two prompts repeats exactly within each of the six individual cells.

## The GPU backbone did not become cheaper

Separate 512-token event diagnostics show almost unchanged AR backbone graph
times. Graph timings exclude prefill here and are per invocation. Draft graph
replay includes logits, argmax and input updates, so its boundary is broader
than the Target backbone graph; it must not be interpreted as backbone-only.

| Model | Sync AR backbone | Async AR backbone | h=8 draft graph | h=8 Target verify graph |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 | 5.943 ms | 5.936 ms | 6.570 ms | 13.051 ms |
| Gemma4 | 6.528 ms | 6.476 ms | 7.593 ms | 12.593 ms |

Each h=8 proposal replays eight draft graphs: the full proposal averages
53.022 ms for Qwen and 60.892 ms for Gemma. Verification adds work after the
proposal. These are nested stream intervals, not additive kernel partitions.

## The timeline explains the apparent contradiction

Separate 64-output CPU/CUDA profiles retain prefill and terminal proposals.
The table uses the union of kernel intervals across GPU streams, not the sum
of overlapping kernel durations. Gaps are the remainder between the first
kernel start and last kernel end. Profiling changes absolute timings; these
numbers are diagnostic and must not replace uninstrumented throughput.

| Model | Mode | CUDA kernel count | Kernel busy union | Gaps |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 | Sync AR | 63,517 | 454.66 ms | 409.85 ms |
| Qwen3.6 | Async AR | 63,517 | 454.94 ms | 134.69 ms |
| Qwen3.6 | h=8/D=8 | 83,127 | 622.13 ms | 130.20 ms |
| Gemma4 | Sync AR | 38,093 | 496.24 ms | 304.27 ms |
| Gemma4 | Async AR | 38,093 | 495.74 ms | 48.11 ms |
| Gemma4 | h=8/D=8 | 42,594 | 604.88 ms | 44.31 ms |

AR kernel counts and busy time are essentially unchanged by asynchronous
scheduling, while gaps shrink substantially. h=8 has more kernel work than
either AR mode. This supports scheduling/host-wait gaps as the explanation
for beating synchronous AR, rather than a reduction in full-model work.
The timeline does not apportion every gap among individual CPU functions.

The code is consistent with this observation: `EngineCore.step()` schedules,
executes, receives output, and updates the scheduler before the next step;
`step_with_batch_queue()` can queue subsequent work before collecting output.
`AsyncOutput.get_output()` waits on the output-copy event. MoE-Skip instead
runs multiple draft graph replays within one worker proposal, amortizing the
outer iteration overhead across accepted output tokens.

## Audit and interpretation limits

- All six cells completed. Dataset fingerprints, output counts, repeated
  outputs and diagnostic output prefixes pass the report audit.
- No post-warmup JIT warnings occur in any cell. Event-instrumented request
  time divided by its uninstrumented same-prompt mean ranges from 0.9905 to
  1.0117; event probes do not explain the approximately 1.3x effect.
- Modes ran sequentially rather than interleaved, with three repeats per
  mode and two prompts. No confidence interval or universal speedup is claimed.
- h=8 output parity with sync AR fails for both prompts in both models. That
  separate issue remains unresolved; it is not the explanation substituted
  for the scheduling-performance finding above.
- The earlier static matrix used sync AR for all speedup denominators. Those
  values are not speedups over the faster asynchronous AR baseline. Do not
  divide the old 16-prompt matrix by this two-prompt AR measurement. A formal
  comparison needs an asynchronous AR control on the same 16 prompts.

## Reproduction and files

Raw `e2e.json`, `diagnostics.json`, `trace.json.gz`, configs, commands and logs
are retained under each model/mode directory. `costs.csv`, `phases.csv` and
`kernel_activity.csv` are generated by `analyze_performance_paths.py`.
`audit.json` and the PNG/PDF figure are generated by `report_performance_paths.py`.
The archived driver and worker snapshots are the measured source versions;
the subsequently published driver additionally invokes analysis on completion.
The background measurement completed successfully; its separate queued
analysis did not produce an artifact, so analysis was run explicitly afterward.

```bash
.venv/bin/python benchmarks/moe_skip/analyze_performance_paths.py --output benchmark_results/moe_skip_path_diagnosis_20260914
MPLCONFIGDIR=/tmp/moe_skip_path_mpl .venv/bin/python benchmarks/moe_skip/report_performance_paths.py --output benchmark_results/moe_skip_path_diagnosis_20260914
```
