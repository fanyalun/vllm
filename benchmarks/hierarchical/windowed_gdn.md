# Windowed private GDN preverification

This opt-in experiment adds `windowed_three_level` to the existing hierarchical
controller. The validated scope is Qwen3.6-35B-A3B, BF16 weights and Conv, FP32
recurrent state, MTP D4, B1, TP1, greedy requests, and at most four inner rounds.
Prefix caching, asynchronous scheduling, quantization, LoRA, and GDN grouping
remain unsupported. The original `exact` and `three_level_p50` policies retain
their update and checkpoint paths.

```python
speculative_config = {
    "method": "hierarchical",
    "inner_method": "mtp",
    "inner_num_speculative_tokens": 4,
    "inner_num_rounds": 4,
    "moe_skip_top_h": 4,
    "hierarchical_stop_policy": "balanced",
    "draft_sample_method": "greedy",
    "preverify_gdn_mode": "replay_tail",
    "preverify_gdn_group_mode": "none",
    "preverify_gdn_update_policy": "windowed_three_level",
    "preverify_gdn_mode_window_size": 5,
    "preverify_gdn_tau_alpha": 0.95,
    "preverify_gdn_tau_beta": 0.36328125,
    "preverify_gdn_optimization": "none",
    "preverify_gdn_tail_policy": "carry",
}
```

Actual `SamplingParams.temperature` must be zero; the CPU request validator
checks this independently of the drafting method. Derived MTP and MoE-Skip
configurations reset all window-specific fields. The fields participate in
the hierarchical graph hash.

## State and call path

`HierarchicalSpeculator.propose` calls `PreverifyState.begin` to copy the
canonical accepted Target prefix. Each GDN layer owns one private
`[1, HV, V, K]` state. Windowed mode allocates no tail or repair tensors and
never swaps recurrent buffers. A request identity, monotonically increasing
outer epoch, CPU initialization flag, and fixed-address GPU validity flag
guard the slot. Finishing a proposal invalidates it; every subsequent proposal
initializes from Target again.

`qwen_gdn_mean_projected` still performs the full QKV/BA projections and Conv.
It passes the actual GPU query offsets to `PreverifyState.update`, which calls
`windowed_replay_tail_update`. Windowed eager preverification explicitly passes
its mutable Conv and SSM tensors to the existing custom op; the op declares
both mutations. Target compilation retains the existing deferred cache lookup.
The context manager restores Target cache bindings on normal and exceptional
exits. Full-head fused RMSNorm/gating and output projection are unchanged.

Each program owns complete K dimensions and disjoint V rows. It loads its SSM
rows once, keeps them resident across subwindows, and stores them at the end
only if at least one subwindow updated the head. A final Skip window cannot
discard an earlier update. Empty inputs and invalid slots return before any
state access. The kernel supports actual lengths 0 through 16; online calls
retain their existing 1 through 5 limit.

At each call-local window start, BF16-rounded sigmoid beta selects Full when
it reaches the beta threshold. Otherwise FP32 `libdevice.exp(g)` selects Skip
strictly above the alpha threshold and Decay at or below it. Using libdevice
for classification avoids an observed adjacent-float boundary discrepancy in
Triton's approximate exponential. Full recurrence still uses the original
FP32 beta, normalization epsilon, exponential, and operation order. Decay and
Skip branch around K/V loads and the delta update. Skip also omits later gate
evaluation within its window.

Inner rejection retains the entire approximate SSM tail. Conv advances by the
real accepted-draft count; the correction becomes the next consumed anchor.
Attention uses the existing shared pool and slot mappings: committed prefix
values must remain unchanged, while speculative suffix writes can be replaced
before Target consumes them. Final verification and Target state commitment
remain in the existing Target path. Even forced-Full single-slot carry is an
approximate online state policy after rejection.

## Optimizations and controls

| Case | Window | Optimization | Beta threshold |
| --- | ---: | --- | ---: |
| V0 | Native exact GDN | Native snapshots/recovery | N/A |
| V1 | 5 | Runtime-forced Full, single-slot carry | 0 |
| V2 | 1 | Per-input three-way update | 0.36328125 |
| V3 | 5 | Windowed update | 0.36328125 |
| V4-D | 5 | `cumulative_decay` | 0.36328125 |
| V4-Q | 5 | `multi_query` | 0.36328125 |
| V4-DQ | 5 | `combined` | 0.36328125 |

Cumulative Decay keeps the window's entering state as an anchor and applies
`exp(sum(g))` to each readout, materializing the state only at the window end.
This changes rounding and has a separate FP32 reference. Multi-query mode
loads and normalizes bounded query tiles only in non-Full branches; Skip and
cumulative Decay share the tiled state reduction. The Q-only Decay variant
preserves stepwise state rounding. No Tensor Core or recurrent dtype change
is involved. Query tiling does not remove additional global SSM reads beyond
the single read already performed by V3.

Graph keys distinguish actual width, policy, window, optimization, and action
audit mode. Gates and validity stay on the device, allowing different actions
on the same captured graph. Capture and warmup restore private caches before
the first real replay. The benchmark's graph/eager check compares one entire
call with the same window semantics.

## Reproduction and interpretation

All artifacts for this run are under
`benchmark_results/windowed_gdn_20260917/`. Large tensors, PTX, profiles, and
per-request tokens are kept locally. The source baseline is
`86be3421aafc971dbe95ea9280320f6b16908e46`; manifests and source snapshots
identify the uncommitted implementation used for each measurement.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  tests/benchmarks/test_hierarchical_measurement.py -q
```

The kernel harness `benchmarks/kernels/benchmark_gdn_windowed.py` performs
50 warmups and 200 individual graph replays per configuration. It restores
the identical initial SSM outside every timed sample and reports raw samples,
median, P95, registers, spills, and PTX. Captured input strides are restored
when available. T6/10/16 made by repeating captured T5 inputs are shape probes,
not new long-window model traces. `audit_gdn_windowed.py` supplies an independent
FP32 reference and separately reports differences from stepwise Decay.

The online harness `run_replay_tail.py --three-level --drain-device` warms every
case and prompt, alternates case order over five repetitions, checks complete
token repeatability, and rejects new timed graph captures or JIT compilation.
The timer includes a worker-side device drain after `LLM.generate`, including
terminal proposal housekeeping and the common RPC/barrier cost. Audit events
and action counters run separately from uninstrumented generation.

`run_windowed_quality.py` compares teacher-forced AR suffixes at boundaries
32/96/160/240, with L5/L16 and continuous 4xD5 at boundary 96. Its boundary
convention is explicit: prefill the prompt and the first `boundary-1` generated
tokens, consume the next AR token as anchor, and compare following predictions.
It checks canonical GDN tensors and the committed attention prefix, and repeats
the same-input native forward after the approximate variants. This fixed-input
quality audit is distinct from actual reject/correction traces in online runs.

`check_windowed_api.py` validates the public combined configuration without
benchmark case switching, including rejection of a stochastic request.
Report coverage, numerical checks, AR token parity, and performance separately.
An AR disagreement must not be labeled lossless merely because the private
state isolation checks pass.

The measured results and limitations are in
[the evaluation report](windowed_gdn_results_20260917.md). To reproduce the
matrix in a new artifact directory with the existing environment:

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
result_dir=benchmark_results/windowed_gdn_reproduction
dataset=benchmark_results/three_level_p50_20260916/final_prompts.jsonl
mkdir -p "$result_dir"
.venv/bin/python benchmarks/hierarchical/run_three_level_ar.py \
  --dataset "$dataset" --output "$result_dir/ar" --repeats 5 --drain-device
.venv/bin/python benchmarks/hierarchical/run_windowed_quality.py \
  --dataset "$dataset" --ar "$result_dir/ar/results.json" \
  --output "$result_dir/quality"
.venv/bin/python benchmarks/kernels/benchmark_gdn_windowed.py \
  --inputs "$result_dir/quality/raw_inputs.pt" --output "$result_dir/kernel_final"
.venv/bin/python benchmarks/kernels/audit_gdn_windowed.py \
  --inputs "$result_dir/quality/raw_inputs.pt" \
  --output "$result_dir/recurrent_correctness_final.json"
.venv/bin/python benchmarks/hierarchical/run_replay_tail.py \
  --inner-method mtp --three-level --drain-device \
  --cases none:carry:balanced three_level:carry:balanced \
    windowed_full:carry:balanced windowed_token:carry:balanced \
    windowed:carry:balanced windowed_decay:carry:balanced \
    windowed_query:carry:balanced windowed_combined:carry:balanced \
  --samples 16 --max-tokens 256 --repeats 5 --seed 42 --dataset "$dataset" \
  --action-audit --profile-output "$result_dir/final/profile.json" \
  --output "$result_dir/final"
.venv/bin/python benchmarks/hierarchical/check_windowed_api.py \
  --dataset "$dataset" --output "$result_dir/public_api.json"
.venv/bin/python benchmarks/hierarchical/run_windowed_quality.py \
  --dataset "$dataset" --ar "$result_dir/ar/results.json" \
  --output "$result_dir/rejections" --samples 1 --boundaries 96 --rejection-audit
.venv/bin/python benchmarks/kernels/benchmark_gdn_windowed.py \
  --inputs "$result_dir/quality/raw_inputs.pt" --output "$result_dir/kernel_warps8" \
  --warps 8
.venv/bin/python benchmarks/hierarchical/summarize_windowed.py --root "$result_dir"
```

The eight-warp experiment is expected to stop at its forced-Full bitwise gate
on the captured input; preserve `failure.json`. A focused reproduction uses
`--warps 8 --lengths 5 --cases v1` with a fresh output directory. Four warps
remain the validated production setting.

The forced-rejection audit constructs mismatching candidates through real
forwards, computes the actual greedy acceptance count, and executes four rounds
without early stopping. It checks `[4,4,4,4]`, `[2,2,2,2]`, `[0,4,4,4]`, and
`[0,0,0,0]` trajectories, preserving the SSM tail bitwise while advancing Conv
and consuming the correction at the next position. It runs inside the same
committed-prefix and same-input Target isolation audit.
