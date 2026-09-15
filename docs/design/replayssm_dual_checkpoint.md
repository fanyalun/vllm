# ReplaySSM dual checkpoint

Qwen3.5/3.6 GDN speculative decode can retain a candidate state at the end of
each verification window. Full acceptance promotes that state to the next
checkpoint by changing a device-resident slot selector. Partial acceptance
retains the existing checkpoint and commits only the accepted input prefix to
the ReplaySSM history.

## Enable

```bash
vllm serve /path/to/qwen_model \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --mamba-cache-mode none \
  --use-replayssm-spec \
  --replayssm-spec-dual-checkpoint \
  --replayssm-buffer-len 16
```

The option defaults to false. It requires the existing Triton GDN ReplaySSM
speculative path and cannot be combined with `replayssm_spec_flush_interval`.
The DSpark speculator can use the same target path; confidence-based draft
length selection is not implemented by this option.

## State boundaries

For `D` drafts, the target processes `T = D + 1` inputs: the preceding sampled
bonus or recovery token followed by the drafts. The starting state precedes
the first input. The candidate tail follows the last draft. The next sampled
token is not included in the candidate state.

The per-layer cache layout is `(conv, state0, d, k, g, state1)`. Both states are
FP32. The shared block-indexed metadata tracks the head selector, ring origin,
committed history length, preceding actual window length, and flush flag.
The preceding window length also indicates whether a candidate was produced.

At the next metadata build:

1. If the preceding input window was fully committed, swap the selector and
   clear the logical history. Compare acceptance to the preceding actual window
   length, not the configured maximum or the next window's length.
2. Otherwise, retain the head and commit the accepted input prefix. Accepting
   `a` drafts commits `a + 1` previously processed inputs. Invalidate the old
   tail by leaving it as the writable slot.
3. If history length `H` plus the next actual window length `T` exceeds the
   hard limit `W`, flush the history into the selected head before verify.
4. Verify writes the new window's `d/k/g` and the candidate tail. No intermediate
   full recurrent states are materialized.

The flush launch completes before any new ring writes. After a flush, verify
uses the advanced ring origin and zero history locally; the next metadata
commit accounts for that advance. Merely setting the flush flag cannot make
overlapping history writes safe.

All-accepted promotion is logical: it neither copies nor clears either full
state. The tensor addresses remain fixed for CUDA Graph replay. Padding blocks
do not update metadata or state. First decode resets metadata to state0, which
prefill initializes. Prefill also clears existing block metadata, so chunked
prefill and recomputation cannot retain a preceding request's selector. V2
supplies the effective prefill lengths, including
recomputed tokens of resumed requests, in request order for this reset.

## Capacity and cost

In dual mode, `replayssm_buffer_len` is `W`, with physical ring length
`next_pow2(W)`. The configured maximum `D + 1` must fit in `W`. Equality
`H + T == W` does not flush. The original mode retains its existing
`L = buffer_len + max_spec_len` allocation and early-flush policy.

The extra checkpoint is included in both model-level page-size estimation and
layer-level cache allocation. For Qwen3.6-35B-A3B at TP1, one extra FP32 state
is 2 MiB per GDN layer per request. The net allocation difference also depends
on the ring size and shared attention/Mamba page padding.

The tail is computed inside verify from the current full-precision window
deltas and the checkpoint/history reconstruction. This adds compute and a
full-state write every round. High full-acceptance frequency reduces history
and flush frequency but does not guarantee faster decoding. The initial A100
B=1, D=4 comparison was slower at the GDN kernel level. The optimized path
skips history reconstruction when the device history length is zero, including
after promotion and flush. It still writes current-window intermediates for
possible rejection. Short windows pad only the tail reduction to 16 positions
for a matrix multiply, retaining the original verify width. Windows of at most
nine inputs use BV=64, NK=2 and four warps. See
`benchmark_results/replayssm_dual_checkpoint_opt_d4_d8_20260915/report.md`
for the bounded A100 before/after measurements.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/config/test_replayssm_dual_checkpoint.py \
  tests/config/test_replayssm_flush_interval.py \
  tests/v1/worker/test_gdn_dual_checkpoint_metadata.py \
  tests/kernels/test_replayssm_dual_checkpoint_gdn.py \
  tests/kernels/test_replayssm_flush_interval.py -q
```

Kernel tests use an independent sequential recurrence and cover variable
lengths, full and partial acceptance, repeated rejection, flush boundaries,
ring wraparound, slot reuse, padding, and graph replay. Floating-point
reassociation means exact token equality across implementations is a separate
end-to-end check, not a consequence of the kernel tolerance tests.

`benchmarks/replayssm/dual_checkpoint_kernel.py` measures complete GDN cycles
with controlled acceptance. `benchmarks/replayssm/dual_checkpoint_smoke.py`
runs a bounded two-request integration check with token IDs, log probabilities,
and repeated-request output. Set `VLLM_USE_V2_MODEL_RUNNER=0` for V1/MTP or `1`
for V2/DSpark. These are smoke checks, not a serving-scale benchmark.
