# Validation and execution provenance

This directory contains only the successful third attempt. The first attempt
failed before measurement because the Gemma MTP assistant path was omitted.
The second failed during CUDA Graph capture because a host-created scalar was
copied to CUDA. Neither attempt supplies numbers to this report. The successful
worker creates the counters directly on the GPU.

GPU reference command (GPU 0, separate from the model measurements on GPU 1):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/hierarchical/check_token_importance.py
```

The saved `gpu_reference.log` reports 24 last-row attention comparisons against
PyTorch and eight expert-pool/GEMM comparisons, plus CUDA Graph capture/replay.
Attention cases cover paged keys, GQA, sliding and full attention, soft caps,
and head dimensions 80, 256 and 512. The expert cases cover random and tied
scores, the ceil-rounded union budget, and both small and aligned assignments.
These are primitive checks, not proof of end-to-end lossless generation.

Source lint command:

```bash
.venv/bin/pre-commit run --files benchmarks/hierarchical/analyze_token_importance.py benchmarks/hierarchical/check_token_importance.py benchmarks/hierarchical/run_token_importance.py benchmarks/hierarchical/token_importance_worker.py
```

All applicable hooks passed. No new dependencies were installed. The runner
and worker hashes still matched the frozen contract after linting. Model
measurements include GPU scoring counters and the per-proposal Python trace
aggregation. Reset/export RPCs and artifact serialization happen outside each
timed generation call; model initialization, compilation and the four complete
warmup requests are excluded.

The output parity table is a separate check from the measurement audit. A
successful measurement contract does not imply matching greedy token streams.
The baseline is two fresh AR (Async) measurements; h=8 is hierarchical MTP plus
full-expert Pre-Verify, so its execution work differs from AR.
