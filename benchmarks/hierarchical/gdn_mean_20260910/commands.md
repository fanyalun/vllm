# Reproduction

Run from `/home/fanya/vllm` using the existing environment and fresh output
directories. No installation or C++/CUDA rebuild is needed for this patch.

```bash
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/benchmarks/hierarchical:$PWD"
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
gdn_results=benchmark_results/gdn_mean_new_run

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/hierarchical/run_cycle_profile.py --method ar \
  --samples 4 --max-tokens 256 --phases e2e e2e_after \
  --output "$gdn_results/ar"

CUDA_VISIBLE_DEVICES=1 bash benchmarks/hierarchical/run_gdn_comparison.sh \
  "$gdn_results/matrix" mtp
CUDA_VISIBLE_DEVICES=0 bash benchmarks/hierarchical/run_gdn_comparison.sh \
  "$gdn_results/matrix" dspark

CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/hierarchical/run_forward_stages.py --model qwen36 \
  --gdn-comparison --output "$gdn_results/stages_final"

gdn_report=benchmarks/hierarchical/gdn_mean_new_report
.venv/bin/python benchmarks/hierarchical/analyze_gdn_mean.py \
  "$gdn_results/matrix/mtp_none" \
  "$gdn_results/matrix/mtp_ssm_mean" \
  "$gdn_results/matrix/mtp_input_mean" \
  "$gdn_results/matrix/dspark_none" \
  "$gdn_results/matrix/dspark_ssm_mean" \
  "$gdn_results/matrix/dspark_input_mean" \
  --ar "$gdn_results/ar" --output "$gdn_report"
.venv/bin/python benchmarks/hierarchical/summarize_gdn_mean_stages.py \
  "$gdn_results/stages_final" --output "$gdn_report"
.venv/bin/python benchmarks/hierarchical/plot_gdn_mean.py \
  --input "$gdn_report/summary.csv" \
  --output "$gdn_report/cycle_comparison"
```

The recorded MTP and DSpark matrix runners overlapped on separate GPUs; each
method's three modes ran sequentially on its fixed GPU. The initial MTP launcher
also listed DSpark and subsequently stopped at its overwrite guard because the
separate DSpark runner already owned those directories. All six individual
cells completed. Pilot runs and that launcher log are archived separately from
the official coverage counters.

Validation commands, all passed:

```bash
.venv/bin/python -m pytest \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/benchmarks/test_hierarchical_measurement.py -q

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/kernels/mamba/test_gdn_fused_mtp.py -q

.venv/bin/pre-commit run --files \
  vllm/config/compilation.py vllm/config/speculative.py \
  vllm/model_executor/layers/mamba/gdn/mean_update.py \
  vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
  vllm/v1/worker/gpu/spec_decode/hierarchical/state.py \
  vllm/v1/worker/gpu/spec_decode/hierarchical/speculator.py \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  tests/benchmarks/test_hierarchical_measurement.py
.venv/bin/pre-commit run mypy-3.12 --hook-stage manual --files \
  vllm/config/compilation.py vllm/config/speculative.py \
  vllm/model_executor/layers/mamba/gdn/mean_update.py \
  vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
  vllm/v1/worker/gpu/spec_decode/hierarchical/state.py \
  vllm/v1/worker/gpu/spec_decode/hierarchical/speculator.py
```

The first command passed 78 tests; the GPU command passed 26. AR equivalence
remained a failed model-output gate, including for the existing `none` controls.
These tests and measurements do not establish task accuracy or sampling
distribution equivalence.
