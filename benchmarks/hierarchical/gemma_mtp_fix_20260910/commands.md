# Reproduction

Run from `/home/fanya/vllm` with the existing environment. No dependencies were
installed. Choose new output directories for a rerun.

```bash
export PATH=/home/fanya/vllm/.venv/bin:$PATH
export PYTHONPATH=/home/fanya/vllm/benchmarks/hierarchical:/home/fanya/vllm
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/hierarchical/run_cycle_profile.py \
  --method mtp --rounds 4 --samples 4 \
  --model /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it \
  --draft-model /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it-assistant \
  --dataset benchmark_results/moe_skip_e2e_16x512_b1_20260908_gemma4/gemma4_16.jsonl \
  --output benchmark_results/hierarchical_gemma_fix_20260910/mtp_n4
```

For graph/current-metadata parity use the same arguments, `CUDA_VISIBLE_DEVICES=1`,
`--phases profile --worker-extension inner_metadata_worker.InnerMetadataWorker`,
and output `metadata_check`. The same-input comparisons run outside the
performance experiment.

For the native D16 control, replace `--rounds 4` with
`--draft-length 16 --async-scheduling --legacy-mm-inputs`, use GPU 1 and output
`mtp_d16_legacy`. The explicit-zero-multimodal-limits attempt (`mtp_d16`) failed
at model initialization with the previously observed compiled-entrypoint
`NoneType.size` error; the successful comparison follows the historical
default-multimodal configuration instead.

For the independent P oracle, use GPU 0, add environment variable
`VLLM_HIERARCHICAL_CHECK_PREVERIFY=1`, replace `--samples 4` with
`--samples 1 --max-tokens 64 --phases e2e`, and use output `preverify_check`.
This command fails during warmup; the failure is retained as a correctness
limitation, not counted as a completed request.

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_hierarchical_measurement.py \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py -q

.venv/bin/python benchmarks/hierarchical/summarize_cycles.py \
  benchmark_results/hierarchical_gemma_fix_20260910/mtp_n4 \
  benchmark_results/hierarchical_gemma_fix_20260910/mtp_d16_legacy \
  --output benchmarks/hierarchical/gemma_mtp_fix_20260910
```
