# Reproduction

Run from `/home/fanya/vllm` using the existing environment. No dependencies were
installed. Use new output directories for a rerun.

```bash
export PATH=/home/fanya/vllm/.venv/bin:$PATH
export PYTHONPATH=/home/fanya/vllm/benchmarks/hierarchical:/home/fanya/vllm
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

.venv/bin/python benchmarks/hierarchical/analyze_failure.py \
  benchmark_results/hierarchical_cycle_20260909/mtp_n4_retry \
  benchmark_results/hierarchical_cycle_20260909/dspark_n4 \
  benchmark_results/hierarchical_gemma4_d16_20260910/mtp_n4 \
  benchmark_results/hierarchical_gemma4_d16_20260910/dspark_n4 \
  --output benchmarks/hierarchical/failure_diagnosis_20260910

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/hierarchical/run_cycle_profile.py \
  --method mtp --rounds 4 --samples 1 --phases profile \
  --worker-extension paired_verify_worker.PairedVerifyWorker \
  --output benchmark_results/hierarchical_diagnosis_20260910/qwen_paired

CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/hierarchical/run_cycle_profile.py \
  --method mtp --rounds 4 --samples 1 --phases profile \
  --model /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it \
  --draft-model /home/fanya/data1/fanya/models/gemma-4-26B-A4B-it-assistant \
  --dataset benchmark_results/moe_skip_e2e_16x512_b1_20260908_gemma4/gemma4_16.jsonl \
  --worker-extension paired_verify_worker.PairedVerifyWorker \
  --output benchmark_results/hierarchical_diagnosis_20260910/gemma_paired
```

For the two four-prompt Gemma metadata probes, use the Gemma command above with
`--samples 4`, extension `inner_metadata_worker.InnerMetadataWorker` or
`inner_buffer_worker.InnerBufferWorker`, and output `gemma_metadata` or
`gemma_buffers`. These workers perform duplicate/triple proposals and explicitly
mark request latency invalid.

For Qwen nsys, use the Qwen command with extension
`diagnostic_worker.DiagnosticWorker` and output `qwen_trace`, prefixed by:

```bash
/usr/local/cuda-12.9/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite=false \
  -o benchmark_results/hierarchical_diagnosis_20260910/qwen_trace/timeline
```

Set `CUDA_VISIBLE_DEVICES=0` on the nsys invocation. The worker starts capture
before Target step 2 and stops before step 12; the rest of the request finishes
normally. This produced `timeline.nsys-rep` for Qwen. The corresponding Gemma
attempt on GPU 1, and a retry on GPU 0 adding `--cuda-event-trace=false`, generated
no nsys report despite completing requests; both logs are archived.

```bash
/usr/local/cuda-12.9/bin/nsys export --type sqlite \
  --output benchmark_results/hierarchical_diagnosis_20260910/qwen_trace/timeline.sqlite \
  benchmark_results/hierarchical_diagnosis_20260910/qwen_trace/timeline.nsys-rep

.venv/bin/python benchmarks/hierarchical/analyze_nsys.py \
  benchmark_results/hierarchical_diagnosis_20260910/qwen_trace/timeline.sqlite \
  --output benchmarks/hierarchical/failure_diagnosis_20260910/qwen_trace

.venv/bin/python -m pytest \
  tests/benchmarks/test_hierarchical_measurement.py \
  tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py -q
```

The raw archive includes config, results, logs, completion markers and the
successful Qwen `.nsys-rep`; SQLite is regenerable with the export command.
Existing four-cell raw evidence is in the previously published experiment
archives linked from `report.md`, fingerprinted by `input_hashes.json`.
