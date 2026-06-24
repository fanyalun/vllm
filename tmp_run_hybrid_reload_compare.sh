#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-root>" >&2
  exit 1
fi

ROOT_DIR="$1"
SCRIPT="examples/features/speculative_decoding/qwen3_6_mtp_ep_load_balance_experiment.py"
MODEL="/home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT_DIR/logs"

run_mode() {
  local mode="$1"
  local out_dir="$ROOT_DIR/$mode"
  local collect_log="$ROOT_DIR/logs/${mode}_collect.log"
  local analyze_log="$ROOT_DIR/logs/${mode}_analyze.log"

  mkdir -p "$out_dir"
  : > "$collect_log"
  : > "$analyze_log"

  echo "[launcher] start mode=${mode} out=${out_dir}"
  .venv/bin/python -u "$SCRIPT" collect \
    --model "$MODEL" \
    --hybrid-spec-state-offload-mode "$mode" \
    --hybrid-spec-state-ewma-alpha 0.5 \
    --local-gpu-ids 0,1 \
    --data-parallel-size 2 \
    --batch-sizes 64 128 \
    --draft-lengths 0 2 4 6 \
    --num-samples 512 \
    --max-model-len 768 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --output-dir "$out_dir" \
    2>&1 | tee -a "$collect_log"

  echo "[launcher] analyze mode=${mode} input=${out_dir}"
  .venv/bin/python -u "$SCRIPT" analyze \
    --input-dir "$out_dir" \
    2>&1 | tee -a "$analyze_log"

  echo "[launcher] finished mode=${mode}"
}

cd /home/fanya/vllm
run_mode disabled
run_mode predict_last
echo "[launcher] all modes finished"
