#!/usr/bin/env bash
set -euo pipefail

TS="${1:?usage: $0 <timestamp>}"
ROOT="/home/fanya/vllm/results/qwen3_6_predict_last_${TS}_group_preload_fix"
SESSION="hybrid_predict_last_${TS}_group_fix"

mkdir -p "$ROOT/logs" "$ROOT/predict_last"

tmux new-session -d -s "$SESSION" "
cd /home/fanya/vllm
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
echo '[launcher] start mode=predict_last out=$ROOT/predict_last' >> '$ROOT/logs/predict_last_launcher.log'
.venv/bin/python -u examples/features/speculative_decoding/qwen3_6_mtp_ep_load_balance_experiment.py collect \
  --model /home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B \
  --hybrid-spec-state-offload-mode predict_last \
  --hybrid-spec-state-ewma-alpha 0.5 \
  --local-gpu-ids 0,1 \
  --data-parallel-size 2 \
  --batch-sizes 64 128 \
  --draft-lengths 0 2 4 6 \
  --num-samples 512 \
  --max-model-len 768 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --output-dir '$ROOT/predict_last' \
  >> '$ROOT/logs/predict_last_collect.log' 2>&1
status=\$?
echo \"[launcher] collect_exit=\$status\" >> '$ROOT/logs/predict_last_launcher.log'
if [ \$status -eq 0 ]; then
  echo '[launcher] analyze mode=predict_last input=$ROOT/predict_last' >> '$ROOT/logs/predict_last_launcher.log'
  .venv/bin/python -u examples/features/speculative_decoding/qwen3_6_mtp_ep_load_balance_experiment.py analyze \
    --input-dir '$ROOT/predict_last' \
    >> '$ROOT/logs/predict_last_analyze.log' 2>&1
  analyze_status=\$?
  echo \"[launcher] analyze_exit=\$analyze_status\" >> '$ROOT/logs/predict_last_launcher.log'
else
  analyze_status=1
fi
echo \"[launcher] finished mode=predict_last collect_exit=\$status analyze_exit=\${analyze_status:-1}\" >> '$ROOT/logs/predict_last_launcher.log'
"

echo "$SESSION"
echo "$ROOT"
