#!/usr/bin/env bash
# Run ShareGuard ablations (1/2/3) then the M3 EP latency benchmark.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/fanya/vllm}"
PY="${PY:-$ROOT/.venv/bin/python}"
EXP="${EXP:-$SCRIPT_DIR}"
MODEL="${MODEL:-/data1/fanya/Qwen/Qwen3.6-35B-A3B}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/fanya/sharedguard/results}"
ABL_OUT="${ABL_OUT:-$RESULTS_ROOT/shareguard_ablations}"
M3_OUT="${M3_OUT:-$RESULTS_ROOT/shareguard_m3}"
TP="${TP:-2}"
CODE_SOURCE="${CODE_SOURCE:-humaneval_mbpp}"
N_PER_DOMAIN="${N_PER_DOMAIN:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LOGIT_EVAL_N="${LOGIT_EVAL_N:-16}"
RESERVOIR_N="${RESERVOIR_N:-2048}"
M3_N_PROMPTS="${M3_N_PROMPTS:-64}"
M3_MAX_TOKENS="${M3_MAX_TOKENS:-64}"
M3_MAX_NUM_SEQS="${M3_MAX_NUM_SEQS:-32}"
M3_WARMUP="${M3_WARMUP:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PATH="$(dirname -- "$PY"):$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

[[ -x "$PY" ]] || { echo "[error] missing Python: $PY" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "[error] missing model: $MODEL" >&2; exit 1; }
[[ -f "$EXP/run_ablations.py" ]] || {
  echo "[error] missing experiment scripts under $EXP" >&2
  exit 1
}
[[ -f "$ROOT/vllm/model_executor/layers/fused_moe/shareguard_runtime.py" ]] || {
  echo "[error] ShareGuard is not integrated into vLLM under $ROOT" >&2
  exit 1
}

GPU_COUNT="$($PY -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < TP )); then
  echo "[error] TP=$TP requires $TP visible GPUs, found $GPU_COUNT" >&2
  exit 1
fi

mkdir -p "$ABL_OUT" "$M3_OUT"

echo "========== ShareGuard environment =========="
echo "python: $PY"
echo "vllm:   $ROOT"
echo "exp:    $EXP"
echo "model:  $MODEL"
echo "gpus:   $CUDA_VISIBLE_DEVICES (TP=$TP)"
echo "data:   $CODE_SOURCE"

echo "========== [1/2] Ablations 1+2+3 =========="
"$PY" -u "$EXP/run_ablations.py" \
  --model "$MODEL" \
  --out-dir "$ABL_OUT" \
  --n-per-domain "$N_PER_DOMAIN" \
  --code-source "$CODE_SOURCE" \
  --max-length "$MAX_LENGTH" \
  --drop-ratios 0.05,0.10,0.15,0.20 \
  --logit-eval-n "$LOGIT_EVAL_N" \
  --reservoir-n "$RESERVOIR_N" \
  --only all \
  2>&1 | tee "$ABL_OUT/run.log"

RHO="$ABL_OUT/m1_rho_epsilon_with_domain.pt"
[[ -f "$RHO" ]] || { echo "[error] missing calibration table: $RHO" >&2; exit 1; }

echo "========== [2/2] M3 EP latency =========="
"$PY" -u "$EXP/run_m3_latency.py" \
  --model "$MODEL" \
  --out-dir "$M3_OUT" \
  --rho-path "$RHO" \
  --tp "$TP" \
  --capacity 0.85 \
  --n-prompts "$M3_N_PROMPTS" \
  --max-tokens "$M3_MAX_TOKENS" \
  --max-num-seqs "$M3_MAX_NUM_SEQS" \
  --warmup "$M3_WARMUP" \
  --modes baseline,min_weight,shareguard \
  2>&1 | tee "$M3_OUT/run.log"

echo "========== DONE =========="
echo "Ablations: $ABL_OUT"
echo "M3:        $M3_OUT"
