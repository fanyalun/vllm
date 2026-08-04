#!/usr/bin/env bash
# 对比策略扫描：只评 ACC / pass@1
#   GSM8K Acc | HumanEval+MBPP pass@1 | MMLU Acc | LiveBench Acc
#
# 用法:
#   cd /home/fanya/sharedguard
#   bash code/judge/run_strategy_scan.sh
#
# 冒烟:
#   LIMIT=16 OUT_DIR=/home/fanya/sharedguard/results/strategy_scan_smoke \
#     bash code/judge/run_strategy_scan.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SHAREDGUARD_ROOT:-/home/fanya/sharedguard}"
JUDGE="${JUDGE:-$SCRIPT_DIR}"
VLLM="${VLLM:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SHAREGUARD_EXP="${SHAREGUARD_EXP:-$SCRIPT_DIR}"
PY="${PY:-$VLLM/.venv/bin/python}"
MODEL="${MODEL:-/data1/fanya/Qwen/Qwen3.6-35B-A3B}"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/dataset/slide/test}"
FULL_DATA_DIR="${FULL_DATA_DIR:-$REPO_ROOT/dataset/full}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/strategy_scan}"
DROP_RATIOS="${DROP_RATIOS:-0.05,0.10,0.15,0.20}"
STRATEGIES="${STRATEGIES:-baseline,min_weight,renorm,shareguard}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
CODE_TIMEOUT_S="${CODE_TIMEOUT_S:-10}"

if [[ -z "${RHO_PATH:-}" ]]; then
  if [[ -f "$REPO_ROOT/code/pt/mixed_bf16.pt" ]]; then
    RHO_PATH="$REPO_ROOT/code/pt/mixed_bf16.pt"
  elif [[ -f "$REPO_ROOT/results/shareguard_ablations/m1_rho_epsilon_with_domain.pt" ]]; then
    RHO_PATH="$REPO_ROOT/results/shareguard_ablations/m1_rho_epsilon_with_domain.pt"
  else
    echo "[error] no rho/epsilon table found; set RHO_PATH" >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SHAREDGUARD_ROOT="$REPO_ROOT"
export SHAREGUARD_EXP
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

[[ -x "$PY" ]] || { echo "[error] missing Python: $PY" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "[error] missing model: $MODEL" >&2; exit 1; }
[[ -d "$TEST_DIR" ]] || { echo "[error] missing test data: $TEST_DIR" >&2; exit 1; }
[[ -d "$FULL_DATA_DIR" ]] || { echo "[error] missing full data: $FULL_DATA_DIR" >&2; exit 1; }
[[ -d "$SHAREGUARD_EXP" ]] || { echo "[error] missing experiments: $SHAREGUARD_EXP" >&2; exit 1; }
[[ -f "$RHO_PATH" ]] || { echo "[error] missing rho/epsilon table: $RHO_PATH" >&2; exit 1; }

mkdir -p "$OUT_DIR"
LOG_FILE="${LOG_FILE:-$OUT_DIR/run.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

EXTRA=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--validate-only)
fi

echo "=============================================="
echo " Strategy scan (ACC only)"
echo " model : $MODEL"
echo " test  : $TEST_DIR"
echo " full  : $FULL_DATA_DIR"
echo " rho   : $RHO_PATH"
echo " drops : $DROP_RATIOS"
echo " modes : $STRATEGIES"
echo " batch : $EVAL_BATCH_SIZE"
echo " out   : $OUT_DIR"
echo " log   : $LOG_FILE"
echo "=============================================="

"$PY" -u "$JUDGE/run_strategy_scan.py" \
  --model "$MODEL" \
  --test-dir "$TEST_DIR" \
  --full-data-dir "$FULL_DATA_DIR" \
  --rho-path "$RHO_PATH" \
  --out-dir "$OUT_DIR" \
  --drop-ratios "$DROP_RATIOS" \
  --strategies "$STRATEGIES" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --code-timeout-s "$CODE_TIMEOUT_S" \
  "${EXTRA[@]}"

echo "========== DONE =========="
ls -lh "$OUT_DIR" || true
