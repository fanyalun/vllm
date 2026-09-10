#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/benchmarks/hierarchical:$PWD"
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
output="${1:?Usage: run_gdn_comparison.sh OUTPUT_DIRECTORY}"
methods=("${@:2}")
if [[ ${#methods[@]} -eq 0 ]]; then
    methods=(mtp dspark)
fi
mkdir -p "$output"
for method in "${methods[@]}"; do
    for mode in none ssm_mean input_mean; do
        cell="$output/${method}_${mode}"
        if [[ -e "$cell" ]]; then
            echo "Refusing to overwrite existing cell: $cell" >&2
            exit 1
        fi
        .venv/bin/python benchmarks/hierarchical/run_cycle_profile.py \
            --method "$method" --rounds 4 --gdn-mode "$mode" \
            --samples 4 --max-tokens 256 --warmup-all --output "$cell" \
            > "$output/${method}_${mode}.log" 2>&1
    done
done
