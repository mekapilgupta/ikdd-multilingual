#!/usr/bin/env bash
# =============================================================================
# run_synthetic.sh — Dual-Pass Synthetic Data Generation (v2)
# Pass 1: Generate with thinking enabled
# Pass 2: Verify + override answers with thinking enabled
# =============================================================================
set -eo pipefail

echo "[synthetic] Activating conda environment: slmtask ..."
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slmtask
set -u

if command -v gcc-12 &>/dev/null; then
    export CC=$(which gcc-12)
    export CXX=$(which g++-12)
fi

unset HOST
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-32B-Instruct}"

python generate_synthetic.py \
    --model-path "$MODEL_PATH" \
    --curated-file "curated_eng_Latn.jsonl" \
    --output-file "synthetic_data_v2.jsonl" \
    --failed-file "failed_synthetic_data_v2.jsonl" \
    --batch-size 32 \
    --verify-batch-size 16 \
    --max-model-len 8192 \
    --gen-max-tokens 4096 \
    --verif-max-tokens 3000

echo "[synthetic] Dual-pass generation finished!"
