#!/usr/bin/env bash
# =============================================================================
# run_curation.sh
# Activates slmtask conda environment and curates English dataset.
# =============================================================================
set -eo pipefail

echo "[curation] Activating conda environment: slmtask ..."
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slmtask
set -u

# Prefer gcc-12 if available to bypass compiler discrepancies in vLLM compilation
if command -v gcc-12 &>/dev/null; then
    export CC=$(which gcc-12)
    export CXX=$(which g++-12)
fi

# High-performance vLLM and HF environment variables
unset HOST
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-32B-Instruct}"
DATASET_FILE="${DATASET_FILE:-eng_Latn.jsonl}"

# Run offline curation
python curate_english.py \
    --model-path "$MODEL_PATH" \
    --dataset-file "$DATASET_FILE" \
    --output-file "curated_eng_Latn.jsonl" \
    --failed-file "failed_eng_Latn.jsonl" \
    --batch-size 64 \
    --max-model-len 4096

echo "[curation] Curation execution finished successfully!"
