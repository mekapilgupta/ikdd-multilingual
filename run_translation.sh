#!/usr/bin/env bash
# =============================================================================
# run_translation.sh — Triple-Pass Multilingual Translation Pipeline Runner
# Translates English sampled data into Odia, Assamese, Sinhala, Maltese
# =============================================================================
set -eo pipefail

echo "[translation] Activating conda environment: slmtask ..."
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slmtask
set -u

if command -v gcc-12 &>/dev/null; then
    export CC=$(which gcc-12)
    export CXX=$(which g++-12)
    export CUDAHOSTCXX=$(which g++-12)
fi

unset HOST
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${MODEL_DIR:-models}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Sample 150 items per difficulty level from curated + synthetic data
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 1] Sampling 150 items per level (L0-L5) ..."
python sample_dataset.py \
    --curated-file "curated_eng_Latn.jsonl" \
    --synthetic-file "synthetic_data_v2.jsonl" \
    --output-file "sampled_eng_Latn.jsonl" \
    --per-level 150 \
    --seed 42

echo "[STEP 1] Sampling complete."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: NMT Forward + Back-translation + chrF/LaBSE Scoring
# ─────────────────────────────────────────────────────────────────────────────
echo "[STEP 2] Running NMT translation & scoring (Pass 1A + 1B + 2) ..."
python translate_pipeline.py \
    --sampled-file "sampled_eng_Latn.jsonl" \
    --output-dir "translated" \
    --cache-file "translate_cache.jsonl" \
    --nllb-large-path "${MODEL_DIR}/nllb-200-3.3B" \
    --nllb-small-path "${MODEL_DIR}/nllb-200-1.3B" \
    --indictrans-fwd-path "${MODEL_DIR}/indictrans2-en-indic-1B" \
    --indictrans-back-path "${MODEL_DIR}/indictrans2-indic-en-1B" \
    --labse-path "${MODEL_DIR}/LaBSE" \
    --judge-model-path "${MODEL_DIR}/Qwen2.5-32B-Instruct" \
    --languages ory_Orya asm_Beng sin_Sinh mlt_Latn \
    --nmt-batch-size 32

echo ""
echo "[STEP 2] Translation and scoring completed."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Standalone vLLM Judge on Flagged Items (Clean Process Isolation)
# ─────────────────────────────────────────────────────────────────────────────
echo "[STEP 3] Running standalone vLLM Judge on flagged items (Pass 3) ..."
python translate_judge.py \
    --sampled-file "sampled_eng_Latn.jsonl" \
    --cache-file "translate_cache.jsonl" \
    --scores-file "translate_scores.json" \
    --decisions-file "translate_decisions.json" \
    --output-dir "translated" \
    --judge-model-path "${MODEL_DIR}/Qwen2.5-32B-Instruct" \
    --languages ory_Orya asm_Beng sin_Sinh mlt_Latn \
    --judge-batch-size 16 \
    --judge-max-tokens 2048 \
    --max-model-len 8192

echo ""
echo "[translation] Pipeline 100% complete!"
echo "Output translated files:"
ls -lh translated/
