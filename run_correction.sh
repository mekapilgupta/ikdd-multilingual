#!/usr/bin/env bash
# run_correction.sh — Full frontier-LLM quality correction pipeline
# Usage:
#   bash ikdd/run_correction.sh              # full run (all 3600 items)
#   bash ikdd/run_correction.sh --pilot-n 10 # pilot 10 items per language first

set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-slmtask}"
LABSE_PATH="${LABSE_PATH:-sentence-transformers/LaBSE}"
TRANSLATED_DIR="translated"
OUTPUT_DIR="corrected"

echo "=================================================="
echo " IKDD Translation Quality Correction Pipeline"
echo " $(date)"
echo "=================================================="

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Check httpx
python3 -c "import httpx" 2>/dev/null || pip install httpx --quiet

mkdir -p "$OUTPUT_DIR" eval_results

# ── STEP A: Evaluate BEFORE correction ──────────────────────────────────────
echo ""
echo "[STEP A] Evaluating translation quality BEFORE correction..."
python3 evaluate_quality.py \
    --input-dir  "$TRANSLATED_DIR" \
    --output-dir eval_results \
    --tag        before_correction \
    --labse-path "$LABSE_PATH" \
    --languages  ory_Orya asm_Beng sin_Sinh mlt_Latn \
    --skip-tier3   # Tier 3 (COMET/BERTScore) only after correction

# ── STEP B: Run correction (Gemini + Qwen) ──────────────────────────────────
echo ""
echo "[STEP B] Running frontier-LLM correction (Stage 1: Gemini, Stage 2: Qwen)..."
python3 quality_correction.py \
    --translated-dir  "$TRANSLATED_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --cache-file      correction_cache.jsonl \
    --token-log       correction_token_usage.jsonl \
    --correction-log  correction_log.jsonl \
    --labse-path      "$LABSE_PATH" \
    --gemini-concurrency 20 \
    --qwen-concurrency   10 \
    --languages ory_Orya asm_Beng sin_Sinh mlt_Latn \
    "$@"  # pass --pilot-n etc.

# ── STEP C: Evaluate AFTER correction ───────────────────────────────────────
echo ""
echo "[STEP C] Evaluating translation quality AFTER correction..."
python3 evaluate_quality.py \
    --input-dir  "$OUTPUT_DIR" \
    --output-dir eval_results \
    --tag        after_correction \
    --labse-path "$LABSE_PATH" \
    --languages  ory_Orya asm_Beng sin_Sinh mlt_Latn

# ── STEP D: Generate comparison report ──────────────────────────────────────
echo ""
echo "[STEP D] Generating before/after comparison report..."
python3 evaluate_quality.py \
    --compare \
    --before eval_results/eval_before_correction_summary.json \
    --after  eval_results/eval_after_correction_summary.json \
    --output-dir eval_results

echo ""
echo "=================================================="
echo " PIPELINE COMPLETE"
echo " Corrected files  : $OUTPUT_DIR/"
echo " Eval results     : eval_results/"
echo " Token usage log  : correction_token_usage.jsonl"
echo " Per-item log     : correction_log.jsonl"
echo " Comparison report: eval_results/comparison_report.md"
echo " $(date)"
echo "=================================================="
