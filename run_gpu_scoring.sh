#!/usr/bin/env bash
# ============================================================
# run_gpu_scoring.sh  [GPU NODE — user runs this]
# ============================================================
# Reads correction_cache.jsonl (generated on login node),
# applies corrections, runs all GPU-based quality metrics,
# and writes final publishable corrected_*.jsonl files.
#
# Prerequisite: run_api_correction.sh must have finished and
# correction_cache.jsonl must be in ikdd/ directory.
#
# Usage:
#   conda activate slmtask
#   bash ikdd/run_gpu_scoring.sh
#   bash ikdd/run_gpu_scoring.sh --skip-comet  # skip COMET (~40min)
# ============================================================
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-slmtask}"
MODEL_DIR="${MODEL_DIR:-models}"
OUTPUT_DIR="corrected"
EVAL_DIR="eval_results"

echo "============================================================"
echo " GPU SCORING PIPELINE  [GPU NODE]"
echo " $(date)"
echo "============================================================"

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || true

# Quick sanity check
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"

CACHE_FILE="correction_cache.jsonl"
if [ ! -f "$CACHE_FILE" ]; then
    echo "ERROR: $CACHE_FILE not found!"
    echo "Run run_api_correction.sh on the login node first."
    exit 1
fi
echo "Cache: $CACHE_FILE ($(wc -l < "$CACHE_FILE") entries)"
echo ""

mkdir -p "$OUTPUT_DIR" "$EVAL_DIR"

# ── STEP A: Evaluate BEFORE correction (Tier 1 only, fast) ──────────────────
echo "[STEP A] Scoring BEFORE — Tier 1 (chrF++, BLEU, METEOR)..."
python3 evaluate_quality.py \
    --input-dir  translated \
    --output-dir "$EVAL_DIR" \
    --tag        before_correction \
    --labse-path "$MODEL_DIR/LaBSE" \
    --languages  ory_Orya asm_Beng sin_Sinh mlt_Latn \
    --skip-tier3

# ── STEP B: Apply API corrections from cache + LaBSE re-score ───────────────
echo ""
echo "[STEP B] Applying corrections + LaBSE re-score..."
python3 quality_correction.py \
    --translated-dir  translated \
    --output-dir      "$OUTPUT_DIR" \
    --cache-file      "$CACHE_FILE" \
    --token-log       correction_token_usage.jsonl \
    --correction-log  correction_log.jsonl \
    --labse-path      "$MODEL_DIR/LaBSE" \
    --gemini-concurrency 0 \
    --qwen-concurrency   0 \
    --skip-stage1 \
    --skip-stage2 \
    --rescore-only \
    --languages ory_Orya asm_Beng sin_Sinh mlt_Latn

python3 apply_corrections.py \
    --translated-dir translated \
    --cache-file     "$CACHE_FILE" \
    --output-dir     "$OUTPUT_DIR" \
    --labse-path     "$MODEL_DIR/LaBSE"

# ── STEP C: Evaluate AFTER — all metrics ────────────────────────────────────
echo ""
echo "[STEP C] Scoring AFTER — all metrics (Tier 1 + 2 + 3)..."

SKIP_COMET=""
for arg in "$@"; do
    [ "$arg" = "--skip-comet" ] && SKIP_COMET="--skip-comet"
done

python3 evaluate_quality.py \
    --input-dir  "$OUTPUT_DIR" \
    --output-dir "$EVAL_DIR" \
    --tag        after_correction \
    --labse-path "$MODEL_DIR/LaBSE" \
    --languages  ory_Orya asm_Beng sin_Sinh mlt_Latn \
    $SKIP_COMET

# ── STEP D: Comparison report ────────────────────────────────────────────────
echo ""
echo "[STEP D] Generating before/after comparison report..."
python3 evaluate_quality.py \
    --compare \
    --before "$EVAL_DIR/eval_before_correction_summary.json" \
    --after  "$EVAL_DIR/eval_after_correction_summary.json" \
    --output-dir "$EVAL_DIR"

echo ""
echo "============================================================"
echo " GPU SCORING COMPLETE"
echo " Corrected files  : $OUTPUT_DIR/corrected_{lang}.jsonl"
echo " Eval JSON        : $EVAL_DIR/eval_*.json"
echo " Comparison table : $EVAL_DIR/comparison_report.md"
echo " Token usage      : correction_token_usage.jsonl"
echo " $(date)"
echo "============================================================"
