#!/usr/bin/env bash
# ============================================================
# run_api_correction.sh  [LOGIN/CPU NODE]
# ============================================================
# Runs ONLY the OpenRouter API calls (Gemini + Qwen).
# No GPU, no heavy models — safe to run on login node.
#
# Output: ikdd/correction_cache.jsonl  ← copy this to GPU node
#
# Usage:
#   bash ikdd/run_api_correction.sh            # full 3600 items
#   bash ikdd/run_api_correction.sh --pilot-n 5 # pilot 5 per lang
# ============================================================
set -eo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo " STAGE 1+2: API Correction (LOGIN NODE — CPU only)"
echo " $(date)"
echo "============================================================"
echo " Models : google/gemini-3.7-flash:batch  +  qwen/qwen3-7-flash"
echo " Output : ikdd/correction_cache.jsonl"
echo "============================================================"
echo ""

# Basic conda activation (slmtask has httpx)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate slmtask 2>/dev/null || true

# Ensure httpx available
python3 -c "import httpx" 2>/dev/null || pip install httpx --quiet

python3 quality_correction.py \
    --translated-dir  translated \
    --output-dir      corrected \
    --cache-file      correction_cache.jsonl \
    --token-log       correction_token_usage.jsonl \
    --correction-log  correction_log_api.jsonl \
    --labse-path      sentence-transformers/LaBSE \
    --gemini-concurrency 20 \
    --qwen-concurrency   10 \
    --no-labse \
    --languages ory_Orya asm_Beng sin_Sinh mlt_Latn \
    "$@"

echo ""
echo "============================================================"
echo " API correction done."
echo " Cache saved to: ikdd/correction_cache.jsonl"
echo ""
echo " NEXT STEP (on GPU node):"
echo "   bash ikdd/run_gpu_scoring.sh"
echo "============================================================"
