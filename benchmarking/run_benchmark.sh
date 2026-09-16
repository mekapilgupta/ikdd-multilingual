#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Multilingual MCQ Benchmark Runner (GPU Node)
# Evaluates Qwen3-8B and Qwen3.5-9B on 4 languages x 6 conditions (C1 to C6)
# =============================================================================
set -eo pipefail

echo "[benchmark] Activating conda environment: slmtask ..."
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slmtask
set -u

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${MODEL_DIR:-models}"
DATA_DIR="${DATA_DIR:-corrected}"
RESULTS_DIR="${RESULTS_DIR:-benchmarking/results}"
mkdir -p "$RESULTS_DIR"

# Target models:
MODELS=(
    "Qwen3-8B"
    "qwen3.5-9b-base"
)

# Backend mode: "hf" (HuggingFace Transformers, recommended for Qwen3/3.5) or "vllm"
MODE="${1:-hf}"

for BENCH_MODEL_NAME in "${MODELS[@]}"; do
    BENCH_MODEL="${MODEL_DIR}/${BENCH_MODEL_NAME}"
    if [ ! -f "${BENCH_MODEL}/config.json" ]; then
        echo "[WARN] config.json not found in ${BENCH_MODEL} — skipping."
        continue
    fi

    echo ""
    echo "============================================================"
    echo " [BENCHMARK] MODEL: ${BENCH_MODEL_NAME} (Mode: ${MODE})"
    echo "============================================================"

    # Step 1: Sanity Load & Inference Test
    python benchmarking/test_local_model.py --model-path "$BENCH_MODEL" --mode "$MODE"

    # Step 2: 6-Condition Evaluation across all 4 languages
    for LANG in ory_Orya asm_Beng sin_Sinh mlt_Latn; do
        CORRECTED_FILE="${DATA_DIR}/corrected_${LANG}.jsonl"
        OUT_FILE="${RESULTS_DIR}/bench_${LANG}_${BENCH_MODEL_NAME}.jsonl"

        if [ ! -f "$CORRECTED_FILE" ]; then
            echo "[WARN] File not found: $CORRECTED_FILE — skipping."
            continue
        fi

        echo ""
        echo "[benchmark] Evaluating ${LANG} on ${BENCH_MODEL_NAME} (${MODE})..."
        python benchmarking/benchmark_runner.py \
            --mode "$MODE" \
            --model-path "$BENCH_MODEL" \
            --input-file "$CORRECTED_FILE" \
            --output-file "$OUT_FILE" \
            --lang-code "$LANG" \
            --batch-size 64 \
            --temperature 0.0 \
            --max-tokens 80
    done

    echo "[benchmark] Completed evaluation for ${BENCH_MODEL_NAME}!"
done

echo ""
echo "============================================================"
echo " ALL LOCAL BENCHMARKS COMPLETED!"
echo " Results written to ${RESULTS_DIR}/"
echo "============================================================"
