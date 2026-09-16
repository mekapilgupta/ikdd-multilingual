#!/usr/bin/env bash
# =============================================================================
# run_paid_benchmarks.sh — Paid API Benchmark Runner (OpenRouter Batch API)
# Evaluates Gemini 3.7 Flash & GPT-5.6 Luna across 4 languages x 6 conditions
# Safe incremental resuming, token limit throttling, and auto-retry
# =============================================================================
set -eo pipefail

echo "============================================================"
echo " 1. Running missing batches for google/gemini-3.7-flash:batch"
echo "============================================================"
python run_api_benchmark_batch.py \
    --model "google/gemini-3.7-flash:batch" \
    --batch-size 500 \
    --max-active-batches 2

echo ""
echo "============================================================"
echo " 2. Running missing batches for openai/gpt-5.6-luna:batch"
echo "============================================================"
python run_api_benchmark_batch.py \
    --model "openai/gpt-5.6-luna:batch" \
    --batch-size 500 \
    --max-active-batches 2

echo ""
echo "============================================================"
echo " ALL PAID BENCHMARKS FULLY COMPLETED & VERIFIED!"
echo "============================================================"
