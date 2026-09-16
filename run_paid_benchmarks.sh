#!/usr/bin/env bash
# =============================================================================
# run_paid_benchmarks.sh — Paid Models Batch API Benchmarking (CPU/Login Node)
# Evaluates Gemini 3.7 Flash & GPT-5.6 Luna across 3,600 items x 6 conditions (C1-C6)
# =============================================================================
set -eo pipefail

echo "============================================================"
echo " STAGE 4: BENCHMARKING PAID MODELS VIA OPENROUTER BATCH API"
echo " Date: $(date)"
echo "============================================================"

# 1. Benchmark google/gemini-3.7-flash:batch
python benchmarking/run_api_benchmark_batch.py --model "google/gemini-3.7-flash:batch"

# 2. Benchmark openai/gpt-5.6-luna:batch
python benchmarking/run_api_benchmark_batch.py --model "openai/gpt-5.6-luna:batch"

echo ""
echo "============================================================"
echo " PAID BENCHMARKING COMPLETE!"
echo " Results written to ikdd/benchmarking/results/"
echo "============================================================"
