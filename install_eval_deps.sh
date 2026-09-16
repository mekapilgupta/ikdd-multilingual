#!/usr/bin/env bash
# install_eval_deps.sh — Install all evaluation dependencies into slmtask env
set -eo pipefail

CONDA_ENV="slmtask"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "Installing evaluation dependencies into conda env: $CONDA_ENV"
echo ""

# ── Tier 1: already have (pure Python) ───────────────────────────────────────
echo "[Tier 1] chrF++, BLEU, METEOR — pure Python (no install needed)"

# ── Tier 2: embeddings ────────────────────────────────────────────────────────
echo "[Tier 2] sentence-transformers, httpx..."
pip install sentence-transformers httpx --quiet
echo "  ✓ sentence-transformers, httpx"

# ── Tier 3: BERTScore ────────────────────────────────────────────────────────
echo "[Tier 3a] bert-score..."
pip install bert-score --quiet
echo "  ✓ bert-score"

# ── Tier 3: COMET ─────────────────────────────────────────────────────────────
echo "[Tier 3b] unbabel-comet..."
pip install unbabel-comet --quiet
echo "  ✓ unbabel-comet"

# ── Download COMET model ───────────────────────────────────────────────────────
echo "[COMET model] Downloading Unbabel/wmt22-comet-da (~1.5GB)..."
python3 -c "
from comet import download_model
path = download_model('Unbabel/wmt22-comet-da')
print(f'COMET model at: {path}')
"

echo ""
echo "All evaluation dependencies installed successfully."
