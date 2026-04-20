#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "[setup] creating venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import typer, huggingface_hub" 2>/dev/null; then
    echo "[setup] installing deps"
    pip install -e .
fi

echo "[1/5] info"
python -m build.cli info

echo "[2/5] download base"
python -m build.cli download

echo "[3/5] convert + package MLX (Mac only; skipped elsewhere)"
python -m build.cli convert-mlx   || true
python -m build.cli package --kind mlx  || true

echo "[4/5] convert + package GGUF"
python -m build.cli convert-gguf
python -m build.cli package --kind gguf

echo "[5/5] upload (requires HF_TOKEN)"
if [ -n "${HF_TOKEN:-}" ]; then
    python -m build.cli upload
else
    echo "  skip — set HF_TOKEN to enable upload"
fi

echo "done."
