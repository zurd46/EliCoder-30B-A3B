#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for envfile in .env ../.env; do
    if [ -f "$envfile" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$envfile"
        set +a
        echo "[env] loaded $envfile"
    fi
done

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

echo "[0/5] prerequisites"
missing=()
for tool in cmake git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        missing+=("$tool")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "  missing: ${missing[*]}"
    case "$(uname -s)" in
        Darwin) echo "  install: brew install ${missing[*]}" ;;
        Linux)  echo "  install: sudo apt install ${missing[*]}" ;;
        *)      echo "  please install: ${missing[*]}" ;;
    esac
    exit 1
fi
echo "  ok: $(cmake --version | head -1), $(git --version)"

TARGET_CTX="${CODERLLM_CTX:-32768}"

echo "[1/3] auto-detect + plan"
python -m build.cli detect --ctx "$TARGET_CTX"

echo
echo "[2/3] build the picked variant(s)"
if [ -n "${HF_TOKEN:-}" ]; then
    python -m build.cli auto --ctx "$TARGET_CTX" --yes --upload
else
    echo "  HF_TOKEN not set — building locally, not uploading"
    python -m build.cli auto --ctx "$TARGET_CTX" --yes
fi

echo
echo "[3/3] done."
echo "  packages: work/packages/"
echo "  import into LM Studio: File → Import Model → select the folder"
