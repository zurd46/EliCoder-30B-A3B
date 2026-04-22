#!/usr/bin/env bash
# RunPod End-to-End Pipeline — führt alle 5 Phasen in Reihenfolge aus.
#
# Idempotent: Jede Phase schreibt /workspace/.phaseNN_done — schon fertige werden
# beim Restart übersprungen. Innerhalb einer Phase nutzt Transformers'
# `get_last_checkpoint` + HF Hub mirror für Auto-Resume.
#
# Run:
#   cd /workspace/CoderLLM/training
#   bash runpod/pipeline.sh 2>&1 | tee -a /workspace/pipeline.log
#
# Stoppt den Pod am Ende automatisch (spart $$) wenn RUNPOD_API_KEY + RUNPOD_POD_ID
# gesetzt sind — sonst läuft er weiter bis manuell gestoppt.

set -euo pipefail

REPO_URL="${CODERLLM_REPO_URL:-https://github.com/zurd46/CoderLLM.git}"
WORKSPACE="${CODERLLM_WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/CoderLLM"

# 1. Repo klonen / updaten
if [ ! -d "$REPO_DIR/.git" ]; then
  echo ">>> cloning $REPO_URL → $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
  echo ">>> pulling latest in $REPO_DIR"
  git -C "$REPO_DIR" pull --ff-only || true
fi

cd "$REPO_DIR/training"

# 2. ENV-Checks
if [ -z "${HF_TOKEN:-}" ] && [ -f "$WORKSPACE/.env" ]; then
  echo ">>> sourcing $WORKSPACE/.env"
  set -a; . "$WORKSPACE/.env"; set +a
fi

: "${HF_TOKEN:?HF_TOKEN muss gesetzt sein (ENV oder /workspace/.env)}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# 3. Phasen ausführen — jede mit Idempotency-Marker
run_phase() {
  local num="$1" name="$2" script="$3"
  local marker="$WORKSPACE/.phase${num}_done"
  if [ -f "$marker" ]; then
    echo ">>> phase $num ($name) already done — skip"
    return 0
  fi
  echo ""
  echo "========================================"
  echo ">>> PHASE $num: $name"
  echo "========================================"
  local start=$(date +%s)
  python "$script"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! phase $num failed with rc=$rc"
    exit $rc
  fi
  local end=$(date +%s)
  echo ">>> phase $num done in $((end - start))s"
}

run_phase 01 "Dataset-Build"   runpod/01_data_build.py
run_phase 02 "SFT"             runpod/02_sft.py
run_phase 03 "DPO"             runpod/03_dpo.py
run_phase 04 "LongCtx"         runpod/04_longctx.py
run_phase 05 "Export"          runpod/05_export.py
run_phase 06 "BFCL-lite Eval"  runpod/06_bfcl_eval.py

echo ""
echo "========================================"
echo ">>> ALL PHASES COMPLETE"
echo "========================================"

# 4. Pod automatisch stoppen (Kosten sparen)
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo ">>> stopping pod $RUNPOD_POD_ID via RunPod API"
  curl -s -X POST "https://api.runpod.io/graphql" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podStop(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) { id desiredStatus } }\"}" \
    || echo "!!! pod-stop call failed — bitte manuell stoppen"
else
  echo ">>> RUNPOD_API_KEY/RUNPOD_POD_ID nicht gesetzt — Pod läuft weiter bis manuell gestoppt"
fi
