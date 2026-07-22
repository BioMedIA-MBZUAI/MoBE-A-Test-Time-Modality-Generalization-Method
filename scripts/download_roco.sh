#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SPLIT="${SPLIT:-train}"
OUT_ROOT="${OUT_ROOT:-"$PROJECT_ROOT/dataset_roco"}"

cd "$PROJECT_ROOT"
python tools/download_roco.py \
  --split "$SPLIT" \
  --out-root "$OUT_ROOT"
