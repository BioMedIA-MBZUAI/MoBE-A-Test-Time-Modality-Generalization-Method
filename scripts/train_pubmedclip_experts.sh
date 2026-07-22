#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROCO_ROOT="${ROCO_ROOT:-"$PROJECT_ROOT/dataset_roco"}"
OUT_DIR="${OUT_DIR:-"$PROJECT_ROOT/experts_pubmedclip_hf"}"
MODALITIES="${MODALITIES:-Angiogram,CT,MRI,Ultrasound,Xray}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"

cd "$PROJECT_ROOT"
python tools/train_pubmedclip_experts.py \
  --roco-root "$ROCO_ROOT" \
  --out-dir "$OUT_DIR" \
  --modalities "$MODALITIES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE"
