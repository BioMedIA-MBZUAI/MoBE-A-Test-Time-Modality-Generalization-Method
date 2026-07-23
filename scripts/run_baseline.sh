#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:-biomedclip}"

DATASETS="${DATASETS:-breastmnist_224}"
DATA_ROOT="${DATA_ROOT:-"/home/razaimam/Documents/Projects/TTW/datasets_all"}"
CONFIG_DIR="${CONFIG_DIR:-"$PROJECT_ROOT/configs"}"
EXPERTS_DIR="${EXPERTS_DIR:-"/home/razaimam/Documents/Projects/TTW/experts"}"
LOG_DIR="${LOG_DIR:-"$PROJECT_ROOT/results/$METHOD"}"

cd "$PROJECT_ROOT"

case "$METHOD" in
  biomedclip|pubmedclip|tda|tpt)
    python -m "baselines.${METHOD}" \
      --datasets "$DATASETS" \
      --data-root "$DATA_ROOT" \
      --log-dir "$LOG_DIR"
    ;;
  mome)
    python -m baselines.mome \
      --config "$CONFIG_DIR" \
      --datasets "$DATASETS" \
      --data-root "$DATA_ROOT" \
      --experts-dir "$EXPERTS_DIR" \
      --log-dir "$LOG_DIR"
    ;;
  *)
    echo "Unknown baseline: $METHOD"
    echo "Choose one of: biomedclip, pubmedclip, tda, tpt, mome"
    exit 2
    ;;
esac
