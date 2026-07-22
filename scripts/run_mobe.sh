#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASETS="${DATASETS:-dermamnist}"
DATA_ROOT="${DATA_ROOT:-"/home/razaimam/Documents/Projects/TTW/datasets_all"}"
EXPERTS_DIR="${EXPERTS_DIR:-"/home/razaimam/Documents/Projects/TTW/experts"}"
LOG_DIR="${LOG_DIR:-"$PROJECT_ROOT/results/mobe"}"
CONFIG_DIR="${CONFIG_DIR:-"$PROJECT_ROOT/configs"}"

cd "$PROJECT_ROOT"
python mobe.py \
  --config "$CONFIG_DIR" \
  --datasets "$DATASETS" \
  --data-root "$DATA_ROOT" \
  --experts-dir "$EXPERTS_DIR" \
  --log-dir "$LOG_DIR"
