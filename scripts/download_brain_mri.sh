#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_DIR="${OUT_DIR:-"$PROJECT_ROOT/datasets_all/hardbench/BTMRI"}"
TMP_DIR="${TMP_DIR:-"$PROJECT_ROOT/.cache/brain_mri_raw"}"

cd "$PROJECT_ROOT"
OUT_DIR="$OUT_DIR" TMP_DIR="$TMP_DIR" python tools/download_brain_mri.py
