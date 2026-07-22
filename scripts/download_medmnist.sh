#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-"$PROJECT_ROOT/datasets_all/medmnist"}"
RECORD_ID="${RECORD_ID:-10519652}"
DATASETS="${DATASETS:-bloodmnist_224 breastmnist_224 chestmnist_224 dermamnist organamnist_224 organcmnist_224 organsmnist_224 pathmnist_224 retinamnist_224 tissuemnist_224 octmnist_224}"

mkdir -p "$OUT_DIR"

for dataset in $DATASETS; do
  url="https://zenodo.org/records/${RECORD_ID}/files/${dataset}.npz"
  echo "Downloading ${dataset}.npz"
  wget -c -P "$OUT_DIR" "$url"
done

echo "MedMNIST files are in: $OUT_DIR"
