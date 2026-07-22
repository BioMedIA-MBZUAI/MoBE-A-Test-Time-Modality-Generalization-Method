#!/usr/bin/env python3
"""Download ROCOv2 radiology images and metadata into the local project tree."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [
    path for path in sys.path
    if Path(path or ".").resolve() != PROJECT_ROOT
]
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "dataset_roco"))
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    return parser.parse_args()


def main():
    args = get_args()
    split_name = "val" if args.split == "validation" else args.split
    out_root = Path(args.out_root).expanduser().resolve() / split_name
    img_dir = out_root / "images"
    meta_path = out_root / "metadata.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("eltorio/ROCOv2-radiology", split=args.split)
    print(ds)

    with meta_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(ds, desc=f"Saving ROCOv2 {args.split} images")):
            img = sample["image"]
            if not isinstance(img, Image.Image):
                raise ValueError("Expected sample['image'] to be a PIL.Image")

            img_name = f"{idx:06d}.jpg"
            img.convert("RGB").save(img_dir / img_name, quality=95)

            meta = {
                "image": img_name,
                "caption": sample.get("caption", ""),
                "modality": sample.get("modality", None),
                "source": sample.get("source", None),
                "idx": idx,
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Images saved to: {img_dir}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
