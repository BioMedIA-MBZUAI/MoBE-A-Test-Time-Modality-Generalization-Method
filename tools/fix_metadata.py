#!/usr/bin/env python3
"""Normalize a ROCO JSONL metadata file into strict JSON."""

import argparse
import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--roco-root", default=str(PROJECT_ROOT / "dataset_roco"))
    parser.add_argument("--input-name", default="metadata.jsonl")
    parser.add_argument("--output-name", default="metadata_fixed.jsonl")
    return parser.parse_args()


def main():
    args = get_args()
    split_dir = Path(args.roco_root).expanduser().resolve() / args.split
    in_path = split_dir / args.input_name
    out_path = split_dir / args.output_name

    bad = 0
    total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                item = json.loads(line)
            except Exception:
                try:
                    item = ast.literal_eval(line)
                except Exception:
                    bad += 1
                    continue

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")
    print(f"Total lines: {total}, bad lines skipped: {bad}")


if __name__ == "__main__":
    main()
