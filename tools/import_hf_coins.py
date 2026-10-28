"""Import the REAL coin dataset "coins-1apki" (Roboflow 100, CC BY 4.0) from Hugging Face.

Source: https://huggingface.co/datasets/Francesco/coins-1apki
       (original: https://universe.roboflow.com/object-detection/coins-1apki)
Smartphone photos on many backgrounds (wood, gray desk, green mat, yellow paper), stretched by Roboflow
to 640x640. Labels: a COCO bounding box per object, classes coin/nail/nut/screw.

Only images containing ONLY coins, with 3 to 30 coins, are kept (the scope of the project spec).
Ground truth = count + bounding boxes, for Precision/Recall evaluation.

Run:
    curl -L -o data/downloads/hf_coins/test.parquet ^
      https://huggingface.co/datasets/Francesco/coins-1apki/resolve/main/data/test-00000-of-00001-83ed470261ddf492.parquet
    python tools/import_hf_coins.py data/downloads/hf_coins/test.parquet data/real/hf_coins
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

COIN = 1  # Class ids: 0 coins (super-class), 1 coin, 2 nail, 3 nut, 4 screw


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    src =Path(sys.argv[1] if len(sys.argv) > 1 else "data/downloads/hf_coins/test.parquet")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/real/hf_coins")
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(src).to_pylist()
    gt = {}
    for r in rows:
        cats = r["objects"]["category"]
        if set(cats) != {COIN} or not 3 <= len(cats) <= 30:
            continue
        name = f"hf_{r['image_id']:04d}.jpg"
        (out / name).write_bytes(r["image"]["bytes"])
        gt[name] = {"count": len(cats),
                    "boxes": [[round(v, 1) for v in b] for b in r["objects"]["bbox"]]}
    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1)
    # Random 50/50 split: dev for parameter tuning, test only for reporting the final result
    names = sorted(gt)
    random.Random(2026).shuffle(names)
    half = len(names) // 2
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump({"dev": sorted(names[:half]), "test": sorted(names[half:])}, f, indent=1)
    (out / "SOURCE.md").write_text(
        "Images from the \"coins-1apki\" dataset (Roboflow 100), license CC BY 4.0.\n"
        "https://universe.roboflow.com/object-detection/coins-1apki\n"
        "https://huggingface.co/datasets/Francesco/coins-1apki (split test)\n"
        "Filtered: only images that contain coins only, 3–30 coins per image.\n", encoding="utf-8")
    print(f"Wrote {len(gt)}/{len(rows)} images to {out}")


if __name__ == "__main__":
    main()
