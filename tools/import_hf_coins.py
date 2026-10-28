"""Nhập bộ ảnh đồng xu THẬT "coins-1apki" (Roboflow 100, CC BY 4.0) từ Hugging Face.

Nguồn: https://huggingface.co/datasets/Francesco/coins-1apki
       (gốc: https://universe.roboflow.com/object-detection/coins-1apki)
Ảnh chụp điện thoại, nhiều nền (gỗ, bàn xám, thảm xanh, giấy vàng), đã bị Roboflow
kéo giãn về 640x640. Nhãn: khung bao COCO cho từng vật, lớp coin/nail/nut/screw.

Chỉ giữ ảnh CHỈ chứa đồng xu và có từ 3 đến 30 xu (đúng phạm vi đặc tả đề tài).
Ground truth gồm số đếm + khung bao để đánh giá Precision/Recall.

Chạy:
    curl -L -o data/downloads/hf_coins/test.parquet ^
      https://huggingface.co/datasets/Francesco/coins-1apki/resolve/main/data/test-00000-of-00001-83ed470261ddf492.parquet
    python tools/import_hf_coins.py data/downloads/hf_coins/test.parquet data/real/hf_coins
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

COIN = 1  # Nhãn lớp: 0 coins (siêu lớp), 1 coin, 2 nail, 3 nut, 4 screw


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
    # Chia ngẫu nhiên 50/50: dev để tinh chỉnh tham số, test chỉ dùng để báo cáo kết quả cuối
    names = sorted(gt)
    random.Random(2026).shuffle(names)
    half = len(names) // 2
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump({"dev": sorted(names[:half]), "test": sorted(names[half:])}, f, indent=1)
    (out / "SOURCE.md").write_text(
        "Ảnh từ bộ dữ liệu \"coins-1apki\" (Roboflow 100), giấy phép CC BY 4.0.\n"
        "https://universe.roboflow.com/object-detection/coins-1apki\n"
        "https://huggingface.co/datasets/Francesco/coins-1apki (split test)\n"
        "Đã lọc: chỉ ảnh chứa toàn đồng xu, 3–30 xu mỗi ảnh.\n", encoding="utf-8")
    print(f"Đã ghi {len(gt)}/{len(rows)} ảnh vào {out}")


if __name__ == "__main__":
    main()
