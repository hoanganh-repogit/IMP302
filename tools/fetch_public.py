"""Lưu các ảnh đồng xu công khai có sẵn trong thư viện scikit-image vào data/public/.

- skimage.data.coins(): ảnh xám 303x384, 24 đồng xu Hy Lạp cổ trên nền tối, ánh sáng KHÔNG đều.
  Dùng làm bài kiểm tra "khó" cho phần hạn chế 7.2 (thử với --flatten).
Chạy:  python tools/fetch_public.py
"""
import json
import sys
from pathlib import Path

import cv2
from skimage import data

OUT = Path(__file__).resolve().parents[1] / "data" / "public"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "skimage_coins.png"), data.coins())
    gt = {"skimage_coins.png": {"count": 24}}
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"Đã lưu vào {OUT}")


if __name__ == "__main__":
    main()
