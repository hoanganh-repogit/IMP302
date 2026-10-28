"""Chia ảnh thật thành 4 "real case" tương ứng 4 kịch bản giả lập.

Chỉ dùng nhãn chuẩn + thống kê ảnh (KHÔNG dùng kết quả thuật toán) để việc chia không thiên vị:
    R4 touching       – có ít nhất một cặp xu (hộp chuẩn) chạm/chồng nhau
    R3 textured       – độ nhám nền thuộc 30% cao nhất (vân gỗ, hạt thảm, vết bẩn)
    R2 low contrast   – chênh lệch màu Lab xu–nền thuộc 30% thấp nhất
    R1 ideal          – còn lại
Thứ tự ưu tiên R4 > R3 > R2 > R1 để mỗi ảnh thuộc đúng một case. Ngưỡng phân vị tính trên cả 960 ảnh.
Chạy:  python tools/real_cases.py   ->  report/data/real_cases.csv
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real" / "hf_coins"
OUT = ROOT / "report" / "data" / "real_cases.csv"
CASES = {1: "R1 Ideal", 2: "R2 Low contrast", 3: "R3 Textured background", 4: "R4 Touching coins"}


def touching(boxes, gap=3.0):
    """Xấp xỉ mỗi xu bằng hình tròn nội tiếp hộp; chạm nhau nếu khoảng cách tâm <= r1 + r2 + gap."""
    c = np.array([[x + w / 2, y + h / 2, (w + h) / 4] for x, y, w, h in boxes])
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if np.hypot(*(c[i, :2] - c[j, :2])) <= c[i, 2] + c[j, 2] + gap:
                return True
    return False


def attributes(img, boxes):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    coin = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        cv2.ellipse(coin, (int(x + bw / 2), int(y + bh / 2)), (int(bw / 2), int(bh / 2)), 0, 0, 360, 1, -1)
    near = cv2.dilate(coin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = near == 0
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    texture = float(np.std(highpass[bg]))                     # độ nhám nền
    contrasts = []
    for x, y, bw, bh in boxes:                                # chênh lệch màu xu – vành nền quanh xu
        m_in = np.zeros((h, w), np.uint8)
        cv2.ellipse(m_in, (int(x + bw / 2), int(y + bh / 2)), (max(1, int(bw / 2.6)), max(1, int(bh / 2.6))),
                    0, 0, 360, 1, -1)
        m_ring = np.zeros((h, w), np.uint8)
        cv2.ellipse(m_ring, (int(x + bw / 2), int(y + bh / 2)), (int(bw / 2) + 10, int(bh / 2) + 10),
                    0, 0, 360, 1, -1)
        m_ring = (m_ring > 0) & (near == 0)
        if m_in.any() and m_ring.any():
            contrasts.append(np.linalg.norm(lab[m_in > 0].mean(0) - lab[m_ring].mean(0)))
    return texture, float(np.median(contrasts)) if contrasts else 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gt = json.load(open(REAL / "ground_truth.json", encoding="utf-8"))
    split = json.load(open(REAL / "split.json", encoding="utf-8"))
    which = {n: s for s, names in split.items() for n in names}
    rows = []
    for name in sorted(gt):
        img = cv2.imread(str(REAL / name))
        tex, con = attributes(img, gt[name]["boxes"])
        rows.append({"image": name, "split": which[name], "touching": int(touching(gt[name]["boxes"])),
                     "texture": round(tex, 3), "contrast": round(con, 2)})
    t_hi = np.percentile([r["texture"] for r in rows], 70)
    c_lo = np.percentile([r["contrast"] for r in rows], 30)
    for r in rows:
        r["case"] = 4 if r["touching"] else 3 if r["texture"] >= t_hi else 2 if r["contrast"] <= c_lo else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Ngưỡng: texture >= {t_hi:.2f}, contrast <= {c_lo:.2f}")
    for k, v in CASES.items():
        n_all = sum(r["case"] == k for r in rows)
        n_test = sum(r["case"] == k and r["split"] == "test" for r in rows)
        print(f"  {v:<24} tổng {n_all:>4}   test {n_test:>4}")


if __name__ == "__main__":
    main()
