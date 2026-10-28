"""Kiểm chứng các thuật toán tự cài đặt (NumPy) so với OpenCV: kết quả & thời gian.

Chạy:  python tools/verify_impl.py [đường_dẫn_ảnh]
Mặc định dùng data/synthetic/case3_noisy_glare.jpg.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ipm import Config  # noqa: E402
from ipm import labeling as lab  # noqa: E402
from ipm import morphology as morph  # noqa: E402
from ipm import preprocess as pre  # noqa: E402
from ipm import segmentation as seg  # noqa: E402


def timed(fn, *args, repeat=3):
    best, out = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t)
    return out, best * 1000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/case3_noisy_glare.jpg"
    img, _ = pre.resize_to_width(pre.load_image(path), 800)
    gray = pre.rgb_to_gray(img)
    cfg = Config()
    rows = []

    (g1, t1) = timed(pre.median_filter, gray, 5, "numpy")
    (g2, t2) = timed(pre.median_filter, gray, 5, "opencv")
    rows.append(("Lọc trung vị 5x5", f"khác biệt tối đa = {np.abs(g1.astype(int) - g2).max()}", t1, t2))

    (s1, t1) = timed(pre.gaussian_filter, g1, 5, 1.0, "numpy")
    (s2, t2) = timed(pre.gaussian_filter, g1, 5, 1.0, "opencv")
    rows.append(("Lọc Gaussian 5x5", f"khác biệt tối đa = {np.abs(s1.astype(int) - s2).max()}", t1, t2))

    ((T1, _), t1) = timed(seg.otsu_threshold, s1, "numpy")
    ((T2, _), t2) = timed(seg.otsu_threshold, s1, "opencv")
    rows.append(("Ngưỡng Otsu", f"T* tự cài = {T1}, OpenCV = {T2}", t1, t2))

    mask, _ = seg.segment(s1, cfg)
    se = morph.structuring_element(5)
    (e1, t1) = timed(morph.erode, mask, se, "numpy")
    (e2, t2) = timed(morph.erode, mask, se, "opencv")
    rows.append(("Phép Co (Erosion)", f"trùng khớp = {np.array_equal(e1, e2)}", t1, t2))
    (d1, t1) = timed(morph.dilate, mask, se, "numpy")
    (d2, t2) = timed(morph.dilate, mask, se, "opencv")
    rows.append(("Phép Giãn (Dilation)", f"trùng khớp = {np.array_equal(d1, d2)}", t1, t2))

    for conn in (4, 8):
        ((l1, n1), t1) = timed(lab.label, mask, conn, "numpy", repeat=1)
        ((l2, n2), t2) = timed(lab.label, mask, conn, "opencv", repeat=1)
        rows.append((f"Gán nhãn {conn}-liên thông",
                     f"số vùng {n1} / {n2}, cùng phân hoạch = {lab.same_partition(l1, l2)}", t1, t2))

    print(f"Ảnh: {path}  ({gray.shape[1]}x{gray.shape[0]})\n")
    print(f"{'Thuật toán':<26}{'Kết quả so sánh':<46}{'Tự cài (ms)':>12}{'OpenCV (ms)':>13}")
    print("-" * 97)
    for name, res, ta, tb in rows:
        print(f"{name:<26}{res:<46}{ta:>12.2f}{tb:>13.2f}")


if __name__ == "__main__":
    main()
