"""Giai đoạn 3b – Tách vật thể chạm nhau: Distance Transform + Watershed (Chương 9).

Đây là lời giải cho hạn chế "Risk of Oversimplification" (Case 4): hai vật dính nhau
tạo thành MỘT vùng liên thông. Vùng như vậy lõm ở chỗ tiếp xúc nên có độ đặc
(solidity = A / A_bao_lồi) thấp; ta chỉ chạy Watershed trên những vùng bị nghi ngờ.
"""
import cv2
import numpy as np
from skimage.segmentation import watershed


def _shape(region):
    """(độ đặc A/A_hull, độ tròn 4·pi·A/L^2) của một vùng nhị phân."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_mask = np.zeros_like(region)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    A = region.sum()
    L = cv2.arcLength(c, True)
    return A / max(1, hull_mask.sum()), (4 * np.pi * A / L ** 2 if L > 0 else 0.0)


def _markers(dist):
    """Đỉnh cục bộ của bản đồ khoảng cách = tâm của từng vật (mỗi đồng xu có 1 đỉnh)."""
    dmax = dist.max()
    k = max(3, int(0.5 * dmax)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peaks = (dist >= cv2.dilate(dist, se)) & (dist >= 0.4 * dmax)
    n, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    return n - 1, markers


def split_touching(labels, n, solidity_thresh=0.94, min_circularity=0.0):
    """Trả về (bản đồ nhãn mới, số vùng mới, danh sách các vùng đã bị tách).

    Vùng quá kém tròn (dải mép bàn, vệt bóng dài) KHÔNG được tách: Watershed sẽ băm
    chúng thành nhiều mảnh tròn giả; cứ để nguyên cho bộ lọc hình dạng phía sau loại bỏ.
    """
    out = labels.copy()
    next_id = n + 1
    log = []
    ys, xs = np.nonzero(labels)
    l = labels[ys, xs]
    for i in range(1, n + 1):
        sel = l == i
        if not sel.any():
            continue
        y0, y1 = ys[sel].min(), ys[sel].max() + 1
        x0, x1 = xs[sel].min(), xs[sel].max() + 1
        region = np.pad((labels[y0:y1, x0:x1] == i).astype(np.uint8), 1)
        sol, circ = _shape(region)
        if sol >= solidity_thresh or circ < min_circularity:
            continue
        dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        n_peaks, markers = _markers(dist)
        if n_peaks < 2:
            continue
        ws = watershed(-dist, markers, mask=region.astype(bool))[1:-1, 1:-1]
        view = out[y0:y1, x0:x1]
        for j in range(2, n_peaks + 1):
            view[ws == j] = next_id
            next_id += 1
        log.append({"label": i, "solidity": round(float(sol), 3), "parts": n_peaks})
    return out, next_id - 1, log
