"""Giai đoạn 3c – Trích xuất đặc trưng biên & vùng (Chương 10.1, 10.2, 11.1)."""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ObjectFeatures:
    id: int
    label: int              # Giá trị nhãn tương ứng trên bản đồ nhãn r(n1, n2)
    cx: float               # Trọng tâm
    cy: float
    x: int                  # Khung bao (x, y, w, h)
    y: int
    w: int
    h: int
    area: int               # A_m (px)
    perimeter: float        # L_m (px)
    diameter: float         # Đường kính biên (px)
    orientation: float      # Góc trục chính (độ)
    circularity: float      # 4·pi·A / L^2
    solidity: float         # A / A_bao_lồi
    eq_diameter: float      # Đường kính tương đương sqrt(4A/pi)
    cls: str = ""

    def to_dict(self):
        return asdict(self)


def boundary_length(pts):
    """L = sum ΔL, ΔL = [(x2-x1)^2 + (y2-y1)^2]^(1/2) dọc theo contour khép kín (Mục 10.1)."""
    if len(pts) < 2:
        return 0.0
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def boundary_diameter(hull_pts):
    """Diameter(B) = max D(p_i, p_j). Hai điểm xa nhất luôn là đỉnh bao lồi nên chỉ xét bao lồi.

    Trả về (đường kính, góc trục chính theo độ). Dùng atan2 thay cho tan^-1(dy/dx)
    để không chia cho 0 khi trục chính thẳng đứng. Trục y của ảnh hướng xuống dưới.
    """
    if len(hull_pts) < 2:
        return 0.0, 0.0
    d = hull_pts[:, None, :] - hull_pts[None, :, :]
    D = np.hypot(d[..., 0], d[..., 1])
    i, j = np.unravel_index(np.argmax(D), D.shape)
    (x1, y1), (x2, y2) = hull_pts[i], hull_pts[j]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 180.0
    return float(D[i, j]), float(angle)


def extract(labels, n):
    """Tính vector đặc trưng cho từng vùng R_1..R_n; ID đánh từ trên xuống, trái sang phải."""
    ys, xs = np.nonzero(labels)
    l = labels[ys, xs]
    area = np.bincount(l, minlength=n + 1)
    sx = np.bincount(l, weights=xs, minlength=n + 1)
    sy = np.bincount(l, weights=ys, minlength=n + 1)
    xmin = np.full(n + 1, labels.shape[1]); np.minimum.at(xmin, l, xs)
    xmax = np.full(n + 1, -1); np.maximum.at(xmax, l, xs)
    ymin = np.full(n + 1, labels.shape[0]); np.minimum.at(ymin, l, ys)
    ymax = np.full(n + 1, -1); np.maximum.at(ymax, l, ys)

    objs = []
    for i in range(1, n + 1):
        A = int(area[i])
        if A == 0:
            continue
        x0, y0, x1, y1 = int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])
        region = np.pad((labels[y0:y1 + 1, x0:x1 + 1] == i).astype(np.uint8), 1)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=len)
        L = boundary_length(c.reshape(-1, 2).astype(np.float64))
        hull = cv2.convexHull(c)
        diam, theta = boundary_diameter(hull.reshape(-1, 2).astype(np.float64))
        hull_mask = np.zeros_like(region)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        objs.append(ObjectFeatures(
            id=0, label=i,
            cx=float(sx[i] / A), cy=float(sy[i] / A),
            x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
            area=A, perimeter=L, diameter=diam, orientation=theta,
            circularity=float(4 * np.pi * A / L ** 2) if L > 0 else 0.0,
            solidity=float(A / max(1, hull_mask.sum())),
            eq_diameter=float(np.sqrt(4 * A / np.pi)),
        ))
    objs.sort(key=lambda o: (o.cy, o.cx))
    for k, o in enumerate(objs, start=1):
        o.id = k
    return objs
