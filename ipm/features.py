"""Stage 3c – Boundary and region feature extraction (Lectures 10.1, 10.2, 11.1)."""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ObjectFeatures:
    id: int
    label: int              # Corresponding value on the label map r(n1, n2)
    cx: float               # Centroid
    cy: float
    x: int                  # Bounding box (x, y, w, h)
    y: int
    w: int
    h: int
    area: int               # A_m (px)
    perimeter: float        # L_m (px)
    diameter: float         # Boundary diameter (px)
    orientation: float      # Major-axis orientation (degrees)
    circularity: float      # 4·pi·A / L^2
    solidity: float         # A / A_convex_hull
    eq_diameter: float      # Equivalent diameter sqrt(4A/pi)
    cls: str = ""

    def to_dict(self):
        return asdict(self)


def boundary_length(pts):
    """L = sum ΔL, ΔL = [(x2-x1)^2 + (y2-y1)^2]^(1/2) along the closed contour (Section 10.1)."""
    if len(pts) < 2:
        return 0.0
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def boundary_diameter(hull_pts):
    """Diameter(B) = max D(p_i, p_j). The farthest pair is always on the convex hull, so only hull points are checked.

    Returns (diameter, major-axis angle in degrees). atan2 is used instead of tan^-1(dy/dx)
    to avoid dividing by zero for a vertical axis. The image y axis points down.
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
    """Compute the feature vector of each region R_1..R_n; IDs run top to bottom, left to right."""
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
