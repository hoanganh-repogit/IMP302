"""Stage 3b – Splitting touching objects: distance transform + watershed (Lecture 9).

This addresses the "Risk of Oversimplification" limitation (Case 4): two touching objects
form ONE connected region. Such a region is concave at the contact point, so its solidity
(A / A_convex_hull) is low; watershed is run only on these suspicious regions.
"""
import cv2
import numpy as np
from skimage.segmentation import watershed


def _shape(region):
    """(solidity A/A_hull, circularity 4·pi·A/L^2) of a binary region."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_mask = np.zeros_like(region)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    A = region.sum()
    L = cv2.arcLength(c, True)
    return A / max(1, hull_mask.sum()), (4 * np.pi * A / L ** 2 if L > 0 else 0.0)


def _markers(dist):
    """Local maxima of the distance map = object centers (one peak per coin)."""
    dmax = dist.max()
    k = max(3, int(0.5 * dmax)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peaks = (dist >= cv2.dilate(dist, se)) & (dist >= 0.4 * dmax)
    n, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    return n - 1, markers


def split_touching(labels, n, solidity_thresh=0.94, min_circularity=0.0):
    """Return (new label map, new number of regions, list of the regions that were split).

    Regions that are far from round (table-edge strips, long shadow streaks) are NOT split:
    watershed would chop them into many fake round pieces; they are left for the shape filter.
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
