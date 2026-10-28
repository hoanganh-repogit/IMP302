"""Stage 2b – Binary morphology (Lectures 1.2 & 2.3)."""
import cv2
import numpy as np


def structuring_element(size):
    """Elliptical size x size SE (centrally symmetric, so S equals its reflection)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _offsets(se):
    return [(dy, dx) for dy, dx in zip(*np.nonzero(se))]


def erode(mask, se, backend="numpy"):
    """(b ⊖ S)(x) = 1  <=>  S placed at x lies entirely inside B (every pixel under the SE is 1)."""
    if backend == "opencv":
        return cv2.erode(mask.astype(np.uint8), se)
    h, w = mask.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    # Outside the border counts as 1 (as in OpenCV) so objects touching the edge are not eroded from outside
    p = np.pad(mask.astype(bool), ((ph, ph), (pw, pw)), constant_values=True)
    out = np.ones((h, w), dtype=bool)
    for dy, dx in _offsets(se):
        out &= p[dy:dy + h, dx:dx + w]
    return out.astype(np.uint8)


def dilate(mask, se, backend="numpy"):
    """(b ⊕ S)(x) = 1  <=>  S placed at x intersects B (at least one pixel under the SE is 1)."""
    if backend == "opencv":
        return cv2.dilate(mask.astype(np.uint8), se)
    h, w = mask.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    p = np.pad(mask.astype(bool), ((ph, ph), (pw, pw)), constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy, dx in _offsets(se):
        out |= p[dy:dy + h, dx:dx + w]
    return out.astype(np.uint8)


def opening(mask, se, iterations=1, backend="numpy"):
    """b ∘ S = (b ⊖ S) ⊕ S : removes background specks smaller than the SE."""
    out = mask
    for _ in range(iterations):
        out = erode(out, se, backend)
    for _ in range(iterations):
        out = dilate(out, se, backend)
    return out


def closing(mask, se, iterations=1, backend="numpy"):
    """b • S = (b ⊕ S) ⊖ S : seals cracks and holes smaller than the SE inside objects."""
    out = mask
    for _ in range(iterations):
        out = dilate(out, se, backend)
    for _ in range(iterations):
        out = erode(out, se, backend)
    return out


def fill_holes(mask):
    """Fill every black hole lying fully inside an object (background not connected to the border).

    Complements closing: glare spots larger than the 5x5 SE cannot be sealed by closing.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = m
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)  # flood the background from the border
    holes = ff[1:-1, 1:-1] == 0
    return (m | holes).astype(np.uint8)
