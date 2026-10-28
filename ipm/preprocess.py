"""Stage 1 – Acquisition & gray-level preprocessing (Lectures 1.1, 2.1, 2.2, 3, 7, 8)."""
from pathlib import Path

import cv2
import numpy as np

K = 256  # Number of gray levels after 8-bit quantization (D = 256)


# ----------------------------------------------------------------------------
# Image read / write (supports non-ASCII, e.g. Vietnamese, paths on Windows)
# ----------------------------------------------------------------------------
def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def save_image(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise ValueError(f"Cannot write image: {path}")
    buf.tofile(str(path))


# ----------------------------------------------------------------------------
# 1.1 Geometric transforms & interpolation (Lecture 1.1)
# ----------------------------------------------------------------------------
def nearest_neighbor_resize(img, out_h, out_w):
    """Inverse mapping + nearest-neighbor interpolation: g(n) = f{INT[a1 + 0.5], INT[a2 + 0.5]}."""
    in_h, in_w = img.shape[:2]
    a1 = np.arange(out_h) * (in_h / out_h)
    a2 = np.arange(out_w) * (in_w / out_w)
    rows = np.clip(np.floor(a1 + 0.5).astype(int), 0, in_h - 1)
    cols = np.clip(np.floor(a2 + 0.5).astype(int), 0, in_w - 1)
    return img[rows[:, None], cols[None, :]]


def resize_to_width(img, width, method="area"):
    """Return (image, scale factor s). Only downscales, never upscales."""
    h, w = img.shape[:2]
    if not width or w <= width:
        return img.copy(), 1.0
    s = width / w
    out_h = round(h * s)
    if method == "nearest":
        return nearest_neighbor_resize(img, out_h, width), s
    return cv2.resize(img, (width, out_h), interpolation=cv2.INTER_AREA), s


# ----------------------------------------------------------------------------
# 1.2 RGB -> gray-level conversion (luminance Y, BT.601)
# ----------------------------------------------------------------------------
def rgb_to_gray(bgr):
    b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.rint(y), 0, K - 1).astype(np.uint8)


# ----------------------------------------------------------------------------
# 1.3 Histogram, AOD, FSHS (Lecture 1.1)
# ----------------------------------------------------------------------------
def histogram(gray):
    """H_f(k) = number of pixels with gray level k; sum_k H_f(k) = M x N."""
    return np.bincount(gray.ravel(), minlength=K).astype(np.int64)


def aod(gray):
    """AOD(f) = (1/MN) * sum_k k * H_f(k)."""
    H = histogram(gray)
    return float((np.arange(K) * H).sum() / H.sum())


def fshs(gray, low_pct=0.0, high_pct=100.0):
    """Full-Scale Histogram Stretch: g = (K-1)/(B-A) * (f - A).

    With low_pct=0, high_pct=100 this is exactly the textbook FSHS (A = min, B = max).
    Taking A, B as percentiles keeps a few outliers (dust specks, glare) from making
    B - A = 255, which would cancel the stretch. Returns (g, A, B).
    """
    A = float(np.percentile(gray, low_pct))
    B = float(np.percentile(gray, high_pct))
    if B <= A:
        return gray.copy(), A, B
    g = (K - 1) / (B - A) * (gray.astype(np.float64) - A)
    return np.clip(np.rint(g), 0, K - 1).astype(np.uint8), A, B


def flatten_illumination(gray, background_bright, se_frac=0.4):
    """Flatten uneven illumination under the multiplicative model f = i(x,y) * r(x,y).

    The background i(x,y) is estimated by closing (bright background) or opening (dark background)
    with an SE LARGER than the largest object (otherwise objects leak into the background). Bright
    background: divide by it. Dark background: subtract it (top-hat), since dividing by values near 0 amplifies noise.
    Computed at 1/4 resolution for speed.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(3, int(min(small.shape) * se_frac)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    op = cv2.MORPH_CLOSE if background_bright else cv2.MORPH_OPEN
    bg = cv2.morphologyEx(small, op, se)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    if background_bright:
        out = gray.astype(np.float64) / np.maximum(bg, 1.0) * bg.mean()
    else:
        out = gray.astype(np.float64) - bg + bg.mean()
    return np.clip(np.rint(out), 0, K - 1).astype(np.uint8)


def background_difference(bgr, kernel_frac=0.25, color=True):
    """Background-difference image: D(n) = ||I(n) - B(n)|| (change-detection idea, Section 9.5).

    The "reference frame" of video is replaced by a background B(n) estimated from the image itself: a
    large median window removes every object smaller than ~0.8x the window and keeps only the background (including shading).
    Objects BRIGHTER or DARKER than the background both give a large D, so no polarity detection is needed.
    With color=True, D is measured in Lab space: it catches yellow coins on a green mat even when their
    brightness is about the same. The background is computed at 1/4 resolution for speed.
    """
    h, w = bgr.shape[:2]
    src = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB) if color else rgb_to_gray(bgr)
    small = cv2.resize(src, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(3, int(min(small.shape[:2]) * kernel_frac)) | 1
    bg = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)
    diff = src.astype(np.float64) - bg.astype(np.float64)
    D = np.sqrt((diff ** 2).sum(axis=-1)) if diff.ndim == 3 else np.abs(diff)
    return np.clip(np.rint(D), 0, K - 1).astype(np.uint8)


# ----------------------------------------------------------------------------
# 1.4 Spatial-domain denoising (Lectures 2.1, 2.2, 3.3)
# ----------------------------------------------------------------------------
def gaussian_kernel_1d(ksize, sigma):
    """w(s) ~ exp(-s^2 / 2sigma^2); the 2-D Gaussian is separable, w(s,t) = w(s)·w(t)."""
    s = np.arange(ksize) - ksize // 2
    w = np.exp(-(s ** 2) / (2.0 * sigma ** 2))
    return w / w.sum()


def gaussian_filter(gray, ksize=5, sigma=1.0, backend="numpy"):
    if backend == "opencv":
        return cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    h, w = gray.shape
    p = ksize // 2
    wk = gaussian_kernel_1d(ksize, sigma)
    f = np.pad(gray.astype(np.float64), p, mode="reflect")  # = OpenCV BORDER_REFLECT_101
    tmp = sum(wk[i] * f[:, i:i + w] for i in range(ksize))      # convolve along rows
    out = sum(wk[i] * tmp[i:i + h, :] for i in range(ksize))    # convolve along columns
    return np.clip(np.rint(out), 0, K - 1).astype(np.uint8)


def median_filter(gray, ksize=5, backend="numpy"):
    """f^(n) = median_{(s,t) in S_xy} g(s,t) – a nonlinear order-statistic filter."""
    if backend == "opencv":
        return cv2.medianBlur(gray, ksize)
    p = ksize // 2
    padded = np.pad(gray, p, mode="edge")  # = BORDER_REPLICATE used by cv2.medianBlur
    win = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    return np.median(win, axis=(-2, -1)).astype(np.uint8)


def denoise(gray, cfg):
    """Median FIRST (kills impulse noise), Gaussian AFTER (smooths sensor noise).

    In the opposite order the Gaussian would "smear" each impulse over its neighborhood
    and the median filter that follows would have a harder time removing it.
    """
    out = gray
    if cfg.median_ksize and cfg.median_ksize > 1:
        out = median_filter(out, cfg.median_ksize, cfg.backend)
    if cfg.gaussian_ksize and cfg.gaussian_ksize > 1:
        out = gaussian_filter(out, cfg.gaussian_ksize, cfg.gaussian_sigma, cfg.backend)
    return out
