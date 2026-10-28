"""Stage 2a – Similarity-based segmentation: Otsu (Lecture 1.2) & K-Means (Lecture 9.1)."""
import cv2
import numpy as np

from .preprocess import K, histogram


def otsu_threshold(gray, backend="numpy"):
    """Find T* = argmax_t sigma_b^2(t), with sigma_b^2 = w0·w1·(mu0 - mu1)^2.

    Background class C0 = {0..t}, object class C1 = {t+1..K-1}  ->  mask = (gray > T*).
    Returns (T*, array of sigma_b^2 over t) – the array is plotted in the report.
    """
    H = histogram(gray)
    N = H.sum()
    k = np.arange(K)
    cum = np.cumsum(H)
    w0 = cum / N
    w1 = (N - cum) / N                       # integer arithmetic -> w1 is exactly 0 at the end of the range
    m_cum = np.cumsum(k * H) / N             # sum_{k<=t} k·p(k)
    mu_T = m_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m_cum / w0
        mu1 = (mu_T - m_cum) / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b = np.nan_to_num(sigma_b, nan=0.0, posinf=0.0, neginf=0.0)
    if backend == "opencv":
        T, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(T), sigma_b
    return int(np.argmax(sigma_b)), sigma_b


def kmeans_threshold(gray, k=2, eps=0.2, max_iter=100):
    """4-step K-Means (Section 9.1) on scalar samples z_q = gray level.

    All pixels with the same gray level always fall into the same cluster, so clustering is done
    directly on the 256 gray levels weighted by H(k) – identical to running on the M x N pixels
    but thousands of times faster. Deterministic initialization at min/max makes results repeatable.
    Returns (T, cluster centers, number of iterations); for k = 2, T is the midpoint of the 2 centers.
    """
    H = histogram(gray).astype(np.float64)
    levels = np.arange(K, dtype=np.float64)
    present = np.flatnonzero(H)
    m = np.linspace(present[0], present[-1], k)                     # Step 1: initialize
    it = 0
    for it in range(1, max_iter + 1):
        assign = np.argmin(np.abs(levels[:, None] - m[None, :]), axis=1)  # Step 2: assign samples
        new_m = m.copy()
        for i in range(k):                                           # Step 3: update centers
            wsum = H[assign == i].sum()
            if wsum > 0:
                new_m[i] = (levels[assign == i] * H[assign == i]).sum() / wsum
        E = np.abs(new_m - m).sum()                                  # Step 4: check convergence
        m = new_m
        if E <= eps:
            break
    T = int(np.floor((m[0] + m[-1]) / 2)) if k == 2 else None
    return T, m, it


def triangle_threshold(gray):
    """Triangle method (Zack 1977) for a SINGLE-peak histogram with a long tail.

    Join the histogram peak to the farthest non-zero gray level; the threshold is the histogram
    point with the largest perpendicular distance to that line. It suits the difference image D(n):
    the background piles up near 0 and the objects, only a few percent, lie in the tail – a case
    where Otsu performs poorly.
    """
    H = histogram(gray).astype(np.float64)
    peak = int(np.argmax(H))
    last = int(np.flatnonzero(H)[-1])
    if last <= peak:
        return peak
    k = np.arange(peak, last + 1)
    x1, y1, x2, y2 = peak, H[peak], last, H[last]
    dist = np.abs((y2 - y1) * k - (x2 - x1) * H[k] + x2 * y1 - y2 * x1)
    return int(k[np.argmax(dist)])


def mad_threshold(gray, k=6.0):
    """T = median + k·1.4826·MAD: a robust statistical threshold that treats objects as "outliers" of the background."""
    v = gray.astype(np.float64).ravel()
    med = np.median(v)
    return int(np.ceil(med + k * 1.4826 * np.median(np.abs(v - med))))


def background_is_bright(gray, T, margin_frac=0.02):
    """Detect polarity: the image border is almost always background, so check whether most border pixels are bright or dark."""
    h, w = gray.shape
    m = max(2, int(min(h, w) * margin_frac))
    border = np.concatenate([gray[:m].ravel(), gray[-m:].ravel(),
                             gray[:, :m].ravel(), gray[:, -m:].ravel()])
    return bool(np.mean(border > T) > 0.5)


def threshold_mask(gray, T, bg_bright):
    """Objects = 1. On a bright background (white paper) the objects are the DARK class -> invert the mask."""
    bright = gray > T
    return (~bright if bg_bright else bright).astype(np.uint8)


def segment(gray, cfg, method=None):
    """Return (binary mask {0,1}, segmentation info)."""
    method = method or cfg.seg_method
    info = {"method": method}
    if method == "bgsub":
        # The input is the difference image D(n): objects are always the high-D class
        rule = cfg.bg_threshold
        if rule == "otsu":
            T, _ = otsu_threshold(gray, cfg.backend)
        elif rule == "mad":
            T = mad_threshold(gray, cfg.bg_mad_k)
        else:
            T = triangle_threshold(gray)
        info.update(T=int(T), threshold_rule=rule, background_bright=False)
        return threshold_mask(gray, T, False), info
    if method == "otsu":
        T, _ = otsu_threshold(gray, cfg.backend)
    elif method == "kmeans":
        T, centers, iters = kmeans_threshold(gray)
        info.update(centers=[round(float(c), 2) for c in centers], iterations=iters)
    else:
        raise ValueError(f"invalid seg_method: {method}")
    if cfg.polarity == "auto":
        bg_bright = background_is_bright(gray, T)
    else:
        bg_bright = cfg.polarity == "dark"
    info.update(T=int(T), background_bright=bg_bright)
    return threshold_mask(gray, T, bg_bright), info
