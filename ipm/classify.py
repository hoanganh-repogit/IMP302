"""Stage 4 – Pattern classification with a decision rule on area (Lecture 12).

Decision function:  A_m >= T_area -> C1 (Large),  A_m < T_area -> C2 (Small).
Unlike the original spec, T_area is not the mean area: the mean is biased when the numbers of
large and small objects differ or when a merged blob is present, so the default is 1-D Otsu on
the set of areas (maximum between-class variance – the same principle as Otsu in Stage 2).
"""
import numpy as np

LARGE, SMALL, SAME = "Large", "Small", "Same"


def otsu_1d(values):
    """Split the sorted areas into 2 groups that maximize w0·w1·(mu0 - mu1)^2."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    best, best_i = -1.0, 1
    for i in range(1, n):
        w0, w1 = i / n, (n - i) / n
        sb = w0 * w1 * (v[:i].mean() - v[i:].mean()) ** 2
        if sb > best:
            best, best_i = sb, i
    return (v[best_i - 1] + v[best_i]) / 2


def largest_gap(values):
    v = np.sort(np.asarray(values, dtype=np.float64))
    i = int(np.argmax(np.diff(v)))
    return (v[i] + v[i + 1]) / 2


def classify(objects, cfg):
    """Set o.cls for every object and return information about the decision threshold."""
    info = {"method": cfg.classify_method, "threshold": None, "warning": None}
    if not objects:
        info.update(n_large=0, n_small=0)
        return info
    areas = np.array([o.area for o in objects], dtype=np.float64)

    if cfg.area_threshold is not None:
        info["method"] = "fixed"
        T = float(cfg.area_threshold)
    elif len(areas) < 2 or areas.max() / areas.min() < cfg.min_size_ratio:
        # All objects are about the same size: an adaptive split would be arbitrary
        info["warning"] = (f"A_max/A_min = {areas.max() / areas.min():.2f} < {cfg.min_size_ratio}: "
                           "single size group, no Large/Small split")
        for o in objects:
            o.cls = SAME
        info.update(n_large=0, n_small=0, n_same=len(objects))
        return info
    elif cfg.classify_method == "mean":
        T = float(areas.mean())
    elif cfg.classify_method == "gap":
        T = largest_gap(areas)
    else:
        T = otsu_1d(areas)

    for o in objects:
        o.cls = LARGE if o.area >= T else SMALL
    info.update(threshold=round(T, 1),
                n_large=sum(o.cls == LARGE for o in objects),
                n_small=sum(o.cls == SMALL for o in objects))
    return info
