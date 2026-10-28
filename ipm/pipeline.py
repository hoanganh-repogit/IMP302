"""Runs the whole 4-stage pipeline and times each stage."""
import time
from dataclasses import dataclass, field

import numpy as np

from . import classify as clf
from . import features as feat
from . import labeling as lab
from . import morphology as morph
from . import preprocess as pre
from . import segmentation as seg
from .config import Config
from .split import split_touching


@dataclass
class PipelineResult:
    cfg: Config
    image: np.ndarray          # BGR color image after size normalization
    scale: float
    gray: np.ndarray           # f(n1, n2)
    contrast: np.ndarray       # After FSHS (and illumination flattening if enabled)
    fshs_range: tuple          # (A, B)
    aod_before: float
    aod_after: float
    smooth: np.ndarray         # g_smooth(n1, n2)
    seg_info: dict
    mask_raw: np.ndarray       # b(n) right after thresholding
    mask_open: np.ndarray
    mask_closed: np.ndarray
    mask_clean: np.ndarray     # b_clean(n)
    labels: np.ndarray         # r(n1, n2)
    objects: list
    class_info: dict
    stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def run(bgr, cfg: Config) -> PipelineResult:
    t = {}
    t0 = tic = time.perf_counter()

    def lap(name):
        nonlocal tic
        now = time.perf_counter()
        t[name] = round((now - tic) * 1000, 2)
        tic = now

    # ===== STAGE 1: Gray-level preprocessing =====
    img, scale = pre.resize_to_width(bgr, cfg.work_width, cfg.resize_method)
    gray = pre.rgb_to_gray(img)
    aod_before = pre.aod(gray)
    contrast, A, B = pre.fshs(gray, cfg.fshs_low_pct, cfg.fshs_high_pct)
    if cfg.flatten_light:
        T0, _ = seg.otsu_threshold(contrast)
        bg_bright = (seg.background_is_bright(contrast, T0) if cfg.polarity == "auto"
                     else cfg.polarity == "dark")
        contrast = pre.flatten_illumination(contrast, bg_bright, cfg.flatten_se_frac)
        contrast, _, _ = pre.fshs(contrast, cfg.fshs_low_pct, cfg.fshs_high_pct)
    aod_after = pre.aod(contrast)
    if cfg.seg_method == "bgsub":
        # Segment the background-difference image D(n) instead of the gray image
        smooth = pre.denoise(pre.background_difference(img, cfg.bg_kernel_frac, cfg.bg_color), cfg)
    else:
        smooth = pre.denoise(contrast, cfg)
    lap("1_preprocess")

    # ===== STAGE 2: Segmentation + morphology =====
    mask_raw, seg_info = seg.segment(smooth, cfg)
    se = morph.structuring_element(cfg.se_size)
    mask_open = morph.opening(mask_raw, se, cfg.open_iter, cfg.backend)
    mask_closed = morph.closing(mask_open, se, cfg.close_iter, cfg.backend)
    mask_clean = morph.fill_holes(mask_closed) if cfg.fill_holes else mask_closed
    lap("2_segmentation")

    # ===== STAGE 3: Labeling + splitting touching objects + features =====
    labels, n = lab.label(mask_clean, cfg.connectivity, cfg.backend)
    n_components = n
    a_min = max(cfg.min_area_px, int(cfg.min_area_frac * mask_clean.size))
    labels, n = lab.remove_small(labels, a_min)
    if cfg.clear_border:
        labels, n = lab.clear_border(labels)
    n_before_split = n
    split_log = []
    if cfg.split_touching and n > 0:
        labels, n, split_log = split_touching(labels, n, cfg.solidity_thresh, cfg.split_min_circularity)
        labels, n = lab.remove_small(labels, a_min)
    objects = feat.extract(labels, n)
    n_rejected = 0
    if cfg.min_circularity > 0 or cfg.max_area_frac < 1 or cfg.min_area_rel_median > 0:
        # Shape filter (Lecture 10): drop elongated regions (table edges, shadow streaks), oversized
        # background patches and crumbs that are abnormally small for the typical object size in the image
        a_max = cfg.max_area_frac * labels.size
        keep = [o for o in objects if o.circularity >= cfg.min_circularity and o.area <= a_max]
        if keep and cfg.min_area_rel_median > 0:
            a_med = float(np.median([o.area for o in keep]))
            keep = [o for o in keep if o.area >= cfg.min_area_rel_median * a_med]
        n_rejected = len(objects) - len(keep)
        if n_rejected:
            lut = np.zeros(labels.max() + 1, dtype=np.int32)
            for o in keep:
                lut[o.label] = o.label
            labels = lut[labels]
            for k, o in enumerate(keep, start=1):
                o.id = k
            objects = keep
    lap("3_label_features")

    # ===== STAGE 4: Classification =====
    class_info = clf.classify(objects, cfg)
    lap("4_classify")
    t["total"] = round((time.perf_counter() - t0) * 1000, 2)

    stats = {
        "regions_raw": lab.count_components(mask_raw, cfg.connectivity),
        "regions_after_morph": n_components,
        "a_min": a_min,
        "regions_after_area_filter": n_before_split,
        "split": split_log,
        "rejected_by_shape": n_rejected,
        "n_obj": len(objects),
    }
    return PipelineResult(cfg, img, scale, gray, contrast, (A, B), aod_before, aod_after, smooth,
                          seg_info, mask_raw, mask_open, mask_closed, mask_clean, labels,
                          objects, class_info, stats, t)
