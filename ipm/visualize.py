"""Figures & tables for the report."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import segmentation as seg  # noqa: E402
from .classify import LARGE, SMALL  # noqa: E402
from .preprocess import histogram, save_image  # noqa: E402

CLASS_COLORS = {LARGE: (40, 180, 40), SMALL: (0, 140, 255)}  # BGR: green / orange
DEFAULT_COLOR = (160, 160, 160)


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def colorize_labels(labels):
    """Pseudo-color label map: one color per region, black background."""
    n = int(labels.max())
    rng = np.random.default_rng(7)
    lut = rng.integers(60, 256, size=(n + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]


def draw_detections(img, objects, class_info):
    out = img.copy()
    W = img.shape[1]
    fs = max(0.45, W / 1500)
    th = max(1, round(W / 450))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for o in objects:
        color = CLASS_COLORS.get(o.cls, DEFAULT_COLOR)
        cv2.rectangle(out, (o.x, o.y), (o.x + o.w - 1, o.y + o.h - 1), color, th)
        cv2.circle(out, (round(o.cx), round(o.cy)), th + 2, (0, 0, 255), -1)
        text = f"#{o.id} {o.cls}"
        (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
        ty = o.y - 6 if o.y - tht - 8 > 0 else o.y + o.h + tht + 6
        cv2.rectangle(out, (o.x, ty - tht - 4), (o.x + tw + 6, ty + 4), color, -1)
        cv2.putText(out, text, (o.x + 3, ty), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    header = f"Count: {len(objects)}  |  Large: {class_info.get('n_large', 0)}  |  Small: {class_info.get('n_small', 0)}"
    (tw, tht), _ = cv2.getTextSize(header, font, fs * 1.1, th)
    cv2.rectangle(out, (0, 0), (tw + 16, tht + 18), (30, 30, 30), -1)
    cv2.putText(out, header, (8, tht + 9), font, fs * 1.1, (255, 255, 255), th, cv2.LINE_AA)
    return out


def _hist_axis(ax, img, title, lines=(), log=False):
    H = histogram(img)
    ax.bar(np.arange(256), H, width=1.0, color="#4a6fa5")
    if log:
        ax.set_yscale("log")
    for x, label, color in lines:
        ax.axvline(x, color=color, linestyle="--", linewidth=1.2, label=label)
    ax.set_xlim(0, 255)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Gray level k")
    ax.set_ylabel("H(k)")
    if lines:
        ax.legend(fontsize=8)


def fig_histogram(res, path):
    """Output 1: original / FSHS gray image + the corresponding histograms with AOD."""
    A, B = res.fshs_range
    T = res.seg_info["T"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs[0, 0].imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    axs[0, 0].set_title(f"Original gray image f(n)  –  AOD = {res.aod_before:.1f}")
    _hist_axis(axs[0, 1], res.gray, "Histogram of the original image",
               [(A, f"A = {A:.0f}", "red"), (B, f"B = {B:.0f}", "red"),
                (res.aod_before, f"AOD = {res.aod_before:.1f}", "green")])
    if res.seg_info["method"] == "bgsub":
        axs[1, 0].imshow(res.smooth, cmap="magma")
        axs[1, 0].set_title("Background-difference image D(n) = ||I(n) - B(n)||")
        _hist_axis(axs[1, 1], res.smooth, "Histogram of D(n) (log y-axis)",
                   [(T, f"T ({res.seg_info['threshold_rule']}) = {T}", "purple")], log=True)
    else:
        axs[1, 0].imshow(res.contrast, cmap="gray", vmin=0, vmax=255)
        axs[1, 0].set_title(f"After FSHS g(n)  –  AOD = {res.aod_after:.1f}")
        _hist_axis(axs[1, 1], res.smooth, "Histogram after FSHS + denoising",
                   [(res.aod_after, f"AOD = {res.aod_after:.1f}", "green"),
                    (T, f"T* ({res.seg_info['method']}) = {T}", "purple")])
    for ax in (axs[0, 0], axs[1, 0]):
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_fshs_vs_threshold(res, path, fixed_T=128):
    """Shows that FSHS helps a FIXED threshold, while Otsu is invariant to linear transforms."""
    bg = res.seg_info["background_bright"]
    T_raw, _ = seg.otsu_threshold(res.gray)
    T_fs, _ = seg.otsu_threshold(res.contrast)
    m = [seg.threshold_mask(res.gray, fixed_T, bg), seg.threshold_mask(res.contrast, fixed_T, bg),
         seg.threshold_mask(res.gray, T_raw, bg), seg.threshold_mask(res.contrast, T_fs, bg)]
    titles = [f"Fixed threshold T={fixed_T} – original", f"Fixed threshold T={fixed_T} – after FSHS",
              f"Otsu – original (T*={T_raw})", f"Otsu – after FSHS (T*={T_fs})"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, mk, tt in zip(axs, m, titles):
        ax.imshow(mk, cmap="gray", vmin=0, vmax=1)
        ax.set_title(tt, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"IoU(Otsu original, Otsu FSHS) = {iou(m[2], m[3]):.3f}  –  "
                 f"IoU(fixed original, fixed FSHS) = {iou(m[0], m[1]):.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_otsu_vs_kmeans(res, path):
    """Compares the 2 segmentation branches on the same g_smooth image."""
    m_o, i_o = seg.segment(res.smooth, res.cfg, "otsu")
    m_k, i_k = seg.segment(res.smooth, res.cfg, "kmeans")
    diff = np.zeros((*m_o.shape, 3), np.uint8)
    diff[(m_o > 0) & (m_k == 0)] = (255, 80, 80)
    diff[(m_o == 0) & (m_k > 0)] = (80, 160, 255)
    diff[(m_o > 0) & (m_k > 0)] = (220, 220, 220)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(m_o, cmap="gray", vmin=0, vmax=1); axs[0].set_title(f"Otsu  T* = {i_o['T']}")
    axs[1].imshow(m_k, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title(f"K-Means k=2  centres = {i_k['centers']}, T = {i_k['T']}, {i_k['iterations']} iterations")
    axs[2].imshow(diff)
    axs[2].set_title(f"Difference (red: Otsu only, blue: K-Means only)  IoU = {iou(m_o, m_k):.4f}")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"otsu_T": i_o["T"], "kmeans_T": i_k["T"], "kmeans_centers": i_k["centers"],
            "kmeans_iterations": i_k["iterations"], "iou": round(iou(m_o, m_k), 4)}


def fig_morphology(res, path):
    """Mask post-processing step by step – illustration for Case 3."""
    st = res.stats
    imgs = [res.mask_raw, res.mask_open, res.mask_closed, res.mask_clean]
    titles = [f"After thresholding ({st['regions_raw']} regions)", "After opening (5x5 elliptical SE)",
              "After closing", f"After hole filling ({st['regions_after_morph']} regions)"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, im, tt in zip(axs, imgs, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(tt, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_overview(res, detect_bgr, path):
    st, ci = res.stats, res.class_info
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    panels = [
        (_rgb(res.image), None, f"1. Input (AOD = {res.aod_before:.1f})"),
        (res.smooth, "gray", "2. Background difference D(n) + denoising" if res.seg_info["method"] == "bgsub"
         else f"2. FSHS + denoising (AOD = {res.aod_after:.1f})"),
        (res.mask_raw, "gray", f"3. {res.seg_info['method'].upper()} (T* = {res.seg_info['T']}) – {st['regions_raw']} regions"),
        (res.mask_clean, "gray", f"4. After morphology – {st['regions_after_morph']} regions"),
        (_rgb(colorize_labels(res.labels)), None, f"5. Region labeling – {st['n_obj']} objects"),
        (_rgb(detect_bgr), None, f"6. Result: {ci.get('n_large', 0)} Large, {ci.get('n_small', 0)} Small"),
    ]
    for ax, (im, cmap, tt) in zip(axs.ravel(), panels):
        if cmap:  # gray images & masks: fix the scale so a single-valued image is not displayed wrongly
            ax.imshow(im, cmap=cmap, vmin=0, vmax=im.max() if im.max() > 1 else 1)
        else:
            ax.imshow(im)
        ax.set_title(tt, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


FEATURE_COLUMNS = ["id", "cx", "cy", "x", "y", "w", "h", "area", "perimeter", "diameter",
                   "orientation", "circularity", "solidity", "eq_diameter", "cls"]


def write_features_csv(objects, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(FEATURE_COLUMNS)
        for o in objects:
            d = o.to_dict()
            wr.writerow([round(d[c], 3) if isinstance(d[c], float) else d[c] for c in FEATURE_COLUMNS])


def print_feature_table(objects, class_info):
    line = "=" * 104
    print(line)
    print(f"{'ID':<4}| {'Center (cx,cy)':<14}| {'Box (x, y, w, h)':<22}| {'A (px)':>8} | {'L (px)':>8} | "
          f"{'D (px)':>7} | {'Circ.':>7} | {'Solid.':>6} | Class")
    print(line)
    for o in objects:
        print(f"#{o.id:<3}| ({o.cx:>5.0f},{o.cy:>5.0f}) | ({o.x:>4},{o.y:>4},{o.w:>4},{o.h:>4})  | {o.area:>8} | "
              f"{o.perimeter:>8.1f} | {o.diameter:>7.1f} | {o.circularity:>7.3f} | {o.solidity:>6.3f} | {o.cls}")
    print(line)
    T = class_info.get("threshold")
    print(f"Area threshold T_area = {T} px ({class_info['method']})" if T is not None
          else f"Note: {class_info.get('warning')}")


def save_all(res, out_dir, level="all"):
    """Write all outputs of one image to out_dir.

    level = "all": every figure for the report; "minimal": only the result image + table + summary
    (used for hundreds of images, since matplotlib plotting dominates the run time).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detect = draw_detections(res.image, res.objects, res.class_info)
    seg_cmp = None
    if level == "all":
        fig_overview(res, detect, out_dir / "00_pipeline_overview.png")
        fig_histogram(res, out_dir / "01_histogram_fshs.png")
        fig_fshs_vs_threshold(res, out_dir / "01b_fshs_vs_threshold.png")
        save_image(out_dir / "02_mask_raw.png", res.mask_raw * 255)
        save_image(out_dir / "02_mask_clean.png", res.mask_clean * 255)
        fig_morphology(res, out_dir / "02b_morphology_steps.png")
        seg_cmp = fig_otsu_vs_kmeans(res, out_dir / "02c_otsu_vs_kmeans.png")
        save_image(out_dir / "03_label_map.png", colorize_labels(res.labels))
    save_image(out_dir / "04_detection.png", detect)
    write_features_csv(res.objects, out_dir / "05_features.csv")
    summary = {
        "config": res.cfg.to_dict(),
        "scale": res.scale,
        "image_size": list(res.image.shape[:2]),
        "fshs_A_B": list(res.fshs_range),
        "aod_before": round(res.aod_before, 2),
        "aod_after": round(res.aod_after, 2),
        "segmentation": res.seg_info,
        "otsu_vs_kmeans": seg_cmp,
        "stats": res.stats,
        "classification": res.class_info,
        "timings_ms": res.timings,
        "objects": [o.to_dict() for o in res.objects],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    return summary
