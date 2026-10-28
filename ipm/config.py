"""System-wide configuration. Every "magic number" of the pipeline lives here."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---------- Stage 1: Preprocessing (Lectures 1, 2, 3, 7, 8) ----------
    work_width: int = 800            # Normalized image width (px); smaller images are kept as is
    resize_method: str = "area"      # "area" (cv2.INTER_AREA, anti-aliased) | "nearest" (own implementation, Lecture 1.1)
    fshs_low_pct: float = 0.5        # Percentile-based FSHS: A = P0.5, B = P99.5 (robust to outliers)
    fshs_high_pct: float = 99.5
    flatten_light: bool = False      # Flatten uneven illumination (future work 7.2)
    flatten_se_frac: float = 0.4     # Background-estimation SE size / short image side (must exceed the largest object)
    median_ksize: int = 5            # Median filter against impulse noise (Lectures 2.2, 3.3)
    gaussian_ksize: int = 5          # Gaussian filter against sensor noise (Lectures 2.1, 3.3)
    gaussian_sigma: float = 1.0

    # ---------- Stage 2: Segmentation & morphology (Lectures 1.2, 2.3, 9.1) ----------
    seg_method: str = "otsu"         # "otsu" | "kmeans" | "bgsub" (background subtraction – real photos, non-uniform background)
    bg_kernel_frac: float = 0.25     # bgsub: median window for the background / short side (must exceed 1.25x object diameter)
    bg_color: bool = True            # bgsub: color difference in Lab space (False = luminance only)
    bg_threshold: str = "triangle"   # bgsub: "triangle" | "otsu" | "mad"
    bg_mad_k: float = 6.0            # bgsub + mad: T = median + k·1.4826·MAD
    polarity: str = "auto"           # "auto" | "bright" (objects brighter than background) | "dark" (objects darker)
    se_size: int = 5                 # 5x5 elliptical structuring element
    open_iter: int = 1
    close_iter: int = 1
    fill_holes: bool = True          # Fill glare holes larger than the SE that closing cannot seal

    # ---------- Stage 3: Labeling & features (Lectures 1.2, 9, 10, 11.1) ----------
    connectivity: int = 4            # 4- or 8-neighborhood
    min_area_px: int = 100           # Absolute A_min
    min_area_frac: float = 0.0003    # A_min relative to the image area -> the larger of the two is used
    split_touching: bool = True      # Split touching objects with distance transform + watershed
    solidity_thresh: float = 0.94    # Regions with solidity (A / A_hull) below this are suspected to be merged
    split_min_circularity: float = 0.35  # Only watershed "coin-cluster-like" regions (2 touching coins ~0.5; table-edge strips < 0.3)
    clear_border: bool = False       # Drop regions touching the image border (objects must lie fully inside – table edges, dark corners)
    min_area_rel_median: float = 0.0 # Drop regions with A < this fraction x the median object area (noise crumbs). 0 = off
    min_circularity: float = 0.0     # Shape filter: drop regions with lower circularity (table edges, shadow streaks). 0 = off
    max_area_frac: float = 1.0       # Drop regions larger than this fraction of the image (mis-split background patches). 1 = off

    # ---------- Stage 4: Classification (Lecture 12) ----------
    classify_method: str = "otsu"    # "otsu" (1-D Otsu on the set of areas) | "mean" (original spec) | "gap"
    min_size_ratio: float = 1.3      # A_max / A_min below this -> treated as a single size group
    area_threshold: float | None = None  # Fixed area threshold (px) when the camera is calibrated

    # ---------- Implementation ----------
    backend: str = "numpy"           # "numpy" = own implementations | "opencv" = library functions

    def to_dict(self):
        return asdict(self)
