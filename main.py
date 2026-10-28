"""Run the pipeline on one image or on a whole folder of images.

Examples:
    python main.py data/synthetic                     # whole synthetic set (reads ground_truth.json automatically)
    python main.py data/raw/case1_ideal.jpg           # a single image
    python main.py data/raw --method kmeans --no-split
    python main.py data/raw/uneven_light.jpg --flatten               # image lit from one side
    python main.py data/real/hf_coins --save minimal --backend opencv   # hundreds of real photos
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from ipm import Config, run
from ipm.evaluate import evaluate_image, load_ground_truth
from ipm.preprocess import load_image
from ipm.visualize import print_feature_table, save_all

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DETAIL_LIMIT = 30  # Folders with more images than this only print the summary table
# Configuration for real photos (selected on the dev split with tools/experiment_real.py)
REAL_PRESET = {"seg_method": "bgsub", "bg_threshold": "triangle", "bg_color": True, "bg_kernel_frac": 0.15,
               "clear_border": True, "split_min_circularity": 0.35, "min_area_rel_median": 0.25,
               "min_circularity": 0.6, "max_area_frac": 0.05}


def parse_args():
    p = argparse.ArgumentParser(description="IMP302m – Object counting & size classification")
    p.add_argument("input", help="Image path or image folder")
    p.add_argument("--out", default="outputs", help="Output folder (default: outputs)")
    p.add_argument("--gt", help="ground_truth.json file (default: looked up in the image folder)")
    p.add_argument("--method", choices=["otsu", "kmeans", "bgsub"], default="otsu")
    p.add_argument("--classify", choices=["otsu", "mean", "gap"], default="otsu")
    p.add_argument("--area-threshold", type=float, help="Fixed area threshold (px)")
    p.add_argument("--polarity", choices=["auto", "bright", "dark"], default="auto",
                   help="bright: objects brighter than the background; dark: objects darker")
    p.add_argument("--backend", choices=["numpy", "opencv"], default="numpy",
                   help="numpy = own implementations, opencv = library functions")
    p.add_argument("--width", type=int, default=800, help="Normalized width (0 = keep original)")
    p.add_argument("--resize", choices=["area", "nearest"], default="area")
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    p.add_argument("--se", type=int, default=5, help="Structuring element size")
    p.add_argument("--no-split", action="store_true", help="Disable splitting of touching objects (watershed)")
    p.add_argument("--no-fill", action="store_true", help="Disable hole filling")
    p.add_argument("--flatten", action="store_true", help="Enable uneven-illumination flattening")
    p.add_argument("--save", choices=["all", "minimal", "none"], default="all",
                   help="all: every report figure; minimal: result image + table only; none: summary.csv only")
    p.add_argument("--limit", type=int, help="Only process the first N images")
    p.add_argument("--subset", choices=["dev", "test"], help="Only process this subset of split.json in the image folder")
    p.add_argument("--bg-threshold", choices=["triangle", "otsu", "mad"], default="triangle",
                   help="Threshold rule for --method bgsub")
    p.add_argument("--bg-gray", action="store_true", help="bgsub: use luminance only instead of Lab color")
    p.add_argument("--min-circularity", type=float, default=0.0, help="Shape filter (0 = off)")
    p.add_argument("--max-area-frac", type=float, default=1.0, help="Drop regions larger than this fraction of the image")
    p.add_argument("--preset", choices=["real"],
                   help="real = configuration for real photos with non-uniform backgrounds (bgsub + shape filter)")
    return p.parse_args()


def build_config(a):
    cfg = Config(work_width=a.width, resize_method=a.resize, seg_method=a.method,
                 classify_method=a.classify, area_threshold=a.area_threshold,
                 polarity=a.polarity, backend=a.backend, connectivity=a.connectivity,
                 se_size=a.se, split_touching=not a.no_split, fill_holes=not a.no_fill,
                 flatten_light=a.flatten, bg_threshold=a.bg_threshold, bg_color=not a.bg_gray,
                 min_circularity=a.min_circularity, max_area_frac=a.max_area_frac)
    if a.preset == "real":
        for k, v in REAL_PRESET.items():
            setattr(cfg, k, v)
    return cfg


def print_rows(rows):
    print("\n" + "=" * 124)
    print(f"{'Image':<26}{'AOD before':>11}{'AOD after':>10}{'T*':>5}{'Raw reg.':>10}{'Morph':>8}"
          f"{'Split':>6}{'Count':>6}{'Large':>6}{'Small':>6}{'GT':>5}{'IoU':>8}{'ms':>9}  Result")
    print("=" * 124)
    for r in rows:
        verdict = ""
        if "count_ok" in r:
            verdict = "OK" if r["count_ok"] and r.get("class_ok", True) else "WRONG"
        iou = f"{r['mask_iou']:.3f}" if "mask_iou" in r else "-"
        print(f"{r['image']:<26}{r['aod_before']:>11}{r['aod_after']:>10}{r['T']:>5}{r['regions_raw']:>10}"
              f"{r['regions_morph']:>8}{r['split']:>6}{r['count']:>6}{r['large']:>6}{r['small']:>6}"
              f"{str(r.get('gt_count', '-')):>5}{iou:>8}{r['time_ms']:>9}  {verdict}")


def aggregate(rows):
    """Set-level metrics: counting accuracy, MAE, object-level Precision/Recall/F1."""
    ev = [r for r in rows if r.get("gt_count") is not None]
    if not ev:
        return None
    err = np.array([r["count_error"] for r in ev])
    agg = {
        "n_images": len(ev),
        "count_exact_acc": float(np.mean(err == 0)),
        "count_within1_acc": float(np.mean(np.abs(err) <= 1)),
        "count_mae": float(np.mean(np.abs(err))),
        "over_count_images": int(np.sum(err > 0)),
        "under_count_images": int(np.sum(err < 0)),
        "mean_time_ms": float(np.mean([r["time_ms"] for r in rows])),
    }
    if all("tp" in r for r in ev):
        tp, fp, fn = (sum(r[k] for r in ev) for k in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        agg.update(tp=tp, fp=fp, fn=fn, precision=p, recall=rc,
                   f1=2 * p * rc / (p + rc) if p + rc else 0.0)
    return agg


def print_aggregate(agg):
    print("\n" + "=" * 60)
    print(f"SUMMARY OVER {agg['n_images']} IMAGES WITH GROUND TRUTH")
    print("=" * 60)
    print(f"Exact count accuracy       : {agg['count_exact_acc'] * 100:6.1f} %")
    print(f"Count within ±1            : {agg['count_within1_acc'] * 100:6.1f} %")
    print(f"Mean abs. count error (MAE): {agg['count_mae']:6.2f} objects/image")
    print(f"Over- / under-counted imgs : {agg['over_count_images']} / {agg['under_count_images']}")
    if "precision" in agg:
        print(f"TP / FP / FN               : {agg['tp']} / {agg['fp']} / {agg['fn']}")
        print(f"Precision / Recall / F1    : {agg['precision']:.3f} / {agg['recall']:.3f} / {agg['f1']:.3f}")
    print(f"Mean time                  : {agg['mean_time_ms']:.1f} ms/image")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args()
    cfg = build_config(a)
    src = Path(a.input)
    if src.is_dir():
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        gt_path = Path(a.gt) if a.gt else src / "ground_truth.json"
    else:
        images = [src]
        gt_path = Path(a.gt) if a.gt else src.parent / "ground_truth.json"
    if a.subset:
        with open(src / "split.json", encoding="utf-8") as f:
            wanted = set(json.load(f)[a.subset])
        images = [p for p in images if p.name in wanted]
    if a.limit:
        images = images[:a.limit]
    if not images:
        sys.exit(f"No images found in {src}")
    gt = load_ground_truth(gt_path)
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    detail = len(images) <= DETAIL_LIMIT
    rows = []
    t_start = time.perf_counter()

    for k, path in enumerate(images, start=1):
        res = run(load_image(path), cfg)
        if a.save != "none":
            save_all(res, out_root / path.stem, a.save)
        if detail:
            print(f"\n### {path.name}")
            print_feature_table(res.objects, res.class_info)
            if res.class_info.get("warning"):
                print("WARNING:", res.class_info["warning"])
        elif k % 50 == 0 or k == len(images):
            print(f"  processed {k}/{len(images)} images ({time.perf_counter() - t_start:.0f} s)")
        st, ci = res.stats, res.class_info
        row = {
            "image": path.name,
            "aod_before": round(res.aod_before, 1),
            "aod_after": round(res.aod_after, 1),
            "T": res.seg_info["T"],
            "bg_bright": res.seg_info["background_bright"],
            "regions_raw": st["regions_raw"],
            "regions_morph": st["regions_after_morph"],
            "split": sum(s["parts"] - 1 for s in st["split"]),
            "count": st["n_obj"],
            "large": ci.get("n_large", 0),
            "small": ci.get("n_small", 0),
            "time_ms": res.timings["total"],
        }
        if path.name in gt:
            row.update(evaluate_image(res, gt[path.name], gt_path.parent))
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    if detail:
        print_rows(rows)
    agg = aggregate(rows)
    if agg and (not detail or "precision" in agg):
        print_aggregate(agg)
    print(f"\nResults saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
