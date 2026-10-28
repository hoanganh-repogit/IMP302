"""Compare many pipeline configurations on real photos with ground truth (ablation table for the report).

Configurations must be SELECTED on the dev split only; the test split is run once with the chosen configuration.
Run:  python tools/experiment_real.py data/real/hf_coins dev
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ipm import Config, run  # noqa: E402
from ipm.evaluate import evaluate_image  # noqa: E402
from ipm.preprocess import load_image  # noqa: E402
from main import aggregate  # noqa: E402

# Same results as the numpy backend (tools/verify_impl.py), but faster. Circularity-gated watershed is
# disabled in the base configuration so rows A–K really are "before the improvements"; later rows re-enable it step by step.
BASE = Config(backend="opencv", split_min_circularity=0.0)
SHAPE = {"seg_method": "bgsub", "min_circularity": 0.5, "max_area_frac": 0.05}
FULL = {**SHAPE, "clear_border": True, "split_min_circularity": 0.35, "min_area_rel_median": 0.25}
EXPERIMENTS = [
    ("A. Baseline: Otsu on gray image", {}),
    ("B. Baseline + illumination flattening", {"flatten_light": True}),
    ("C. K-Means on gray image", {"seg_method": "kmeans"}),
    ("D. Background diff. (gray) + Otsu", {"seg_method": "bgsub", "bg_color": False, "bg_threshold": "otsu"}),
    ("E. Background diff. (gray) + Triangle", {"seg_method": "bgsub", "bg_color": False}),
    ("F. Background diff. (Lab) + Otsu", {"seg_method": "bgsub", "bg_threshold": "otsu"}),
    ("G. Background diff. (Lab) + Triangle", {"seg_method": "bgsub"}),
    ("H. Background diff. (Lab) + MAD k=6", {"seg_method": "bgsub", "bg_threshold": "mad"}),
    ("I. G + shape filter (circ>=0.5, A<=5%)",
     {"seg_method": "bgsub", "min_circularity": 0.5, "max_area_frac": 0.05}),
    ("J. G + shape filter (circ>=0.6, A<=5%)",
     {"seg_method": "bgsub", "min_circularity": 0.6, "max_area_frac": 0.05}),
    ("K. I without watershed", {**SHAPE, "split_touching": False}),
    ("L. I + border clearing", {**SHAPE, "clear_border": True}),
    ("M. L + watershed only if circ>=0.35",
     {**SHAPE, "clear_border": True, "split_min_circularity": 0.35}),
    ("N. M + drop crumbs < 0.25 x median", FULL),
    ("O. N with circ>=0.6", {**FULL, "min_circularity": 0.6}),
    ("P. N with circ>=0.4", {**FULL, "min_circularity": 0.4}),
    ("Q. N with Otsu threshold", {**FULL, "bg_threshold": "otsu"}),
    ("R. N with gray image (no color)", {**FULL, "bg_color": False}),
    ("S. N with background window 0.15", {**FULL, "bg_kernel_frac": 0.15}),
    ("T. N with background window 0.35", {**FULL, "bg_kernel_frac": 0.35}),
    ("U. N without watershed", {**FULL, "split_touching": False}),
    ("V. N + window 0.15 + circ>=0.6 [FINAL]", {**FULL, "bg_kernel_frac": 0.15, "min_circularity": 0.6}),
    ("X. V without watershed",
     {**FULL, "bg_kernel_frac": 0.15, "min_circularity": 0.6, "split_touching": False}),
    ("W. N + window 0.20 + circ>=0.6", {**FULL, "bg_kernel_frac": 0.20, "min_circularity": 0.6}),
]


def run_one(args):
    path, gt_entry, overrides = args
    res = run(load_image(path), replace(BASE, **overrides))
    row = {"time_ms": res.timings["total"]}
    row.update(evaluate_image(res, gt_entry))
    return row


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/real/hf_coins")
    subset = sys.argv[2] if len(sys.argv) > 2 else "dev"
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    gt = json.load(open(root / "ground_truth.json", encoding="utf-8"))
    names = json.load(open(root / "split.json", encoding="utf-8"))[subset]
    print(f"{subset} split: {len(names)} images\n")
    print(f"{'Configuration':<42}{'Exact':>9}{'±1':>7}{'MAE':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'ms':>7}")
    print("-" * 93)
    results = {}
    with ProcessPoolExecutor() as ex:
        for name, ov in EXPERIMENTS:
            if only and name.split(".")[0] not in only:
                continue
            rows = list(ex.map(run_one, [(root / n, gt[n], ov) for n in names], chunksize=8))
            a = aggregate(rows)
            results[name] = {"overrides": ov, **a}
            print(f"{name:<42}{a['count_exact_acc'] * 100:>8.1f}%{a['count_within1_acc'] * 100:>6.1f}%"
                  f"{a['count_mae']:>7.2f}{a['precision']:>7.3f}{a['recall']:>7.3f}{a['f1']:>7.3f}"
                  f"{a['mean_time_ms']:>7.0f}", flush=True)
    out = Path("outputs") / f"experiment_{root.name}_{subset}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
