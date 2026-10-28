"""So sánh nhiều cấu hình pipeline trên ảnh thật có ground truth (bảng ablation cho báo cáo).

Chỉ nên CHỌN cấu hình trên tập dev; tập test chỉ chạy một lần với cấu hình đã chọn.
Chạy:  python tools/experiment_real.py data/real/hf_coins dev
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

# Kết quả giống hệt backend numpy (tools/verify_impl.py), nhanh hơn. Tắt chặn Watershed theo độ tròn
# ở cấu hình nền để các dòng A–K đúng là "trước khi cải tiến"; các dòng sau bật lại từng bước.
BASE = Config(backend="opencv", split_min_circularity=0.0)
SHAPE = {"seg_method": "bgsub", "min_circularity": 0.5, "max_area_frac": 0.05}
FULL = {**SHAPE, "clear_border": True, "split_min_circularity": 0.35, "min_area_rel_median": 0.25}
EXPERIMENTS = [
    ("A. Gốc: Otsu trên ảnh xám", {}),
    ("B. Gốc + san phẳng ánh sáng", {"flatten_light": True}),
    ("C. K-Means trên ảnh xám", {"seg_method": "kmeans"}),
    ("D. Trừ nền (xám) + Otsu", {"seg_method": "bgsub", "bg_color": False, "bg_threshold": "otsu"}),
    ("E. Trừ nền (xám) + Triangle", {"seg_method": "bgsub", "bg_color": False}),
    ("F. Trừ nền (Lab) + Otsu", {"seg_method": "bgsub", "bg_threshold": "otsu"}),
    ("G. Trừ nền (Lab) + Triangle", {"seg_method": "bgsub"}),
    ("H. Trừ nền (Lab) + MAD k=6", {"seg_method": "bgsub", "bg_threshold": "mad"}),
    ("I. G + lọc hình dạng (tròn>=0.5, A<=5%)",
     {"seg_method": "bgsub", "min_circularity": 0.5, "max_area_frac": 0.05}),
    ("J. G + lọc hình dạng (tròn>=0.6, A<=5%)",
     {"seg_method": "bgsub", "min_circularity": 0.6, "max_area_frac": 0.05}),
    ("K. I nhưng tắt Watershed", {**SHAPE, "split_touching": False}),
    ("L. I + bỏ vùng chạm viền", {**SHAPE, "clear_border": True}),
    ("M. L + chỉ Watershed vùng tròn>=0.35",
     {**SHAPE, "clear_border": True, "split_min_circularity": 0.35}),
    ("N. M + bỏ vụn < 0.25 x trung vị", FULL),
    ("O. N với tròn>=0.6", {**FULL, "min_circularity": 0.6}),
    ("P. N với tròn>=0.4", {**FULL, "min_circularity": 0.4}),
    ("Q. N với ngưỡng Otsu", {**FULL, "bg_threshold": "otsu"}),
    ("R. N với ảnh xám (không màu)", {**FULL, "bg_color": False}),
    ("S. N với cửa sổ nền 0.15", {**FULL, "bg_kernel_frac": 0.15}),
    ("T. N với cửa sổ nền 0.35", {**FULL, "bg_kernel_frac": 0.35}),
    ("U. N nhưng tắt Watershed", {**FULL, "split_touching": False}),
    ("V. N + cửa sổ 0.15 + tròn>=0.6 [CHỐT]", {**FULL, "bg_kernel_frac": 0.15, "min_circularity": 0.6}),
    ("X. V nhưng tắt Watershed",
     {**FULL, "bg_kernel_frac": 0.15, "min_circularity": 0.6, "split_touching": False}),
    ("W. N + cửa sổ 0.20 + tròn>=0.6", {**FULL, "bg_kernel_frac": 0.20, "min_circularity": 0.6}),
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
    print(f"Tập {subset}: {len(names)} ảnh\n")
    print(f"{'Cấu hình':<42}{'Đếm đúng':>9}{'±1':>7}{'MAE':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'ms':>7}")
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
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
