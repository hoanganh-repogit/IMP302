"""Chạy pipeline trên 1 ảnh hoặc cả thư mục ảnh.

Ví dụ:
    python main.py data/synthetic                     # chạy cả bộ ảnh giả lập (tự đọc ground_truth.json)
    python main.py data/raw/case1_ideal.jpg           # chạy 1 ảnh
    python main.py data/raw --method kmeans --no-split
    python main.py data/raw/anh_lech_sang.jpg --flatten            # ảnh chiếu sáng lệch một phía
    python main.py data/real/hf_coins --save minimal --backend opencv   # chạy hàng trăm ảnh thật
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
DETAIL_LIMIT = 30  # Thư mục nhiều ảnh hơn thế thì chỉ in bảng tổng hợp
# Cấu hình cho ảnh thật (chọn trên tập dev bằng tools/experiment_real.py, xem PLAN.md)
REAL_PRESET = {"seg_method": "bgsub", "bg_threshold": "triangle", "bg_color": True, "bg_kernel_frac": 0.15,
               "clear_border": True, "split_min_circularity": 0.35, "min_area_rel_median": 0.25,
               "min_circularity": 0.6, "max_area_frac": 0.05}


def parse_args():
    p = argparse.ArgumentParser(description="IMP302m – Đếm & phân loại kích thước vật thể")
    p.add_argument("input", help="Đường dẫn ảnh hoặc thư mục ảnh")
    p.add_argument("--out", default="outputs", help="Thư mục kết quả (mặc định: outputs)")
    p.add_argument("--gt", help="File ground_truth.json (mặc định: tìm trong thư mục ảnh)")
    p.add_argument("--method", choices=["otsu", "kmeans", "bgsub"], default="otsu")
    p.add_argument("--classify", choices=["otsu", "mean", "gap"], default="otsu")
    p.add_argument("--area-threshold", type=float, help="Ngưỡng diện tích cố định (px)")
    p.add_argument("--polarity", choices=["auto", "bright", "dark"], default="auto",
                   help="bright: vật sáng hơn nền; dark: vật tối hơn nền")
    p.add_argument("--backend", choices=["numpy", "opencv"], default="numpy",
                   help="numpy = thuật toán tự cài đặt, opencv = hàm thư viện")
    p.add_argument("--width", type=int, default=800, help="Chiều rộng chuẩn hóa (0 = giữ nguyên)")
    p.add_argument("--resize", choices=["area", "nearest"], default="area")
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    p.add_argument("--se", type=int, default=5, help="Kích thước phần tử cấu trúc")
    p.add_argument("--no-split", action="store_true", help="Tắt tách vật dính nhau (Watershed)")
    p.add_argument("--no-fill", action="store_true", help="Tắt lấp lỗ")
    p.add_argument("--flatten", action="store_true", help="Bật san phẳng ánh sáng không đều")
    p.add_argument("--save", choices=["all", "minimal", "none"], default="all",
                   help="all: đủ hình báo cáo; minimal: chỉ ảnh kết quả + bảng; none: chỉ summary.csv")
    p.add_argument("--limit", type=int, help="Chỉ chạy N ảnh đầu tiên")
    p.add_argument("--subset", choices=["dev", "test"], help="Chỉ chạy tập con trong split.json của thư mục ảnh")
    p.add_argument("--bg-threshold", choices=["triangle", "otsu", "mad"], default="triangle",
                   help="Quy tắc ngưỡng cho --method bgsub")
    p.add_argument("--bg-gray", action="store_true", help="bgsub: chỉ dùng độ sáng thay vì màu Lab")
    p.add_argument("--min-circularity", type=float, default=0.0, help="Lọc hình dạng (0 = tắt)")
    p.add_argument("--max-area-frac", type=float, default=1.0, help="Bỏ vùng lớn hơn tỉ lệ này của ảnh")
    p.add_argument("--preset", choices=["real"],
                   help="real = cấu hình cho ảnh thật nền không đều (bgsub + lọc hình dạng)")
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
    print("\n" + "=" * 118)
    print(f"{'Ảnh':<26}{'AOD trước':>10}{'AOD sau':>9}{'T*':>5}{'Vùng thô':>10}{'Sau morph':>10}"
          f"{'Tách':>6}{'Đếm':>6}{'Lớn':>5}{'Nhỏ':>5}{'GT':>5}{'IoU':>8}{'ms':>9}  Kết quả")
    print("=" * 118)
    for r in rows:
        verdict = ""
        if "count_ok" in r:
            verdict = "ĐÚNG" if r["count_ok"] and r.get("class_ok", True) else "SAI"
        iou = f"{r['mask_iou']:.3f}" if "mask_iou" in r else "-"
        print(f"{r['image']:<26}{r['aod_before']:>10}{r['aod_after']:>9}{r['T']:>5}{r['regions_raw']:>10}"
              f"{r['regions_morph']:>10}{r['split']:>6}{r['count']:>6}{r['large']:>5}{r['small']:>5}"
              f"{str(r.get('gt_count', '-')):>5}{iou:>8}{r['time_ms']:>9}  {verdict}")


def aggregate(rows):
    """Chỉ số trên toàn tập: độ chính xác đếm, MAE, Precision/Recall/F1 cấp vật thể."""
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
    print(f"TỔNG HỢP TRÊN {agg['n_images']} ẢNH CÓ GROUND TRUTH")
    print("=" * 60)
    print(f"Đếm đúng tuyệt đối         : {agg['count_exact_acc'] * 100:6.1f} %")
    print(f"Đếm sai lệch không quá ±1  : {agg['count_within1_acc'] * 100:6.1f} %")
    print(f"Sai số đếm trung bình (MAE): {agg['count_mae']:6.2f} vật/ảnh")
    print(f"Ảnh đếm thừa / đếm thiếu   : {agg['over_count_images']} / {agg['under_count_images']}")
    if "precision" in agg:
        print(f"TP / FP / FN               : {agg['tp']} / {agg['fp']} / {agg['fn']}")
        print(f"Precision / Recall / F1    : {agg['precision']:.3f} / {agg['recall']:.3f} / {agg['f1']:.3f}")
    print(f"Thời gian trung bình       : {agg['mean_time_ms']:.1f} ms/ảnh")


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
        sys.exit(f"Không tìm thấy ảnh nào trong {src}")
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
                print("CẢNH BÁO:", res.class_info["warning"])
        elif k % 50 == 0 or k == len(images):
            print(f"  đã xử lý {k}/{len(images)} ảnh ({time.perf_counter() - t_start:.0f} s)")
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
    print(f"\nĐã lưu kết quả vào: {out_root.resolve()}")


if __name__ == "__main__":
    main()
