"""Sinh số liệu, biểu đồ và hình minh họa (chữ tiếng Anh) cho báo cáo LaTeX (report/).

Yêu cầu đã chạy trước: tools/generate_synthetic.py, tools/import_hf_coins.py, tools/real_cases.py,
tools/experiment_real.py (dev + test).
Chạy:  python tools/make_report_figures.py            (đầy đủ, khoảng 5 phút)
       python tools/make_report_figures.py --charts-only   (chỉ vẽ lại biểu đồ từ số liệu đã lưu)
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ipm import Config, run  # noqa: E402
from ipm import segmentation as seg  # noqa: E402
from ipm.evaluate import evaluate_image  # noqa: E402
from ipm.preprocess import load_image  # noqa: E402
from ipm.visualize import draw_detections, fig_fshs_vs_threshold, fig_histogram, fig_morphology  # noqa: E402
from main import REAL_PRESET  # noqa: E402

OUT = ROOT / "report" / "figures"
DATA = ROOT / "report" / "data"
REAL = ROOT / "data" / "real" / "hf_coins"
SYN = ROOT / "data" / "synthetic"

# Bảng màu đã kiểm định (dataviz validator: PASS ở chế độ sáng, nền trắng)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
GT_BGR, PRED_BGR = (72, 73, 227), (214, 120, 42)   # đỏ #e34948 = nhãn chuẩn, xanh #2a78d6 = dự đoán
COL_W = 3.45  # Độ rộng 1 cột IEEE (inch)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "savefig.dpi": 300,
})

BASELINE = Config(backend="opencv", split_min_circularity=0.0)
FINAL = replace(Config(backend="opencv"), **REAL_PRESET)
SYN_CASES = [("case1_ideal", "Case 1 – Ideal"), ("case2_low_contrast", "Case 2 – Low contrast"),
             ("case3_noisy_glare", "Case 3 – Noise & glare"), ("case4_overlapping", "Case 4 – Overlapping")]
REAL_CASES = {1: "R1 – Ideal", 2: "R2 – Low contrast", 3: "R3 – Textured background", 4: "R4 – Touching coins"}


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _boxes(img, boxes, color, scale=1.0, pad=0, th=3):
    for x, y, w, h in boxes:
        x, y, w, h = (int(round(v * scale)) for v in (x, y, w, h))
        cv2.rectangle(img, (x - pad, y - pad), (x + w + pad, y + h + pad), color, th)
    return img


def _panel_grid(rows, path, width, title_size=6.5):
    """rows: list các hàng, mỗi hàng là list (ảnh, tiêu đề)."""
    nr, nc = len(rows), len(rows[0])
    h0, w0 = rows[0][0][0].shape[:2]
    fig, axs = plt.subplots(nr, nc, figsize=(width, width / nc * h0 / w0 * nr + 0.22 * nr), squeeze=False)
    for r, row in enumerate(rows):
        for c, (im, title) in enumerate(row):
            ax = axs[r, c]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0 if im.ndim == 2 else None,
                      vmax=(1 if im.max() <= 1 else 255) if im.ndim == 2 else None)
            ax.set_title(title, fontsize=title_size, color=INK, pad=2.5)
            ax.axis("off")
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.5)
    fig.savefig(path, dpi=230)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Kết quả từng ảnh trên tập test (A và V)
# ---------------------------------------------------------------------------
def _eval(args):
    name, gt_entry, which = args
    res = run(load_image(REAL / name), BASELINE if which == "A" else FINAL)
    row = {"image": name, "config": which, "count": len(res.objects)}
    row.update(evaluate_image(res, gt_entry))
    return row


def per_image_test():
    gt = json.load(open(REAL / "ground_truth.json", encoding="utf-8"))
    names = json.load(open(REAL / "split.json", encoding="utf-8"))["test"]
    jobs = [(n, gt[n], w) for w in ("A", "V") for n in names]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_eval, jobs, chunksize=8))
    with open(DATA / "test_per_image.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "config", "count", "gt_count", "count_error", "tp", "fp", "fn"],
                            extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return rows


def load_rows():
    with open(DATA / "test_per_image.csv", encoding="utf-8") as f:
        return [{k: (v if k in ("image", "config") else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def load_cases():
    with open(DATA / "real_cases.csv", encoding="utf-8") as f:
        return {r["image"]: int(r["case"]) for r in csv.DictReader(f)}


def _metrics(rs):
    e = np.array([r["count_error"] for r in rs])
    tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"n_images": len(rs), "n_coins": int(sum(r["gt_count"] for r in rs)),
            "exact": float(np.mean(e == 0) * 100), "within1": float(np.mean(np.abs(e) <= 1) * 100),
            "mae": float(np.mean(np.abs(e))), "precision": p, "recall": rc,
            "f1": 2 * p * rc / (p + rc) if p + rc else 0.0, "tp": tp, "fp": fp, "fn": fn}


def case_metrics(rows, cases):
    out = {}
    for cfg in ("A", "V"):
        out[cfg] = {"all": _metrics([r for r in rows if r["config"] == cfg])}
        for k in REAL_CASES:
            out[cfg][k] = _metrics([r for r in rows if r["config"] == cfg and cases[r["image"]] == k])
    with open(DATA / "real_case_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# 2. Biểu đồ
# ---------------------------------------------------------------------------
def chart_count_error(rows):
    bins = list(range(-4, 5))
    labels = ["≤−4"] + [f"{b:+d}" if b else "0" for b in bins[1:-1]] + ["≥+4"]

    def dist(cfg):
        e = np.clip([r["count_error"] for r in rows if r["config"] == cfg], -4, 4)
        return np.array([np.mean(e == b) * 100 for b in bins])

    a, v = dist("A"), dist("V")
    x = np.arange(len(bins))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.bar(x - w / 2 - 0.01, v, w, color=BLUE, label="Final configuration (V)", edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2 + 0.01, a, w, color=ORANGE, label="Baseline (A)", edgecolor="white", linewidth=0.8)
    for xi, val in zip(x, v):
        if val >= 5:
            ax.text(xi - w / 2, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Counting error (predicted − true), objects per image")
    ax.set_ylabel("Images (%)")
    _grid(ax, "y")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.savefig(OUT / "chart_count_error.png")
    plt.close(fig)


ABLATION = [
    ("A. Gốc: Otsu trên ảnh xám", "A  Baseline: Otsu, gray"),
    ("G. Trừ nền (Lab) + Triangle", "G  Lab background diff. + Triangle"),
    ("I. G + lọc hình dạng (tròn>=0.5, A<=5%)", "I  + shape filter"),
    ("L. I + bỏ vùng chạm viền", "L  + border clearing"),
    ("M. L + chỉ Watershed vùng tròn>=0.35", "M  + gated watershed"),
    ("N. M + bỏ vụn < 0.25 x trung vị", "N  + small-fragment removal"),
    ("V. N + cửa sổ 0.15 + tròn>=0.6 [CHỐT]", "V  + tuning (final)"),
]


def chart_ablation():
    dev = json.load(open(ROOT / "outputs" / "experiment_hf_coins_dev.json", encoding="utf-8"))
    labels = [lab for _, lab in ABLATION]
    acc = [dev[k]["count_exact_acc"] * 100 for k, _ in ABLATION]
    f1 = [dev[k]["f1"] for k, _ in ABLATION]
    y = np.arange(len(labels))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.2), sharey=True, gridspec_kw={"wspace": 0.22})
    a1.barh(y, acc, 0.62, color=BLUE)
    a2.barh(y, f1, 0.62, color=BLUE)
    for yi, v in zip(y, acc):
        a1.text(v + 1.5, yi, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
    for yi, v in zip(y, f1):
        a2.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    a1.set_yticks(y, labels)
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 50, 100])
    a2.set_xlim(0, 1.18)
    a2.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    a1.set_title("Exact-count accuracy (%)", loc="left", color=INK)
    a2.set_title("Object-level F1", loc="left", color=INK)
    for ax in (a1, a2):
        _grid(ax)
        ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "chart_ablation.png")
    plt.close(fig)


def chart_real_cases(m):
    keys = list(REAL_CASES)
    short = ["R1", "R2", "R3", "R4", "All"]
    x = np.arange(len(short))
    w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw={"wspace": 0.28})
    for ax, metric, scale, fmt in ((a1, "exact", 1, "{:.0f}"), (a2, "f1", 1, "{:.2f}")):
        v = [m["V"][str(k) if str(k) in m["V"] else k][metric] for k in keys] + [m["V"]["all"][metric]]
        a = [m["A"][str(k) if str(k) in m["A"] else k][metric] for k in keys] + [m["A"]["all"][metric]]
        ax.bar(x - w / 2 - 0.01, v, w, color=BLUE, label="Final (V)", edgecolor="white", linewidth=0.8)
        ax.bar(x + w / 2 + 0.01, a, w, color=ORANGE, label="Baseline (A)", edgecolor="white", linewidth=0.8)
        top = 100 if metric == "exact" else 1.0
        for xi, val in zip(x, v):
            ax.text(xi - w / 2, val + top * 0.015, fmt.format(val), ha="center", va="bottom", fontsize=5.8, color=INK)
        ax.set_ylim(0, top * 1.12)
        ax.set_xticks(x, short, fontsize=6)
        _grid(ax, "y")
    a1.set_title("Exact-count accuracy (%)", loc="left", color=INK)
    a2.set_title("Object-level F1", loc="left", color=INK)
    a1.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, bbox_to_anchor=(0, -0.28))
    fig.savefig(OUT / "chart_real_cases.png")
    plt.close(fig)


def chart_synth_regions(stats):
    stages = [("regions_raw", "After thresholding"), ("regions_after_morph", "After morphology"),
              ("regions_after_area_filter", "After area filter"), ("n_obj", "Final (after watershed)")]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    x = np.arange(len(stats))
    w = 0.2
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    for i, ((k, lab), c) in enumerate(zip(stages, colors)):
        vals = [s[k] for s in stats]
        xs = x + (i - 1.5) * (w + 0.01)
        ax.bar(xs, vals, w, color=c, label=lab, edgecolor="white", linewidth=0.6)
        for xi, val in zip(xs, vals):
            ax.text(xi, val + 0.6, str(val), ha="center", va="bottom", fontsize=5.6, color=INK)
    ax.axhline(6, color=INK2, linestyle=(0, (3, 2)), linewidth=0.8, label="Ground truth (6)")
    ax.set_xticks(x, [lab.replace(" – ", "\n") for _, lab in SYN_CASES], fontsize=6.3)
    ax.set_ylabel("Number of regions")
    _grid(ax, "y")
    ax.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.set_ylim(0, max(s["regions_raw"] for s in stats) * 1.12)
    fig.savefig(OUT / "chart_synth_regions.png")
    plt.close(fig)


STAGES = [("1_preprocess", "Stage 1 Preprocessing"), ("2_segmentation", "Stage 2 Segmentation + morph."),
          ("3_label_features", "Stage 3 Labeling + features"), ("4_classify", "Stage 4 Classification")]


def measure_timing(n=100):
    names = json.load(open(REAL / "split.json", encoding="utf-8"))["test"][:n]
    imgs = [load_image(REAL / nm) for nm in names]
    out = {}
    for backend in ("numpy", "opencv"):
        cfg = replace(FINAL, backend=backend)
        run(imgs[0], cfg)
        acc = {k: [] for k, _ in STAGES}
        for im in imgs:
            t = run(im, cfg).timings
            for k, _ in STAGES:
                acc[k].append(t[k])
        out[backend] = {k: float(np.mean(v)) for k, v in acc.items()}
    with open(DATA / "timing.json", "w", encoding="utf-8") as f:
        json.dump({"n_images": n, "image_size": "640x640", **out}, f, indent=2)
    return out


def chart_timing(t):
    rows = [("NumPy\n(own impl.)", t["numpy"]), ("OpenCV", t["opencv"])]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(COL_W, 1.35))
    for yi, (_, d) in zip([1, 0], rows):
        left = 0.0
        for (k, lab), c in zip(STAGES, colors):
            w = d[k]
            ax.barh(yi, w, 0.55, left=left, color=c, edgecolor="white", linewidth=1.0, label=lab if yi == 1 else None)
            if w > 12:
                ax.text(left + w / 2, yi, f"{w:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#0b0b0b" if c == YELLOW else "white")
            left += w
        ax.text(left + 4, yi, f"{left:.0f} ms", va="center", fontsize=7, color=INK)
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.set_xlabel("Mean time per 640×640 image (ms)")
    ax.set_xlim(0, max(sum(d.values()) for _, d in rows) * 1.18)
    _grid(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.42))
    fig.savefig(OUT / "chart_timing.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Hình ảnh giả lập
# ---------------------------------------------------------------------------
def synthetic_figures():
    gt = json.load(open(SYN / "ground_truth.json", encoding="utf-8"))
    cfg = Config()  # backend numpy (tự cài đặt) – khớp với bảng số liệu giả lập trong báo cáo
    rows, stats = [], []
    results = {}
    for stem, title in SYN_CASES:
        res = run(load_image(SYN / f"{stem}.jpg"), cfg)
        results[stem] = res
        st, ci = res.stats, res.class_info
        stats.append(st)
        n_gt = gt[f"{stem}.jpg"]["count"]
        det = draw_detections(res.image, res.objects, ci)
        rows.append([
            (_rgb(res.image), f"{title}\nGround truth: {n_gt}"),
            (res.smooth, f"FSHS + denoising\nAOD {res.aod_before:.0f} → {res.aod_after:.0f}"),
            (res.mask_raw, f"Otsu mask (T* = {res.seg_info['T']})\n{st['regions_raw']} regions"),
            (res.mask_clean, f"After morphology\n{st['regions_after_morph']} regions"),
            (_rgb(det), f"Result: {st['n_obj']}/{n_gt} objects\n{ci['n_large']} Large, {ci['n_small']} Small"),
        ])
    _panel_grid(rows, OUT / "synth_grid.jpg", 7.16)
    chart_synth_regions(stats)
    # Hình chuyên đề cho từng case
    fig_histogram(results["case2_low_contrast"], OUT / "case2_histogram.png")
    fig_fshs_vs_threshold(results["case2_low_contrast"], OUT / "case2_fshs_threshold.png")
    fig_morphology(results["case3_noisy_glare"], OUT / "case3_morphology.png")
    nosplit = run(load_image(SYN / "case4_overlapping.jpg"), replace(cfg, split_touching=False))
    r4 = results["case4_overlapping"]
    a = draw_detections(r4.image, r4.objects, r4.class_info)
    b = draw_detections(nosplit.image, nosplit.objects, nosplit.class_info)
    h, w = a.shape[:2]
    crop = (slice(int(0.25 * h), int(0.95 * h)), slice(int(0.25 * w), w))
    _panel_grid([[(_rgb(a[crop]), "(a) With watershed\n6 objects: 3 Large, 3 Small"),
                  (_rgb(b[crop]), "(b) Without watershed\n5 objects: 4 Large, 1 Small")]],
                OUT / "case4_split.jpg", COL_W, 6.3)
    with open(DATA / "synthetic_results.json", "w", encoding="utf-8") as f:
        json.dump({stem: {"stats": results[stem].stats, "T": results[stem].seg_info["T"],
                          "aod": [results[stem].aod_before, results[stem].aod_after],
                          "class": results[stem].class_info} for stem, _ in SYN_CASES}
                  | {"case4_nosplit": {"stats": nosplit.stats, "class": nosplit.class_info}},
                  f, indent=2, default=float)


# ---------------------------------------------------------------------------
# 4. Hình ảnh thật
# ---------------------------------------------------------------------------
def real_case_grid(rows, cases):
    gt = json.load(open(REAL / "ground_truth.json", encoding="utf-8"))
    v = {r["image"]: r for r in rows if r["config"] == "V"}
    rng = np.random.default_rng(11)
    grid, picks = [], {}
    for k, title in REAL_CASES.items():
        name = str(rng.choice(sorted(n for n in v if cases[n] == k)))  # chọn ngẫu nhiên (seed cố định)
        picks[k] = name
        img = load_image(REAL / name)
        ra, rv = run(img, BASELINE), run(img, FINAL)
        boxes = gt[name]["boxes"]
        inp = _boxes(img.copy(), boxes, GT_BGR, th=2)
        out = _boxes(rv.image.copy(), boxes, GT_BGR, th=2)
        _boxes(out, [(o.x, o.y, o.w, o.h) for o in rv.objects], PRED_BGR, pad=3, th=2)
        grid.append([
            (_rgb(inp), f"{title}\nGround truth: {len(boxes)}"),
            (ra.mask_clean, f"Baseline A (Otsu mask)\n{len(ra.objects)} objects"),
            (rv.smooth, "Background difference\nD(n) = ||I − B||"),
            ((rv.labels > 0).astype(np.uint8), f"Final V: kept regions\n{len(rv.objects)} objects"),
            (_rgb(out), f"Final V: {len(rv.objects)}/{len(boxes)}\nTP {v[name]['tp']}, FP {v[name]['fp']}, FN {v[name]['fn']}"),
        ])
    _panel_grid(grid, OUT / "real_grid.jpg", 7.16)
    return picks


def real_examples(rows):
    v = {r["image"]: r for r in rows if r["config"] == "V"}
    gt = json.load(open(REAL / "ground_truth.json", encoding="utf-8"))
    picks = ["hf_0003.jpg", "hf_1512.jpg", "hf_1042.jpg", "hf_1355.jpg", "hf_0024.jpg", "hf_1153.jpg"]
    panels = []
    for name in picks:
        res = run(load_image(REAL / name), FINAL)
        im = _boxes(res.image.copy(), gt[name]["boxes"], GT_BGR)
        _boxes(im, [(o.x, o.y, o.w, o.h) for o in res.objects], PRED_BGR, pad=3)
        panels.append((_rgb(im), f"{name[:-4]}: true {v[name]['gt_count']}, predicted {v[name]['count']}"))
    _panel_grid([panels[:3], panels[3:]], OUT / "real_examples.jpg", 7.16, 7.5)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if "--charts-only" in sys.argv:
        rows = load_rows()
    else:
        print("1/5 Chạy A và V trên tập test ...")
        rows = per_image_test()
    m = case_metrics(rows, cases)
    for cfg in ("A", "V"):
        for k in ["all", *REAL_CASES]:
            d = m[cfg][k]
            print(f"   {cfg} {str(k):<4} n={d['n_images']:>3} coins={d['n_coins']:>4} exact={d['exact']:5.1f}% "
                  f"±1={d['within1']:5.1f}% MAE={d['mae']:.2f} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")
    print("2/5 Biểu đồ ...")
    chart_count_error(rows)
    chart_ablation()
    chart_real_cases(m)
    if "--charts-only" in sys.argv:
        chart_timing(json.load(open(DATA / "timing.json", encoding="utf-8")))
        real_case_grid(rows, cases)
        real_examples(rows)
        synthetic_figures()
        return
    print("3/5 Đo thời gian đơn luồng ...")
    t = measure_timing()
    print("   ", json.dumps(t))
    chart_timing(t)
    print("4/5 Hình ảnh thật ...")
    print("   ", real_case_grid(rows, cases))
    real_examples(rows)
    print("5/5 Hình ảnh giả lập ...")
    synthetic_figures()
    print(f"Xong: {OUT}")


if __name__ == "__main__":
    main()
