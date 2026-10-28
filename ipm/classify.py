"""Giai đoạn 4 – Phân lớp mẫu theo quy tắc quyết định trên diện tích (Chương 12).

Hàm quyết định:  A_m >= T_area -> C1 (Large),  A_m < T_area -> C2 (Small).
Khác bản gốc ở cách chọn T_area: diện tích trung bình bị kéo lệch khi số vật lớn/nhỏ
chênh nhau hoặc có một khối dính nhau, nên mặc định dùng Otsu 1-D trên tập diện tích
(cực đại phương sai giữa hai nhóm – cùng nguyên lý với Otsu ở Giai đoạn 2).
"""
import numpy as np

LARGE, SMALL, SAME = "Large", "Small", "Same"


def otsu_1d(values):
    """Chia dãy diện tích đã sắp xếp thành 2 nhóm sao cho w0·w1·(mu0 - mu1)^2 lớn nhất."""
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
    """Gán o.cls cho từng vật thể, trả về thông tin ngưỡng quyết định."""
    info = {"method": cfg.classify_method, "threshold": None, "warning": None}
    if not objects:
        info.update(n_large=0, n_small=0)
        return info
    areas = np.array([o.area for o in objects], dtype=np.float64)

    if cfg.area_threshold is not None:
        info["method"] = "fixed"
        T = float(cfg.area_threshold)
    elif len(areas) < 2 or areas.max() / areas.min() < cfg.min_size_ratio:
        # Mọi vật gần như cùng cỡ: chia đôi bằng ngưỡng thích nghi sẽ là chia bừa
        info["warning"] = (f"A_max/A_min = {areas.max() / areas.min():.2f} < {cfg.min_size_ratio}: "
                           "chỉ có 1 nhóm kích thước, không chia Large/Small")
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
