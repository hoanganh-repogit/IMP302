"""Giai đoạn 2a – Phân đoạn dựa trên tính tương đồng: Otsu (Chương 1.2) & K-Means (Chương 9.1)."""
import cv2
import numpy as np

from .preprocess import K, histogram


def otsu_threshold(gray, backend="numpy"):
    """Tìm T* = argmax_t sigma_b^2(t), với sigma_b^2 = w0·w1·(mu0 - mu1)^2.

    Lớp nền C0 = {0..t}, lớp vật thể C1 = {t+1..K-1}  ->  mặt nạ = (gray > T*).
    Trả về (T*, mảng sigma_b^2 theo t) – mảng này dùng để vẽ đồ thị trong báo cáo.
    """
    H = histogram(gray)
    N = H.sum()
    k = np.arange(K)
    cum = np.cumsum(H)
    w0 = cum / N
    w1 = (N - cum) / N                       # tính bằng số nguyên -> w1 = 0 chính xác ở cuối dải
    m_cum = np.cumsum(k * H) / N             # sum_{k<=t} k·p(k)
    mu_T = m_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m_cum / w0
        mu1 = (mu_T - m_cum) / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b = np.nan_to_num(sigma_b, nan=0.0, posinf=0.0, neginf=0.0)
    if backend == "opencv":
        T, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(T), sigma_b
    return int(np.argmax(sigma_b)), sigma_b


def kmeans_threshold(gray, k=2, eps=0.2, max_iter=100):
    """K-Means 4 bước (Mục 9.1) trên các mẫu vô hướng z_q = mức xám.

    Vì mọi pixel cùng mức xám luôn rơi vào cùng một cụm, ta gom cụm trực tiếp trên
    256 mức xám có trọng số H(k) – kết quả y hệt chạy trên M x N pixel nhưng nhanh hơn
    hàng nghìn lần. Khởi tạo tất định tại min/max để kết quả lặp lại được.
    Trả về (T, các tâm cụm, số vòng lặp); với k = 2, T là trung điểm của 2 tâm.
    """
    H = histogram(gray).astype(np.float64)
    levels = np.arange(K, dtype=np.float64)
    present = np.flatnonzero(H)
    m = np.linspace(present[0], present[-1], k)                     # Bước 1: khởi tạo
    it = 0
    for it in range(1, max_iter + 1):
        assign = np.argmin(np.abs(levels[:, None] - m[None, :]), axis=1)  # Bước 2: gán mẫu
        new_m = m.copy()
        for i in range(k):                                           # Bước 3: cập nhật tâm
            wsum = H[assign == i].sum()
            if wsum > 0:
                new_m[i] = (levels[assign == i] * H[assign == i]).sum() / wsum
        E = np.abs(new_m - m).sum()                                  # Bước 4: kiểm tra hội tụ
        m = new_m
        if E <= eps:
            break
    T = int(np.floor((m[0] + m[-1]) / 2)) if k == 2 else None
    return T, m, it


def triangle_threshold(gray):
    """Phương pháp tam giác (Zack 1977) cho histogram MỘT đỉnh có đuôi dài.

    Nối đỉnh histogram với mức xám khác 0 xa nhất; ngưỡng là điểm trên histogram có
    khoảng cách vuông góc lớn nhất tới đường nối đó. Hợp với ảnh khác biệt D(n): nền dồn
    về gần 0, vật thể chỉ chiếm vài % nằm ở đuôi – trường hợp mà Otsu hoạt động kém.
    """
    H = histogram(gray).astype(np.float64)
    peak = int(np.argmax(H))
    last = int(np.flatnonzero(H)[-1])
    if last <= peak:
        return peak
    k = np.arange(peak, last + 1)
    x1, y1, x2, y2 = peak, H[peak], last, H[last]
    dist = np.abs((y2 - y1) * k - (x2 - x1) * H[k] + x2 * y1 - y2 * x1)
    return int(k[np.argmax(dist)])


def mad_threshold(gray, k=6.0):
    """T = median + k·1.4826·MAD: ngưỡng thống kê bền vững, coi vật thể là "ngoại lai" so với nền."""
    v = gray.astype(np.float64).ravel()
    med = np.median(v)
    return int(np.ceil(med + k * 1.4826 * np.median(np.abs(v - med))))


def background_is_bright(gray, T, margin_frac=0.02):
    """Xác định cực tính: viền ảnh gần như luôn là nền, xem đa số pixel viền thuộc lớp sáng hay tối."""
    h, w = gray.shape
    m = max(2, int(min(h, w) * margin_frac))
    border = np.concatenate([gray[:m].ravel(), gray[-m:].ravel(),
                             gray[:, :m].ravel(), gray[:, -m:].ravel()])
    return bool(np.mean(border > T) > 0.5)


def threshold_mask(gray, T, bg_bright):
    """Vật thể = 1. Nếu nền sáng (giấy trắng) thì vật thể là lớp TỐI -> đảo mặt nạ."""
    bright = gray > T
    return (~bright if bg_bright else bright).astype(np.uint8)


def segment(gray, cfg, method=None):
    """Trả về (mặt nạ nhị phân {0,1}, thông tin phân đoạn)."""
    method = method or cfg.seg_method
    info = {"method": method}
    if method == "bgsub":
        # Đầu vào là ảnh khác biệt D(n): vật thể luôn là lớp có D lớn
        rule = cfg.bg_threshold
        if rule == "otsu":
            T, _ = otsu_threshold(gray, cfg.backend)
        elif rule == "mad":
            T = mad_threshold(gray, cfg.bg_mad_k)
        else:
            T = triangle_threshold(gray)
        info.update(T=int(T), threshold_rule=rule, background_bright=False)
        return threshold_mask(gray, T, False), info
    if method == "otsu":
        T, _ = otsu_threshold(gray, cfg.backend)
    elif method == "kmeans":
        T, centers, iters = kmeans_threshold(gray)
        info.update(centers=[round(float(c), 2) for c in centers], iterations=iters)
    else:
        raise ValueError(f"seg_method không hợp lệ: {method}")
    if cfg.polarity == "auto":
        bg_bright = background_is_bright(gray, T)
    else:
        bg_bright = cfg.polarity == "dark"
    info.update(T=int(T), background_bright=bg_bright)
    return threshold_mask(gray, T, bg_bright), info
