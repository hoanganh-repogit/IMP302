"""Tham số cấu hình toàn hệ thống. Mọi "con số ma thuật" của pipeline đều nằm ở đây."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---------- Giai đoạn 1: Tiền xử lý (Chương 1, 2, 3, 7, 8) ----------
    work_width: int = 800            # Chuẩn hóa chiều rộng ảnh (px); ảnh nhỏ hơn giữ nguyên
    resize_method: str = "area"      # "area" (cv2.INTER_AREA, chống răng cưa) | "nearest" (tự cài đặt, Chương 1.1)
    fshs_low_pct: float = 0.5        # FSHS theo phân vị: A = P0.5, B = P99.5 (chống điểm ngoại lai)
    fshs_high_pct: float = 99.5
    flatten_light: bool = False      # San phẳng ánh sáng không đều (Hướng phát triển 7.2)
    flatten_se_frac: float = 0.4     # Kích thước SE ước lượng nền / cạnh ngắn của ảnh (phải > vật lớn nhất)
    median_ksize: int = 5            # Lọc trung vị khử nhiễu xung (Chương 2.2, 3.3)
    gaussian_ksize: int = 5          # Lọc Gaussian khử nhiễu cảm biến (Chương 2.1, 3.3)
    gaussian_sigma: float = 1.0

    # ---------- Giai đoạn 2: Phân đoạn & Hình thái học (Chương 1.2, 2.3, 9.1) ----------
    seg_method: str = "otsu"         # "otsu" | "kmeans" | "bgsub" (trừ nền – cho ảnh thật, nền không đều)
    bg_kernel_frac: float = 0.25     # bgsub: cửa sổ trung vị ước lượng nền / cạnh ngắn (phải > 1.25 lần đường kính vật)
    bg_color: bool = True            # bgsub: khác biệt màu trong không gian Lab (False = chỉ độ sáng)
    bg_threshold: str = "triangle"   # bgsub: "triangle" | "otsu" | "mad"
    bg_mad_k: float = 6.0            # bgsub + mad: T = median + k·1.4826·MAD
    polarity: str = "auto"           # "auto" | "bright" (vật sáng hơn nền) | "dark" (vật tối hơn nền)
    se_size: int = 5                 # Phần tử cấu trúc elip 5x5
    open_iter: int = 1
    close_iter: int = 1
    fill_holes: bool = True          # Lấp lỗ do lóa lớn hơn SE mà phép Closing không bít được

    # ---------- Giai đoạn 3: Gán nhãn & Đặc trưng (Chương 1.2, 9, 10, 11.1) ----------
    connectivity: int = 4            # 4 hoặc 8 lân cận
    min_area_px: int = 100           # A_min tuyệt đối
    min_area_frac: float = 0.0003    # A_min tương đối theo diện tích ảnh -> lấy max của hai giá trị
    split_touching: bool = True      # Tách vật chạm nhau bằng Distance Transform + Watershed
    solidity_thresh: float = 0.94    # Vùng có độ đặc (A / A_hull) thấp hơn ngưỡng này bị nghi là dính nhau
    split_min_circularity: float = 0.35  # Chỉ Watershed vùng "giống cụm xu" (2 xu chạm nhau ~0.5; dải mép bàn < 0.3)
    clear_border: bool = False       # Bỏ vùng chạm viền ảnh (vật thể phải nằm trọn trong khung – mép bàn, góc tối)
    min_area_rel_median: float = 0.0 # Bỏ vùng có A < tỉ lệ này x trung vị diện tích các vật (vụn nhiễu). 0 = tắt
    min_circularity: float = 0.0     # Lọc hình dạng: bỏ vùng có độ tròn thấp hơn (mép bàn, vệt bóng). 0 = tắt
    max_area_frac: float = 1.0       # Bỏ vùng lớn hơn tỉ lệ này của ảnh (mảng nền bị tách nhầm). 1 = tắt

    # ---------- Giai đoạn 4: Phân loại (Chương 12) ----------
    classify_method: str = "otsu"    # "otsu" (Otsu 1-D trên tập diện tích) | "mean" (bản gốc) | "gap"
    min_size_ratio: float = 1.3      # A_max / A_min < ngưỡng này -> coi như chỉ có 1 nhóm kích thước
    area_threshold: float | None = None  # Ngưỡng diện tích cố định (px) nếu đã hiệu chuẩn camera

    # ---------- Cài đặt ----------
    backend: str = "numpy"           # "numpy" = thuật toán tự cài đặt | "opencv" = hàm thư viện

    def to_dict(self):
        return asdict(self)
