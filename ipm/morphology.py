"""Giai đoạn 2b – Hình thái học nhị phân (Chương 1.2 & 2.3)."""
import cv2
import numpy as np


def structuring_element(size):
    """SE hình elip size x size (đối xứng tâm nên S và ảnh phản xạ của S trùng nhau)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _offsets(se):
    return [(dy, dx) for dy, dx in zip(*np.nonzero(se))]


def erode(mask, se, backend="numpy"):
    """(b ⊖ S)(x) = 1  <=>  S đặt tại x nằm trọn trong B (mọi pixel dưới SE đều bằng 1)."""
    if backend == "opencv":
        return cv2.erode(mask.astype(np.uint8), se)
    h, w = mask.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    # Ngoài biên coi như = 1 (giống OpenCV) để vật chạm mép ảnh không bị ăn mòn từ phía ngoài
    p = np.pad(mask.astype(bool), ((ph, ph), (pw, pw)), constant_values=True)
    out = np.ones((h, w), dtype=bool)
    for dy, dx in _offsets(se):
        out &= p[dy:dy + h, dx:dx + w]
    return out.astype(np.uint8)


def dilate(mask, se, backend="numpy"):
    """(b ⊕ S)(x) = 1  <=>  S đặt tại x giao B khác rỗng (ít nhất 1 pixel dưới SE bằng 1)."""
    if backend == "opencv":
        return cv2.dilate(mask.astype(np.uint8), se)
    h, w = mask.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    p = np.pad(mask.astype(bool), ((ph, ph), (pw, pw)), constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy, dx in _offsets(se):
        out |= p[dy:dy + h, dx:dx + w]
    return out.astype(np.uint8)


def opening(mask, se, iterations=1, backend="numpy"):
    """b ∘ S = (b ⊖ S) ⊕ S : xóa nhiễu hạt nhỏ hơn SE ngoài nền."""
    out = mask
    for _ in range(iterations):
        out = erode(out, se, backend)
    for _ in range(iterations):
        out = dilate(out, se, backend)
    return out


def closing(mask, se, iterations=1, backend="numpy"):
    """b • S = (b ⊕ S) ⊖ S : hàn khe nứt, bít lỗ nhỏ hơn SE bên trong vật thể."""
    out = mask
    for _ in range(iterations):
        out = dilate(out, se, backend)
    for _ in range(iterations):
        out = erode(out, se, backend)
    return out


def fill_holes(mask):
    """Lấp mọi lỗ đen nằm trọn bên trong vật thể (vùng nền không thông ra viền ảnh).

    Bổ sung cho Closing: vết lóa lớn hơn SE 5x5 thì Closing không bít được.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = m
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)  # loang nền từ viền
    holes = ff[1:-1, 1:-1] == 0
    return (m | holes).astype(np.uint8)
