"""Giai đoạn 3a – Gán nhãn vùng liên thông 2 lượt quét (Region Labeling – Chương 1.2)."""
import cv2
import numpy as np


def two_pass_labeling(mask, connectivity=4):
    """Thuật toán 2 lượt quét với Bảng tương đương (cài đặt bằng Union-Find).

    Lượt 1: quét trái->phải, trên->dưới. Tại pixel vật thể, xét các lân cận đã đi qua
            (trái, trên; thêm trên-trái, trên-phải nếu 8-liên thông):
            - không có lân cận nào mang nhãn  -> cấp nhãn mới k, k <- k + 1
            - có lân cận mang nhãn            -> nhận nhãn nhỏ nhất, các nhãn khác
                                                 được ghi vào bảng tương đương.
    Lượt 2: thay mỗi nhãn bằng nhãn đại diện của lớp tương đương, đánh số lại 1..N.
    Chỉ duyệt các pixel vật thể (np.flatnonzero giữ đúng thứ tự quét mành).
    """
    h, w = mask.shape
    lab = [0] * (h * w)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    next_label = 1
    eight = connectivity == 8
    for idx in np.flatnonzero(mask).tolist():
        x = idx % w
        neigh = []
        if x > 0 and lab[idx - 1]:
            neigh.append(lab[idx - 1])
        if idx >= w:
            up = idx - w
            if lab[up]:
                neigh.append(lab[up])
            if eight:
                if x > 0 and lab[up - 1]:
                    neigh.append(lab[up - 1])
                if x < w - 1 and lab[up + 1]:
                    neigh.append(lab[up + 1])
        if not neigh:
            lab[idx] = next_label
            parent.append(next_label)
            next_label += 1
        else:
            m = min(neigh)
            lab[idx] = m
            for n in neigh:
                if n != m:
                    union(m, n)

    # Lượt 2: gốc của mỗi lớp là nhãn nhỏ nhất -> luôn được đánh số trước các thành viên
    lut = np.zeros(next_label, dtype=np.int32)
    n_final = 0
    for l in range(1, next_label):
        r = find(l)
        if r == l:
            n_final += 1
            lut[l] = n_final
        else:
            lut[l] = lut[r]
    labels = lut[np.asarray(lab, dtype=np.int32)].reshape(h, w)
    return labels, n_final


def label(mask, connectivity=4, backend="numpy"):
    if backend == "opencv":
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity,
                                            ltype=cv2.CV_32S)
        return labels, n - 1
    return two_pass_labeling(mask, connectivity)


def count_components(mask, connectivity=4):
    """Đếm nhanh số vùng (chỉ dùng cho số liệu chẩn đoán, không nằm trong pipeline chính)."""
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return n - 1


def remove_small(labels, a_min):
    """Bỏ vùng có diện tích < A_min rồi đánh số lại liên tục 1..N_obj."""
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= a_min)
    keep = keep[keep > 0]
    lut = np.zeros(len(areas), dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels], len(keep)


def clear_border(labels):
    """Xóa mọi vùng có pixel nằm trên viền ảnh, rồi đánh số lại."""
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    out = labels.copy()
    out[np.isin(out, border[border > 0])] = 0
    return remove_small(out, 1)


def same_partition(a, b):
    """Hai bản đồ nhãn có cùng cách chia vùng không (bỏ qua việc đánh số khác nhau)?"""
    if a.shape != b.shape or not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a.ravel(), b.ravel()], axis=1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))
