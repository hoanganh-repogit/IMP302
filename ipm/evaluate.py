"""Đánh giá định lượng so với nhãn thật (ground truth)."""
import json
from pathlib import Path

import cv2
import numpy as np

from .preprocess import load_image


def load_ground_truth(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match_boxes(res, boxes):
    """Ghép 1-1 vật thể dự đoán với khung bao thật (tọa độ ảnh gốc, COCO [x, y, w, h]).

    Một dự đoán là đúng (TP) khi trọng tâm của nó nằm trong khung thật chưa được ghép;
    ưu tiên cặp có trọng tâm gần tâm khung nhất. Dư -> FP (đếm thừa), thiếu -> FN (đếm sót).
    """
    s = res.scale
    gt = np.array(boxes, dtype=np.float64).reshape(-1, 4) * s
    pred = np.array([[o.cx, o.cy] for o in res.objects], dtype=np.float64).reshape(-1, 2)
    pairs = []
    for i, (px, py) in enumerate(pred):
        for j, (x, y, w, h) in enumerate(gt):
            if x <= px <= x + w and y <= py <= y + h:
                pairs.append((np.hypot(px - (x + w / 2), py - (y + h / 2)), i, j))
    used_p, used_g = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}


def evaluate_image(res, gt_entry, gt_dir=None):
    """So sánh số đếm, số Large/Small và (nếu có mặt nạ thật) IoU của mặt nạ phân đoạn."""
    ci = res.class_info
    out = {
        "gt_count": gt_entry.get("count"),
        "gt_large": gt_entry.get("large"),
        "gt_small": gt_entry.get("small"),
    }
    n = len(res.objects)
    if out["gt_count"] is not None:
        out["count_ok"] = n == out["gt_count"]
        out["count_error"] = n - out["gt_count"]
    if out["gt_large"] is not None and out["gt_small"] is not None:
        out["class_ok"] = (ci.get("n_large") == out["gt_large"] and ci.get("n_small") == out["gt_small"])
    if gt_entry.get("boxes") is not None:
        out.update(match_boxes(res, gt_entry["boxes"]))
    mask_file = gt_entry.get("mask")
    if mask_file and gt_dir:
        gt = load_image(Path(gt_dir) / mask_file)[..., 0] > 127
        h, w = res.mask_clean.shape
        gt = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        pred = res.mask_clean > 0
        union = np.logical_or(gt, pred).sum()
        out["mask_iou"] = round(float(np.logical_and(gt, pred).sum() / union), 4) if union else 1.0
    return out
