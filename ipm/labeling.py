"""Stage 3a – Two-pass connected-component labeling (Region Labeling – Lecture 1.2)."""
import cv2
import numpy as np


def two_pass_labeling(mask, connectivity=4):
    """Two-pass algorithm with an equivalence table (implemented with union-find).

    Pass 1: scan left->right, top->bottom. At an object pixel, look at the already visited
            neighbors (left, up; plus up-left and up-right for 8-connectivity):
            - no labeled neighbor  -> assign a new label k, k <- k + 1
            - labeled neighbors    -> take the smallest label; the other labels are
                                      recorded in the equivalence table.
    Pass 2: replace every label by the representative of its equivalence class, renumber 1..N.
    Only object pixels are visited (np.flatnonzero keeps raster-scan order).
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

    # Pass 2: the root of each class is its smallest label -> it is always numbered before its members
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
    """Quick region count (diagnostic statistics only, not part of the main pipeline)."""
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return n - 1


def remove_small(labels, a_min):
    """Drop regions with area < A_min and renumber them consecutively 1..N_obj."""
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= a_min)
    keep = keep[keep > 0]
    lut = np.zeros(len(areas), dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels], len(keep)


def clear_border(labels):
    """Remove every region with a pixel on the image border, then renumber."""
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    out = labels.copy()
    out[np.isin(out, border[border > 0])] = 0
    return remove_small(out, 1)


def same_partition(a, b):
    """Do two label maps define the same partition (ignoring different numbering)?"""
    if a.shape != b.shape or not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a.ravel(), b.ravel()], axis=1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))
