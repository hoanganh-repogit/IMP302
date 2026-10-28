"""Generate synthetic test images with exact ground truth.

Each case deliberately stresses one stage of the pipeline (as in Chapter 3 of the spec):
    case1_ideal         – white paper background, uniform lighting       -> baseline
    case2_low_contrast  – underexposed, gray levels squeezed into [30, 100] -> FSHS
    case3_noisy_glare   – dust, hairs, salt-and-pepper noise, glare on coins -> median + morphology + hole filling
    case4_overlapping   – 2 small coins touching/overlapping             -> watershed
Uneven illumination is tested on the real photos (data/real/hf_coins).

Run:  python tools/generate_synthetic.py   (output in data/synthetic/)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 1600, 1067                 # Larger than work_width so the resizing step is exercised too
R_LARGE, R_SMALL = 95, 62
GOLD = np.array([70, 150, 200], float)     # BGR – gold coin (large)
SILVER = np.array([150, 150, 146], float)  # BGR – silver coin (small)
OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def paper(rng, base=(232, 234, 236)):
    img = np.empty((H, W, 3), float)
    img[:] = base
    grain = cv2.GaussianBlur(rng.normal(0, 5, (H, W)), (0, 0), 2.5)
    return img + grain[..., None]


def place(rng, radii, gap=40, margin=30, fixed=()):
    """Randomly place non-overlapping circles at least `gap` px apart."""
    placed = list(fixed)
    for r in radii:
        for _ in range(5000):
            cx = int(rng.integers(margin + r, W - margin - r))
            cy = int(rng.integers(margin + r, H - margin - r))
            if all(np.hypot(cx - x, cy - y) >= r + rr + gap for x, y, rr in placed):
                placed.append((cx, cy, r))
                break
        else:
            raise RuntimeError("Not enough room to place the objects")
    return placed


def draw_coin(img, mask, cx, cy, r, color, label, rng, glare=False):
    # Soft drop shadow (light from the top-left)
    sh = np.zeros((H, W), np.float32)
    cv2.circle(sh, (cx + 7, cy + 9), r, 1.0, -1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (0, 0), 9)
    img *= (1 - 0.22 * sh)[..., None]

    y0, y1, x0, x1 = max(cy - r - 2, 0), min(cy + r + 3, H), max(cx - r - 2, 0), min(cx + r + 3, W)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - cx, yy - cy)
    d = dist / r
    alpha = np.clip(r - dist + 0.5, 0, 1)                 # anti-aliased edge
    shade = 1.0 - 0.16 * d ** 2                          # bright center, darker towards the rim
    shade -= 0.12 * ((d > 0.84) & (d < 0.92))            # rim ridge
    # Embossed pattern: denomination + random texture
    pat = np.zeros((y1 - y0, x1 - x0), np.uint8)
    fs = r / 55
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    cv2.putText(pat, label, (cx - x0 - tw // 2, cy - y0 + th // 2), cv2.FONT_HERSHEY_DUPLEX, fs, 255, 2)
    for _ in range(6):
        a = rng.uniform(0, 2 * np.pi)
        cv2.ellipse(pat, (cx - x0, cy - y0), (int(r * 0.7), int(r * 0.7)), np.degrees(a), 0, 40, 150, 2)
    pat = pat.astype(np.float32) / 255
    shade -= 0.14 * cv2.GaussianBlur(pat, (0, 0), 1.0)
    shade += rng.normal(0, 0.02, shade.shape)
    coin = color[None, None, :] * shade[..., None]
    if glare:  # Specular glare spot – large enough that a 5x5 closing cannot seal it
        g = np.zeros_like(pat)
        cv2.ellipse(g, (cx - x0 - int(0.25 * r), cy - y0 - int(0.2 * r)),
                    (int(0.38 * r), int(0.22 * r)), -30, 0, 360, 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), 2.0)[..., None]
        coin = coin * (1 - g) + 252 * g
    region = img[y0:y1, x0:x1]
    region[:] = region * (1 - alpha[..., None]) + coin * alpha[..., None]
    cv2.circle(mask, (cx, cy), r, 255, -1)


def render(rng, coins, glare=False):
    img = paper(rng)
    mask = np.zeros((H, W), np.uint8)
    objects = []
    for cx, cy, r in coins:
        large = r == R_LARGE
        draw_coin(img, mask, cx, cy, r, GOLD if large else SILVER, "5000" if large else "1000", rng,
                  glare=glare)
        objects.append({"cx": cx, "cy": cy, "r": r, "cls": "Large" if large else "Small"})
    return img, mask, objects


def finish(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gt").mkdir(exist_ok=True)
    gt = {}

    def save(name, img, mask, objects):
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        mname = f"gt/{Path(name).stem}_mask.png"
        cv2.imwrite(str(OUT / mname), mask)
        gt[name] = {"count": len(objects),
                    "large": sum(o["cls"] == "Large" for o in objects),
                    "small": sum(o["cls"] == "Small" for o in objects),
                    "mask": mname, "objects": objects}
        print(f"  {name}: {gt[name]['count']} objects ({gt[name]['large']} large, {gt[name]['small']} small)")

    radii6 = [R_LARGE] * 3 + [R_SMALL] * 3

    # Case 1 – ideal
    rng = np.random.default_rng(1)
    img, mask, objs = render(rng, place(rng, radii6))
    save("case1_ideal.jpg", finish(img), mask, objs)

    # Case 2 – underexposed, low contrast: squeeze all gray levels into the dark range [30, 100]
    rng = np.random.default_rng(2)
    img, mask, objs = render(rng, place(rng, radii6))
    img = 30 + img * (70 / 255) + rng.normal(0, 2.5, img.shape)
    save("case2_low_contrast.jpg", finish(img), mask, objs)

    # Case 3 – background specks + glare
    rng = np.random.default_rng(3)
    img, mask, objs = render(rng, place(rng, radii6), glare=True)
    coin_area = mask > 0
    for _ in range(60):                                   # dust
        x, y = int(rng.integers(5, W - 5)), int(rng.integers(5, H - 5))
        if not coin_area[y, x]:
            v = float(rng.integers(60, 150))
            cv2.circle(img, (x, y), int(rng.integers(2, 8)), (v, v, v), -1, cv2.LINE_AA)
    for _ in range(3):                                    # hairs
        x, y = int(rng.integers(50, W - 200)), int(rng.integers(50, H - 50))
        cv2.line(img, (x, y), (x + int(rng.integers(80, 180)), y + int(rng.integers(-60, 60))),
                 (70, 70, 70), 2, cv2.LINE_AA)
    img += rng.normal(0, 6, img.shape)                     # Gaussian sensor noise
    sp = rng.random((H, W))                               # 2% salt-and-pepper noise
    img[sp < 0.01] = 0
    img[sp > 0.99] = 255
    save("case3_noisy_glare.jpg", finish(img), mask, objs)

    # Case 4 – 2 small coins slightly overlapping
    rng = np.random.default_rng(4)
    pair = [(600, 560, R_SMALL), (600 + 2 * R_SMALL - 8, 560, R_SMALL)]
    coins = place(rng, [R_LARGE] * 3 + [R_SMALL], fixed=pair)
    img, mask, objs = render(rng, coins)
    save("case4_overlapping.jpg", finish(img), mask, objs)

    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(gt)} images + ground_truth.json to {OUT}")


if __name__ == "__main__":
    main()
