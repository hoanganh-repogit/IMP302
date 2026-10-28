"""Save the public coin images bundled with scikit-image into data/public/.

- skimage.data.coins(): 303x384 gray image, 24 ancient Greek coins on a dark background, UNEVEN lighting.
  Used as a "hard" test for limitation 7.2 (try it with --flatten).
Run:  python tools/fetch_public.py
"""
import json
import sys
from pathlib import Path

import cv2
from skimage import data

OUT = Path(__file__).resolve().parents[1] / "data" / "public"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "skimage_coins.png"), data.coins())
    gt = {"skimage_coins.png": {"count": 24}}
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
