#!/usr/bin/env python3
"""Build the Re-ID crop dataset from labelled frames.

Annotations: CSV ``frame,x1,y1,x2,y2,label`` per sequence (``label`` = anatomical
branch name/label of the lumen, as in the paper's annotation, Sec. IV-A). Frames:
a folder of images sorted by name (``frame`` = 1-based index) or a video.

Each (sequence, label) pair is one identity -> <out>/<seq>_<label>/*.png, 128x128.

python tools/make_reid_crops.py --seq case01 frames/case01 ann/case01.csv \
                                --seq case02 case02.mp4 ann/case02.csv --out reid_crops
"""

import argparse
import csv
import os

import cv2


def iter_frames(src):
    if os.path.isdir(src):
        for f in sorted(os.listdir(src)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                yield cv2.imread(os.path.join(src, f))
    else:
        cap = cv2.VideoCapture(src)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs=3, action="append", metavar=("NAME", "FRAMES", "CSV"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--min-side", type=int, default=6)
    a = ap.parse_args()
    n = 0
    for name, src, ann in a.seq:
        rows = {}
        for r in csv.DictReader(open(ann)):
            rows.setdefault(int(r["frame"]), []).append(r)
        for i, fr in enumerate(iter_frames(src), start=1):
            for k, r in enumerate(rows.get(i, [])):
                x1, y1, x2, y2 = (int(round(float(r[c]))) for c in ("x1", "y1", "x2", "y2"))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fr.shape[1], x2), min(fr.shape[0], y2)
                if x2 - x1 < a.min_side or y2 - y1 < a.min_side:
                    continue
                d = os.path.join(a.out, f"{name}_{r['label']}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{i:06d}_{k}.png"),
                            cv2.resize(fr[y1:y2, x1:x2], (a.size, a.size)))
                n += 1
    print(f"{n} crops -> {a.out}")


if __name__ == "__main__":
    main()
