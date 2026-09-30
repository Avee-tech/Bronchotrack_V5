#!/usr/bin/env python3
"""Export lumen boxes to a YOLO dataset (images/ + labels/ + data.yaml), usable by YOLOv12
and YOLOv7 alike (training, or scoring with tools/compare_detectors.py). One class: ``lumen``.

Sources
  --bench DIR   runs made by tools/make_vb_benchmark.py (DIR/run*/frames/*.png + gt_boxes.csv);
                split by run (--val-runs), so validation frames come from unseen trajectories
  --seq FRAMES CSV   any frames folder (or video) + CSV ``frame,x1,y1,x2,y2[,...]`` (1-based
                frame index), repeatable; split by --val-frac of each sequence's frames

python tools/export_yolo_dataset.py --bench vb_bench --val-runs run03 run06 --every 2 --out yolo_lumen
"""

import argparse
import csv
import glob
import os
import random

import cv2


def iter_frames(src):
    if os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg"))):
            yield int(os.path.splitext(os.path.basename(f))[0]) if os.path.basename(f)[:-4].isdigit() else None, \
                cv2.imread(f)
    else:
        cap = cv2.VideoCapture(src)
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            k += 1
            yield k, fr


def read_boxes(csv_path):
    out = {}
    for r in csv.DictReader(open(csv_path)):
        out.setdefault(int(r["frame"]), []).append([float(r[c]) for c in ("x1", "y1", "x2", "y2")])
    return out


def write_item(out, split, name, img, boxes, grayscale):
    h, w = img.shape[:2]
    if grayscale:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(out, "labels", split, name + ".txt"), "w") as f:
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench")
    ap.add_argument("--val-runs", nargs="*", default=[])
    ap.add_argument("--seq", nargs=2, action="append", metavar=("FRAMES", "CSV"), default=[])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame (consecutive frames are near-duplicates)")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for sp in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", sp), exist_ok=True)
    n = {"train": 0, "val": 0}
    if a.bench:
        for rd in sorted(d for d in glob.glob(os.path.join(a.bench, "run*")) if os.path.isdir(d)):
            run = os.path.basename(rd)
            split = "val" if run in a.val_runs else "train"
            boxes = read_boxes(os.path.join(rd, "gt_boxes.csv"))
            for k, img in iter_frames(os.path.join(rd, "frames")):
                if k is None or (k - 1) % a.every:
                    continue
                write_item(a.out, split, f"{run}_{k:05d}", img, boxes.get(k, []), a.grayscale)
                n[split] += 1
    rnd = random.Random(0)
    for i, (src, csv_path) in enumerate(a.seq):
        boxes = read_boxes(csv_path)
        for k, img in iter_frames(src):
            if k is None or (k - 1) % a.every:
                continue
            split = "val" if rnd.random() < a.val_frac else "train"
            write_item(a.out, split, f"seq{i}_{k:05d}", img, boxes.get(k, []), a.grayscale)
            n[split] += 1
    root = os.path.abspath(a.out).replace("\\", "/")
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['lumen']\n")
    print(f"{n['train']} train / {n['val']} val images -> {os.path.join(a.out, 'data.yaml')}")


if __name__ == "__main__":
    main()
