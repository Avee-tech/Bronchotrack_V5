#!/usr/bin/env python3
"""Compare lumen detectors on the same labelled images with one metric implementation
(so YOLOv12 and YOLOv7 are scored identically, independent of each framework's own val code).

Reports AP50, AP50-95 (COCO-style, 101-point), precision / recall / F1 at a fixed confidence,
recall at IoU 0.3 (loose match, as for lumen openings) and CPU/GPU time per frame.

python tools/compare_detectors.py --data yolo_lumen --split val \
    --det yolov12 runs/lumen_yolo12n/weights/best.pt 640 \
    --det yolov7  yolov7/runs/train/lumen_y7tiny/weights/best.pt 640 --yolov7-repo yolov7
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.detector import build_detector  # noqa: E402


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred, scores, gt, thr):
    """Greedy (by score) one-to-one matching; returns a TP flag per prediction."""
    order = np.argsort(-scores)
    tp = np.zeros(len(pred), bool)
    if len(gt) == 0:
        return tp
    M = iou_matrix(pred[order], gt)
    used = np.zeros(len(gt), bool)
    for i in range(len(order)):
        j = np.argmax(np.where(used, -1, M[i]))
        if M[i, j] >= thr and not used[j]:
            used[j] = True
            tp[order[i]] = True
    return tp


def average_precision(tp, scores, n_gt):
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-scores)
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side="left")
        ap += prec[k] if k < len(prec) else 0.0
    return ap / 101


def load_items(root, split):
    items = []
    for im in sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")) +
                     glob.glob(os.path.join(root, "images", split, "*.png"))):
        lab = os.path.join(root, "labels", split, os.path.splitext(os.path.basename(im))[0] + ".txt")
        img = cv2.imread(im)
        h, w = img.shape[:2]
        g = np.loadtxt(lab, ndmin=2) if os.path.exists(lab) and os.path.getsize(lab) else np.zeros((0, 5))
        b = np.stack([(g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h,
                      (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h], 1) if len(g) else np.zeros((0, 4))
        items.append((im, img, b))
    return items


def evaluate(det, items, conf_op=0.25, warmup=3):
    for _, img, _ in items[:warmup]:
        det(img)
    thr = np.round(np.arange(0.5, 0.96, 0.05), 2)
    per_thr = {t: [] for t in thr}
    loose, scores_all, n_gt, t_tot = [], [], 0, 0.0
    for _, img, gt in items:
        t0 = time.perf_counter()
        d = det(img)
        t_tot += time.perf_counter() - t0
        n_gt += len(gt)
        scores_all.append(d.scores)
        for t in thr:
            per_thr[t].append(match(d.boxes, d.scores, gt, t))
        loose.append(match(d.boxes, d.scores, gt, 0.3))
    S = np.concatenate(scores_all) if scores_all else np.zeros(0)
    aps = {t: average_precision(np.concatenate(per_thr[t]), S, n_gt) for t in thr}
    tp50 = np.concatenate(per_thr[0.5])
    keep = S >= conf_op
    tp_c = int(tp50[keep].sum())
    P = tp_c / max(int(keep.sum()), 1)
    R = tp_c / max(n_gt, 1)
    R30 = int(np.concatenate(loose)[keep].sum()) / max(n_gt, 1)
    return {"AP50": aps[0.5], "AP75": aps[0.75], "AP50_95": float(np.mean(list(aps.values()))),
            "P": P, "R": R, "F1": 2 * P * R / max(P + R, 1e-9), "R_iou30": R30,
            "boxes_per_frame": float(keep.sum()) / len(items), "gt_per_frame": n_gt / len(items),
            "ms_per_frame": 1000 * t_tot / len(items), "n_images": len(items), "conf": conf_op}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="YOLO dataset root (images/<split>, labels/<split>)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--det", nargs=3, action="append", metavar=("KIND", "WEIGHTS", "IMGSZ"), required=True,
                    help="yolov12 / yolov7 / ultralytics, weights, input size; repeatable")
    ap.add_argument("--name", action="append", help="display name per --det (default: kind)")
    ap.add_argument("--yolov7-repo", default=os.environ.get("YOLOV7_REPO", "yolov7"))
    ap.add_argument("--conf", type=float, default=0.25, help="operating confidence for P/R/F1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="save the table as JSON")
    a = ap.parse_args()
    items = load_items(a.data, a.split)
    print(f"{len(items)} images, {sum(len(b) for _, _, b in items)} boxes")
    res = {}
    for i, (kind, w, sz) in enumerate(a.det):
        name = a.name[i] if a.name and i < len(a.name) else kind
        extra = {"repo": a.yolov7_repo} if kind == "yolov7" else {}
        det = build_detector(kind, w, img_size=int(sz), conf_thr=0.001, iou_thr=0.6, device=a.device, **extra)
        res[name] = evaluate(det, items, a.conf)
        res[name].update(weights=w, imgsz=int(sz))
    cols = ["AP50", "AP75", "AP50_95", "P", "R", "F1", "R_iou30", "boxes_per_frame", "ms_per_frame"]
    print(f"{'detector':24s}" + "".join(f"{c:>13s}" for c in cols))
    for n, r in res.items():
        print(f"{n:24s}" + "".join(f"{r[c]:13.3f}" for c in cols))
    print(f"(P/R/F1/R_iou30 at conf >= {a.conf}; gt per frame {next(iter(res.values()))['gt_per_frame']:.2f})")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
