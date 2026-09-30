#!/usr/bin/env python3
"""Run the lumen detector once over a video and save MOT-style detections
(frame,-1,x,y,w,h,score) so tracking/association variants can be re-run fast
with  tools/run.py --detector mot --weights <this file>."""
import argparse, os, sys
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.detector import build_detector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--detector", default="yolov12", choices=["yolov12", "ultralytics", "yolov7"])
ap.add_argument("--weights", required=True)
ap.add_argument("--yolov7-repo", default="yolov7")
ap.add_argument("--img-size", type=int, default=640)
ap.add_argument("--classes", type=int, nargs="*")
ap.add_argument("--grayscale", action="store_true")
ap.add_argument("--conf", type=float, default=0.1)
ap.add_argument("--nms-iou", type=float, default=0.6)
ap.add_argument("--out", required=True)
a = ap.parse_args()
kw = dict(conf_thr=a.conf, iou_thr=a.nms_iou, img_size=a.img_size)
extra = {"repo": a.yolov7_repo} if a.detector == "yolov7" else {"classes": a.classes}
if a.detector == "yolov12":
    extra["grayscale"] = a.grayscale
det = build_detector(a.detector, a.weights, **extra, **kw)
cap = cv2.VideoCapture(a.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with open(a.out, "w") as f:
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        d = det(fr)
        for (x1, y1, x2, y2), s in zip(d.boxes, d.scores):
            f.write(f"{i},-1,{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{s:.4f}\n")
        if i % 100 == 0:
            print(f"{i}/{n}", flush=True)
print("done", a.out)
