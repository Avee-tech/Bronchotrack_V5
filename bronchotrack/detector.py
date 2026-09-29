"""Lumen detector (paper Sec. III-A).

The paper uses YOLOv7 (ref. [28]) trained on a single class ``lumen`` with
256x256 inputs, a detection threshold of 0.1 and an NMS IoU threshold of 0.6
(patient) / 0.7 (porcine) (Sec. IV-A). The relaxed IoU threshold keeps
overlapping parent/child lumen boxes (Fig. 3).

Three back-ends are provided behind a common ``__call__(frame) -> Detections``:

* ``YOLOv7Detector``        official WongKinYiu/yolov7 checkpoint (paper setting)
* ``UltralyticsDetector``   any ultralytics YOLO checkpoint (v5/v8/11 ...)
* ``AnnotationDetector``    replays ground-truth / pre-computed boxes (evaluation)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray   # (N,4) xyxy in original-frame pixels
    scores: np.ndarray  # (N,)

    def __len__(self):
        return len(self.scores)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain NMS; returns kept indices sorted by score."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a + b - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, int)


class YOLOv7Detector:
    """Official YOLOv7 checkpoint (https://github.com/WongKinYiu/yolov7).

    ``repo`` must point to a clone of the yolov7 repository (its ``models``
    package is needed to unpickle the checkpoint).
    """

    def __init__(self, weights: str, repo: str, img_size: int = 256, conf_thr: float = 0.1,
                 iou_thr: float = 0.6, device: str = "cuda"):
        import torch

        if repo not in sys.path:
            sys.path.insert(0, repo)
        from models.experimental import attempt_load  # type: ignore

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(weights, map_location=self.device).eval()
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.img_size, self.conf_thr, self.iou_thr = img_size, conf_thr, iou_thr

    def __call__(self, frame: np.ndarray) -> Detections:
        torch = self.torch
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (self.img_size, self.img_size))
        x = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).to(self.device)
        x = (x.half() if self.half else x.float()) / 255.0
        with torch.no_grad():
            pred = self.model(x[None])[0][0].float().cpu().numpy()  # (N, 5+nc) cx,cy,w,h,obj,cls..
        conf = pred[:, 4] * pred[:, 5:].max(1)
        m = conf >= self.conf_thr
        pred, conf = pred[m], conf[m]
        if len(pred) == 0:
            return Detections.empty()
        b = np.stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                      pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2], 1)
        keep = nms(b, conf, self.iou_thr)
        b, conf = b[keep], conf[keep]
        b *= np.array([w0 / self.img_size, h0 / self.img_size] * 2)
        return Detections(b.astype(np.float32), conf.astype(np.float32))


class UltralyticsDetector:
    """Any ultralytics checkpoint (drop-in alternative to YOLOv7)."""

    def __init__(self, weights: str, img_size: int = 256, conf_thr: float = 0.1, iou_thr: float = 0.6,
                 device: Optional[str] = None, classes: Optional[List[int]] = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.kw = dict(imgsz=img_size, conf=conf_thr, iou=iou_thr, verbose=False, classes=classes)
        if device is not None:
            self.kw["device"] = device

    def __call__(self, frame: np.ndarray) -> Detections:
        r = self.model.predict(frame, **self.kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Detections.empty()
        return Detections(r.boxes.xyxy.cpu().numpy().astype(np.float32),
                          r.boxes.conf.cpu().numpy().astype(np.float32))


class AnnotationDetector:
    """Replays boxes from a dict ``frame_idx -> [(x1,y1,x2,y2,score), ...]``.

    Useful to evaluate tracking/association in isolation from the detector, or to
    run on detections cached from an external model.
    """

    def __init__(self, table: Dict[int, List], jitter: float = 0.0, drop: float = 0.0, seed: int = 0):
        self.table, self.t = table, 0
        self.jitter, self.drop = jitter, drop
        self.rng = np.random.default_rng(seed)

    def __call__(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Detections:
        t = self.t if frame_idx is None else frame_idx
        self.t = t + 1
        rows = self.table.get(t, [])
        if not rows:
            return Detections.empty()
        a = np.asarray(rows, np.float32)
        if a.shape[1] == 4:
            a = np.concatenate([a, np.ones((len(a), 1), np.float32)], 1)
        if self.drop > 0:
            a = a[self.rng.random(len(a)) >= self.drop]
        if self.jitter > 0 and len(a):
            wh = np.stack([a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]] * 2, 1)
            a[:, :4] += self.rng.normal(0, self.jitter, a[:, :4].shape) * wh
        return Detections(a[:, :4], a[:, 4])

    @staticmethod
    def from_mot_txt(path: str) -> "AnnotationDetector":
        """MOT-style ``frame,id,x,y,w,h,score,...`` (frame 1-based)."""
        table: Dict[int, List] = {}
        for line in open(path):
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            f, x, y, w, h = int(float(p[0])) - 1, *map(float, p[2:6])
            s = float(p[6]) if len(p) > 6 and float(p[6]) >= 0 else 1.0
            table.setdefault(f, []).append((x, y, x + w, y + h, s))
        return AnnotationDetector(table)


def build_detector(kind: str, weights: Optional[str] = None, **kw):
    kind = kind.lower()
    if kind == "yolov7":
        repo = kw.pop("repo", os.environ.get("YOLOV7_REPO", "yolov7"))
        return YOLOv7Detector(weights, repo=repo, **kw)
    if kind in ("ultralytics", "yolo"):
        return UltralyticsDetector(weights, **kw)
    if kind in ("mot", "annotation"):
        return AnnotationDetector.from_mot_txt(weights)
    raise ValueError(f"unknown detector kind {kind}")
