"""Lumen detector (paper Sec. III-A).

**Default: YOLOv12** (Ultralytics ``yolo12{n,s,m,l,x}``) lumen weights. The paper used YOLOv7 (ref. [28]), trained on a single class ``lumen`` with
256x256 inputs, a detection threshold of 0.1 and an NMS IoU threshold of 0.6
(patient) / 0.7 (porcine) (Sec. IV-A). The relaxed IoU threshold keeps
overlapping parent/child lumen boxes (Fig. 3).

Three back-ends are provided behind a common ``__call__(frame) -> Detections``:

* ``YOLOv12Detector``       YOLOv12 checkpoint through Ultralytics (default)
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
        load = torch.load  # the official checkpoints are full pickles: torch>=2.6 needs weights_only=False
        torch.load = lambda *a, **k: load(*a, **{"weights_only": False, **k})
        try:
            self.model = attempt_load(weights, map_location=self.device).eval()
        finally:
            torch.load = load
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.img_size, self.conf_thr, self.iou_thr = img_size, conf_thr, iou_thr

    def __call__(self, frame: np.ndarray) -> Detections:
        torch = self.torch
        h0, w0 = frame.shape[:2]
        # letterbox as in YOLOv7 training (keep the aspect ratio, pad with grey 114)
        r = self.img_size / max(h0, w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        px, py = (self.img_size - nw) // 2, (self.img_size - nh) // 2
        img = np.full((self.img_size, self.img_size, 3), 114, np.uint8)
        img[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
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
        b = (b - np.array([px, py, px, py])) / r
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w0)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h0)
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


class YOLOv12Detector(UltralyticsDetector):
    """YOLOv12 lumen detector (Ultralytics >= 8.3.78; attention-centric YOLO with an NMS head).

    ``classes``: keep only these class ids (e.g. a lumen-only subset of a 2-class model);
    ``grayscale``: convert frames to grey before detection, for weights trained on grey images.
    """

    def __init__(self, weights: str, img_size: int = 640, conf_thr: float = 0.1, iou_thr: float = 0.6,
                 device: Optional[str] = None, classes: Optional[List[int]] = None, grayscale: bool = False,
                 strict: bool = False):
        super().__init__(weights, img_size, conf_thr, iou_thr, device, classes)
        self.grayscale = grayscale
        self.family = self._family()
        if self.family != "yolo12":
            msg = (f"{os.path.basename(str(weights))} looks like a {self.family or 'non-YOLOv12'} model, "
                   "not YOLOv12; running it anyway")
            if strict:
                raise ValueError(msg)
            print(f"[BronchoTrack] warning: {msg}")
        self.names = getattr(self.model, "names", {})

    def _family(self) -> Optional[str]:
        yml = getattr(getattr(self.model, "model", None), "yaml", {}) or {}
        name = str(yml.get("yaml_file", "")) + " " + str(getattr(self.model, "ckpt_path", "") or "")
        low = name.lower()
        for fam in ("yolov12", "yolo12", "yolo11", "yolov8", "yolov5", "yolo26", "yolov10", "rtdetr"):
            if fam in low:
                return "yolo12" if fam in ("yolov12", "yolo12") else fam
        # a checkpoint trained from yolo12*.yaml keeps its A2C2f (area-attention) blocks
        mods = {type(m).__name__ for m in getattr(self.model, "model", self.model).modules()}
        return "yolo12" if "A2C2f" in mods else None

    def __call__(self, frame: np.ndarray) -> Detections:
        if self.grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return super().__call__(frame)


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
    if kind in ("yolov12", "yolo12"):
        return YOLOv12Detector(weights, **kw)
    if kind in ("ultralytics", "yolo"):
        kw.pop("grayscale", None)
        return UltralyticsDetector(weights, **kw)
    if kind in ("mot", "annotation"):
        return AnnotationDetector.from_mot_txt(weights)
    raise ValueError(f"unknown detector kind {kind}")
