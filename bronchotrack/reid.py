"""Re-ID appearance model (paper Sec. III-B-2 and Sec. IV-A).

* ResNet50 backbone trained as an image classifier with softmax loss; each
  physical lumen is one class (3630 crops, resized to 128x128).
* At inference the final FC layer is dropped and the pooled CNN feature is the
  appearance embedding (L2-normalised so that Eq. (4) is a cosine distance).
* Tracklet embeddings are updated with an exponential moving average, Eq. (3),
  momentum alpha = 0.9.

``HistogramEmbedder`` is a torch-free fallback (HSV histogram) that is *not* part
of the paper; it only exists so the pipeline can run on machines without torch.
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

REID_SIZE = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_boxes(frame: np.ndarray, boxes: np.ndarray, size: int = REID_SIZE) -> np.ndarray:
    """Crop each xyxy box and resize to ``size`` x ``size`` (RGB, float32 in [0,1])."""
    H, W = frame.shape[:2]
    out = np.zeros((len(boxes), size, size, 3), np.float32)
    for i, b in enumerate(np.asarray(boxes)):
        x1, y1 = int(np.clip(np.floor(b[0]), 0, W - 1)), int(np.clip(np.floor(b[1]), 0, H - 1))
        x2, y2 = int(np.clip(np.ceil(b[2]), x1 + 1, W)), int(np.clip(np.ceil(b[3]), y1 + 1, H))
        crop = cv2.resize(frame[y1:y2, x1:x2], (size, size))
        out[i] = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    return out


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def ema_update(e_prev: Optional[np.ndarray], f: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Eq. (3): e_t = alpha * e_{t-1} + (1 - alpha) * f_t   (re-normalised)."""
    f = l2norm(f)
    if e_prev is None:
        return f
    return l2norm(alpha * e_prev + (1 - alpha) * f)


def appearance_cost(track_emb: np.ndarray, det_emb: np.ndarray) -> np.ndarray:
    """Eq. (4): C_a(i,j) = 1 - f_j^T e_i   ->   matrix (n_tracks, n_dets), clipped to [0,1]."""
    if len(track_emb) == 0 or len(det_emb) == 0:
        return np.ones((len(track_emb), len(det_emb)))
    return np.clip(1.0 - l2norm(track_emb) @ l2norm(det_emb).T, 0.0, 1.0)


# --------------------------------------------------------------------------- #
class ResNet50ReID:
    """ResNet50 appearance embedder (2048-d pooled feature)."""

    def __init__(self, weights: Optional[str] = None, device: str = "cuda", num_classes: int = 0,
                 imagenet_init: bool = True):
        import torch
        import torchvision

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        init = None
        if imagenet_init and weights is None:
            try:
                init = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            except Exception:  # offline
                init = None
        try:
            net = torchvision.models.resnet50(weights=init)
        except Exception:
            net = torchvision.models.resnet50(weights=None)
        net.fc = torch.nn.Linear(2048, max(1, num_classes))
        if weights is not None and os.path.exists(weights):
            sd = torch.load(weights, map_location="cpu")
            sd = sd.get("state_dict", sd)
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}  # FC dropped at inference
            net.load_state_dict(sd, strict=False)
        net.fc = torch.nn.Identity()
        self.net = net.to(self.device).eval()

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 2048), np.float32)
        torch = self.torch
        x = (crop_boxes(frame, boxes) - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device)
        with torch.no_grad():
            f = self.net(x).float().cpu().numpy()
        return l2norm(f)


class HistogramEmbedder:
    """Torch-free fallback: normalised HSV colour histogram (NOT the paper's model)."""

    def __init__(self, bins=(8, 8, 4)):
        self.bins = bins

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        crops = (crop_boxes(frame, boxes, 64) * 255).astype(np.uint8)
        feats: List[np.ndarray] = []
        for c in crops:
            hsv = cv2.cvtColor(c[:, :, ::-1], cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]).ravel()
            feats.append(np.sqrt(h))
        return l2norm(np.asarray(feats, np.float32)) if feats else np.zeros((0, int(np.prod(self.bins))), np.float32)


def build_reid(weights: Optional[str] = None, device: str = "cuda", backend: str = "auto"):
    if backend == "hist":
        return HistogramEmbedder()
    try:
        return ResNet50ReID(weights, device=device)
    except ImportError:
        if backend == "resnet50":
            raise
        print("[BronchoTrack] torch not available -> HistogramEmbedder fallback for Re-ID")
        return HistogramEmbedder()
