"""Adapted loop closure for tracking recovery – BronchoTrack-LC (paper Sec. III-D).

* The airway gallery G is extended with image keyframes.
* When the bronchoscope enters a new branch, the frame becomes a keyframe and a
  loop is searched for against the lambda most recently updated gallery records
  (lambda = 1).
* Matching uses LoFTR (ref. [39]) dense matching; a loop is detected when more
  than eta matched keypoint pairs are found (eta = 100).
* On a loop, tracklet associations between the matched keyframe and the current
  frame are recomputed; otherwise the new keyframe is inserted into G.

``SIFTMatcher`` is a fallback (not in the paper) for machines without kornia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

LC_SIZE = 256


def _gray(img: np.ndarray, size: int = LC_SIZE) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (size, size))


class LoFTRMatcher:
    def __init__(self, pretrained: str = "indoor_new", device: str = "cuda", conf_thr: float = 0.5):
        import torch
        import kornia.feature as KF

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = KF.LoFTR(pretrained=pretrained).to(self.device).eval()
        self.conf_thr = conf_thr

    def __call__(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        ta = torch.from_numpy(_gray(a)).float()[None, None].to(self.device) / 255.0
        tb = torch.from_numpy(_gray(b)).float()[None, None].to(self.device) / 255.0
        with torch.no_grad():
            out = self.model({"image0": ta, "image1": tb})
        m = out["confidence"] >= self.conf_thr
        return out["keypoints0"][m].cpu().numpy(), out["keypoints1"][m].cpu().numpy()


class SIFTMatcher:
    """Fallback matcher (NOT the paper's LoFTR)."""

    def __init__(self, ratio: float = 0.8):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.ratio = ratio

    def __call__(self, a, b):
        ga, gb = _gray(a), _gray(b)
        ka, da = self.sift.detectAndCompute(ga, None)
        kb, db = self.sift.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return np.zeros((0, 2)), np.zeros((0, 2))
        good = [m for m, n in self.bf.knnMatch(da, db, k=2) if m.distance < self.ratio * n.distance]
        return (np.array([ka[m.queryIdx].pt for m in good]).reshape(-1, 2),
                np.array([kb[m.trainIdx].pt for m in good]).reshape(-1, 2))


def build_matcher(kind: str = "auto", device: str = "cuda"):
    if kind in ("auto", "loftr"):
        try:
            return LoFTRMatcher(device=device)
        except Exception as e:  # missing kornia / weights offline
            if kind == "loftr":
                raise
            print(f"[BronchoTrack-LC] LoFTR unavailable ({type(e).__name__}: {e}) -> SIFT fallback")
    return SIFTMatcher()


@dataclass
class LoopResult:
    detected: bool
    record_label: Optional[str] = None
    n_matches: int = 0


class LoopClosure:
    def __init__(self, matcher=None, eta: int = 100, lam: int = 1):
        self.matcher = matcher if matcher is not None else build_matcher()
        self.eta, self.lam = eta, lam

    def search(self, assoc, S, frame: np.ndarray, t: int) -> LoopResult:
        """Called when ``assoc.new_branch_entered``: compare the new keyframe with
        the ``lam`` most recently updated gallery records."""
        new_label = assoc.loc
        recs = [r for r in assoc.gallery.values() if r.label != new_label and r.keyframe is not None]
        recs.sort(key=lambda r: -r.updated)
        best = None
        for rec in recs[: self.lam]:
            k0, k1 = self.matcher(rec.keyframe, frame)
            if len(k0) > self.eta and (best is None or len(k0) > best[1]):
                best = (rec, len(k0), k0, k1)
        if best is None:
            return LoopResult(False)  # new keyframe stays in G (inserted by the associator)
        rec, n, k0, k1 = best
        self._reassociate(assoc, rec, S, frame, k0, k1)
        return LoopResult(True, rec.label, n)

    @staticmethod
    def _reassociate(assoc, rec, S, frame, k0, k1):
        """Recompute tracklet associations between the matched keyframe and the
        current frame: warp the keyframe's labelled lumens into the current frame
        with the geometry from the dense matches and re-assign labels."""
        H0, W0 = rec.keyframe.shape[:2]
        H1, W1 = frame.shape[:2]
        p0 = k0 * np.array([W0 / LC_SIZE, H0 / LC_SIZE])
        p1 = k1 * np.array([W1 / LC_SIZE, H1 / LC_SIZE])
        A, _ = cv2.estimateAffinePartial2D(p0.astype(np.float32), p1.astype(np.float32),
                                           method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if A is None:
            A = np.array([[1, 0, 0], [0, 1, 0]], float)
        items = [(tid, c, rec.labels.get(tid)) for tid, c in rec.centers.items() if rec.labels.get(tid)]
        nodes = list(S)
        # the new branch record was a false branch transition -> drop it
        assoc.gallery.pop(assoc.loc, None)
        if items and nodes:
            src = np.array([c for _, c, _ in items])
            warped = src @ A[:, :2].T + A[:, 2]
            cur = np.array([n.center for n in nodes])
            cost = np.linalg.norm(warped[:, None] - cur[None], axis=2)
            diag = np.hypot(W1, H1)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] < 0.25 * diag:
                    nodes[j].label = items[i][2]
        # roll follows the in-plane rotation of the similarity transform
        dtheta = float(np.arctan2(A[1, 0], A[0, 0]))
        assoc.roll = float((rec.roll + dtheta + np.pi) % (2 * np.pi) - np.pi)
        assoc.loc = rec.label
        rec.updated = max(rec.updated, max((r_.updated for r_ in assoc.gallery.values()), default=0)) + 1
