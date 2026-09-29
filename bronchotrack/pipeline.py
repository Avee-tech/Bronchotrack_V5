"""The full BronchoTrack / BronchoTrack-LC pipeline (paper Fig. 1)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .airway_graph import AirwayGraph
from .association import AirwayAssociator, AssociationConfig
from .detector import Detections
from .subgraph import build_subgraph
from .tracker import LumenTracker, TrackerConfig


@dataclass
class BronchoTrackConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    use_graph: bool = True          # ablation "w/o Graph"
    use_lc: bool = False            # BronchoTrack-LC
    lc_eta: int = 100               # matched keypoint pairs for a loop (Sec. III-D)
    lc_lambda: int = 1              # most-recent gallery records compared
    contain_thr: float = 0.8        # box inclusion ratio for the lumen hierarchy
    prune_iou: float = 0.6          # single-child IoU above which parent/child are duplicates


@dataclass
class TrackOut:
    track_id: int
    box: np.ndarray
    score: float
    label: Optional[str]
    level: int
    activated: bool


@dataclass
class FrameResult:
    t: int
    location: Optional[str]
    roll: float
    tracks: List[TrackOut]
    loop_closed: bool = False
    runtime_ms: Dict[str, float] = field(default_factory=dict)


class BronchoTrack:
    def __init__(self, detector, graph: Optional[AirwayGraph], reid=None,
                 cfg: Optional[BronchoTrackConfig] = None, matcher=None):
        cfg = cfg if cfg is not None else BronchoTrackConfig()
        self.detector = detector
        self.reid = reid
        self.M = graph
        self.cfg = cfg
        if reid is None:
            cfg.tracker.use_reid = False
        self.tracker = LumenTracker(cfg.tracker)
        self.assoc = AirwayAssociator(graph, cfg.association) if (graph is not None and cfg.use_graph) else None
        self.lc = None
        if cfg.use_lc and self.assoc is not None:
            from .loop_closure import LoopClosure
            self.lc = LoopClosure(matcher, eta=cfg.lc_eta, lam=cfg.lc_lambda)
        self.t = -1

    @property
    def location(self) -> Optional[str]:
        return None if self.assoc is None else self.assoc.loc

    def process(self, frame: np.ndarray, detections: Optional[Detections] = None) -> FrameResult:
        self.t += 1
        t = self.t
        H, W = frame.shape[:2]
        rt: Dict[str, float] = {}

        # ---- detection (Sec. III-A)
        t0 = time.perf_counter()
        dets = detections if detections is not None else self.detector(frame)
        rt["detection"] = 1e3 * (time.perf_counter() - t0)

        # ---- Re-ID embeddings (Sec. III-B-2)
        t0 = time.perf_counter()
        embs = self.reid(frame, dets.boxes) if (self.reid is not None and self.cfg.tracker.use_reid and len(dets)) else None
        rt["reid"] = 1e3 * (time.perf_counter() - t0)

        # ---- multi-lumen tracking (Sec. III-B)
        t0 = time.perf_counter()
        prev_loc = self.location
        self.tracker.update(dets.boxes, dets.scores, embs, t,
                            graph=self.M if self.assoc is not None else None, prev_loc=prev_loc)
        rt["tracking"] = 1e3 * (time.perf_counter() - t0)

        # detections that belong to a tracklet in this frame (others = background)
        by_det = {tr.det_index: tr for tr in self.tracker.tracks if tr.t_end == t and tr.det_index >= 0}
        dets_idx = sorted(by_det)
        S = build_subgraph(dets.boxes[dets_idx] if dets_idx else np.zeros((0, 4)),
                           dets.scores[dets_idx] if dets_idx else np.zeros(0),
                           contain_thr=self.cfg.contain_thr, prune_iou=self.cfg.prune_iou,
                           track_ids=[by_det[d].track_id for d in dets_idx],
                           track_ages=[by_det[d].age(t) for d in dets_idx], dets=dets_idx)

        # ---- airway graph association (Sec. III-C)
        loop = False
        t0 = time.perf_counter()
        if self.assoc is not None:
            for n in S:  # inter-frame association: labels travel with tracklets
                n.label = by_det[n.det].label
            self.assoc.step(S, t, (W, H), frame if self.lc is not None else None)
            rt["graph"] = 1e3 * (time.perf_counter() - t0)

            # ---- adapted loop closure (Sec. III-D)
            if self.lc is not None and self.assoc.new_branch_entered and len(self.assoc.gallery) > 1:
                t0 = time.perf_counter()
                res = self.lc.search(self.assoc, S, frame, t)
                if res.detected:
                    loop = True
                    self.assoc._intra_frame(S)
                    self.assoc.loc = self.assoc._vote(S) or res.record_label
                    self.assoc.gallery[res.record_label].updated = t
                rt["lc"] = 1e3 * (time.perf_counter() - t0)

            # write refined labels back to the tracklets
            for n in S:
                by_det[n.det].label = n.label

        levels = {n.det: n.level for n in S}
        out = [TrackOut(tr.track_id, tr.box.copy(), tr.score, tr.label, levels.get(d, 1), tr.activated)
               for d, tr in sorted(by_det.items()) if d in levels]
        return FrameResult(t, self.location, 0.0 if self.assoc is None else self.assoc.roll, out, loop, rt)


# --------------------------------------------------------------------------- #
def draw(frame: np.ndarray, res: FrameResult, show_roll: bool = True) -> np.ndarray:
    """Overlay in the style of Fig. 4: [Branch]-[Tracklet]-[confidence]."""
    img = frame.copy()
    for tr in res.tracks:
        if not tr.activated:
            continue
        rng = np.random.default_rng(tr.track_id * 7919)
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        x1, y1, x2, y2 = tr.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if tr.level == 1 else 1)
        txt = f"{tr.label if tr.label is not None else '?'}-{tr.track_id}-{tr.score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, max(th + 4, y1)), col, -1)
        cv2.putText(img, txt, (x1 + 1, max(th + 1, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    if res.location is not None:
        s = f"loc: {res.location}"
        if show_roll:
            s += f"  roll: {np.degrees(res.roll):+.0f}deg"
        if res.loop_closed:
            s += "  [loop]"
        cv2.putText(img, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


class AirwayMap:
    """Small 2-D map of the airway graph (x/y of the standard frame) for overlays:
    current branch in red, branches of the labelled lumens in green."""

    def __init__(self, graph: AirwayGraph, size: int = 260, margin: int = 12):
        self.g, self.size = graph, size
        pts = np.concatenate([(b.points if b.points is not None else np.stack([b.start, b.end]))[:, :2]
                              for b in graph.branches.values()])
        lo, hi = pts.min(0), pts.max(0)
        self.s = (size - 2 * margin) / max(hi - lo)
        self.off = margin - lo * self.s + (size - 2 * margin - (hi - lo) * self.s) / 2
        self.base = np.full((size, size, 3), 30, np.uint8)
        for b in graph.branches.values():
            cv2.polylines(self.base, [self._px(b)], False, (150, 150, 150), max(1, 3 - b.generation // 2))
        # standard frame: +x points to the patient's right main bronchus
        cv2.putText(self.base, "R", (size - 18, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(self.base, "L", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def _px(self, b):
        p = (b.points if b.points is not None else np.stack([b.start, b.end]))[:, :2]
        return (p * self.s + self.off).astype(np.int32).reshape(-1, 1, 2)

    def render(self, loc: Optional[str], seen: List[str]) -> np.ndarray:
        img = self.base.copy()
        for l in seen:
            if l in self.g:
                cv2.polylines(img, [self._px(self.g[l])], False, (80, 220, 80), 2)
        if loc in self.g:
            cv2.polylines(img, [self._px(self.g[loc])], False, (40, 40, 255), 4)
            e = self.g[loc].end[:2] * self.s + self.off
            cv2.putText(img, loc, (int(e[0]) + 4, int(e[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 255), 1)
        return img


def draw_with_map(frame: np.ndarray, res: FrameResult, amap: AirwayMap) -> np.ndarray:
    img = draw(frame, res)
    inset = amap.render(res.location, [t.label for t in res.tracks if t.activated and t.label])
    h = img.shape[0]
    if inset.shape[0] != h:
        k = h / inset.shape[0]
        inset = cv2.resize(inset, (int(inset.shape[1] * k), h))
    return np.concatenate([img, inset], 1)
