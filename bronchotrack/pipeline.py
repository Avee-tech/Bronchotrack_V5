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
from .geometry import FusionConfig, GeometricFusion, apply_fusion
from .motion import MotionConfig, MotionModel
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
    use_geometry: bool = False      # extension: fuse the roll-corrected diameter:distance cue
    fusion: FusionConfig = field(default_factory=FusionConfig)
    use_motion: bool = False        # version 3: speed-constrained motion model of the scope
    motion: MotionConfig = field(default_factory=MotionConfig)
    export_particles: int = 0       # particles to return per frame for drawing (0 = none)


@dataclass
class TrackOut:
    track_id: int
    box: np.ndarray
    score: float
    label: Optional[str]
    level: int
    activated: bool
    prob: Optional[float] = None    # fused label probability (geometry extension)
    assoc: Optional[tuple] = None   # (label, prob) from the association method alone
    ratio: Optional[tuple] = None   # (label, prob) from the diameter:distance ratio method alone


@dataclass
class FrameResult:
    t: int
    location: Optional[str]
    roll: float
    tracks: List[TrackOut]
    loop_closed: bool = False
    runtime_ms: Dict[str, float] = field(default_factory=dict)
    loc_prob: Optional[float] = None          # geometry extension
    assoc_location: Optional[str] = None      # association-only location (before fusion)
    evidence_location: Optional[str] = None   # per-frame Eq. (9) location before the motion model
    particles: Optional[np.ndarray] = None    # motion-model particle positions (standard frame)
    ratio_err_assoc: Optional[float] = None
    ratio_err_fused: Optional[float] = None


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
        self.fusion = GeometricFusion(graph, cfg.fusion) if (cfg.use_geometry and self.assoc is not None) else None
        self.motion = MotionModel(graph, cfg.motion) if (cfg.use_motion and self.assoc is not None) else None
        if self.motion is not None and self.fusion is not None and cfg.motion.use_label_age:
            self.fusion.cfg.use_label_age = True
        self._label_hist: Dict[int, tuple] = {}  # track id -> (label, frame it was assigned)
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
            was_init = self.assoc.initialized
            for n in S:  # inter-frame association: labels travel with tracklets
                n.label = by_det[n.det].label
                n.prev_label = n.label
                h = self._label_hist.get(n.track_id)
                n.label_age = (t - h[1]) if (h is not None and h[0] == n.label) else 0
            self.assoc.step(S, t, (W, H), frame if self.lc is not None else None)
            if self.motion is not None and self.assoc.initialized and not was_init:
                # the carina was just recognised: the tip is within view range of it
                self.motion.reset(self.M.root, last_mm=self.cfg.motion.init_range)
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

        # ---- extension: roll-corrected diameter:distance cue fused with the association
        fres = None
        assoc_loc = self.location
        prior = None
        if self.motion is not None:
            self.motion.predict()                 # where can the scope be now?
            prior = self.motion.reach_probs()     # ... and within the next second
        if self.fusion is not None and self.assoc.initialized:
            t0 = time.perf_counter()
            fres = self.fusion.fuse(S, self.assoc.loc, self.assoc.roll, loc_prior=prior,
                                    prior_weight=self.cfg.motion.prior_weight)
            if self.cfg.fusion.feedback:
                apply_fusion(S, fres)
                if fres.location is not None:
                    self.assoc.loc = fres.location
            rt["geometry"] = 1e3 * (time.perf_counter() - t0)
        evidence_loc = fres.location if fres is not None else self.location
        mloc, mprob = None, None
        if self.motion is not None:
            t0 = time.perf_counter()
            if self.assoc.initialized:
                votes = fres.votes if fres is not None else dict(self.assoc.last_votes)
            else:
                votes = {self.M.root: 1.0}
            self.motion.update(votes)
            mloc, mprob = self.motion.estimate()
            self.assoc.loc = mloc                  # feedback: gating / recovery next frame
            rt["motion"] = 1e3 * (time.perf_counter() - t0)

        if self.assoc is not None:
            # write refined labels back to the tracklets
            for n in S:
                by_det[n.det].label = n.label
                h = self._label_hist.get(n.track_id)
                if h is None or h[0] != n.label:
                    self._label_hist[n.track_id] = (n.label, t)

        levels = {n.det: n.level for n in S}
        out = []
        for d, tr in sorted(by_det.items()):
            if d not in levels:
                continue
            lab, prob, a_, r_ = tr.label, None, None, None
            if fres is not None and d in fres.labels:
                lab, prob = fres.labels[d]
                a_, r_ = fres.assoc.get(d), fres.ratio.get(d)
            elif self.assoc is not None and tr.label is not None:
                # original pipeline: the association's own confidence in the label (track age)
                a_ = (tr.label, 0.55 + 0.40 * (1 - np.exp(-max(tr.age(t), 0) / 10.0)))
            out.append(TrackOut(tr.track_id, tr.box.copy(), tr.score, lab, levels[d], tr.activated, prob, a_, r_))
        r = FrameResult(t, mloc if mloc is not None else evidence_loc,
                        0.0 if self.assoc is None else self.assoc.roll, out, loop, rt)
        r.assoc_location = assoc_loc
        r.evidence_location = evidence_loc
        if fres is not None:
            r.loc_prob, r.ratio_err_assoc, r.ratio_err_fused = fres.loc_prob, fres.ratio_err_assoc, fres.ratio_err_fused
        if mloc is not None:
            r.loc_prob = mprob
            if self.cfg.export_particles:
                r.particles = self.motion.positions(self.cfg.export_particles)
        return r


# --------------------------------------------------------------------------- #
def draw(frame: np.ndarray, res: FrameResult, show_roll: bool = True) -> np.ndarray:
    """Overlay in the style of Fig. 4: [Branch]-[Tracklet]-[confidence] (+ p=fused probability)."""
    img = frame.copy()
    k = max(0.4, img.shape[1] / 1000.0)            # text scale follows the frame size
    fs, th_ = 0.45 * k, max(1, int(round(k)))
    for tr in res.tracks:
        if not tr.activated:
            continue
        rng = np.random.default_rng(tr.track_id * 7919)
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        x1, y1, x2, y2 = tr.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, (3 if tr.level == 1 else 2) * th_)
        txt = f"{tr.label if tr.label is not None else '?'}-{tr.track_id}-{tr.score:.2f}"
        if tr.prob is not None:
            txt += f" F={tr.prob:.2f}"
        lines = [txt]
        if tr.assoc is not None or tr.ratio is not None:
            def fmt(tag, v):
                return f"{tag}: -" if v is None or v[0] is None else f"{tag}: {v[0]} {v[1]:.2f}"
            parts = [fmt("A", tr.assoc)]
            if tr.prob is not None:  # the ratio method only exists in the modified pipeline
                parts.append(fmt("R", tr.ratio))
            lines.append("  ".join(parts))
        sizes = [cv2.getTextSize(l_, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)[0] for l_ in lines]
        th = max(h_ for _, h_ in sizes)
        tw = max(w_ for w_, _ in sizes)
        lh = th + 6
        y0 = max(lh * len(lines), y1)
        cv2.rectangle(img, (x1, y0 - lh * len(lines)), (x1 + tw + 4, y0), col, -1)
        for li, l_ in enumerate(lines):
            cv2.putText(img, l_, (x1 + 2, y0 - lh * (len(lines) - 1 - li) - 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), th_)
    if res.location is not None:
        s = f"loc: {res.location}"
        if res.loc_prob is not None:
            s += f"  p={res.loc_prob:.2f}"
        if res.particles is not None and res.evidence_location is not None:
            s += f"  (evidence: {res.evidence_location})"
        if show_roll:
            s += f"  roll {np.degrees(res.roll):+.0f}deg"
        if res.loop_closed:
            s += "  [loop]"
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, 2 * th_)
        cv2.rectangle(img, (0, 0), (tw + 16, th + 18), (20, 20, 20), -1)
        cv2.putText(img, s, (8, th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, (255, 255, 255), 2 * th_)
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

    def render(self, loc: Optional[str], seen: List[str], particles: Optional[np.ndarray] = None) -> np.ndarray:
        img = self.base.copy()
        for l in seen:
            if l in self.g:
                cv2.polylines(img, [self._px(self.g[l])], False, (80, 220, 80), 2)
        if loc in self.g:
            cv2.polylines(img, [self._px(self.g[loc])], False, (40, 40, 255), 4)
            e = self.g[loc].end[:2] * self.s + self.off
            cv2.putText(img, loc, (int(e[0]) + 4, int(e[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 255), 1)
        if particles is not None and len(particles):  # motion model belief (version 3)
            q = (particles[:, :2] * self.s + self.off).astype(int)
            for x, y in q:
                if 0 <= x < self.size and 0 <= y < self.size:
                    cv2.circle(img, (int(x), int(y)), 2, (0, 200, 255), -1)
            cv2.putText(img, "motion model", (6, self.size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        return img


def draw_with_map(frame: np.ndarray, res: FrameResult, amap: AirwayMap) -> np.ndarray:
    img = draw(frame, res)
    inset = amap.render(res.location, [t.label for t in res.tracks if t.activated and t.label], res.particles)
    h = img.shape[0]
    if inset.shape[0] != h:
        k = h / inset.shape[0]
        inset = cv2.resize(inset, (int(inset.shape[1] * k), h))
    return np.concatenate([img, inset], 1)
