"""Multi-lumen tracker (paper Sec. III-B, thresholds from Sec. IV-A).

Association follows the ByteTrack-style two-stage scheme the paper modifies
(ref. [34]):

1. Detections are split into high- and low-confidence candidates.
2. Currently tracked *and* temporarily lost tracklets are candidates; those whose
   airway label is more than three generations away from the previous
   bronchoscope location are filtered out.
3. High-confidence detections are matched with the fused cost of Eq. (5),
       C = lambda * C_a + (1 - lambda) * C_m,  lambda = 0.5,
   with C_m = 1 - IoU (Eq. 2, KF-predicted boxes) and C_a = 1 - f^T e (Eq. 4).
   High-cost pairs are gated (matching threshold 0.4) and the rest solved by the
   Hungarian algorithm.
4. Low-confidence detections + unmatched high-confidence detections are matched
   to the unmatched tracklets with motion distance only; threshold 0.7 if stage 1
   produced matches, otherwise 0.9.
5. Unmatched high-confidence detections start new tracklets; unmatched low
   confidence detections are discarded as background.
6. Lost tracklets are removed after 50 frames without a match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanFilter7, iou_matrix, xyha_to_xyxy, xyxy_to_xyha
from .reid import appearance_cost, ema_update


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass(eq=False)
class Tracklet:
    """T = {ID, {t_start, t_end}, {b_start, ..., b_end}}  (Eq. 1) + KF state, embedding, label."""

    track_id: int
    t_start: int
    t_end: int
    box: np.ndarray                 # last observed / updated xyxy
    score: float
    mean: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    emb: Optional[np.ndarray] = None
    state: TrackState = TrackState.New
    activated: bool = False
    label: Optional[str] = None     # anatomical branch label propagated over time
    boxes: Dict[int, np.ndarray] = field(default_factory=dict)  # {b_start, ..., b_end}
    det_index: int = -1             # index of the matched detection in the current frame
    hits: int = 0

    @property
    def pred_box(self) -> np.ndarray:
        if self.mean is None:
            return self.box
        return xyha_to_xyxy(self.mean[:4])

    @property
    def center(self) -> np.ndarray:
        b = self.box
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    def age(self, t: int) -> int:
        return t - self.t_start


@dataclass
class TrackerConfig:
    det_thresh: float = 0.1          # detection threshold (Sec. IV-A)
    high_thresh: float = 0.5         # high/low split (ByteTrack default; not stated in paper)
    new_track_thresh: float = 0.5    # min score to spawn a tracklet (= high_thresh)
    match_thresh_high: float = 0.4   # stage-1 gate (Sec. IV-A)
    match_thresh_low: float = 0.7    # stage-2 gate if stage 1 matched something
    match_thresh_low_empty: float = 0.9  # stage-2 gate otherwise
    unconfirmed_thresh: float = 0.7  # ByteTrack: gate for tracks seen once
    lam: float = 0.5                 # lambda in Eq. (5)
    ema_alpha: float = 0.9           # momentum in Eq. (3)
    track_buffer: int = 50           # frames a lost tracklet is kept (Sec. III-B)
    label_gate_generations: int = 3  # filter tracklets > 3 generations from location
    use_kf: bool = True              # ablation "w/o KF"
    use_reid: bool = True            # ablation "w/o Re-ID"
    history: int = 300               # boxes kept per tracklet


def _assign(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = np.where(cost > thresh, thresh + 1e4, cost)
    r, k = linear_sum_assignment(c)
    matches = [(i, j) for i, j in zip(r, k) if cost[i, j] <= thresh]
    mi = {i for i, _ in matches}
    mj = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in mi], [j for j in range(cost.shape[1]) if j not in mj]


class LumenTracker:
    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg if cfg is not None else TrackerConfig()
        self.kf = KalmanFilter7()
        self.tracks: List[Tracklet] = []
        self.removed: List[Tracklet] = []
        self._next_id = 1
        self.frame = -1

    # --------------------------------------------------------------- helpers
    def _new(self, box, score, emb, t) -> Tracklet:
        tr = Tracklet(self._next_id, t, t, np.asarray(box, float), float(score))
        self._next_id += 1
        if self.cfg.use_kf:
            tr.mean, tr.cov = self.kf.initiate(xyxy_to_xyha(tr.box))
        tr.emb = None if emb is None else ema_update(None, emb)
        tr.boxes[t] = tr.box.copy()
        tr.hits = 1
        return tr

    def _update(self, tr: Tracklet, box, score, emb, t, det_idx):
        box = np.asarray(box, float)
        if self.cfg.use_kf:
            tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, xyxy_to_xyha(box))
        if emb is not None:
            tr.emb = ema_update(tr.emb, emb, self.cfg.ema_alpha)
        tr.box, tr.score, tr.t_end, tr.det_index = box, float(score), t, det_idx
        tr.boxes[t] = box.copy()
        if len(tr.boxes) > self.cfg.history:
            tr.boxes.pop(min(tr.boxes))
        tr.state, tr.activated = TrackState.Tracked, True
        tr.hits += 1

    def _motion_cost(self, tracks: List[Tracklet], boxes: np.ndarray) -> np.ndarray:
        if not tracks or len(boxes) == 0:
            return np.ones((len(tracks), len(boxes)))
        pb = np.stack([t.pred_box for t in tracks])
        return 1.0 - iou_matrix(pb, boxes)                            # Eq. (2)

    def _fused_cost(self, tracks, boxes, embs) -> np.ndarray:
        cm = self._motion_cost(tracks, boxes)
        if not self.cfg.use_reid or embs is None or not tracks:
            return cm
        te = np.stack([t.emb if t.emb is not None else np.zeros(embs.shape[1]) for t in tracks])
        ca = appearance_cost(te, embs)                                # Eq. (4)
        return self.cfg.lam * ca + (1 - self.cfg.lam) * cm            # Eq. (5)

    # ---------------------------------------------------------------- update
    def update(self, boxes: np.ndarray, scores: np.ndarray, embs: Optional[np.ndarray], t: int,
               graph=None, prev_loc: Optional[str] = None) -> List[Tracklet]:
        cfg = self.cfg
        self.frame = t
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        for tr in self.tracks:
            tr.det_index = -1

        keep = scores >= cfg.det_thresh
        idx_all = np.where(keep)[0]
        hi = [i for i in idx_all if scores[i] >= cfg.high_thresh]
        lo = [i for i in idx_all if scores[i] < cfg.high_thresh]

        confirmed = [tr for tr in self.tracks if tr.activated and tr.state in (TrackState.Tracked, TrackState.Lost)]
        unconfirmed = [tr for tr in self.tracks if not tr.activated and tr.state != TrackState.Removed]

        # ---- KF prediction
        if cfg.use_kf:
            for tr in confirmed + unconfirmed:
                m = tr.mean.copy()
                if tr.state != TrackState.Tracked:
                    m[6] = 0.0  # freeze height velocity for lost tracks (ByteTrack)
                tr.mean, tr.cov = self.kf.predict(m, tr.cov)

        # ---- airway-label gating (> 3 generations from previous location)
        pool = confirmed
        if graph is not None and prev_loc is not None and prev_loc in graph:
            pool = [tr for tr in confirmed if tr.label is None or tr.label not in graph
                    or graph.tree_distance(tr.label, prev_loc) <= cfg.label_gate_generations]

        # ---- stage 1: high-confidence, appearance + motion
        e_hi = embs[hi] if (embs is not None and len(hi)) else None
        c1 = self._fused_cost(pool, boxes[hi], e_hi)
        m1, u_tr1, u_hi = _assign(c1, cfg.match_thresh_high)
        for i, j in m1:
            d = hi[j]
            self._update(pool[i], boxes[d], scores[d], None if embs is None else embs[d], t, d)

        # ---- stage 2: low + unmatched high, motion only. As in ByteTrack (ref. [34])
        # only tracklets that were tracked in the previous frame take part: a lost
        # tracklet is recovered by appearance (stage 1), never by IoU alone.
        rem_tracks = [pool[i] for i in u_tr1 if pool[i].state == TrackState.Tracked]
        rem_dets = lo + [hi[j] for j in u_hi]
        thr2 = cfg.match_thresh_low if m1 else cfg.match_thresh_low_empty
        c2 = self._motion_cost(rem_tracks, boxes[rem_dets])
        m2, u_tr2, u_d2 = _assign(c2, thr2)
        for i, j in m2:
            d = rem_dets[j]
            self._update(rem_tracks[i], boxes[d], scores[d], None if embs is None else embs[d], t, d)
        for i in u_tr2:
            tr = rem_tracks[i]
            if tr.state == TrackState.Tracked:
                tr.state = TrackState.Lost
        for tr in confirmed:  # tracks removed by label gating also count as not matched
            if tr not in pool and tr.state == TrackState.Tracked:
                tr.state = TrackState.Lost

        # ---- unconfirmed tracks vs remaining high-confidence detections
        left_hi = [rem_dets[j] for j in u_d2 if scores[rem_dets[j]] >= cfg.high_thresh]
        e_l = embs[left_hi] if (embs is not None and len(left_hi)) else None
        c3 = self._fused_cost(unconfirmed, boxes[left_hi], e_l)
        m3, u_un, u_l = _assign(c3, cfg.unconfirmed_thresh)
        for i, j in m3:
            d = left_hi[j]
            self._update(unconfirmed[i], boxes[d], scores[d], None if embs is None else embs[d], t, d)
        for i in u_un:
            unconfirmed[i].state = TrackState.Removed

        # ---- new tracklets from unmatched high-confidence detections
        for j in u_l:
            d = left_hi[j]
            if scores[d] < cfg.new_track_thresh:
                continue
            tr = self._new(boxes[d], scores[d], None if embs is None else embs[d], t)
            tr.det_index = d
            if t == 0 or not self.tracks:  # first frame: activate immediately (ByteTrack)
                tr.activated = True
            tr.state = TrackState.Tracked
            self.tracks.append(tr)

        # ---- remove stale lost tracklets
        for tr in self.tracks:
            if tr.state == TrackState.Lost and t - tr.t_end > cfg.track_buffer:
                tr.state = TrackState.Removed
        self.removed += [tr for tr in self.tracks if tr.state == TrackState.Removed]
        self.tracks = [tr for tr in self.tracks if tr.state != TrackState.Removed]
        return self.active()

    def active(self) -> List[Tracklet]:
        """Activated tracklets matched to a detection in the current frame."""
        return [tr for tr in self.tracks
                if tr.state == TrackState.Tracked and tr.t_end == self.frame and tr.det_index >= 0]

    def get(self, track_id: int) -> Optional[Tracklet]:
        for tr in self.tracks:
            if tr.track_id == track_id:
                return tr
        return None
