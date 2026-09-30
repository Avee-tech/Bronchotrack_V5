"""Speed-constrained motion model of the bronchoscope along the airway (version 3).

Extension to BronchoTrack (not in the paper).

The scope tip moves along the airway centre lines at a bounded speed, so between two
frames it can only advance or retract by a few millimetres. The model keeps a
belief over the tip position (branch, arc length s) with a particle filter:

* **Predict.** Every particle has a position (branch, s) and a signed velocity v
  along the centre line (+ = insertion, - = retraction). Velocities start uniform
  in [-v_max, v_max], v_max = max_factor x average speed, and change gradually
  (random walk: over ``tau`` seconds by about one average speed, clipped to
  +-v_max), so a particle keeps advancing or retracting like a real scope and can
  stop or reverse. Passing the end of a branch it continues into one of the children
  (chosen at random); passing the start it goes back into the parent. The reachable
  region grows at most v_max per second: the camera cannot be reported in a branch
  it could not have reached yet.
* **Update.** The per-frame localisation evidence (Eq. (9) votes of the fused
  labels, each weighted by its probability) gives, for a particle at (b, s),
      L = eps + share(b)
            + adjacent x sum share(children(b)) x exp(-(len(b) - s) / view_range)
            + adjacent x share(parent(b))       x exp(-s / view_range)
  eps covers frames where the evidence is wrong. The two adjacency terms cover the
  Eq. (9) one-generation ambiguity at a bifurcation *and* let the belief move: a
  child's evidence pulls particles to the end of the parent, where they can cross.
* **Output.** The branch holding the largest posterior mass, with that mass as its
  probability. The fused labels are scored with the probability that their Eq. (9)
  location is *reachable within the next ``horizon`` seconds* (one-generation moves
  near a bifurcation are allowed, instant multi-generation jumps are not), and the
  location is fed back to the association (label gating, recovery).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .airway_graph import AirwayGraph


@dataclass
class MotionConfig:
    speed: float = 7.0           # average speed of the scope tip along the airway, mm/s (ModelV3 videos: 6-8)
    fps: float = 30.0            # frames per second of the video
    max_factor: float = 2.0      # the tip never moves faster than max_factor x average speed
    tau: float = 1.0             # s over which the velocity changes by about one average speed
    n_particles: int = 2000
    eps: float = 0.05            # likelihood floor (evidence can be wrong)
    adjacent: float = 0.3        # likelihood share for parent/children of a voted branch
    prior_weight: float = 1.0    # weight of log P_reach(location) in the fusion hypotheses
    horizon: float = 1.0         # s: fusion prior = branches reachable within this time
    view_range: float = 20.0     # mm: evidence for a child also supports the last ~20 mm of its parent
    init_range: float = 40.0     # mm: when the carina is first recognised the tip is within this of it
    reach_floor: float = 0.02    # prior of a branch the scope cannot reach in time
    use_label_age: bool = True   # association confidence from how long a lumen kept its label
    seed: int = 0

    @property
    def step_max(self) -> float:
        """Largest displacement per frame, mm."""
        return self.max_factor * self.speed / self.fps


class MotionModel:
    def __init__(self, graph: AirwayGraph, cfg: Optional[MotionConfig] = None, start: Optional[str] = None):
        self.g = graph
        self.cfg = cfg if cfg is not None else MotionConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.labels = list(graph.labels())
        self.idx = {l: i for i, l in enumerate(self.labels)}
        self.L = np.array([max(graph[l].length, 1e-3) for l in self.labels])
        self.parent = np.array([self.idx[graph.parent(l)] if graph.parent(l) is not None else -1
                                for l in self.labels])
        self.children = [[self.idx[c] for c in graph.children(l)] for l in self.labels]
        self._geo = {}
        for l in self.labels:  # arc-length parametrised centre lines for drawing
            b = graph[l]
            p = b.points if b.points is not None and len(b.points) > 1 else np.stack([b.start, b.end])
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            self._geo[l] = (p, cum)
        self.reset(start)

    # ------------------------------------------------------------------ state
    def reset(self, start: Optional[str] = None, last_mm: Optional[float] = None):
        """Particles spread uniformly along ``start`` (default: the trachea), or over its
        last ``last_mm`` millimetres."""
        b = self.idx[start if start is not None else self.g.root]
        n = self.cfg.n_particles
        lo = 0.0 if last_mm is None else max(0.0, self.L[b] - last_mm)
        self.b = np.full(n, b, int)
        self.s = self.rng.uniform(lo, self.L[b], n)
        self.v = self.rng.uniform(-self.cfg.step_max, self.cfg.step_max, n)  # mm per frame
        self.w = np.full(n, 1.0 / n)

    def reach_probs(self, horizon_s: Optional[float] = None) -> Dict[str, float]:
        """Probability that the tip can be in each branch within ``horizon_s`` seconds."""
        r = self.cfg.step_max * self.cfg.fps * (self.cfg.horizon if horizon_s is None else horizon_s)
        nb = len(self.labels)
        own = np.bincount(self.b, weights=self.w, minlength=nb)
        near_end = np.bincount(self.b, weights=self.w * ((self.L[self.b] - self.s) <= r), minlength=nb)
        near_start = np.bincount(self.b, weights=self.w * (self.s <= r), minlength=nb)
        reach = own.copy()
        for i in range(nb):
            for c in self.children[i]:
                reach[c] += near_end[i]
            if self.parent[i] >= 0:
                reach[self.parent[i]] += near_start[i]
        return {self.labels[i]: float(min(1.0, v)) for i, v in enumerate(reach) if v > 0}

    def branch_probs(self) -> Dict[str, float]:
        m = np.bincount(self.b, weights=self.w, minlength=len(self.labels))
        return {self.labels[i]: float(v) for i, v in enumerate(m) if v > 0}

    def estimate(self):
        """(branch with the largest mass, its probability)."""
        m = np.bincount(self.b, weights=self.w, minlength=len(self.labels))
        i = int(m.argmax())
        return self.labels[i], float(m[i])

    # ---------------------------------------------------------------- predict
    def predict(self):
        c = self.cfg
        sd = (c.speed / c.fps) * np.sqrt(1.0 / (c.fps * c.tau))  # velocity change per frame
        self.v = np.clip(self.v + self.rng.normal(0, sd, len(self.v)), -c.step_max, c.step_max)
        self.s = self.s + self.v
        for _ in range(8):  # a step is short; a few crossings at most
            over = np.where(self.s > self.L[self.b])[0]
            under = np.where(self.s < 0)[0]
            if len(over) == 0 and len(under) == 0:
                break
            for i in over:
                ch = self.children[self.b[i]]
                if ch:
                    self.s[i] -= self.L[self.b[i]]
                    self.b[i] = ch[self.rng.integers(len(ch))]
                else:
                    self.s[i] = self.L[self.b[i]]
            for i in under:
                p = self.parent[self.b[i]]
                if p >= 0:
                    self.b[i] = p
                    self.s[i] += self.L[p]
                else:
                    self.s[i] = 0.0
        np.clip(self.s, 0, self.L[self.b], out=self.s)

    # ----------------------------------------------------------------- update
    def likelihood(self, votes: Dict[str, float]) -> np.ndarray:
        """Per-particle likelihood of this frame's location votes."""
        tot = sum(v for k, v in votes.items() if k in self.idx)
        if tot <= 0:
            return np.ones(len(self.b))
        share = np.zeros(len(self.labels))
        for k, v in votes.items():
            if k in self.idx:
                share[self.idx[k]] += v / tot
        child_share = np.array([share[ch].sum() if ch else 0.0 for ch in self.children])
        par_share = np.where(self.parent >= 0, share[np.maximum(self.parent, 0)], 0.0)
        lam = self.cfg.view_range
        return (self.cfg.eps + share[self.b]
                + self.cfg.adjacent * child_share[self.b] * np.exp(-(self.L[self.b] - self.s) / lam)
                + self.cfg.adjacent * par_share[self.b] * np.exp(-self.s / lam))

    def update(self, votes: Dict[str, float]):
        self.w = self.w * self.likelihood(votes)
        tot = self.w.sum()
        if not np.isfinite(tot) or tot <= 0:
            self.w = np.full(len(self.w), 1.0 / len(self.w))
        else:
            self.w /= tot
        if 1.0 / np.sum(self.w ** 2) < 0.5 * len(self.w):
            self._resample()

    def _resample(self):
        n = len(self.w)
        pos = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), n - 1)
        self.b, self.s, self.v = self.b[idx].copy(), self.s[idx].copy(), self.v[idx].copy()
        self.w = np.full(n, 1.0 / n)

    # ---------------------------------------------------------------- drawing
    def positions(self, k: Optional[int] = None) -> np.ndarray:
        """3-D positions (standard frame) of ``k`` particles (all if None)."""
        sel = np.arange(len(self.b)) if k is None or k >= len(self.b) else \
            self.rng.choice(len(self.b), k, replace=False, p=self.w)
        out = np.zeros((len(sel), 3))
        for bi in np.unique(self.b[sel]):
            m = self.b[sel] == bi
            p, cum = self._geo[self.labels[bi]]
            ss = self.s[sel][m] * (cum[-1] / self.L[bi])
            out[m] = np.stack([np.interp(ss, cum, p[:, j]) for j in range(3)], 1)
        return out
