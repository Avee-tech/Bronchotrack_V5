"""Learned, branch-dependent bronchoscope speed (version 4 of the motion model).

Extension to BronchoTrack (not in the paper).

A doctor does not drive the scope at one speed: wide, low-generation airways are passed
quickly, narrow distal branches, sharp turns and crowded bifurcations slowly. The speed the
scope tip progresses through branch ``b`` is modelled as log-linear in the branch geometry,

    ln v_b = w0 + w1 ln(D_b / 5 mm) + w2 g_b + w3 theta_b + w4 c_b + w5 r      (+ noise)

  D_b      median lumen diameter of the branch (mm)
  g_b      generation (trachea = 0)
  theta_b  angle (rad) between the parent and the branch: how sharp the turn into it is
  c_b      complexity of the bifurcation at the end of the branch, the one the doctor is
           approaching:  0 for a terminal branch, else (n_children - 1) + (1 - min_sep / pi)
           with min_sep the smallest angle between two children (crowded = more complex)
  r        1 when retracting through the branch, 0 when advancing

The weights are learned with Bayesian linear regression (Gaussian prior, closed-form
posterior), so a handful of runs already gives a usable estimate and the prior keeps
the model sensible where data is missing. Every doctor has a profile (posterior mean and
covariance) stored as JSON; a new doctor starts from the population profile, which is
updated with every run of every doctor.

Observations come from completed runs: each time the scope passes all the way through a
branch (entered from one end, left through the other), the speed is branch length /
time in the branch. The location sequence can be ground truth (annotated, EM-tracked,
benchmark) or BronchoTrack's own output.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .airway_graph import AirwayGraph

FEATURES = ["intercept", "ln(D/5mm)", "generation", "entry angle (rad)", "bifurcation complexity",
            "retracting"]
V_MIN, V_MAX = 0.5, 40.0  # mm/s, physically plausible range


# --------------------------------------------------------------------------- features
def branch_diameter(g: AirwayGraph, lab: str) -> float:
    b = g[lab]
    if b.diam is not None and len(b.diam):
        d = float(np.median(b.diam))
        if d > 0:
            return d
    return 2.0 * float(b.radius or 2.5)


def bifurcation_complexity(g: AirwayGraph, lab: str) -> float:
    ch = g.children(lab)
    if len(ch) < 2:
        return 0.0
    dirs = [g[c].direction for c in ch]
    sep = min(float(np.arccos(np.clip(np.dot(a, b), -1, 1)))
              for i, a in enumerate(dirs) for b in dirs[i + 1:])
    return (len(ch) - 1) + (1.0 - sep / np.pi)


def branch_features(g: AirwayGraph, lab: str, retract: bool = False) -> np.ndarray:
    p = g.parent(lab)
    theta = g.intersection_angle(p, lab) if p is not None else 0.0
    return np.array([1.0, np.log(branch_diameter(g, lab) / 5.0), float(g.generation(lab)), theta,
                     bifurcation_complexity(g, lab), 1.0 if retract else 0.0])


# --------------------------------------------------------------------------- model
@dataclass
class SpeedModel:
    """Gaussian posterior over the weights of the log-speed model."""

    mean: np.ndarray
    cov: np.ndarray
    noise: float = 0.35          # sd of ln(speed) between traversals of similar branches
    n_obs: int = 0
    n_runs: int = 0
    name: str = "population"

    @staticmethod
    def prior(speed: float = 3.5, slope_sd: Sequence[float] = (0.5, 0.15, 0.5, 0.3, 0.3),
              intercept_sd: float = 0.5, name: str = "population") -> "SpeedModel":
        """Prior centred on one constant ``speed`` (mm/s): all geometry slopes zero."""
        m = np.zeros(len(FEATURES))
        m[0] = np.log(speed)
        return SpeedModel(m, np.diag(np.r_[intercept_sd, slope_sd] ** 2), name=name)

    # ---- learning
    def update(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "SpeedModel":
        """Bayesian update with observations ln(speed) ``y`` of feature rows ``X``
        (``w``: per-observation reliability in (0, 1])."""
        X, y = np.atleast_2d(X), np.asarray(y, float)
        if len(y) == 0:
            return self
        w = np.ones(len(y)) if w is None else np.clip(np.asarray(w, float), 1e-3, 1.0)
        P0 = np.linalg.inv(self.cov)
        Pn = P0 + (X.T * w) @ X / self.noise ** 2
        cov = np.linalg.inv(Pn)
        mean = cov @ (P0 @ self.mean + (X.T * w) @ y / self.noise ** 2)
        self.mean, self.cov, self.n_obs = mean, 0.5 * (cov + cov.T), self.n_obs + len(y)
        return self

    def as_prior_for(self, name: str, pace_sd: float = 0.3, inflate: float = 1.2) -> "SpeedModel":
        """A new doctor starts from this (population) profile. Doctors differ mainly in their
        overall pace (the intercept: wide new uncertainty, sd ``pace_sd`` in ln speed); how much
        geometry slows them down is assumed similar between doctors (slopes only slightly widened)."""
        extra = np.diag([pace_sd ** 2] + [0.02 ** 2] * (len(FEATURES) - 1))
        return SpeedModel(self.mean.copy(), self.cov * inflate ** 2 + extra, self.noise, 0, 0, name)

    # ---- prediction
    def predict_log(self, x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, float)
        return float(x @ self.mean), float(np.sqrt(x @ self.cov @ x))

    def speed(self, x: np.ndarray) -> float:
        return float(np.clip(np.exp(self.predict_log(x)[0]), V_MIN, V_MAX))

    def branch_speeds(self, g: AirwayGraph, retract: bool = False) -> Dict[str, float]:
        """Expected speed (mm/s) of the scope through every branch."""
        return {l: self.speed(branch_features(g, l, retract)) for l in g.labels()}

    # ---- storage
    def to_dict(self) -> dict:
        return {"name": self.name, "features": FEATURES, "mean": self.mean.tolist(),
                "cov": self.cov.tolist(), "noise": self.noise, "n_obs": self.n_obs, "n_runs": self.n_runs}

    @staticmethod
    def from_dict(d: dict) -> "SpeedModel":
        return SpeedModel(np.asarray(d["mean"], float), np.asarray(d["cov"], float), float(d.get("noise", 0.35)),
                          int(d.get("n_obs", 0)), int(d.get("n_runs", 0)), d.get("name", "population"))

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @staticmethod
    def load(path: str) -> "SpeedModel":
        return SpeedModel.from_dict(json.load(open(path)))

    def describe(self) -> str:
        sd = np.sqrt(np.diag(self.cov))
        lines = [f"speed profile '{self.name}': {self.n_runs} runs, {self.n_obs} branch traversals",
                 f"  base speed {np.exp(self.mean[0]):.1f} mm/s (reference branch: 5 mm wide, generation 0, straight, no bifurcation)"]
        for k in range(1, len(FEATURES)):
            lines.append(f"  {FEATURES[k]:24s} {self.mean[k]:+.3f} +- {sd[k]:.3f}  "
                         f"(x{np.exp(self.mean[k]):.2f} per unit)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- profiles
class SpeedProfiles:
    """Directory of speed profiles: ``population.json`` and one ``<doctor>.json`` each."""

    def __init__(self, root: str, default_speed: float = 3.5):
        self.root, self.default_speed = root, default_speed

    def _path(self, name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        return os.path.join(self.root, f"{safe}.json")

    def population(self) -> SpeedModel:
        p = self._path("population")
        return SpeedModel.load(p) if os.path.exists(p) else SpeedModel.prior(self.default_speed)

    def get(self, operator: Optional[str]) -> SpeedModel:
        if not operator:
            return self.population()
        p = self._path(operator)
        if os.path.exists(p):
            return SpeedModel.load(p)
        return self.population().as_prior_for(operator)

    def learn(self, operator: Optional[str], X: np.ndarray, y: np.ndarray, w: np.ndarray) -> SpeedModel:
        """Add one run's traversals to the doctor's profile and to the population."""
        pop = self.population()
        pop.update(X, y, w)
        pop.n_runs += 1
        pop.save(self._path("population"))
        if not operator:
            return pop
        m = self.get(operator)
        m.update(X, y, w)
        m.n_runs += 1
        m.save(self._path(operator))
        return m


# --------------------------------------------------------------------------- observations
@dataclass
class Traversal:
    branch: str
    start: int       # first frame in the branch
    end: int         # last frame in the branch
    retract: bool
    speed: float     # mm/s
    weight: float


def _segments(locs: Sequence[Optional[str]], probs: Sequence[float], min_len: int):
    """Run-length segments [label, start, end, mean prob]; segments shorter than ``min_len``
    frames (flicker) are absorbed by their neighbours."""
    seg = []
    for i, (l, p) in enumerate(zip(locs, probs)):
        if seg and seg[-1][0] == l:
            seg[-1][2] = i
            seg[-1][3].append(p)
        else:
            seg.append([l, i, i, [p]])
    changed = True
    while changed and len(seg) > 2:
        changed = False
        k = min(range(len(seg)), key=lambda j: seg[j][2] - seg[j][1])
        if seg[k][2] - seg[k][1] + 1 >= min_len:
            break
        # merge into the longer neighbour
        if k == 0:
            tgt = 1
        elif k == len(seg) - 1:
            tgt = k - 1
        else:
            tgt = k - 1 if (seg[k - 1][2] - seg[k - 1][1]) >= (seg[k + 1][2] - seg[k + 1][1]) else k + 1
        a, b = min(k, tgt), max(k, tgt)
        seg[a] = [seg[tgt][0], seg[a][1], seg[b][2], seg[a][3] + seg[b][3]]
        del seg[b]
        # re-join equal neighbours
        j = 0
        while j < len(seg) - 1:
            if seg[j][0] == seg[j + 1][0]:
                seg[j] = [seg[j][0], seg[j][1], seg[j + 1][2], seg[j][3] + seg[j + 1][3]]
                del seg[j + 1]
            else:
                j += 1
        changed = True
    return [(l, s, e, float(np.mean(p))) for l, s, e, p in seg]


def extract_traversals(g: AirwayGraph, locs: Sequence[Optional[str]], fps: float,
                       probs: Optional[Sequence[float]] = None, frames: Optional[Sequence[int]] = None,
                       min_dur: float = 0.3) -> List[Traversal]:
    """Complete passes through a branch: entered from the parent and left into a child
    (advancing), or entered from a child and left into the parent (retracting)."""
    probs = [1.0] * len(locs) if probs is None else [1.0 if p is None or not np.isfinite(p) else float(p)
                                                    for p in probs]
    frames = list(range(1, len(locs) + 1)) if frames is None else list(frames)
    seg = _segments(list(locs), probs, max(1, int(round(min_dur * fps))))
    out = []
    for k in range(1, len(seg) - 1):
        (lp, _, _, _), (lb, s, e, pr), (ln, _, _, _) = seg[k - 1], seg[k], seg[k + 1]
        if lb is None or lb not in g.branches or lp not in g.branches or ln not in g.branches:
            continue
        fwd = g.parent(lb) == lp and ln in g.children(lb)
        back = lp in g.children(lb) and g.parent(lb) == ln
        if not (fwd or back):
            continue
        dur = (frames[e] - frames[s] + 1) / fps
        v = g[lb].length / dur
        if V_MIN <= v <= V_MAX:
            out.append(Traversal(lb, frames[s], frames[e], back, v, pr))
    return out


def traversal_data(g: AirwayGraph, trav: Iterable[Traversal]):
    trav = list(trav)
    if not trav:
        return np.zeros((0, len(FEATURES))), np.zeros(0), np.zeros(0)
    X = np.stack([branch_features(g, t.branch, t.retract) for t in trav])
    return X, np.log([t.speed for t in trav]), np.array([t.weight for t in trav])


def read_location_csv(path: str):
    """frames, locations, probabilities from tools/run.py location.csv or a GT csv (frame,location)."""
    fr, loc, pr = [], [], []
    for r in csv.DictReader(open(path)):
        fr.append(int(r["frame"]))
        loc.append(r["location"] or None)
        p = r.get("loc_prob")
        pr.append(float(p) if p not in (None, "") else 1.0)
    return fr, loc, pr


# --------------------------------------------------------------------------- online
class OnlineSpeedLearner:
    """Learns during a run: whenever a traversal completes, the profile is updated and new
    per-branch speeds are returned (fed to the motion model)."""

    def __init__(self, g: AirwayGraph, model: SpeedModel, fps: float, min_dur: float = 0.3):
        self.g, self.model, self.fps, self.min_dur = g, model, fps, min_dur
        self.frames, self.locs, self.probs = [], [], []
        self.seen = set()
        self.traversals: List[Traversal] = []

    def step(self, frame: int, loc: Optional[str], prob: Optional[float]):
        """Returns (advancing, retracting) per-branch speeds when the profile changed, else None."""
        changed = bool(self.locs) and loc != self.locs[-1]
        self.frames.append(frame)
        self.locs.append(loc)
        self.probs.append(1.0 if prob is None else prob)
        if not changed:
            return None
        tol = 2 * max(1, int(round(self.min_dur * self.fps)))
        new = [t for t in extract_traversals(self.g, self.locs, self.fps, self.probs, self.frames, self.min_dur)
               if not any(b == t.branch and abs(s - t.start) <= tol for b, s in self.seen)]
        if not new:
            return None
        for t in new:
            self.seen.add((t.branch, t.start))
        self.traversals += new
        self.model.update(*traversal_data(self.g, new))
        return self.model.branch_speeds(self.g), self.model.branch_speeds(self.g, retract=True)
