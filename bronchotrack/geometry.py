"""Roll-corrected diameter:distance ratio cue, fused with the airway association.

Extension to BronchoTrack (not in the paper).

1. **Straighten the boxes.** The tracker returns axis-aligned boxes in the image.
   With the scope rolled by ``roll`` the lumen appears rotated, so its box is
   distorted. Modelling the lumen opening as an ellipse with axes (w', h'), the
   axis-aligned box after a rotation theta obeys
       W^2 = w'^2 cos^2 + h'^2 sin^2,   H^2 = w'^2 sin^2 + h'^2 cos^2
   which is inverted to recover (w', h') "as if the scope were straight"
   (a rectangle model is available too; it wrongly shrinks round lumens).
2. **Diameter : distance.** The longest edge of the straightened box is the
   lumen's apparent diameter D (an obliquely viewed circle keeps its diameter as
   the major axis). For two lumens i, j seen together, d_ij is the distance
   between their box centres, giving the observed ratio  D_i / d_ij.
3. **Actual ratio.** From the airway graph: go 1.1 x the parent's radius from the
   bifurcation into each child branch; D_a is the lumen diameter measured at that
   point from the CT segmentation and s_ab the distance between the two points,
   projected on the parent's tangent plane: R_a = D_a / s_ab. For lumens at a
   similar depth the perspective scale cancels, so observed and actual ratios are
   directly comparable.
4. **Fusion.** For every group of sibling lumens the association's labels act as
   a prior (confidence grows with tracklet age); the ratio match is a
   log-normal likelihood. Hypotheses cover the association's parent branch,
   its parent and its children (so a one-generation slip can be corrected).
   The posterior gives fused labels, per-lumen probabilities, and through
   Eq. (9) a location with a probability.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .airway_graph import AirwayGraph
from .subgraph import LumenNode, LumenSubgraph


# --------------------------------------------------------------------------- #
# 1. straighten a box
# --------------------------------------------------------------------------- #
def derotate_box(W: float, H: float, theta: float, model: str = "ellipse",
                 cond: float = 0.25) -> Tuple[float, float]:
    """Width/height of the object before the image rotation ``theta``.

    ``model="ellipse"`` (default): a lumen opening is an ellipse with axes (w', h');
    its axis-aligned box after rotation satisfies
        W^2 = w'^2 cos^2 + h'^2 sin^2,   H^2 = w'^2 sin^2 + h'^2 cos^2,
    which is linear in (w'^2, h'^2). A circle is left unchanged by any roll.
    ``model="rect"``: rectangle, [W, H] = [[c, s], [s, c]] [w', h'].
    Both are ill-conditioned near 45 deg (det = cos 2theta -> 0); there, or when the
    solution is not physical, the object is taken as round/square.
    """
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    if model == "rect":
        det = c * c - s * s
        if abs(det) > cond:
            w, h = (c * W - s * H) / det, (c * H - s * W) / det
            if w > 0.1 * min(W, H) and h > 0.1 * min(W, H):
                return float(w), float(h)
        side = (W + H) / (2 * (c + s))
        return float(side), float(side)
    c2, s2 = c * c, s * s
    det = c2 * c2 - s2 * s2  # = cos(2 theta)
    if abs(det) > cond:
        A = (c2 * W * W - s2 * H * H) / det
        B = (c2 * H * H - s2 * W * W) / det
        if A > 0.01 * min(W, H) ** 2 and B > 0.01 * min(W, H) ** 2:
            return float(np.sqrt(A)), float(np.sqrt(B))
    side = float(np.sqrt((W * W + H * H) / 2))
    return side, side


def corrected_diameter(box: np.ndarray, roll: float) -> float:
    w, h = derotate_box(box[2] - box[0], box[3] - box[1], roll)
    return max(w, h)


# --------------------------------------------------------------------------- #
# 2-3. observed vs actual ratios
# --------------------------------------------------------------------------- #
class RatioModel:
    """Actual diameter:distance ratios from the airway graph.

    For sibling branches a, b with parent P: starting at the bifurcation (the start of
    each child's centre line) go ``entry_factor`` x r_P along each child to points
    p_a, p_b. The diameter is the local lumen diameter at that point
    (``Branch.diameter_at``, measured from the segmentation by ``measure_diameters``;
    2 x the branch radius if the graph has no diameter profile). The distance is
    |p_a - p_b| projected on P's tangent plane, i.e. as the camera looking down P sees it.
        R_a = D_a(p_a) / s_ab,   R_b = D_b(p_b) / s_ab
    """

    def __init__(self, graph: AirwayGraph, entry_factor: float = 1.1, default_radius: float = 2.0):
        self.g, self.f, self.r0 = graph, entry_factor, default_radius
        self._cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def radius(self, lab: str) -> float:
        r = self.g[lab].radius
        return float(r) if r else self.r0

    def entry(self, lab: str) -> Tuple[np.ndarray, float]:
        """(point, local diameter) at entry_factor x parent radius into branch ``lab``."""
        br = self.g[lab]
        par = self.g.parent(lab)
        s = self.f * (self.radius(par) if par is not None else self.radius(lab))
        s = min(s, br.length)
        d = br.diameter_at(s)
        return br.point_at(s), (d if d is not None and d > 0 else 2 * self.radius(lab))

    def ratio(self, a: str, b: str) -> Tuple[float, float]:
        """(D_a / s_ab, D_b / s_ab) for sibling branches a, b."""
        key = (a, b)
        if key not in self._cache:
            pa, Da = self.entry(a)
            pb, Db = self.entry(b)
            par = self.g.parent(a)
            e1, e2, _ = self.g.frame(par) if par is not None else self.g.frame(a)
            v = pa - pb
            sep = float(np.hypot(np.dot(v, e1), np.dot(v, e2)))
            sep = max(sep, 0.25 * (Da + Db))  # guard against degenerate geometry
            self._cache[key] = (Da / sep, Db / sep)
            self._cache[(b, a)] = (Db / sep, Da / sep)
        return self._cache[key]


def observed_ratio(ni: LumenNode, nj: LumenNode, roll: float) -> Optional[Tuple[float, float]]:
    d = float(np.linalg.norm(ni.center - nj.center))
    if d < 5.0:
        return None
    return corrected_diameter(ni.box, roll) / d, corrected_diameter(nj.box, roll) / d


# --------------------------------------------------------------------------- #
# 4. fusion
# --------------------------------------------------------------------------- #
@dataclass
class FusionConfig:
    sigma: float = 0.55          # spread of log(observed / actual) ratio (fitted on synthetic GT)
    w_geo: float = 1.0           # weight of the ratio likelihood against the association prior
    q_min: float = 0.55          # association confidence for a brand-new tracklet ...
    q_max: float = 0.95          # ... rising to this for an old one
    tau_age: float = 10.0        # frames
    n_alt: int = 6               # alternative-label mass spread (prior of a non-association label)
    none_logp: float = -4.0      # log-prior of leaving a lumen unexplained (false detection)
    explore_parent: bool = True  # also test the association parent's parent
    explore_children: bool = True  # ... and its children (one-generation slips)
    entry_factor: float = 1.1    # measure each child at 1.1 x parent radius from the bifurcation
    feedback: bool = True        # write fused labels/location back for the next frame
    use_label_age: bool = False  # confidence from how long a lumen kept its label (version 3)


@dataclass
class FusionResult:
    labels: Dict[int, Tuple[Optional[str], float]] = field(default_factory=dict)   # det -> (label, prob)
    # each method on its own: association prior only (w_geo = 0) and ratio likelihood only
    # (uniform prior over the same hypotheses); det -> (label, prob), R missing if no sibling pair
    assoc: Dict[int, Tuple[Optional[str], float]] = field(default_factory=dict)
    ratio: Dict[int, Tuple[Optional[str], float]] = field(default_factory=dict)
    location: Optional[str] = None
    loc_prob: float = 0.0
    ratio_err_assoc: Optional[float] = None   # mean |log(obs/actual)| with association labels
    ratio_err_fused: Optional[float] = None   # ... with fused labels
    votes: Dict[str, float] = field(default_factory=dict)  # probability-weighted Eq. (9) votes
    pairs: List[Tuple[int, int, float, float, float, float]] = field(default_factory=list)
    # (det_i, det_j, obs_i, actual_i, obs_j, actual_j) for the fused labels


class GeometricFusion:
    def __init__(self, graph: AirwayGraph, cfg: Optional[FusionConfig] = None):
        self.g = graph
        self.cfg = cfg if cfg is not None else FusionConfig()
        self.model = RatioModel(graph, self.cfg.entry_factor)

    # ------------------------------------------------------------------ utils
    def _q(self, n: LumenNode) -> float:
        c = self.cfg
        age = n.label_age if (c.use_label_age and n.label_age >= 0) else n.track_age
        return c.q_min + (c.q_max - c.q_min) * (1 - np.exp(-max(age, 0) / c.tau_age))

    def _implied_location(self, S: LumenSubgraph, members, perm, n_p: int) -> Optional[str]:
        """Location a label hypothesis implies through Eq. (9)."""
        for m, lab in zip(members, perm):
            if lab is not None and lab in self.g:
                return self.g.ancestor_strict(lab, m.level - 1 if n_p == 1 else m.level)
        return None

    def _log_prior(self, n: LumenNode, lab: Optional[str]) -> float:
        if lab is None:
            return self.cfg.none_logp
        if self.cfg.use_label_age and n.prev_label is not None and n.label is not None \
                and n.prev_label != n.label:
            # version 3: the association relabelled a tracked lumen this frame; the new label
            # competes with the one the lumen had kept for label_age frames
            w_prev, w_cur = self._q(n), self.cfg.q_min
            if lab == n.label:
                return float(np.log(0.95 * w_cur / (w_prev + w_cur)))
            if lab == n.prev_label:
                return float(np.log(0.95 * w_prev / (w_prev + w_cur)))
            return float(np.log(0.05 / self.cfg.n_alt))
        if n.label is None:
            return float(np.log(1.0 / self.cfg.n_alt))
        q = self._q(n)
        return float(np.log(q)) if lab == n.label else float(np.log((1 - q) / self.cfg.n_alt))

    def _pair_ll(self, obs: Tuple[float, float], a: str, b: str) -> float:
        ra, rb = self.model.ratio(a, b)
        s2 = 2 * self.cfg.sigma ** 2
        return -((np.log(obs[0] / ra)) ** 2 + (np.log(obs[1] / rb)) ** 2) / s2

    def _pair_err(self, obs, a, b) -> float:
        ra, rb = self.model.ratio(a, b)
        return 0.5 * (abs(np.log(obs[0] / ra)) + abs(np.log(obs[1] / rb)))

    def _hyp_parents(self, P0: str, n: int) -> List[str]:
        out = [P0]
        if self.cfg.explore_parent and self.g.parent(P0) is not None:
            out.append(self.g.parent(P0))
        if self.cfg.explore_children:
            out += [c for c in self.g.children(P0) if len(self.g.children(c)) >= 2]
        return [p for p in dict.fromkeys(out) if len(self.g.children(p)) >= 2]

    def _base_parent(self, S: LumenSubgraph, members: List[LumenNode], assoc_loc: str,
                     is_root: bool) -> Optional[str]:
        votes: Dict[str, float] = {}
        for m in members:
            if m.label in self.g and self.g.parent(m.label) is not None:
                p = self.g.parent(m.label)
                votes[p] = votes.get(p, 0) + 1 + 1e-3 * max(m.track_age, 0)
        if votes:
            return max(votes, key=votes.get)
        if not is_root:
            par = S.nodes[members[0].parent]
            return par.label if par.label in self.g else None
        if assoc_loc in self.g and len(self.g.children(assoc_loc)) >= 2:
            return assoc_loc
        return self.g.parent(assoc_loc) if assoc_loc in self.g else None

    # ------------------------------------------------------------------- main
    def fuse(self, S: LumenSubgraph, assoc_loc: str, roll: float,
             loc_prior: Optional[Dict[str, float]] = None, prior_weight: float = 1.0) -> FusionResult:
        """``loc_prior`` (version 3): predicted probability of each branch from the motion
        model; every hypothesis is scored with prior_weight x log P_motion(its Eq. (9) location)."""
        res = FusionResult()
        n_p_all = len(S.primary())
        nodes = list(S)
        if not nodes:
            res.location, res.loc_prob = assoc_loc, 0.0
            return res
        # sibling groups: children of a node, and the primary lumens
        groups: List[Tuple[List[LumenNode], bool]] = []
        for n in nodes:
            ch = S.children(n.det)
            if len(ch) >= 2:
                groups.append((ch, False))
        roots = S.roots()
        if len(roots) >= 2:
            groups.append((roots, True))

        fused: Dict[int, Tuple[Optional[str], float]] = {}
        errs_a, errs_f = [], []
        for members, is_root in groups:
            obs = {}
            for i, j in itertools.combinations(range(len(members)), 2):
                o = observed_ratio(members[i], members[j], roll)
                if o is not None:
                    obs[(i, j)] = o
            # how well do the association's labels match the actual ratios?
            for (i, j), o in obs.items():
                a, b = members[i].label, members[j].label
                if a in self.g and b in self.g and a != b and self.g.parent(a) == self.g.parent(b):
                    errs_a.append(self._pair_err(o, a, b))
            P0 = self._base_parent(S, members, assoc_loc, is_root)
            if P0 is None or P0 not in self.g:
                continue
            hyps, scores, lps, lgs = [], [], [], []
            for P in self._hyp_parents(P0, len(members)):
                cand = list(self.g.children(P))
                cand += [None] * max(0, len(members) - len(cand))
                seen = set()
                for perm in itertools.permutations(cand, len(members)):
                    if perm in seen:
                        continue
                    seen.add(perm)
                    lp = sum(self._log_prior(m, l) for m, l in zip(members, perm))
                    if not is_root:  # the enclosing lumen's own reading supports P
                        lp += self._log_prior(S.nodes[members[0].parent], P)
                    if loc_prior is not None:  # version 3: can the scope be there yet?
                        loc = self._implied_location(S, members, perm, n_p_all)
                        lp += prior_weight * float(np.log(0.02 + loc_prior.get(loc, 0.0)))
                    lg = 0.0
                    for (i, j), o in obs.items():
                        if perm[i] is not None and perm[j] is not None:
                            lg += self._pair_ll(o, perm[i], perm[j])
                    hyps.append((P, perm))
                    scores.append(lp + self.cfg.w_geo * lg)
                    lps.append(lp)
                    lgs.append(lg)
            if not hyps:
                continue
            sc = np.asarray(scores)
            post = np.exp(sc - sc.max())
            post /= post.sum()
            k = int(post.argmax())
            P_best, best = hyps[k]
            for i, m in enumerate(members):
                lab = best[i]
                p = float(sum(pp for (P, h), pp in zip(hyps, post) if h[i] == lab))
                if m.det not in fused or fused[m.det][1] < p:
                    fused[m.det] = (lab, p)
            if not is_root and members[0].parent is not None:  # the enclosing lumen is P
                pd = members[0].parent
                pP = float(sum(pp for (P, _), pp in zip(hyps, post) if P == P_best))
                if pd not in fused or fused[pd][1] < pP:
                    fused[pd] = (P_best, pP)
            # the two methods separately, over the same hypotheses
            for store, sc_m, needs_pairs in ((res.assoc, lps, False), (res.ratio, lgs, True)):
                if needs_pairs and not obs:
                    continue
                q_ = np.asarray(sc_m, float)
                pm = np.exp(q_ - q_.max())
                pm /= pm.sum()
                for i, m in enumerate(members):
                    marg: Dict[Optional[str], float] = {}
                    for (P, h), pp in zip(hyps, pm):
                        marg[h[i]] = marg.get(h[i], 0.0) + float(pp)
                    lab = max(marg, key=marg.get)
                    if m.det not in store or store[m.det][1] < marg[lab]:
                        store[m.det] = (lab, marg[lab])
                if not is_root and members[0].parent is not None:
                    margP: Dict[str, float] = {}
                    for (P, _), pp in zip(hyps, pm):
                        margP[P] = margP.get(P, 0.0) + float(pp)
                    lab = max(margP, key=margP.get)
                    pd = members[0].parent
                    if pd not in store or store[pd][1] < margP[lab]:
                        store[pd] = (lab, margP[lab])
            for (i, j), o in obs.items():
                if best[i] is not None and best[j] is not None:
                    ra, rb = self.model.ratio(best[i], best[j])
                    res.pairs.append((members[i].det, members[j].det, o[0], ra, o[1], rb))
                    errs_f.append(self._pair_err(o, best[i], best[j]))
        # lumens outside any sibling group keep the association's reading
        for n in nodes:
            if n.det not in fused:
                fused[n.det] = (n.label, self._q(n) if n.label is not None else 0.0)
            if n.det not in res.assoc:
                res.assoc[n.det] = (n.label, self._q(n) if n.label is not None else 0.0)
        res.labels = fused
        res.ratio_err_assoc = float(np.mean(errs_a)) if errs_a else None
        res.ratio_err_fused = float(np.mean(errs_f)) if errs_f else None

        # location through Eq. (9), each vote weighted by its lumen's probability
        # P(loc) = average over the voting lumens of P(its label) * [its vote = loc], so the
        # probability drops both when lumens disagree and when their labels are uncertain
        n_p = len(S.primary())
        votes: Dict[str, float] = {}
        n_vote = 0
        for n in nodes:
            lab, p = fused[n.det]
            if lab is None or lab not in self.g:
                continue
            loc = self.g.ancestor_strict(lab, n.level - 1 if n_p == 1 else n.level)
            if loc is not None:
                votes[loc] = votes.get(loc, 0.0) + p
                n_vote += 1
        res.votes = dict(votes)
        if votes:
            res.location = max(votes, key=votes.get)
            res.loc_prob = votes[res.location] / n_vote
        else:
            res.location, res.loc_prob = assoc_loc, 0.0
        return res


def apply_fusion(S: LumenSubgraph, res: FusionResult):
    """Write fused labels into the lumen subgraph (feedback to the tracklets)."""
    for n in S:
        if n.det in res.labels:
            n.label = res.labels[n.det][0]


def summarize_pairs(pairs: Sequence[Tuple[int, int, float, float, float, float]]) -> Dict[str, float]:
    if not pairs:
        return {}
    o = np.array([[p[2], p[3]] for p in pairs] + [[p[4], p[5]] for p in pairs])
    e = np.log(o[:, 0] / o[:, 1])
    return {"n": len(e), "median_abs_log_err": float(np.median(np.abs(e))), "bias": float(np.mean(e))}
