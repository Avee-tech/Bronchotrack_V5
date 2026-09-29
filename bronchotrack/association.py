"""Airway graph association and branch-level localisation (paper Sec. III-C).

Implements Algorithm 1 (airway graph association) and Algorithm 2 (subgraph
association) together with

* Eq. (6)   roll initialisation from the left/right main bronchi,
* Eq. (7-8) roll propagation from the two oldest tracklets and the gallery,
* the gallery G of visited bronchi (Sec. III-C-2),
* intra-frame label propagation (Sec. III-C-4),
* Eq. (9)   voting-based branch-level localisation.

Conventions
-----------
Image coordinates are (x right, y down). The 2-D graph of a branch's children is
the projection on the branch's tangent plane using the basis (e1, e2) of
``AirwayGraph.tangent_basis`` (e1 = projected standard x-axis). At roll = 0 the
graph's e1 is the image x axis. ``rotate(p, roll)`` maps the graph into the
image; roll is a signed angle (the paper writes arccos, which drops the sign –
we use the equivalent signed ``atan2`` so that clockwise and anti-clockwise
rotations are distinguished).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .airway_graph import AirwayGraph
from .subgraph import LumenNode, LumenSubgraph


# --------------------------------------------------------------------------- #
def signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle rotating vector a onto b in image coordinates."""
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def rotate(p: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(p, float) @ R.T


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass(eq=False)
class GalleryRecord:
    """G_{l_p} = (T^t, roll^t): tracklets observed while inspecting branch l_p."""

    label: str
    t: int
    roll: float
    centers: Dict[int, np.ndarray]           # track_id -> box centre at time t
    boxes: Dict[int, np.ndarray]
    labels: Dict[int, Optional[str]]
    n_lumens: int
    keyframe: Optional[np.ndarray] = None    # image (BronchoTrack-LC)
    updated: int = 0                         # time of last update (for "lambda most recent")


@dataclass
class AssociationConfig:
    truncate: float = 10.0          # length (graph units, mm) at which children are truncated
    mirror: bool = False            # flip image x (display convention of the scope)
    init_roll: float = 0.0          # prior roll used to tell left from right at the carina
    observed_selection: str = "paper"   # "paper": n_hat most aligned children; "all": all children
    min_roll_baseline: float = 8.0  # px, min distance of the two oldest tracklets for Eq. (7)
    use_gallery_roll: bool = True
    max_roll_step: float = np.radians(30)  # reject roll updates faster than this per frame
                                           # (implausible at 15-30 fps; guards against ID switches)
    relabel_inconsistent: bool = False  # aggressive re-matching of inconsistent labels (off)
    refine: bool = True             # anatomical-constraint refinement of tracklet labels
    init_requires_two: bool = True  # initialise when two main-bronchus lumens are visible
    init_stable_frames: int = 5     # ...by the same two tracklets for this many consecutive frames
    entry_area: float = 0.3         # recovery: a primary lumen this large (fraction of the
                                    # image) that vanished is taken as the branch just entered


class AirwayAssociator:
    def __init__(self, graph: AirwayGraph, cfg: Optional[AssociationConfig] = None):
        self.M = graph
        self.cfg = cfg if cfg is not None else AssociationConfig()
        self.reset()

    def reset(self):
        self.initialized = False
        self.roll = float(self.cfg.init_roll)
        self.loc: str = self.M.root
        self.gallery: Dict[str, GalleryRecord] = {}
        self._prev_centers: Dict[int, np.ndarray] = {}
        self._prev_roll = self.roll
        self.new_branch_entered = False
        self.last_votes: Dict[str, float] = {}
        self._last_primary: List[Tuple[str, float]] = []
        self._init_pair: Optional[Tuple] = None
        self._init_count = 0

    def force_init(self, loc: str, roll: float = 0.0):
        """Start mid-airway (e.g. video starting after the carina)."""
        self.initialized, self.loc, self.roll = True, loc, float(roll)

    # ================================================================ Alg. 1
    def step(self, S: LumenSubgraph, t: int, image_size: Tuple[int, int],
             frame: Optional[np.ndarray] = None) -> str:
        """Process one frame. Node labels in ``S`` must already carry the labels
        of their tracklets (inter-frame association). Returns the location."""
        W, H = image_size
        self.image_center = np.array([W / 2.0, H / 2.0])
        self.new_branch_entered = False

        if not self.initialized:
            if not self._try_initialize(S, t, frame):
                for n in S:
                    n.label = None
                self.loc = self.M.root
                return self.loc

        # ---- roll estimation, Eqs. (7)-(8)
        self._estimate_roll(S)

        # ---- intra-frame association (Alg. 2 driven by labelled detections)
        self._intra_frame(S)

        # ---- localisation, Eq. (9)
        prev = self.loc
        self.loc = self._vote(S) or prev

        # ---- gallery update
        self._update_gallery(S, t, frame)
        self._prev_centers = {n.track_id: n.center for n in S if n.track_id is not None}
        self._prev_roll = self.roll
        self._last_primary = [(n.label, n.area / float(W * H)) for n in S.primary() if n.label is not None]
        return self.loc

    # ------------------------------------------------------- initialisation
    def _main_pair(self, S: LumenSubgraph) -> Optional[List[LumenNode]]:
        prim = S.primary()
        if len(prim) == 2:
            return prim
        if len(prim) == 1 and len(S.children(prim[0].det)) == 2:
            return S.children(prim[0].det)
        return None

    def _try_initialize(self, S: LumenSubgraph, t: int, frame) -> bool:
        pair = self._main_pair(S)
        if pair is None:
            self._init_pair, self._init_count = None, 0
            return False
        key = tuple(sorted(str(n.track_id) for n in pair))
        self._init_count = self._init_count + 1 if key == self._init_pair else 1
        self._init_pair = key
        if self._init_count < self.cfg.init_stable_frames:
            return False
        root = self.M.root
        mains = self.M.children(root)
        if len(mains) < 2:
            return False
        # identify left / right main bronchus with the 2-D graph at the prior roll
        ref = self._reference_center(S, pair[0])
        labels = self._match(root, pair, ref, fixed={})
        if labels is None:
            return False
        for n, l in zip(pair, labels):
            n.label = l
        # roll_0, Eq. (6): angle between (c_right - c_left) and the graph's (right - left)
        right = self._named(mains, ("RMB", "right"), default=None)
        left = self._named(mains, ("LMB", "left"), default=None)
        if right is None or left is None:
            left, right = mains[0], mains[1]
        g2d = self.M.projected_children(root, self.cfg.truncate, mirror=self.cfg.mirror)
        byl = {n.label: n for n in pair}
        if right in byl and left in byl:
            obs = byl[right].center - byl[left].center
            refv = g2d[right] - g2d[left]
            self.roll = signed_angle(refv, obs)  # == arccos((c_r - c_l).e_x / |.|) up to sign
        prim = S.primary()
        if len(prim) == 1 and all(prim[0].det != n.det for n in pair):
            prim[0].label = root
        self.initialized = True
        self.loc = root
        # G = {G_carina : (T^0, roll^0)}
        self.gallery = {}
        self._add_record(root, S, t, frame)
        return True

    def _named(self, labels, keys, default=None):
        for l in labels:
            nm = (self.M[l].name or "").lower()
            if any(k.lower() == nm or k.lower() in nm for k in keys):
                return l
        return default

    # --------------------------------------------------------- roll, Eq. 7-8
    @staticmethod
    def _nested(S: LumenSubgraph, a: LumenNode, b: LumenNode) -> bool:
        def anc(n):
            out, p = set(), n.parent
            while p is not None:
                out.add(p)
                p = S.nodes[p].parent
            return out
        return a.det in anc(b) or b.det in anc(a)

    def _oldest_pair(self, S: LumenSubgraph, tracked: List[LumenNode], allowed) -> Optional[Tuple[int, int]]:
        """Two oldest tracklets (Sec. III-C-3). Nested boxes (parent/child) share
        nearly the same centre, so a pair that is not nested is preferred."""
        cand = [n for n in tracked if n.track_id in allowed]
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                if not self._nested(S, a, b):
                    return a.track_id, b.track_id
        return (cand[0].track_id, cand[1].track_id) if len(cand) >= 2 else None

    def _estimate_roll(self, S: LumenSubgraph):
        tracked = sorted([n for n in S if n.track_id is not None], key=lambda n: -n.track_age)
        if len(tracked) < 2:
            return
        cur = {n.track_id: n.center for n in tracked}
        # gallery record sharing the two oldest tracklets (most recently updated first)
        if self.cfg.use_gallery_roll:
            for rec in sorted(self.gallery.values(), key=lambda r: -r.updated):
                pair = self._oldest_pair(S, tracked, rec.centers)
                if pair is None:
                    continue
                a, b = pair
                v0 = rec.centers[a] - rec.centers[b]
                v1 = cur[a] - cur[b]
                if min(np.linalg.norm(v0), np.linalg.norm(v1)) < self.cfg.min_roll_baseline:
                    continue
                r = wrap(rec.roll + signed_angle(v0, v1))
                if abs(wrap(r - self._prev_roll)) > self.cfg.max_roll_step:
                    continue
                self.roll = r
                return
        # fall-back: previous frame (t - 1) acts as the reference observation
        pair = self._oldest_pair(S, tracked, self._prev_centers)
        if pair is not None:
            a, b = pair
            v0 = self._prev_centers[a] - self._prev_centers[b]
            v1 = cur[a] - cur[b]
            if min(np.linalg.norm(v0), np.linalg.norm(v1)) >= self.cfg.min_roll_baseline:
                d = signed_angle(v0, v1)
                if abs(d) <= self.cfg.max_roll_step:
                    self.roll = wrap(self._prev_roll + d)

    # ============================================================ Alg. 2
    def _observed_children(self, lp: str, n_det: int, exclude: Sequence[str] = ()) -> List[str]:
        """ĉh(l_p): children most likely observed – likelihood decreases with the
        intersection angle between l_p and the child."""
        ch = [c for c in self.M.children(lp) if c not in exclude]
        ch.sort(key=lambda c: self.M.intersection_angle(lp, c))
        if self.cfg.observed_selection == "all":
            return ch
        return ch[:max(0, n_det)]

    def _graph2d(self, lp: str, labels: Sequence[str]) -> Dict[str, np.ndarray]:
        g = self.M.projected_children(lp, self.cfg.truncate, children=labels, mirror=self.cfg.mirror)
        return {k: rotate(v, self.roll) for k, v in g.items()}   # rotate by roll^t

    def _reference_center(self, S: LumenSubgraph, node: LumenNode) -> np.ndarray:
        if node.parent is not None and node.parent in S.nodes:
            return S.nodes[node.parent].center
        return self.image_center

    def _match(self, lp: str, nodes: Sequence[LumenNode], ref: np.ndarray,
               fixed: Dict[int, str]) -> Optional[List[str]]:
        """Hungarian graph matching between detected lumens and ch(l_p)."""
        if not nodes:
            return []
        fixed_labels = [fixed[n.det] for n in nodes if n.det in fixed]
        free = [n for n in nodes if n.det not in fixed]
        cand = self._observed_children(lp, len(free), exclude=fixed_labels)
        if not self.M.children(lp) or (free and not cand):
            return None
        labs = fixed_labels + cand
        g = self._graph2d(lp, labs)
        det_pts = np.array([n.center for n in nodes])
        lab_pts = np.array([g[l] for l in labs])
        # similarity normalisation (unknown image <-> graph scale/translation)
        if len(nodes) >= 2 and len(labs) >= 2:
            dn = det_pts - det_pts.mean(0)
            ln = lab_pts - lab_pts.mean(0)
        else:
            dn = det_pts - ref
            ln = lab_pts
        dn = dn / (np.linalg.norm(dn, axis=1).max() + 1e-9)
        ln = ln / (np.linalg.norm(ln, axis=1).max() + 1e-9)
        cost = np.linalg.norm(dn[:, None, :] - ln[None, :, :], axis=2)
        big = 1e6
        for i, n in enumerate(nodes):
            if n.det in fixed:
                cost[i, :] = big
                cost[i, labs.index(fixed[n.det])] = 0.0
        r, c = linear_sum_assignment(cost)
        out: List[Optional[str]] = [None] * len(nodes)
        for i, j in zip(r, c):
            out[i] = labs[j]
        return out

    # ---------------------------------------------------- intra-frame (4)
    def _consistent(self, S: LumenSubgraph, n: LumenNode) -> bool:
        if n.label is None or n.label not in self.M:
            return False
        if n.parent is not None and S.nodes[n.parent].label is not None:
            return self.M.parent(n.label) == S.nodes[n.parent].label
        return True

    def _intra_frame(self, S: LumenSubgraph):
        for n in S:
            if n.label is not None and n.label not in self.M:
                n.label = None
        if self.cfg.refine:
            self._refine(S)
        labelled = sorted([n for n in S if n.label is not None], key=lambda n: -n.track_age)
        if not labelled:
            self._recover(S)
            labelled = sorted([n for n in S if n.label is not None], key=lambda n: -n.track_age)
        queue = list(labelled)
        done: set = set()
        while queue:
            z = queue.pop(0)
            if z.det in done or z.label is None:
                continue
            done.add(z.det)
            lp = z.label
            newly: List[LumenNode] = []
            # parent lumen
            if z.parent is not None:
                p = S.nodes[z.parent]
                pl = self.M.parent(lp)
                if pl is not None and (p.label is None or (self.cfg.relabel_inconsistent and p.det not in done and p.label != pl)):
                    p.label = pl
                    newly.append(p)
            # child lumens -> M(l_p)
            kids = S.children(z.det)
            if kids and self.M.children(lp):
                need = any(k.label is None for k in kids) or (
                    self.cfg.relabel_inconsistent and any(self.M.parent(k.label) != lp for k in kids if k.label in self.M))
                if need:
                    fixed = {k.det: k.label for k in kids if k.det in done and k.label is not None
                             and self.M.parent(k.label) == lp}
                    labs = self._match(lp, kids, z.center, fixed)
                    if labs is not None:
                        for k, l in zip(kids, labs):
                            if l is not None and k.label != l:
                                k.label = l
                                newly.append(k)
            # sibling lumens -> M(pa(l_p))
            sibs = S.siblings(z.det)
            pl = self.M.parent(lp)
            if sibs and pl is not None:
                need = any(s.label is None for s in sibs) or (
                    self.cfg.relabel_inconsistent and any(s.label not in self.M.children(pl) or s.label == lp for s in sibs
                                                          if s.label is not None))
                if need:
                    group = [z] + sibs
                    fixed = {z.det: lp}
                    for s in sibs:
                        if s.det in done and s.label in self.M.children(pl) and s.label != lp:
                            fixed[s.det] = s.label
                    ref = self._reference_center(S, z)
                    labs = self._match(pl, group, ref, fixed)
                    if labs is not None:
                        for s, l in zip(group, labs):
                            if l is not None and s.label != l:
                                s.label = l
                                newly.append(s)
            queue.extend(sorted(newly, key=lambda n: -n.track_age))

    def _implied_location(self, n: LumenNode, n_p: int) -> Optional[str]:
        """Location a labelled lumen implies through Eq. (9); None if infeasible."""
        return self.M.ancestor_strict(n.label, n.level - 1 if n_p == 1 else n.level)

    def _refine(self, S: LumenSubgraph):
        """Refine tracklet labels with anatomical constraints (Fig. 1 caption:
        "the labels are refined based on contextual information and anatomical
        constraints from pre-operative airway graph").

        Every labelled lumen implies a scope location through Eq. (9). The
        hypothesis supported by the largest track-age-weighted mass wins; labels
        that are infeasible (no such ancestor), imply another location, violate
        the observed parent/child relation, or duplicate an older sibling's label
        are cleared and later re-derived by intra-frame propagation.
        """
        # ambiguous inclusion (a lumen inside several boxes): pick the container
        # whose label is the lumen's anatomical parent
        for n in S:
            if n.label is None or n.label not in self.M or len(n.parent_candidates) < 2:
                continue
            pa = self.M.parent(n.label)
            ok = [c for c in n.parent_candidates if c in S.nodes and S.nodes[c].label == pa]
            if ok and ok[0] != n.parent:
                S.reparent(n.det, ok[0])
        n_p = len(S.primary())
        lab = [n for n in S if n.label is not None]
        if not lab:
            return
        w = {n.det: 1.0 + max(n.track_age, 0) for n in lab}
        support: Dict[str, float] = {}
        implied = {}
        for n in lab:
            loc = self._implied_location(n, n_p)
            implied[n.det] = loc
            if loc is not None:
                support[loc] = support.get(loc, 0.0) + w[n.det]
        if not support:
            for n in lab:
                n.label = None
            return
        best = max(support, key=support.get)
        for n in lab:
            if implied[n.det] != best:
                n.label = None
        # parent/child consistency (older lumen wins)
        for n in sorted(S, key=lambda x: -x.track_age):
            if n.label is None or n.parent is None:
                continue
            p = S.nodes[n.parent]
            if p.label is not None and self.M.parent(n.label) != p.label:
                if p.track_age > n.track_age:
                    n.label = None
                else:
                    p.label = None
        # duplicated labels among the lumens of one frame (older lumen wins)
        seen = set()
        for n in sorted(S, key=lambda x: -x.track_age):
            if n.label is None:
                continue
            if n.label in seen:
                n.label = None
            else:
                seen.add(n.label)

    def _recover(self, S: LumenSubgraph):
        """No labelled detection (all tracks lost): re-associate the primary lumens
        with the airway around the last known location."""
        prim = S.primary()
        if not prim:
            return
        # the scope most likely advanced into the large lumen that just filled the view
        big = [(l, a) for l, a in self._last_primary if a >= self.cfg.entry_area and l in self.M
               and (l == self.loc or self.M.parent(l) == self.loc)]
        if big:
            self.loc = max(big, key=lambda x: x[1])[0]
        if len(prim) == 1:
            n = prim[0]
            if S.children(n.det) or not self.M.children(self.loc):
                n.label = self.loc
            else:
                labs = self._match(self.loc, prim, self.image_center, {})
                n.label = labs[0] if labs else self.loc
            return
        base = self.loc if len(self.M.children(self.loc)) >= 2 else (self.M.parent(self.loc) or self.loc)
        labs = self._match(base, prim, self.image_center, {})
        if labs:
            for n, l in zip(prim, labs):
                n.label = l

    # ------------------------------------------------------ Eq. 9 voting
    def _vote(self, S: LumenSubgraph) -> Optional[str]:
        n_p = len(S.primary())
        votes: Dict[str, float] = {}
        for n in S:
            if n.label is None or n.label not in self.M:
                continue
            k = n.level
            loc = self.M.ancestor_strict(n.label, k - 1 if n_p == 1 else k)
            if loc is None:
                continue
            votes[loc] = votes.get(loc, 0.0) + 1.0 + 1e-6 * max(n.track_age, 0)  # older tracks break ties
        self.last_votes = votes
        if not votes:
            return None
        return max(votes, key=votes.get)

    # ------------------------------------------------------------ gallery
    def _add_record(self, label: str, S: LumenSubgraph, t: int, frame):
        tr = [n for n in S if n.track_id is not None]
        self.gallery[label] = GalleryRecord(
            label=label, t=t, roll=self.roll,
            centers={n.track_id: n.center for n in tr}, boxes={n.track_id: n.box.copy() for n in tr},
            labels={n.track_id: n.label for n in tr}, n_lumens=len(S),
            keyframe=None if frame is None else frame.copy(), updated=t)

    def _update_gallery(self, S: LumenSubgraph, t: int, frame):
        rec = self.gallery.get(self.loc)
        if rec is None:
            self.new_branch_entered = True
            self._add_record(self.loc, S, t, frame)
        elif len(S) > rec.n_lumens:  # more comprehensive hierarchy of l_p
            kf = rec.keyframe
            self._add_record(self.loc, S, t, frame)
            if kf is not None and self.gallery[self.loc].keyframe is None:
                self.gallery[self.loc].keyframe = kf
