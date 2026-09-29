"""Semantic airway graph M (paper Sec. III, preamble and Fig. 1(d)).

Pre-operatively the airway is segmented, skeletonised (Lee et al. 3-D medial-axis
thinning, ref. [27]) and the centre line is split at every bifurcation. The start
and end point of each segment are kept to form a topological tree. The tree is
transformed into a *standard coordinate frame* in which

    * y is aligned with the direction of the trachea,
    * x lies in the plane spanned by the origin and the ends of the left and
      right main bronchi,
    * z is orthogonal to both.

Every branch receives a unique label and stores its parent and children. Labels
follow Fig. 1(d): the trachea is ``"0"``, its children ``"00"``, ``"01"``, their
children ``"000"``, ``"001"``, ``"010"`` ... i.e. child *m* of branch *l* is
``l + str(m)``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Branch:
    """One airway segment between two bifurcations."""

    label: str
    parent: Optional[str]
    children: List[str] = field(default_factory=list)
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end: np.ndarray = field(default_factory=lambda: np.zeros(3))
    points: Optional[np.ndarray] = None  # (N,3) centre-line polyline start -> end
    radius: Optional[float] = None
    generation: int = 0
    name: Optional[str] = None  # optional anatomical name, e.g. "RMB"
    diam_s: Optional[np.ndarray] = None  # arc lengths (from the start) where the lumen diameter was measured
    diam: Optional[np.ndarray] = None    # lumen diameter at those points (see measure_diameters)
    diam_area: Optional[np.ndarray] = None  # equivalent-circle diameter of the whole cross-section

    @property
    def direction(self) -> np.ndarray:
        d = np.asarray(self.end, float) - np.asarray(self.start, float)
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([0.0, 1.0, 0.0])

    @property
    def length(self) -> float:
        if self.points is not None and len(self.points) > 1:
            return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())
        return float(np.linalg.norm(np.asarray(self.end) - np.asarray(self.start)))

    def point_at(self, s: float) -> np.ndarray:
        """Point at arc length ``s`` from the start (clamped to the segment)."""
        pts = self.points if self.points is not None and len(self.points) > 1 else np.stack(
            [self.start, self.end])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = float(np.clip(s, 0.0, cum[-1]))
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = min(i, len(seg) - 1)
        if seg[i] == 0:
            return pts[i].astype(float)
        t = (s - cum[i]) / seg[i]
        return (1 - t) * pts[i] + t * pts[i + 1]

    def diameter_at(self, s: float) -> Optional[float]:
        """Local lumen diameter at arc length ``s`` from the branch start (None if unmeasured)."""
        if self.diam is None or self.diam_s is None or len(self.diam) == 0:
            return None
        return float(np.interp(s, self.diam_s, self.diam))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "parent": self.parent,
            "children": list(self.children),
            "start": np.asarray(self.start, float).tolist(),
            "end": np.asarray(self.end, float).tolist(),
            "points": None if self.points is None else np.asarray(self.points, float).tolist(),
            "radius": self.radius,
            "generation": self.generation,
            "name": self.name,
            "diam_s": None if self.diam_s is None else np.round(np.asarray(self.diam_s, float), 3).tolist(),
            "diam": None if self.diam is None else np.round(np.asarray(self.diam, float), 3).tolist(),
            "diam_area": None if self.diam_area is None else np.round(np.asarray(self.diam_area, float), 3).tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Branch":
        return Branch(
            label=str(d["label"]),
            parent=None if d.get("parent") in (None, "") else str(d["parent"]),
            children=[str(c) for c in d.get("children", [])],
            start=np.asarray(d["start"], float),
            end=np.asarray(d["end"], float),
            points=None if d.get("points") is None else np.asarray(d["points"], float),
            radius=d.get("radius"),
            generation=int(d.get("generation", 0)),
            name=d.get("name"),
            diam_s=None if d.get("diam_s") is None else np.asarray(d["diam_s"], float),
            diam=None if d.get("diam") is None else np.asarray(d["diam"], float),
            diam_area=None if d.get("diam_area") is None else np.asarray(d["diam_area"], float),
        )


# --------------------------------------------------------------------------- #
# Airway graph
# --------------------------------------------------------------------------- #
class AirwayGraph:
    """Labelled airway tree ``M`` in the standard coordinate frame."""

    def __init__(self, branches: Dict[str, Branch], root: str = "0"):
        self.branches: Dict[str, Branch] = branches
        self.root = root
        self._recompute_generations()

    # ------------------------------------------------------------------ basics
    def __contains__(self, label) -> bool:
        return label in self.branches

    def __getitem__(self, label: str) -> Branch:
        return self.branches[label]

    def __len__(self) -> int:
        return len(self.branches)

    def labels(self) -> List[str]:
        return list(self.branches.keys())

    def parent(self, label: str) -> Optional[str]:
        return self.branches[label].parent

    def children(self, label: str) -> List[str]:
        return list(self.branches[label].children)

    def siblings(self, label: str) -> List[str]:
        p = self.parent(label)
        return [] if p is None else [c for c in self.children(p) if c != label]

    def generation(self, label: str) -> int:
        return self.branches[label].generation

    def ancestor(self, label: str, k: int) -> str:
        """g^k(l): the branch k levels above ``label`` (clamped at the trachea)."""
        cur = label
        for _ in range(max(0, int(k))):
            p = self.branches[cur].parent
            if p is None:
                break
            cur = p
        return cur

    def ancestor_strict(self, label: str, k: int) -> Optional[str]:
        """g^k(l) without clamping: None if the tree has fewer than k levels above l."""
        cur = label
        for _ in range(max(0, int(k))):
            cur = self.branches[cur].parent
            if cur is None:
                return None
        return cur

    def ancestors(self, label: str) -> List[str]:
        out, cur = [], self.branches[label].parent
        while cur is not None:
            out.append(cur)
            cur = self.branches[cur].parent
        return out

    def lca(self, a: str, b: str) -> str:
        anc_a = [a] + self.ancestors(a)
        set_b = set([b] + self.ancestors(b))
        for x in anc_a:
            if x in set_b:
                return x
        return self.root

    def tree_distance(self, a: str, b: str) -> int:
        """Number of edges between two branches in the airway tree."""
        if a == b:
            return 0
        c = self.lca(a, b)
        return self.generation(a) + self.generation(b) - 2 * self.generation(c)

    def is_descendant(self, a: str, b: str) -> bool:
        """True if ``a`` is (strictly) below ``b``."""
        return b in self.ancestors(a)

    def subgraph(self, label: str) -> List[str]:
        """M(l_p): labels of the subtree rooted at ``label``."""
        out, q = [], deque([label])
        while q:
            x = q.popleft()
            out.append(x)
            q.extend(self.branches[x].children)
        return out

    def max_generation(self) -> int:
        return max(b.generation for b in self.branches.values())

    def _recompute_generations(self):
        self.invalidate()
        if self.root not in self.branches:
            return
        q = deque([(self.root, 0)])
        while q:
            lab, g = q.popleft()
            self.branches[lab].generation = g
            for c in self.branches[lab].children:
                q.append((c, g + 1))

    # ------------------------------------------------------- geometry for Alg.2
    @staticmethod
    def tangent_basis(direction: np.ndarray, ref_x: np.ndarray = np.array([1.0, 0.0, 0.0])
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal basis (e1, e2, d) of the plane perpendicular to ``direction``.

        e1 is the projection of the standard x-axis onto the tangent plane, so the
        2-D graph is expressed with a consistent "zero roll" reference. With
        e2 = d x e1 the triple (e1, e2, d) is a right-handed camera frame
        (image-right, image-down, viewing direction).
        """
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        ref = np.asarray(ref_x, float)
        e1 = ref - np.dot(ref, d) * d
        if np.linalg.norm(e1) < 1e-6:  # direction parallel to x -> fall back to z
            ref = np.array([0.0, 0.0, 1.0])
            e1 = ref - np.dot(ref, d) * d
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        return e1, e2, d

    @staticmethod
    def _min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Rotation matrix taking unit vector a onto unit vector b with no twist."""
        v = np.cross(a, b)
        s, c = np.linalg.norm(v), float(np.dot(a, b))
        if s < 1e-9:
            return np.eye(3)
        k = v / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + s * K + (1 - c) * K @ K

    def frame(self, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """In-plane basis (e1, e2, d) of a branch's tangent plane.

        The trachea uses the projected standard x-axis (so that Eq. (6) measures
        roll against e_x). Every other branch inherits its parent's basis by
        minimal-rotation (parallel) transport along the tree, so a scope that
        advances without twisting keeps the same roll in every branch and the
        roll accumulated by Eq. (8) stays meaningful across bifurcations.
        (With ``transport_frames=False`` every branch uses the projected x-axis.)
        """
        if not getattr(self, "transport_frames", True):
            return self.tangent_basis(self.branches[label].direction)
        cache = self.__dict__.setdefault("_frames", {})
        if label in cache:
            return cache[label]
        b = self.branches[label]
        if b.parent is None or b.parent not in self.branches:
            f = self.tangent_basis(b.direction)
        else:
            e1p, e2p, dp = self.frame(b.parent)
            R = self._min_rotation(dp, b.direction)
            e1 = R @ e1p
            e1 = e1 - np.dot(e1, b.direction) * b.direction
            e1 /= np.linalg.norm(e1) + 1e-12
            f = (e1, np.cross(b.direction, e1), b.direction)
        cache[label] = f
        return f

    def invalidate(self):
        self.__dict__.pop("_frames", None)

    def intersection_angle(self, parent: str, child: str) -> float:
        """Angle (rad) between a branch and one of its children."""
        a, b = self.branches[parent].direction, self.branches[child].direction
        return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))

    def projected_children(self, label: str, truncate: float,
                           children: Optional[Sequence[str]] = None,
                           mirror: bool = False) -> Dict[str, np.ndarray]:
        """2-D graph of ``label``'s children on its tangent plane (Fig. 2, step 2).

        Each child is truncated ``truncate`` (graph units, e.g. mm) along its
        centre line from its start and the truncated point, taken relative to the
        end of ``label``, is projected onto ``label``'s tangent plane.
        Returned coordinates are *image-like* (x right, y down) at zero roll.
        """
        br = self.branches[label]
        e1, e2, _ = self.frame(label)
        origin = np.asarray(br.end, float)
        out = {}
        for c in (children if children is not None else br.children):
            cb = self.branches[c]
            p = cb.point_at(min(truncate, cb.length)) - origin
            q = np.array([np.dot(p, e1), np.dot(p, e2)])
            if mirror:
                q[0] = -q[0]
            out[c] = q
        return out

    # ---------------------------------------------------------------- standard
    def to_standard_frame(self, left_main: str, right_main: str) -> "AirwayGraph":
        """Transform into the paper's standard coordinate frame (in place).

        Origin = carina (end of trachea), y = trachea direction,
        x = in-plane component of (right-main end - left-main end), z = x cross y.
        """
        tr = self.branches[self.root]
        origin = np.asarray(tr.end, float)
        y = tr.direction
        v = np.asarray(self.branches[right_main].end, float) - np.asarray(
            self.branches[left_main].end, float)
        x = v - np.dot(v, y) * y
        x /= np.linalg.norm(x) + 1e-12
        z = np.cross(x, y)
        R = np.stack([x, y, z])  # rows = new axes

        def tf(p):
            return (R @ (np.asarray(p, float) - origin).T).T

        for b in self.branches.values():
            b.start, b.end = tf(b.start), tf(b.end)
            if b.points is not None:
                b.points = tf(b.points)
        # remember the transform so results can be mapped back to CT coordinates:
        # p_std = R (p_ct - origin)   <=>   p_ct = R^T p_std + origin
        self.ct_origin, self.ct_R = origin, R
        self.invalidate()
        return self

    def to_ct(self, p: np.ndarray) -> np.ndarray:
        """Map standard-frame points back to the input (CT) coordinates."""
        R = getattr(self, "ct_R", None)
        if R is None:
            return np.asarray(p, float)
        return np.asarray(p, float) @ R + self.ct_origin

    # --------------------------------------------------------------- relabel
    def relabel_canonical(self) -> "AirwayGraph":
        """Assign Fig. 1(d)-style labels ('0', '00', '01', ...).

        Children are ordered by the angle of their projection on the parent's
        tangent plane, measured from e1 (the projected standard x-axis). This is a
        deterministic, rotation-invariant ordering; the actual anatomical name
        (if known) is preserved in ``Branch.name``.
        """
        mapping = {self.root: "0"}
        q = deque([self.root])
        while q:
            old = q.popleft()
            br = self.branches[old]
            proj = self.projected_children(old, truncate=max(1e-3, 0.5 * np.mean(
                [self.branches[c].length for c in br.children])) if br.children else 1.0)
            ordered = sorted(br.children, key=lambda c: np.arctan2(proj[c][1], proj[c][0]) % (2 * np.pi))
            for m, c in enumerate(ordered):
                mapping[c] = mapping[old] + str(m) if m < 10 else mapping[old] + f"[{m}]"
                q.append(c)
        new = {}
        for old, b in self.branches.items():
            nb = Branch(label=mapping[old], parent=None if b.parent is None else mapping[b.parent],
                        children=[mapping[c] for c in b.children], start=b.start, end=b.end,
                        points=b.points, radius=b.radius, name=b.name, diam_s=b.diam_s, diam=b.diam,
                        diam_area=b.diam_area)
            new[nb.label] = nb
        for nb in new.values():  # keep children ordered like their labels
            nb.children.sort()
        self.branches, self.root = new, "0"
        self.invalidate()
        self._recompute_generations()
        return self

    # ------------------------------------------------------------------- I/O
    def to_json(self, path: str):
        with open(path, "w") as f:
            d = {"root": self.root, "branches": [b.to_dict() for b in self.branches.values()]}
            if getattr(self, "ct_R", None) is not None:
                d["ct_frame"] = {"origin": np.asarray(self.ct_origin).tolist(), "R": np.asarray(self.ct_R).tolist()}
            json.dump(d, f, indent=1)

    @staticmethod
    def from_json(path: str) -> "AirwayGraph":
        with open(path) as f:
            d = json.load(f)
        brs = {str(b["label"]): Branch.from_dict(b) for b in d["branches"]}
        g = AirwayGraph(brs, root=str(d.get("root", "0")))
        if "ct_frame" in d:
            g.ct_origin = np.asarray(d["ct_frame"]["origin"], float)
            g.ct_R = np.asarray(d["ct_frame"]["R"], float)
        return g

    # --------------------------------------------------------------- builders
    @staticmethod
    def from_polylines(polylines: Iterable[np.ndarray], root_hint: Optional[np.ndarray] = None,
                       patient_right: Optional[np.ndarray] = None, merge_tol: float = 1.0,
                       min_branch_length: float = 0.0) -> "AirwayGraph":
        """Build M from centre-line polylines (e.g. a VMTK/3D-Slicer centre-line).

        Polylines are split at shared end points (within ``merge_tol``) to form a
        tree. ``root_hint`` is a point near the top of the trachea (default: the
        end point with the largest coordinate along the polyline's principal axis
        that has degree 1 and is the most "superior" – we pick the end point
        farthest from the tree's centroid). ``patient_right`` is a 3-D vector
        pointing to the patient's right in input coordinates, used to identify the
        right main bronchus; without it the right main bronchus is taken as the
        more vertical main bronchus (anatomical prior).
        """
        polylines = [np.asarray(p, float) for p in polylines if len(p) >= 2]
        # ---- cluster end points into nodes
        ends = []
        for i, p in enumerate(polylines):
            ends.append((i, 0, p[0]))
            ends.append((i, 1, p[-1]))
        node_pos: List[np.ndarray] = []
        end_node = {}
        for i, e, pt in ends:
            for n, q in enumerate(node_pos):
                if np.linalg.norm(q - pt) <= merge_tol:
                    end_node[(i, e)] = n
                    break
            else:
                node_pos.append(pt.copy())
                end_node[(i, e)] = len(node_pos) - 1
        adj: Dict[int, List[Tuple[int, int]]] = {n: [] for n in range(len(node_pos))}
        for i, p in enumerate(polylines):
            a, b = end_node[(i, 0)], end_node[(i, 1)]
            adj[a].append((b, i))
            adj[b].append((a, i))
        return AirwayGraph._from_node_graph(node_pos, adj, polylines, root_hint, patient_right,
                                            min_branch_length)

    @staticmethod
    def _from_node_graph(node_pos, adj, polylines, root_hint, patient_right, min_branch_length,
                         radii: Optional[Sequence[float]] = None):
        leaves = [n for n, nb in adj.items() if len(nb) == 1]
        if root_hint is not None:
            root_node = min(leaves or adj.keys(), key=lambda n: np.linalg.norm(node_pos[n] - root_hint))
        else:
            # the trachea is the largest terminal segment: volume ~ length * r^2 for a mask,
            # length for a bare centre line
            def score(n):
                pid = adj[n][0][1]
                length = float(np.linalg.norm(np.diff(polylines[pid], axis=0), axis=1).sum())
                return length * (radii[pid] ** 2 if radii is not None else 1.0)
            root_node = max(leaves, key=score)
        # ---- BFS rooted tree
        branches: Dict[str, Branch] = {}
        visited = {root_node}
        q = deque()
        (nb, pid), = adj[root_node][:1] or [(None, None)]
        if nb is None:
            raise ValueError("empty centre-line graph")
        tmp_id = 0

        def make(pid_, a, b, parent):
            nonlocal tmp_id
            pts = polylines[pid_]
            rad = None if radii is None else float(radii[pid_])
            if np.linalg.norm(pts[0] - node_pos[a]) > np.linalg.norm(pts[-1] - node_pos[a]):
                pts = pts[::-1]
            lab = f"t{tmp_id}"
            tmp_id += 1
            branches[lab] = Branch(label=lab, parent=parent, start=pts[0].copy(), end=pts[-1].copy(),
                                   points=pts.copy(), radius=rad)
            if parent is not None:
                branches[parent].children.append(lab)
            return lab

        root_lab = make(pid, root_node, nb, None)
        if radii is not None:
            branches[root_lab].radius = float(radii[pid])
        visited.add(nb)
        q.append((nb, root_lab))
        while q:
            node, lab = q.popleft()
            for nxt, pid_ in adj[node]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                c = make(pid_, node, nxt, lab)
                q.append((nxt, c))
        g = AirwayGraph(branches, root=root_lab)
        g._collapse_degree_two()
        if min_branch_length > 0:
            g._prune_short_leaves(min_branch_length)
            g._collapse_degree_two()
        # ---- standard frame
        tr = g.branches[g.root]
        if len(tr.children) < 2:
            raise ValueError("trachea must bifurcate into two main bronchi")
        mains = sorted(tr.children, key=lambda c: g.branches[c].length, reverse=True)[:2]
        if patient_right is not None:
            pr = np.asarray(patient_right, float)
            right = max(mains, key=lambda c: np.dot(g.branches[c].end - tr.end, pr))
        else:  # right main bronchus is more vertical (smaller angle to trachea)
            right = min(mains, key=lambda c: g.intersection_angle(g.root, c))
        left = [c for c in mains if c != right][0]
        g.branches[right].name = g.branches[right].name or "RMB"
        g.branches[left].name = g.branches[left].name or "LMB"
        g.branches[g.root].name = g.branches[g.root].name or "Trachea"
        g.to_standard_frame(left, right)
        return g.relabel_canonical()

    def _collapse_degree_two(self):
        """Merge branches that have exactly one child (no real bifurcation)."""
        changed = True
        while changed:
            changed = False
            for lab in list(self.branches):
                b = self.branches.get(lab)
                if b is None or len(b.children) != 1:
                    continue
                c = self.branches.pop(b.children[0])
                pts = np.concatenate([b.points, c.points[1:]]) if (b.points is not None and c.points is not None) else None
                b.end, b.points, b.children = c.end, pts, c.children
                for cc in c.children:
                    self.branches[cc].parent = lab
                changed = True
        self._recompute_generations()

    def _prune_short_leaves(self, min_len):
        changed = True
        while changed:
            changed = False
            for lab in list(self.branches):
                b = self.branches.get(lab)
                if b is None or b.children or b.parent is None:
                    continue
                if b.length < min_len and len(self.branches[b.parent].children) > 1:
                    self.branches[b.parent].children.remove(lab)
                    del self.branches[lab]
                    changed = True

    @staticmethod
    def from_mask(mask: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0),
                  origin: Sequence[float] = (0.0, 0.0, 0.0), root_hint: Optional[np.ndarray] = None,
                  patient_right: Optional[np.ndarray] = None, min_branch_length: float = 3.0,
                  smooth_sigma: float = 1.0) -> "AirwayGraph":
        """Build M from a binary airway segmentation (ref. [26]) by 3-D Lee
        skeletonisation (ref. [27]) and centre-line splitting at bifurcations.

        ``mask`` is indexed (i, j, k); ``spacing``/``origin`` map indices to mm.
        """
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        from skimage.morphology import skeletonize

        m = mask.astype(bool)
        if smooth_sigma > 0:  # Lee thinning can erase very regular voxel tubes; smooth first
            m = gaussian_filter(m.astype(np.float32), smooth_sigma) > 0.5
        m = np.pad(m, 2)
        skel = skeletonize(m, method="lee")[2:-2, 2:-2, 2:-2] > 0
        edt = distance_transform_edt(m, sampling=spacing)[2:-2, 2:-2, 2:-2]
        idx = np.argwhere(skel)
        if len(idx) == 0:
            raise ValueError("empty skeleton")
        pos = idx * np.asarray(spacing, float) + np.asarray(origin, float)
        index = {tuple(v): n for n, v in enumerate(idx)}
        offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
        nbrs = [[index[t] for t in ((v[0] + a, v[1] + b, v[2] + c) for a, b, c in offs) if t in index]
                for v in idx]
        deg = np.array([len(n) for n in nbrs])
        is_node = deg != 2
        # cluster adjacent junction voxels into one node
        node_of = -np.ones(len(idx), int)
        node_pos = []
        for v in np.where(is_node)[0]:
            if node_of[v] >= 0:
                continue
            comp, q = [v], deque([v])
            node_of[v] = len(node_pos)
            while q:
                u = q.popleft()
                for w in nbrs[u]:
                    if is_node[w] and deg[w] > 2 and deg[u] > 2 and node_of[w] < 0:
                        node_of[w] = node_of[v]
                        comp.append(w)
                        q.append(w)
            node_pos.append(pos[comp].mean(0))
        # trace edges between nodes
        polylines, adj, radii = [], {n: [] for n in range(len(node_pos))}, []
        seen_edges = set()
        for v in np.where(is_node)[0]:
            for w in nbrs[v]:
                if node_of[w] == node_of[v] and is_node[w]:
                    continue
                path, prev, cur = [v, w], v, w
                while not is_node[cur]:
                    nxt = [x for x in nbrs[cur] if x != prev]
                    if not nxt:
                        break
                    prev, cur = cur, nxt[0]
                    path.append(cur)
                a, b = node_of[v], node_of[cur] if is_node[cur] else -1
                if b < 0:
                    continue
                key = (min(a, b), max(a, b), min(path[1], path[-2]) if len(path) > 2 else -1)
                if key in seen_edges or a == b:
                    continue
                seen_edges.add(key)
                pl = np.concatenate([[node_pos[a]], pos[path[1:-1]], [node_pos[b]]])
                adj[a].append((b, len(polylines)))
                adj[b].append((a, len(polylines)))
                polylines.append(pl)
                inner = path[1:-1] or path
                radii.append(float(np.median([edt[tuple(idx[v])] for v in inner])))
        return AirwayGraph._from_node_graph(node_pos, adj, polylines, root_hint, patient_right,
                                            min_branch_length, radii)

    def measure_diameters(self, mask: np.ndarray, spacing: Sequence[float], origin: Sequence[float],
                          step: float = 0.5, grid: float = 0.25) -> "AirwayGraph":
        """Measure the lumen diameter along every centre line from the segmentation.

        At points every ``step`` mm the mask is sampled on the plane perpendicular to the
        local centre-line direction. Two diameters are stored per point:

        * ``diam``       2 x the in-plane distance from the centre-line point to the nearest
                         lumen wall (the diameter *at that point*; stays local even where
                         the plane still cuts the neighbouring lumen at a bifurcation);
        * ``diam_area``  equivalent-circle diameter 2*sqrt(A/pi) of the connected lumen
                         region containing the point.

        ``mask`` is indexed (x, y, z) in the input (CT) frame; needs ``ct_R``/``ct_origin``.
        """
        from scipy.ndimage import distance_transform_edt, label as cc_label, map_coordinates

        R = getattr(self, "ct_R", None)
        if R is None:
            raise ValueError("graph has no CT transform (build it with from_mask / from_polylines)")
        m = mask.astype(np.float32)
        sp, org = np.asarray(spacing, float), np.asarray(origin, float)
        for b in self.branches.values():
            L = b.length
            ss = np.unique(np.r_[np.arange(0.0, L, step), L])
            half = max(3.0 * (b.radius or 2.0), 6.0)
            n = int(round(2 * half / grid)) + 1
            ax = np.linspace(-half, half, n)
            U, V = np.meshgrid(ax, ax, indexing="ij")
            ctr = n // 2
            d_in, d_ar = [], []
            for s_ in ss:
                p = b.point_at(s_)
                q = b.point_at(min(L, s_ + 1.5)) - b.point_at(max(0.0, s_ - 1.5))
                d = q / (np.linalg.norm(q) + 1e-9)
                e1, e2, _ = self.tangent_basis(d)
                P = p[None, None, :] + U[..., None] * e1 + V[..., None] * e2       # standard frame
                Pct = P @ R + self.ct_origin                                        # -> CT (mm)
                idx = ((Pct - org) / sp).reshape(-1, 3).T
                sec = map_coordinates(m, idx, order=1, mode="constant", cval=0.0).reshape(n, n) > 0.5
                lab, _ = cc_label(sec)
                k = lab[ctr, ctr]
                if k == 0:  # centre-line point just outside the voxel lumen: nearest region
                    ys, xs = np.nonzero(lab)
                    if len(ys) == 0:
                        d_in.append(np.nan)
                        d_ar.append(np.nan)
                        continue
                    j = np.argmin((ys - ctr) ** 2 + (xs - ctr) ** 2)
                    k = lab[ys[j], xs[j]]
                    cy, cx = ys[j], xs[j]
                else:
                    cy, cx = ctr, ctr
                reg = lab == k
                d_ar.append(2.0 * np.sqrt(float(reg.sum()) * grid * grid / np.pi))
                d_in.append(2.0 * float(distance_transform_edt(reg)[cy, cx]) * grid)
            d_in, d_ar = np.asarray(d_in), np.asarray(d_ar)
            good = np.isfinite(d_in)
            if good.any():
                b.diam_s, b.diam, b.diam_area = ss[good], d_in[good], d_ar[good]
        return self

    @staticmethod
    def from_vtk(path: str, **kw) -> "AirwayGraph":
        """Build M from a VTK/VTP poly-data centre line (e.g. 3D Slicer / VMTK)."""
        import vtk  # optional dependency
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkXMLPolyDataReader() if path.endswith(".vtp") else vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        pd = reader.GetOutput()
        pts = vtk_to_numpy(pd.GetPoints().GetData())
        lines, cells = [], pd.GetLines()
        cells.InitTraversal()
        ids = vtk.vtkIdList()
        while cells.GetNextCell(ids):
            lines.append(pts[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]])
        return AirwayGraph.from_polylines(lines, **kw)
