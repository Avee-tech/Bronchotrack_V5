"""Synthetic airway + virtual bronchoscopy sequence generator.

Not part of the paper – used to verify the pipeline end-to-end without patient
data. It produces

* a procedural airway tree (``AirwayGraph`` in the standard frame),
* a camera trajectory with forward *and* retraction movements (Fig. 4),
* rendered frames, ground-truth lumen boxes that follow the Fig. 3 annotation
  rule (primary / secondary level, parent box = minimum area enclosing its
  children), GT branch labels per box and the GT camera branch per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import zlib

import numpy as np

from .airway_graph import AirwayGraph, Branch


# --------------------------------------------------------------------------- #
def make_tree(generations: int = 5, seed: int = 0, trachea_len: float = 100.0,
              trachea_r: float = 8.0) -> AirwayGraph:
    """Random bifurcating (occasionally trifurcating) tree, Horsfield-like scaling."""
    rng = np.random.default_rng(seed)
    branches: Dict[str, Branch] = {}
    pts = np.array([[0, -trachea_len, 0], [0, 0, 0]], float)
    branches["0"] = Branch("0", None, [], pts[0], pts[1], pts, trachea_r, 0, "Trachea")

    def grow(lab, gen):
        if gen >= generations:
            return
        p = branches[lab]
        d = p.direction
        e1, e2, _ = AirwayGraph.tangent_basis(d)
        n = 3 if (gen >= 2 and rng.random() < 0.15) else 2
        phi0 = rng.uniform(0, 2 * np.pi) if gen > 0 else 0.0
        for m in range(n):
            phi = phi0 + 2 * np.pi * m / n + rng.normal(0, 0.25)
            ang = np.radians(rng.uniform(25, 50) if gen > 0 else (25 if m == 0 else 45))
            v = np.cos(ang) * d + np.sin(ang) * (np.cos(phi) * e1 + np.sin(phi) * e2)
            L = p.length * rng.uniform(0.55, 0.8) if gen > 0 else (25.0 if m == 0 else 45.0)
            r = p.radius * rng.uniform(0.7, 0.82)
            c = lab + str(m)
            s = p.end
            e = s + L * v
            branches[c] = Branch(c, lab, [], s.copy(), e, np.linspace(s, e, 12), r, gen + 1)
            p.children.append(c)
            grow(c, gen + 1)

    grow("0", 0)
    branches["00"].name, branches["01"].name = "RMB", "LMB"
    g = AirwayGraph(branches, "0")
    return g.to_standard_frame(left_main="01", right_main="00")


# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    pos: np.ndarray
    fwd: np.ndarray
    right: np.ndarray  # image x
    down: np.ndarray   # image y


def _transport(cam: Camera, new_fwd: np.ndarray) -> Camera:
    """Rotate the camera frame minimally so that fwd -> new_fwd (no added roll)."""
    a, b = cam.fwd, new_fwd / np.linalg.norm(new_fwd)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.dot(a, b))
    if s < 1e-9:
        return Camera(cam.pos, b, cam.right, cam.down)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * K @ K
    return Camera(cam.pos, b, R @ cam.right, R @ cam.down)


def _roll(cam: Camera, ang: float) -> Camera:
    c, s = np.cos(ang), np.sin(ang)
    return Camera(cam.pos, cam.fwd, c * cam.right + s * cam.down, -s * cam.right + c * cam.down)


def plan_path(g: AirwayGraph, targets: List[str], depth_frac: float = 0.6) -> List[Tuple[str, float]]:
    """Waypoints (branch, arc length) visiting ``targets`` in order with retraction
    to the lowest common ancestor between consecutive targets."""
    way = [("0", 5.0)]
    cur = "0"
    for tg in targets:
        lca = g.lca(cur, tg)
        # retract from cur up to lca
        node = cur
        while node != lca:
            way.append((node, 2.0))
            node = g.parent(node)
        # descend from lca to tg
        chain = [tg] + [a for a in g.ancestors(tg) if a != lca and g.is_descendant(a, lca)]
        chain = chain[::-1]
        way.append((lca, g[lca].length - 2.0))
        for b in chain:
            way.append((b, g[b].length * (depth_frac if b == tg else 0.9)))
        cur = tg
    return way


def _pos(g: AirwayGraph, b: str, s: float) -> np.ndarray:
    return g[b].point_at(s)


def interpolate_path(g: AirwayGraph, way: List[Tuple[str, float]], speed: float = 1.2
                     ) -> List[Tuple[str, float]]:
    """Dense (branch, s) samples moving at ``speed`` mm/frame along the tree."""
    out = []
    for (b0, s0), (b1, s1) in zip(way[:-1], way[1:]):
        if b0 == b1:
            n = max(1, int(abs(s1 - s0) / speed))
            out += [(b0, s0 + (s1 - s0) * k / n) for k in range(n)]
        elif g.parent(b1) == b0:  # forward: finish b0, enter b1
            n = max(1, int(abs(g[b0].length - s0) / speed))
            out += [(b0, s0 + (g[b0].length - s0) * k / n) for k in range(n)]
            n = max(1, int(s1 / speed))
            out += [(b1, s1 * k / n) for k in range(n)]
        elif g.parent(b0) == b1:  # retract: back to b0 start, into b1 end
            n = max(1, int(s0 / speed))
            out += [(b0, s0 - s0 * k / n) for k in range(n)]
            n = max(1, int(abs(g[b1].length - s1) / speed))
            out += [(b1, g[b1].length - (g[b1].length - s1) * k / n) for k in range(n)]
        else:
            raise ValueError(f"non-adjacent waypoints {b0}->{b1}")
    out.append(way[-1])
    return out


# --------------------------------------------------------------------------- #
@dataclass
class SynthFrame:
    image: np.ndarray
    gt: List[Tuple[int, np.ndarray, str, int]]   # (gt_id, box, label, level)
    location: str
    roll: float


class Renderer:
    def __init__(self, g: AirwayGraph, size: int = 256, fov_deg: float = 100.0, seed: int = 0,
                 near_bif: float = 12.0, far_child: float = 45.0, anchor: float = 8.0,
                 min_child_px: float = 6.0, max_frac: float = 0.9):
        self.g, self.W = g, size
        self.f = (size / 2) / np.tan(np.radians(fov_deg) / 2)
        self.rng = np.random.default_rng(seed)
        self.near_bif, self.far_child, self.anchor = near_bif, far_child, anchor
        self.min_child_px = min_child_px
        self.max_frac = max_frac
        self.ids = {l: i + 1 for i, l in enumerate(sorted(g.labels()))}
        self.tex = {l: np.random.default_rng(zlib.crc32(l.encode())) for l in g.labels()}

    def project(self, cam: Camera, P: np.ndarray, r: float) -> Optional[np.ndarray]:
        q = P - cam.pos
        z = float(np.dot(q, cam.fwd))
        if z <= 1.0:
            return None
        u = self.W / 2 + self.f * np.dot(q, cam.right) / z
        v = self.W / 2 + self.f * np.dot(q, cam.down) / z
        h = self.f * r / z
        return np.array([u - h, v - h, u + h, v + h])

    def lumen_box(self, cam, lab):
        b = self.g[lab]
        return self.project(cam, b.point_at(min(self.anchor, b.length)), b.radius)

    def gt_boxes(self, cam: Camera, branch: str, s: float) -> List[Tuple[str, np.ndarray, int]]:
        """Fig. 3 rule. Far from the bifurcation the current branch's far end is one
        primary lumen enclosing its children; near it, children are primary and
        their resolvable children are secondary."""
        g = self.g
        dist = g[branch].length - s
        out = []

        def inside(bx):
            return bx is not None and bx[2] > 2 and bx[3] > 2 and bx[0] < self.W - 2 and bx[1] < self.W - 2 \
                and (bx[2] - bx[0]) >= 4

        def visible_kids(lab):
            ks = [(c, self.lumen_box(cam, c)) for c in g.children(lab)]
            return [(c, bx) for c, bx in ks if inside(bx) and (bx[2] - bx[0]) >= self.min_child_px]

        def union(bx, others):
            if not others:
                return bx
            o = np.array([b2 for _, b2 in others])
            return np.r_[np.minimum(bx[:2], o[:, :2].min(0) - 1), np.maximum(bx[2:], o[:, 2:].max(0) + 1)]

        kids = visible_kids(branch)
        if dist > self.near_bif or not kids:
            e = g[branch]
            bx = self.project(cam, e.end, e.radius)
            if inside(bx):
                out.append((branch, union(bx, kids), 1))       # parent = min area enclosing children
                out += [(c, b2, 2) for c, b2 in kids]
        else:
            for c, bx in kids:
                gk = visible_kids(c)
                out.append((c, union(bx, gk), 1))
                out += [(k, b2, 2) for k, b2 in gk]
        clipped = []
        for lab, bx, lvl in out:
            if (bx[2] - bx[0]) > self.max_frac * self.W or (bx[3] - bx[1]) > self.max_frac * self.W:
                continue  # a lumen filling the view is wall, not annotated as a lumen
            bx = np.clip(bx, 0, self.W - 1)
            if bx[2] - bx[0] >= 3 and bx[3] - bx[1] >= 3:
                clipped.append((lab, bx, lvl))
        return clipped

    def render(self, boxes) -> np.ndarray:
        W = self.W
        yy, xx = np.mgrid[0:W, 0:W]
        rr = np.hypot(xx - W / 2, yy - W / 2) / (W / 2)
        base = np.stack([150 - 40 * rr, 110 - 40 * rr, 215 - 60 * rr], -1)  # BGR pink/red mucosa
        img = base + self.rng.normal(0, 6, base.shape)
        img = img.astype(np.float32)
        # leaf-level lumens are dark holes; parents are only implied by their children
        leafs = [(l, b) for l, b, lvl in boxes if not any(
            lvl2 > lvl and np.all(b2[:2] >= b[:2] - 1) and np.all(b2[2:] <= b[2:] + 1) for _, b2, lvl2 in boxes)]
        for lab, b in sorted(leafs, key=lambda x: -(x[1][2] - x[1][0])):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            ax, ay = max(2, (b[2] - b[0]) / 2), max(2, (b[3] - b[1]) / 2)
            d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
            shade = np.clip(1.15 - d, 0, 1)[..., None]
            tint = self.tex[lab].uniform(0.6, 1.0, 3)
            ring = (np.abs(d - 0.8) < 0.08)[..., None] * 60 * tint[None, None]
            img = img * (1 - 0.85 * shade) + ring
        return np.clip(img, 0, 255).astype(np.uint8)


def simulate(g: AirwayGraph, targets: List[str], seed: int = 0, speed: float = 1.2,
             roll_walk_deg: float = 0.6, init_roll_deg: float = 0.0, size: int = 256,
             lookahead: int = 8) -> List[SynthFrame]:
    rng = np.random.default_rng(seed)
    path = interpolate_path(g, plan_path(g, targets), speed)
    ren = Renderer(g, size=size, seed=seed)
    tr = g["0"]
    fwd = tr.direction
    e1, e2, _ = AirwayGraph.tangent_basis(fwd)
    cam = Camera(tr.point_at(5.0), fwd, e1, e2)
    cam = _roll(cam, np.radians(init_roll_deg))
    frames = []
    roll_acc = np.radians(init_roll_deg)
    fwd_s = fwd.copy()
    pts = [g[b].point_at(s) for b, s in path]
    for i, (b, s) in enumerate(path):
        pos = pts[i]
        # look ahead along the path while advancing distally; otherwise look down the branch
        target = g[b].direction
        j = min(len(path) - 1, i + lookahead)
        bj, sj = path[j]
        distal = (bj == b and sj > s) or g.is_descendant(bj, b)
        if distal and np.linalg.norm(pts[j] - pos) > 1e-3:
            target = pts[j] - pos
            target = target / np.linalg.norm(target)
        fwd_s = 0.7 * fwd_s + 0.3 * target
        fwd_s /= np.linalg.norm(fwd_s)
        cam = _transport(Camera(pos, cam.fwd, cam.right, cam.down), fwd_s)
        dr = np.radians(rng.normal(0, roll_walk_deg))
        cam = _roll(cam, dr)
        roll_acc += dr
        boxes = ren.gt_boxes(cam, b, s)
        img = ren.render(boxes)
        gt = [(ren.ids[l], bx, l, lvl) for l, bx, lvl in boxes]
        e1b, _, _ = g.frame(b)
        roll_gt = float(np.arctan2(np.dot(e1b, cam.down), np.dot(e1b, cam.right)))
        frames.append(SynthFrame(img, gt, b, roll_gt))
    return frames


class OracleEmbedder:
    """Test-only stand-in for a *trained* Re-ID network on synthetic data.

    Returns a fixed random unit vector per ground-truth lumen identity (matched
    by IoU) plus Gaussian noise, i.e. it behaves like a Re-ID model whose
    embeddings cluster by lumen with a controllable spread ``noise``. It lets
    the association logic be tested independently of Re-ID training.
    """

    def __init__(self, frames: List[SynthFrame], dim: int = 64, noise: float = 0.3, seed: int = 0):
        self.frames, self.dim, self.noise = frames, dim, noise
        self.rng = np.random.default_rng(seed)
        self.proto: Dict[str, np.ndarray] = {}
        self.t = 0
        self.frame_index: Optional[int] = None

    def _p(self, lab):
        if lab not in self.proto:
            v = self.rng.normal(size=self.dim)
            self.proto[lab] = v / np.linalg.norm(v)
        return self.proto[lab]

    def __call__(self, frame, boxes):
        from .kalman import iou_matrix

        # the pipeline only calls the Re-ID on frames with detections, so the caller
        # sets ``frame_index`` before each frame (a plain counter would drift)
        t = self.frame_index if self.frame_index is not None else self.t
        f = self.frames[min(t, len(self.frames) - 1)]
        self.t += 1
        out = []
        gtb = np.array([g[1] for g in f.gt]).reshape(-1, 4)
        iou = iou_matrix(np.asarray(boxes).reshape(-1, 4), gtb) if len(gtb) else np.zeros((len(boxes), 0))
        for i in range(len(boxes)):
            if iou.shape[1] and iou[i].max() > 0.5:
                v = self._p(f.gt[int(iou[i].argmax())][2])
            else:
                v = self.rng.normal(size=self.dim)
                v /= np.linalg.norm(v)
            v = v + self.noise * self.rng.normal(size=self.dim) / np.sqrt(self.dim)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, np.float32).reshape(-1, self.dim)
