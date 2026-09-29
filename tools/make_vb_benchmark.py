#!/usr/bin/env python3
"""Virtual-bronchoscopy test set with exact ground truth from an airway mesh.

Renders camera runs through the airway (forward insertion, retraction to a
parent branch and re-insertion into another branch, as in the paper's patient
data, Sec. IV-B) and writes, per frame:

* the image (PNG, circular endoscope field like 3D Slicer's virtual endoscopy),
* ground-truth lumen boxes following the paper's annotation rule (Fig. 3):
  far from a bifurcation the current branch's far lumen is one primary box
  enclosing its visible children; near it the children are primary and their
  visible children secondary. A lumen is annotated only if its opening is
  visible (depth-buffer test), at least ``min_px`` wide and not filling the view.
  Each lumen's opening is the circle at 1.1 x parent radius into the branch with
  the local lumen diameter there (``AirwayGraph.measure_diameters``).
* the true branch-level location and camera pose.

Output per run:  frames/%05d.png, gt_boxes.csv (frame,id,x1,y1,x2,y2,label,level),
gt_loc.csv (frame,location), poses.csv.

python tools/make_vb_benchmark.py --mesh ModelV3.vtk --graph results/ModelV3/airway_v3.json \
       --out vb_bench --runs 6
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.synthetic import Camera, _roll, _transport, interpolate_path, plan_path  # noqa: E402
from render_lib import Renderer  # noqa: E402


def smoothed_copy(g: AirwayGraph, k: int = 9) -> AirwayGraph:
    """Centre lines from skeletons are jagged; smooth them for camera motion."""
    import copy
    h = copy.deepcopy(g)
    for b in h.branches.values():
        p = b.points
        if p is None or len(p) < k:
            continue
        ker = np.ones(k) / k
        q = np.stack([np.convolve(np.pad(p[:, i], (k // 2, k // 2), mode="edge"), ker, "valid") for i in range(3)], 1)
        q[0], q[-1] = p[0], p[-1]
        b.points = q
    h.invalidate()
    return h


class VBRenderer:
    def __init__(self, mesh, g: AirwayGraph, size=512, fov=60.0, atten=0.003, intensity=2.5, min_px=10,
                 max_frac=0.9, entry_factor=1.1):
        self.g, self.S, self.fov = g, size, fov
        self.R = Renderer(mesh, size, fov, atten, intensity)  # tuned to look like 3D Slicer's endoscopy view
        self.f = (size / 2) / np.tan(np.radians(fov) / 2)
        self.min_px, self.max_frac, self.ef = min_px, max_frac, entry_factor
        yy, xx = np.mgrid[0:size, 0:size]
        self.circle = ((xx - size / 2 + .5) ** 2 + (yy - size / 2 + .5) ** 2) <= (size / 2) ** 2
        import vtk
        self.zf = vtk.vtkWindowToImageFilter()
        self.zf.SetInput(self.R.rw)
        self.zf.SetInputBufferTypeToZBuffer()
        self.zf.ReadFrontBufferOff()

    # ------------------------------------------------------------ geometry
    def radius(self, lab):
        return float(self.g[lab].radius or 2.0)

    def opening(self, lab):
        """(centre, normal, diameter) of the lumen opening of branch ``lab``."""
        b = self.g[lab]
        par = self.g.parent(lab)
        s = min(self.ef * (self.radius(par) if par else self.radius(lab)), b.length)
        c = b.point_at(s)
        q = b.point_at(min(b.length, s + 1.5)) - b.point_at(max(0, s - 1.5))
        d = b.diameter_at(s) or 2 * self.radius(lab)
        return c, q / (np.linalg.norm(q) + 1e-9), d

    def far_end(self, lab):
        b = self.g[lab]
        s = max(0.0, b.length - 1.0)
        q = b.point_at(b.length) - b.point_at(max(0, b.length - 3))
        d = b.diameter_at(s) or 2 * self.radius(lab)
        return b.point_at(s), q / (np.linalg.norm(q) + 1e-9), d

    def project_circle(self, cam: Camera, c, n, d):
        a = np.cross(n, [1, 0, 0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(n, [0, 1, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        P = c[None] + (d / 2) * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
        Q = P - cam.pos
        z = Q @ cam.fwd
        if (z <= 0.5).any():
            return None, None
        u = self.S / 2 + self.f * (Q @ cam.right) / z
        v = self.S / 2 + self.f * (Q @ cam.down) / z
        zc = float((c - cam.pos) @ cam.fwd)
        return np.array([u.min(), v.min(), u.max(), v.max()]), zc

    def visible(self, box, zc, depth):
        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        if not (0 <= cx < self.S and 0 <= cy < self.S):
            return False
        r = max(1, int(0.2 * min(box[2] - box[0], box[3] - box[1])))
        pts = [(cy, cx), (cy - r, cx), (cy + r, cx), (cy, cx - r), (cy, cx + r)]
        ok = sum(1 for (y, x) in pts if 0 <= y < self.S and 0 <= x < self.S and depth[y, x] >= zc - 1.0)
        return ok >= 3

    def usable(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = (cx - self.S / 2) ** 2 + (cy - self.S / 2) ** 2 <= (self.S / 2) ** 2
        return inside and min(w, h) >= self.min_px and max(w, h) <= self.max_frac * self.S

    # ------------------------------------------------------------ render
    def render(self, cam: Camera):
        g = self.g
        pos = g.to_ct(cam.pos)
        fwd = cam.fwd @ g.ct_R
        up = -(cam.down @ g.ct_R)
        img = self.R.render(pos, fwd, up)
        self.zf.Modified()
        self.zf.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        zb = vtk_to_numpy(self.zf.GetOutput().GetPointData().GetScalars()).reshape(self.S, self.S)[::-1]
        n, f = 0.3, 400.0
        depth = 2 * n * f / (f + n - (2 * zb - 1) * (f - n))
        img[~self.circle] = 0
        return img, depth

    def gt_boxes(self, cam: Camera, branch: str, s: float, depth):
        g = self.g
        out = []

        def lumen(lab, far=False):
            c, nrm, d = self.far_end(lab) if far else self.opening(lab)
            box, zc = self.project_circle(cam, c, nrm, d)
            if box is None or not self.visible(box, zc, depth):
                return None
            return box

        def union(bx, kids):
            if not kids:
                return bx
            o = np.array([b2 for _, b2 in kids])
            return np.r_[np.minimum(bx[:2], o[:, :2].min(0) - 2), np.maximum(bx[2:], o[:, 2:].max(0) + 2)]

        def kids_of(lab):
            ks = [(c, lumen(c)) for c in g.children(lab)]
            return [(c, b) for c, b in ks if b is not None and self.usable(b)]

        def ahead(lab):
            """The current branch's lumen ahead: the farthest visible centre-line point."""
            br = g[lab]
            for s2 in np.arange(br.length - 1.0, s + 6.0, -2.0):
                c = br.point_at(s2)
                q = br.point_at(min(br.length, s2 + 1.5)) - br.point_at(max(0, s2 - 1.5))
                d = br.diameter_at(s2) or 2 * self.radius(lab)
                box, zc = self.project_circle(cam, c, q / (np.linalg.norm(q) + 1e-9), d)
                if box is not None and self.usable(box) and self.visible(box, zc, depth):
                    return box
            return None

        dist = g[branch].length - s
        near = max(6.0, 1.2 * (g[branch].diameter_at(g[branch].length) or 2 * self.radius(branch)))
        kids = kids_of(branch)
        if dist > near or not kids:
            bx = ahead(branch)
            if bx is not None:
                out.append((branch, union(bx, kids), 1))
                out += [(c, b, 2) for c, b in kids]
        else:
            for c, bx in kids:
                gk = kids_of(c)
                out.append((c, union(bx, gk), 1))
                out += [(k, b, 2) for k, b in gk]
        res = []
        for lab, bx, lvl in out:
            if max(bx[2] - bx[0], bx[3] - bx[1]) > self.max_frac * self.S:
                continue
            bx = np.clip(bx, 0, self.S - 1)
            if bx[2] - bx[0] >= 3 and bx[3] - bx[1] >= 3:
                res.append((lab, bx, lvl))
        return res


def pick_targets(g, rng, min_gen=3, min_radius=1.4):
    """Two targets >= 3 branches apart (forward, retraction, re-insertion). Targets are
    generation >= 3 branches wide enough (radius >= 1.4 mm) for the camera to stay inside."""
    cand = [l for l in g.labels() if g.generation(l) >= min_gen and (g[l].radius or 0) >= min_radius]
    a = str(rng.choice(cand))
    other = [l for l in cand if g.tree_distance(a, l) >= 3]
    b = str(rng.choice(other or cand))
    return [a, b]


def run(mesh, graph_path, out, n_runs=6, seed=0, size=512, fov=60.0, speed=0.5, roll_walk=0.4,
        lookahead=16, depth_frac=0.6, first=0):
    g = AirwayGraph.from_json(graph_path)
    gs = smoothed_copy(g)
    vb = VBRenderer(mesh, gs, size, fov)
    ids = {l: i + 1 for i, l in enumerate(sorted(g.labels()))}
    summary = []
    for r in range(first, first + n_runs):
        rng = np.random.default_rng(seed + 1000 * r)
        targets = pick_targets(g, rng)
        path = interpolate_path(gs, plan_path(gs, targets, depth_frac), speed)
        d = os.path.join(out, f"run{r + 1:02d}")
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        tr = gs["0"]
        fwd = tr.direction
        e1, e2, _ = gs.frame("0")
        cam = Camera(tr.point_at(5.0), fwd, e1, e2)
        cam = _roll(cam, np.radians(rng.normal(0, 5)))
        fwd_s = fwd.copy()
        pts = [gs[b].point_at(s) for b, s in path]
        with open(os.path.join(d, "gt_boxes.csv"), "w", newline="") as fb, \
                open(os.path.join(d, "gt_loc.csv"), "w", newline="") as fl, \
                open(os.path.join(d, "poses.csv"), "w", newline="") as fp:
            wb, wl, wp = csv.writer(fb), csv.writer(fl), csv.writer(fp)
            wb.writerow(["frame", "id", "x1", "y1", "x2", "y2", "label", "level"])
            wl.writerow(["frame", "location"])
            wp.writerow(["frame", "branch", "s", "px", "py", "pz", "fx", "fy", "fz", "rx", "ry", "rz", "roll_deg"])
            for i, (b, s) in enumerate(path):
                pos = pts[i]
                target = gs[b].direction
                j = min(len(path) - 1, i + lookahead)
                bj, sj = path[j]
                distal = (bj == b and sj > s) or gs.is_descendant(bj, b)
                if distal and np.linalg.norm(pts[j] - pos) > 1e-3:
                    target = (pts[j] - pos) / np.linalg.norm(pts[j] - pos)
                fwd_s = 0.85 * fwd_s + 0.15 * target
                fwd_s /= np.linalg.norm(fwd_s)
                cam = _transport(Camera(pos, cam.fwd, cam.right, cam.down), fwd_s)
                cam = _roll(cam, np.radians(rng.normal(0, roll_walk)))
                img, depth = vb.render(cam)
                boxes = vb.gt_boxes(cam, b, s, depth)
                k = i + 1
                cv2.imwrite(os.path.join(d, "frames", f"{k:05d}.png"), img)
                wl.writerow([k, b])
                for lab, bx, lvl in boxes:
                    wb.writerow([k, ids[lab], *[f"{x:.1f}" for x in bx], lab, lvl])
                e1b, _, _ = gs.frame(b)
                roll = np.degrees(np.arctan2(np.dot(e1b, cam.down), np.dot(e1b, cam.right)))
                wp.writerow([k, b, f"{s:.2f}", *[f"{x:.3f}" for x in np.r_[cam.pos, cam.fwd, cam.right]], f"{roll:.1f}"])
        summary.append((f"run{r + 1:02d}", targets, len(path)))
        print(f"run{r + 1:02d}: targets {targets}, {len(path)} frames", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--graph", required=True, help="graph JSON with diameter profiles and ct_frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=0.5, help="mm per frame")
    ap.add_argument("--first", type=int, default=0, help="index of the first run (to generate runs in parallel)")
    a = ap.parse_args()
    run(a.mesh, a.graph, a.out, a.runs, a.seed, a.size, a.fov, a.speed, first=a.first)
