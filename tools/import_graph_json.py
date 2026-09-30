#!/usr/bin/env python3
"""Convert an airway graph exported by the thesis pipeline (``{"nodes": [...]}`` with
``label``, ``parent``, ``centerline``, per-point ``radius`` and ``median_radius_mm``) into
BronchoTrack's graph JSON.

Labels are kept as given (e.g. trachea / L / R / L1 ...). The main bronchi are named
LMB / RMB for the carina initialisation, the graph is moved into the paper's standard
frame, and the per-point radii become the diameter profile used by the ratio cue.

python tools/import_graph_json.py airway_graph.json --out airway_bt.json --preview airway_bt.png \
       [--left L --right R]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.airway_graph import AirwayGraph, Branch  # noqa: E402


def convert(src: dict, left: str = "L", right: str = "R") -> AirwayGraph:
    nodes = src["nodes"]
    branches = {}
    for n in nodes:
        pts = np.asarray(n["centerline"], float)
        rad = np.asarray(n.get("radius") or [], float)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        b = Branch(label=str(n["label"]), parent=None if n["parent"] is None else str(n["parent"]),
                   start=pts[0].copy(), end=pts[-1].copy(), points=pts,
                   radius=float(n.get("median_radius_mm") or (np.median(rad) if len(rad) else 2.0)))
        if len(rad) == len(pts):
            b.diam_s, b.diam, b.diam_area = cum, 2.0 * rad, 2.0 * rad
        branches[b.label] = b
    for b in branches.values():
        if b.parent is not None:
            branches[b.parent].children.append(b.label)
    roots = [b.label for b in branches.values() if b.parent is None]
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {roots}")
    g = AirwayGraph(branches, root=roots[0])
    g[g.root].name = g[g.root].name or "Trachea"
    g[left].name, g[right].name = "LMB", "RMB"
    return g.to_standard_frame(left_main=left, right_main=right)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--left", default="L", help="label of the left main bronchus")
    ap.add_argument("--right", default="R", help="label of the right main bronchus")
    ap.add_argument("--preview")
    a = ap.parse_args()
    g = convert(json.load(open(a.src)), a.left, a.right)
    g.to_json(a.out)
    print(f"{len(g)} branches, max generation {g.max_generation()}, root {g.root} -> {a.out}")
    if a.preview:
        from build_graph import preview
        preview(g, a.preview)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
