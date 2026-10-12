#!/usr/bin/env python3
"""Cross-check a benchmark run's ground-truth location against its camera poses.

gt_loc.csv is not computed from poses.csv: tools/make_vb_benchmark.py plans the camera path
as (branch, arc length) samples on the airway graph and writes, for every frame, the branch
of the sample (gt_loc.csv) and the camera pose placed at that sample (poses.csv; its
``branch`` and ``s`` columns are the sample itself). This script checks both:

1. gt_loc.csv equals the ``branch`` column of poses.csv;
2. assigning each camera position to the nearest point of the (smoothed) centre lines gives
   the same branch, apart from frames right at a bifurcation where two branches are equally close.

python tools/check_gt_from_poses.py --graph results/ModelV3/airway_v3.json results/ModelV3/benchmark/run0*
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from make_vb_benchmark import smoothed_copy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders containing gt_loc.csv and poses.csv")
    ap.add_argument("--graph", required=True, help="the graph used to render the runs")
    a = ap.parse_args()
    g = smoothed_copy(AirwayGraph.from_json(a.graph))  # the renderer moves the camera on the smoothed centre lines
    pts, labs = [], []
    for lab in g.labels():
        b = g[lab]
        for s in np.arange(0.0, b.length, 0.25):
            pts.append(b.point_at(s))
            labs.append(lab)
    pts = np.array(pts)
    tot_same = tot_near = tot = 0
    for rd in a.runs:
        gl = {int(r["frame"]): r["location"] for r in csv.DictReader(open(os.path.join(rd, "gt_loc.csv")))}
        po = list(csv.DictReader(open(os.path.join(rd, "poses.csv"))))
        same = sum(gl[int(r["frame"])] == r["branch"] for r in po)
        near = 0
        for r in po:
            p = np.array([float(r["px"]), float(r["py"]), float(r["pz"])])
            near += labs[int(np.argmin(((pts - p) ** 2).sum(1)))] == gl[int(r["frame"])]
        print(f"{os.path.basename(rd.rstrip('/'))}: {len(po)} frames; gt_loc == poses.branch: {same}; "
              f"nearest centre line agrees: {near} ({100 * near / len(po):.1f} %)")
        tot_same, tot_near, tot = tot_same + same, tot_near + near, tot + len(po)
    print(f"all: {tot} frames; gt_loc == poses.branch: {tot_same}; nearest centre line agrees: "
          f"{tot_near} ({100 * tot_near / max(tot, 1):.2f} %)")


if __name__ == "__main__":
    main()
