#!/usr/bin/env python3
"""Evaluate BronchoTrack outputs against ground truth (paper Sec. IV-C).

Ground truth per sequence:
  --gt-boxes   CSV frame,id,x1,y1,x2,y2,label      (id = lumen identity, label = branch)
  --gt-loc     CSV frame,location                   (branch containing the scope)
Predictions (written by tools/run.py):
  --pred-dir   folder with tracks_labels.csv and location.csv

Several sequences can be given (repeat --seq); a summary + per-sequence table
is printed, and --compare-dir runs the paired significance test between two
methods' per-sequence localisation accuracy.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.metrics import (compare_accuracy, label_ap, localization_metrics,  # noqa: E402
                                  tracking_metrics)


def read_gt(path):
    gt, gtl = {}, {}
    for r in csv.DictReader(open(path)):
        f = int(r["frame"])
        b = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        gt.setdefault(f, []).append((r["id"], b))
        gtl.setdefault(f, []).append((r["id"], b, r["label"]))
    return gt, gtl


def read_pred(d):
    pr, prl = {}, {}
    for r in csv.DictReader(open(os.path.join(d, "tracks_labels.csv"))):
        f = int(r["frame"])
        b = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        pr.setdefault(f, []).append((int(r["track_id"]), b))
        prl.setdefault(f, []).append((int(r["track_id"]), b, float(r["score"]), r["label"] or None))
    return pr, prl


def read_loc(path, key="location"):
    return {int(r["frame"]): (r[key] or None) for r in csv.DictReader(open(path))}


def evaluate(seq_gt_boxes, seq_gt_loc, pred_dir, graph):
    gt, gtl = read_gt(seq_gt_boxes)
    pr, prl = read_pred(pred_dir)
    m = tracking_metrics(gt, pr)
    m.update({"label_" + k: v for k, v in label_ap(gtl, prl).items() if k == "mAP"})
    if seq_gt_loc:
        gl = read_loc(seq_gt_loc)
        pl = read_loc(os.path.join(pred_dir, "location.csv"))
        fr = sorted(gl)
        L = localization_metrics([gl[f] for f in fr], [pl.get(f) for f in fr], graph)
        m["LocAcc"] = L["LocAcc"]
        m["gen_error_mean"] = L.get("gen_error_mean")
        m["per_branch"] = L["per_branch"]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs=3, action="append", metavar=("GT_BOXES", "GT_LOC", "PRED_DIR"), required=True)
    ap.add_argument("--graph", help="airway graph JSON (for generation error)")
    ap.add_argument("--compare-dir", nargs="*", help="second method's PRED_DIRs (same order) for the stats test")
    ap.add_argument("--json")
    a = ap.parse_args()
    g = AirwayGraph.from_json(a.graph) if a.graph else None
    rows = []
    keys = ["MOTA", "IDF1", "HOTA", "FP", "FN", "IDs", "label_mAP", "LocAcc"]
    print("seq".ljust(24) + "".join(k.rjust(10) for k in keys))
    for gb, gl, pd in a.seq:
        m = evaluate(gb, gl if gl != "-" else None, pd, g)
        rows.append(m)
        print(os.path.basename(pd.rstrip("/")).ljust(24) + "".join(
            (f"{m[k]:10.2f}" if isinstance(m.get(k), float) else str(m.get(k, "-")).rjust(10)) for k in keys))
    mean = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys if any(r.get(k) is not None for r in rows)}
    print("mean".ljust(24) + "".join(f"{mean.get(k, float('nan')):10.2f}" for k in keys))
    if a.compare_dir:
        other = [evaluate(gb, gl, pd, g)["LocAcc"] for (gb, gl, _), pd in zip(a.seq, a.compare_dir)]
        res = compare_accuracy([r["LocAcc"] for r in rows], other)
        print("significance:", res)
    if a.json:
        json.dump({"per_sequence": rows, "mean": mean}, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
