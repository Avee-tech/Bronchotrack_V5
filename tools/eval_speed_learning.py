#!/usr/bin/env python3
"""Does a learned, branch-dependent speed help the motion model? (version 4 evaluation)

Uses a benchmark rendered with a simulated doctor whose speed depends on the branch
(tools/make_vb_benchmark.py --speed-profile). Leave-one-run-out: the speed profile used on
a run is learned only from the other runs. Version 3 (ratio fusion + motion model) is run
with:

  A  constant 3.5 mm/s                       (current default)
  B  constant, the doctor's own average      (one speed per doctor, set before the run)
  C  learned profile, reference paths         (branch-dependent; learned from the other runs' true paths)
  D  learned profile, own output              (branch-dependent; learned from BronchoTrack's own
                                               locations on the other runs, run B: no annotation needed)
  E  the simulated doctor's true profile      (upper bound)

python tools/eval_speed_learning.py --bench vb_speed --doctor doctor_A.json --out vb_speed_results
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.detector import AnnotationDetector  # noqa: E402
from bronchotrack.metrics import compare_accuracy, localization_metrics  # noqa: E402
from bronchotrack.pipeline import BronchoTrack, BronchoTrackConfig  # noqa: E402
from bronchotrack.reid import build_reid  # noqa: E402
from bronchotrack.speed_model import (SpeedModel, extract_traversals,  # noqa: E402
                                      traversal_data)
from eval_benchmark import oracle_detections  # noqa: E402

FPS = 30.0


def gt_locations(rd):
    rows = list(csv.DictReader(open(os.path.join(rd, "gt_loc.csv"))))
    return [int(r["frame"]) for r in rows], [r["location"] for r in rows]


def run_v3(rd, g, dets, speed=3.5, model=None):
    cfg = BronchoTrackConfig(use_geometry=True, use_motion=True)
    cfg.motion.speed, cfg.motion.fps = speed, FPS
    cfg.tracker.high_thresh = cfg.tracker.new_track_thresh = 0.3
    bt = BronchoTrack(AnnotationDetector.from_mot_txt(dets), g, build_reid(None, "cpu", "hist"), cfg,
                      speed_model=model)
    locs, probs = [], []
    for p in sorted(glob.glob(os.path.join(rd, "frames", "*.png"))):
        r = bt.process(cv2.imread(p))
        locs.append(r.location)
        probs.append(r.loc_prob)
    return locs, probs


def learned(g, runs_xyw, prior_speed=3.5):
    m = SpeedModel.prior(prior_speed)
    for X, y, w in runs_xyw:
        m.update(X, y, w)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--graph", default=os.path.join(os.path.dirname(__file__), "..", "results", "ModelV3",
                                                     "airway_v3.json"))
    ap.add_argument("--doctor", help="true speed profile of the simulated doctor (condition E)")
    ap.add_argument("--dets", default="oracle", help="'oracle' (jittered GT boxes) or a dets file name per run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = AirwayGraph.from_json(a.graph)
    runs = sorted(d for d in glob.glob(os.path.join(a.bench, "run*")) if os.path.isdir(d))
    os.makedirs(a.out, exist_ok=True)
    names = [os.path.basename(r) for r in runs]
    dets = {n: (oracle_detections(r) if a.dets == "oracle" else os.path.join(r, a.dets)) for n, r in zip(names, runs)}
    gt = {n: gt_locations(r) for n, r in zip(names, runs)}
    gt_trav = {n: extract_traversals(g, gt[n][1], FPS, frames=gt[n][0]) for n in names}
    gt_xyw = {n: traversal_data(g, gt_trav[n]) for n in names}
    doctor = SpeedModel.load(a.doctor) if a.doctor else None

    res = {c: {} for c in "ABCDE"}
    out_b = {}

    def record(c, n, locs, model):
        L = localization_metrics(gt[n][1], locs, g)
        sw = sum(x != y for x, y in zip(locs, locs[1:]))
        X, y, _ = gt_xyw[n]
        if model is None or len(y) == 0:
            err = None
        else:
            err = float(np.median(np.abs(np.array([np.log(model.speed(x)) for x in X]) - y)))
        gen_ok = [p is not None and g.generation(p) == g.generation(q) for q, p in zip(gt[n][1], locs)]
        res[c][n] = dict(LocAcc=L["LocAcc"], GenAcc=100 * float(np.mean(gen_ok)), switches=sw,
                         correct=[p == q for p, q in zip(gt[n][1], locs)], gen_correct=gen_ok,
                         speed_log_err=err, locations=list(locs))
        print(f"{n} {c}: LocAcc {L['LocAcc']:6.2f}  GenAcc {res[c][n]['GenAcc']:6.2f}  switches {sw:3d}"
              + ("" if err is None else f"  speed error x{np.exp(err):.2f}"), flush=True)

    for n, rd in zip(names, runs):
        others = [m for m in names if m != n]
        locs, _ = run_v3(rd, g, dets[n], 3.5)
        record("A", n, locs, SpeedModel.prior(3.5))
        ys = np.concatenate([gt_xyw[m][1] for m in others])
        v_doc = float(np.exp(np.mean(ys)))
        locs, probs = run_v3(rd, g, dets[n], v_doc)
        out_b[n] = (locs, probs)
        record("B", n, locs, SpeedModel.prior(v_doc))
        mC = learned(g, [gt_xyw[m] for m in others])
        locs, _ = run_v3(rd, g, dets[n], 3.5, mC)
        record("C", n, locs, mC)
        if doctor is not None:
            locs, _ = run_v3(rd, g, dets[n], 3.5, doctor)
            record("E", n, locs, doctor)
    for n, rd in zip(names, runs):  # D: learned from the pipeline's own output on the other runs
        others = [m for m in names if m != n]
        own = []
        for m in others:
            fr = list(range(1, len(out_b[m][0]) + 1))
            own.append(traversal_data(g, extract_traversals(g, out_b[m][0], FPS, out_b[m][1], fr)))
        mD = learned(g, own)
        locs, _ = run_v3(rd, g, dets[n], 3.5, mD)
        record("D", n, locs, mD)
        if n == names[-1]:
            print(mD.describe())
    print(learned(g, [gt_xyw[m] for m in names]).describe())

    summary = {}
    for c in "ABCDE":
        if not res[c]:
            continue
        acc = [res[c][n]["LocAcc"] for n in names]
        allc = np.concatenate([res[c][n]["correct"] for n in names])
        errs = [res[c][n]["speed_log_err"] for n in names if res[c][n]["speed_log_err"] is not None]
        allg = np.concatenate([res[c][n]["gen_correct"] for n in names])
        summary[c] = dict(LocAcc_frames=100 * float(np.mean(allc)), LocAcc_runs=float(np.mean(acc)),
                          GenAcc_frames=100 * float(np.mean(allg)),
                          switches=float(np.mean([res[c][n]["switches"] for n in names])),
                          speed_err_factor=float(np.exp(np.median(errs))) if errs else None)
    for c in summary:
        if c != "A":
            t = compare_accuracy([res["A"][n]["LocAcc"] for n in names], [res[c][n]["LocAcc"] for n in names])
            summary[c]["p_vs_A"] = t.get("p")
            t = compare_accuracy([res["A"][n]["GenAcc"] for n in names], [res[c][n]["GenAcc"] for n in names])
            summary[c]["GenAcc_p_vs_A"] = t.get("p")
        if c not in ("A", "B"):
            t = compare_accuracy([res["B"][n]["LocAcc"] for n in names], [res[c][n]["LocAcc"] for n in names])
            summary[c]["p_vs_B"] = t.get("p")
    for c, s in summary.items():
        print(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
    json.dump({"per_run": {c: {n: {k: v for k, v in d.items() if k not in ("correct", "gen_correct")} for n, d in r.items()}
                           for c, r in res.items()}, "summary": summary},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
