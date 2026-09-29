#!/usr/bin/env python3
"""End-to-end check on synthetic airways + an ablation table in the layout of
the paper's Table V. Also exports one sequence to disk (frames, GT CSVs, graph
JSON, MOT detections) so tools/run.py and tools/evaluate.py can be exercised.

python tools/demo_synthetic.py --seqs 6 --export demo_data
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.detector import AnnotationDetector  # noqa: E402
from bronchotrack.metrics import compare_accuracy, localization_metrics, tracking_metrics  # noqa: E402
from bronchotrack.pipeline import BronchoTrack, BronchoTrackConfig  # noqa: E402
from bronchotrack.reid import HistogramEmbedder, build_reid  # noqa: E402
from bronchotrack.synthetic import OracleEmbedder, make_tree, simulate  # noqa: E402

VARIANTS = {
    "BronchoTrack w/o KF": dict(use_kf=False),
    "BronchoTrack w/o Re-ID": dict(use_reid=False),
    "BronchoTrack w/o Graph": dict(use_graph=False),
    "BronchoTrack": dict(),
    "BronchoTrack-LC": dict(use_lc=True),
}


def pick_targets(g, rng, gen=4):
    leaves = [l for l in g.labels() if g.generation(l) == gen]
    a = rng.choice(leaves)
    b = rng.choice([l for l in leaves if g.lca(a, l) != g.parent(a) and l != a] or leaves)
    return [str(a), str(b)]


def run_variant(g, frames, reid, v, seed):
    cfg = BronchoTrackConfig(use_graph=v.get("use_graph", True), use_lc=v.get("use_lc", False))
    cfg.tracker.use_kf = v.get("use_kf", True)
    cfg.tracker.use_reid = v.get("use_reid", True)
    table = {i: [(*b, float(np.clip(0.9 - 0.02 * lvl, 0, 1))) for _, b, _, lvl in f.gt] for i, f in enumerate(frames)}
    det = AnnotationDetector(table, jitter=0.02, drop=0.03, seed=seed)
    matcher = None
    if cfg.use_lc:
        from bronchotrack.loop_closure import SIFTMatcher
        matcher = SIFTMatcher()
    bt = BronchoTrack(det, g, reid if cfg.tracker.use_reid else None, cfg, matcher)
    gt, pr, gl, pl = {}, {}, [], []
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        r = bt.process(f.image)
        gt[i] = [(gid, b) for gid, b, _, _ in f.gt]
        pr[i] = [(t.track_id, t.box) for t in r.tracks if t.activated]
        gl.append(f.location)
        pl.append(r.location)
    fps = len(frames) / (time.perf_counter() - t0)
    m = tracking_metrics(gt, pr)
    if cfg.use_graph:
        m["LocAcc"] = localization_metrics(gl, pl, g)["LocAcc"]
    m["FPS"] = fps
    return m


def export(g, frames, out):
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    g.to_json(os.path.join(out, "airway.json"))
    with open(os.path.join(out, "gt_boxes.csv"), "w", newline="") as fb, \
            open(os.path.join(out, "gt_loc.csv"), "w", newline="") as fl, \
            open(os.path.join(out, "dets_mot.txt"), "w") as fd:
        wb, wl = csv.writer(fb), csv.writer(fl)
        wb.writerow(["frame", "id", "x1", "y1", "x2", "y2", "label"])
        wl.writerow(["frame", "location"])
        for i, f in enumerate(frames, start=1):
            cv2.imwrite(os.path.join(out, "frames", f"{i:05d}.png"), f.image)
            wl.writerow([i, f.location])
            for gid, b, lab, lvl in f.gt:
                wb.writerow([i, gid, *[f"{x:.1f}" for x in b], lab])
                fd.write(f"{i},-1,{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},{0.9 - 0.02 * lvl:.2f}\n")
    print(f"exported {len(frames)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--reid", default="oracle", choices=["oracle", "hist", "resnet50"],
                    help="oracle = simulated trained Re-ID (GT identities + noise); hist = colour histogram")
    ap.add_argument("--export")
    a = ap.parse_args()
    reid = None if a.reid == "oracle" else (HistogramEmbedder() if a.reid == "hist" else build_reid(None, "cpu", "resnet50"))
    res = {k: [] for k in VARIANTS}
    for s in range(a.seqs):
        rng = np.random.default_rng(100 + s)
        g = make_tree(a.gens, seed=100 + s)
        frames = simulate(g, pick_targets(g, rng), seed=100 + s)
        if a.export and s == 0:
            export(g, frames, a.export)
        for k, v in VARIANTS.items():
            emb = OracleEmbedder(frames, seed=s) if reid is None else reid
            res[k].append(run_variant(g, frames, emb, v, s))
        print(f"seq {s}: {len(frames)} frames, LocAcc " + ", ".join(
            f"{k.replace('BronchoTrack', 'BT')}={res[k][-1].get('LocAcc', float('nan')):.1f}" for k in VARIANTS))
    keys = ["MOTA", "IDF1", "HOTA", "IDs", "LocAcc", "FPS"]
    print("\n" + "Method".ljust(26) + "".join(k.rjust(9) for k in keys))
    for k in VARIANTS:
        row = [np.mean([m.get(c, np.nan) for m in res[k]]) for c in keys]
        print(k.ljust(26) + "".join(f"{x:9.2f}" for x in row))
    base = [m["LocAcc"] for m in res["BronchoTrack"]]
    lc = [m["LocAcc"] for m in res["BronchoTrack-LC"]]
    if len(base) >= 3:
        print("\nBronchoTrack-LC vs BronchoTrack LocAcc:", compare_accuracy(lc, base))


if __name__ == "__main__":
    main()
