#!/usr/bin/env python3
"""Evaluate BronchoTrack variants on a benchmark made by make_vb_benchmark.py,
following the paper's protocol (Sec. IV-C):

* tracking: MOTA, IDF1, HOTA, FP, FN, IDs (Table III)
* localisation accuracy over all frames, per trajectory and per branch; signed
  generation error (Fig. 6/7); paired test between methods: Anderson-Darling on
  the differences -> paired t-test or Wilcoxon (Table IV)
* AP of identifying each visible lumen as its branch
* FPS of tracking + association (detections are cached, so detector time excluded)

python tools/eval_benchmark.py --bench vb_bench --weights best.pt --out vb_results
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.detector import AnnotationDetector, build_detector  # noqa: E402
from bronchotrack.metrics import label_ap, localization_metrics, tracking_metrics  # noqa: E402
from bronchotrack.pipeline import BronchoTrack, BronchoTrackConfig  # noqa: E402
from bronchotrack.reid import build_reid  # noqa: E402

VARIANTS = {
    "BronchoTrack w/o KF": dict(use_kf=False),
    "BronchoTrack w/o Re-ID": dict(use_reid=False),
    "BronchoTrack w/o Graph": dict(use_graph=False),
    "BronchoTrack": dict(),
    "BronchoTrack-LC": dict(use_lc=True),
    "BronchoTrack + ratio fusion": dict(use_geometry=True),
    "BronchoTrack-LC + ratio fusion": dict(use_lc=True, use_geometry=True),
}


def cache_detections(run_dir, weights, conf=0.1, iou=0.6, imgsz=640):
    out = os.path.join(run_dir, "dets.txt")
    if os.path.exists(out):
        return out
    det = build_detector("ultralytics", weights, conf_thr=conf, iou_thr=iou, img_size=imgsz)
    with open(out, "w") as f:
        for p in sorted(glob.glob(os.path.join(run_dir, "frames", "*.png"))):
            k = int(os.path.basename(p)[:-4])
            d = det(cv2.imread(p))
            for (x1, y1, x2, y2), s in zip(d.boxes, d.scores):
                f.write(f"{k},-1,{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{s:.4f}\n")
    return out


def oracle_detections(run_dir, jitter=0.03, drop=0.05, seed=0):
    """Ground-truth boxes as detections (jittered, some dropped) to separate the
    detector from tracking + association, cf. the paper reporting the detector
    separately (Table II)."""
    out = os.path.join(run_dir, "dets_oracle.txt")
    rng = np.random.default_rng(seed)
    with open(out, "w") as f:
        for r in csv.DictReader(open(os.path.join(run_dir, "gt_boxes.csv"))):
            if rng.random() < drop:
                continue
            b = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
            wh = np.r_[b[2] - b[0], b[3] - b[1], b[2] - b[0], b[3] - b[1]]
            b = b + rng.normal(0, jitter, 4) * wh
            s = float(np.clip(rng.normal(0.8, 0.1), 0.35, 0.99))
            f.write(f"{r['frame']},-1,{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},{s:.3f}\n")
    return out


def load_gt(run_dir):
    boxes, labels = {}, {}
    for r in csv.DictReader(open(os.path.join(run_dir, "gt_boxes.csv"))):
        f = int(r["frame"])
        b = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        boxes.setdefault(f, []).append((int(r["id"]), b))
        labels.setdefault(f, []).append((int(r["id"]), b, r["label"]))
    loc = {int(r["frame"]): r["location"] for r in csv.DictReader(open(os.path.join(run_dir, "gt_loc.csv")))}
    return boxes, labels, loc


class GTFrame:
    def __init__(self, gt):
        self.gt = gt


def oracle_reid(run_dir, noise=0.3, seed=0):
    """Simulated *trained* Re-ID: embeddings cluster by ground-truth lumen identity."""
    from bronchotrack.synthetic import OracleEmbedder
    gt = {}
    for r in csv.DictReader(open(os.path.join(run_dir, "gt_boxes.csv"))):
        b = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        gt.setdefault(int(r["frame"]), []).append((int(r["id"]), b, r["label"], int(r["level"])))
    n = len(glob.glob(os.path.join(run_dir, "frames", "*.png")))
    return OracleEmbedder([GTFrame(gt.get(k, [])) for k in range(1, n + 1)], noise=noise, seed=seed)


def run_variant(run_dir, g, dets_path, v, reid_kind, high_thresh):
    cfg = BronchoTrackConfig(use_graph=v.get("use_graph", True), use_lc=v.get("use_lc", False),
                             use_geometry=v.get("use_geometry", False))
    cfg.tracker.use_kf = v.get("use_kf", True)
    cfg.tracker.high_thresh = cfg.tracker.new_track_thresh = high_thresh
    if not v.get("use_reid", True):
        reid = None
    elif reid_kind == "oracle":
        reid = oracle_reid(run_dir)
    else:
        reid = build_reid(None, "cpu", reid_kind)
    matcher = None
    if cfg.use_lc:
        from bronchotrack.loop_closure import build_matcher
        matcher = build_matcher("auto", "cpu")
    bt = BronchoTrack(AnnotationDetector.from_mot_txt(dets_path), g, reid, cfg, matcher)
    pr, prl, loc, probs = {}, {}, {}, {}
    frames = sorted(glob.glob(os.path.join(run_dir, "frames", "*.png")))
    t_tot = 0.0
    for p in frames:
        k = int(os.path.basename(p)[:-4])
        img = cv2.imread(p)
        if hasattr(reid, "frame_index"):
            reid.frame_index = k - 1
        t0 = time.perf_counter()
        r = bt.process(img)
        t_tot += time.perf_counter() - t0 - r.runtime_ms.get("detection", 0) / 1e3
        act = [t for t in r.tracks if t.activated]
        pr[k] = [(t.track_id, t.box) for t in act]
        prl[k] = [(t.track_id, t.box, t.prob if t.prob is not None else t.score, t.label) for t in act]
        loc[k] = r.location
        probs[k] = r.loc_prob
    return pr, prl, loc, probs, len(frames) / max(t_tot, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--graph", default=None, help="default: the graph used to render (results/ModelV3/airway_v3.json)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reid", default="hist", choices=["hist", "resnet50", "oracle"],
                    help="oracle = simulated trained Re-ID (embeddings cluster by GT lumen identity)")
    ap.add_argument("--high-thresh", type=float, default=0.3)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-dets", action="store_true", help="use jittered GT boxes instead of the detector")
    a = ap.parse_args()
    graph = a.graph or os.path.join(os.path.dirname(__file__), "..", "results", "ModelV3", "airway_v3.json")
    g = AirwayGraph.from_json(graph)
    runs = sorted(d for d in glob.glob(os.path.join(a.bench, "run*")) if os.path.isdir(d))
    os.makedirs(a.out, exist_ok=True)
    res = {v: {} for v in a.variants}
    for rd in runs:
        name = os.path.basename(rd)
        dets = oracle_detections(rd) if a.oracle_dets else cache_detections(rd, a.weights)
        gtb, gtl, gloc = load_gt(rd)
        for v in a.variants:
            pr, prl, loc, probs, fps = run_variant(rd, g, dets, VARIANTS[v], a.reid, a.high_thresh)
            m = tracking_metrics(gtb, pr)
            fr = sorted(gloc)
            L = localization_metrics([gloc[f] for f in fr], [loc.get(f) for f in fr], g)
            m.update(LocAcc=L["LocAcc"], per_branch=L["per_branch"], gen_err=L.get("gen_error", []),
                     tree_dist=L.get("tree_dist_mean"), AP=label_ap(gtl, prl)["mAP"], FPS=fps, n=len(fr),
                     correct=[gloc[f] == loc.get(f) for f in fr],
                     loc_prob=[probs.get(f) for f in fr])
            res[v][name] = m
            print(f"{name} {v:32s} LocAcc {m['LocAcc']:6.2f}  MOTA {m['MOTA']:6.2f}  IDF1 {m['IDF1']:6.2f}  "
                  f"HOTA {m['HOTA']:6.2f}  AP {m['AP']:6.2f}  FPS {fps:6.1f}", flush=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("saved", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
