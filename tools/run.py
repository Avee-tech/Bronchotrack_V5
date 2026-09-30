#!/usr/bin/env python3
"""Run BronchoTrack / BronchoTrack-LC on a bronchoscopy video.

Examples
--------
# paper setting: YOLOv7 detector + ResNet50 Re-ID + airway graph (+ loop closure)
python tools/run.py --video case01.mp4 --graph case01_airway.json \
    --detector yolov7 --weights lumen_yolov7.pt --yolov7-repo ~/yolov7 \
    --reid-weights reid_resnet50.pth --lc --out-dir out/case01

# ultralytics checkpoint (e.g. YOLO11) instead of YOLOv7
python tools/run.py --video case01.mp4 --graph airway.json --detector ultralytics --weights best.pt
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.detector import build_detector  # noqa: E402
from bronchotrack.pipeline import AirwayMap, BronchoTrack, BronchoTrackConfig, draw, draw_with_map  # noqa: E402
from bronchotrack.reid import build_reid  # noqa: E402


def frames_from(src):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    elif os.path.isdir(src):
        files = sorted(f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        for f in files:
            yield cv2.imread(os.path.join(src, f))
        return
    else:
        cap = cv2.VideoCapture(src)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file, image folder or camera index")
    ap.add_argument("--graph", help="airway graph JSON (tools/build_graph.py)")
    ap.add_argument("--detector", default="yolov7", choices=["yolov7", "ultralytics", "mot"])
    ap.add_argument("--weights", required=True, help="detector weights (or MOT txt for --detector mot)")
    ap.add_argument("--yolov7-repo", default=os.environ.get("YOLOV7_REPO", "yolov7"))
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.1, help="detection threshold (paper: 0.1)")
    ap.add_argument("--nms-iou", type=float, default=0.6, help="0.6 patient / 0.7 porcine")
    ap.add_argument("--reid", default="auto", choices=["auto", "resnet50", "hist", "none"])
    ap.add_argument("--reid-weights")
    ap.add_argument("--lc", action="store_true", help="BronchoTrack-LC (loop closure)")
    ap.add_argument("--lc-matcher", default="auto", choices=["auto", "loftr", "sift"])
    ap.add_argument("--no-kf", action="store_true", help="ablation: w/o Kalman filter")
    ap.add_argument("--no-graph", action="store_true", help="ablation: w/o airway graph")
    ap.add_argument("--init-roll", type=float, default=0.0, help="prior roll (deg) at the carina")
    ap.add_argument("--init-loc", help="start at this branch instead of before the carina")
    ap.add_argument("--mirror", action="store_true", help="flip graph x (scope display convention)")
    ap.add_argument("--truncate", type=float, default=10.0, help="child truncation length (graph units)")
    ap.add_argument("--high-thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--map", action="store_true", help="add an airway-map inset to the overlay video")
    ap.add_argument("--geometry", action="store_true",
                    help="extension: fuse the roll-corrected diameter:distance cue with the association")
    ap.add_argument("--geo-sigma", type=float, default=0.55, help="spread of log(observed/actual) ratio")
    ap.add_argument("--geo-weight", type=float, default=1.0, help="weight of the ratio cue vs the association")
    ap.add_argument("--motion", action="store_true",
                    help="version 3: speed-constrained motion model (use with --geometry)")
    ap.add_argument("--speed", type=float, default=7.0, help="average scope speed along the airway, mm/s")
    ap.add_argument("--speed-max-factor", type=float, default=2.0, help="max speed = factor x average")
    ap.add_argument("--fps", type=float, default=None, help="overlay fps (default: source fps or 15)")
    a = ap.parse_args()

    kw = dict(conf_thr=a.conf, iou_thr=a.nms_iou, img_size=a.img_size)
    if a.detector == "yolov7":
        detector = build_detector("yolov7", a.weights, repo=a.yolov7_repo, device=a.device, **kw)
    elif a.detector == "ultralytics":
        detector = build_detector("ultralytics", a.weights, **kw)
    else:
        detector = build_detector("mot", a.weights)
    reid = None if a.reid == "none" else build_reid(a.reid_weights, a.device, a.reid)
    graph = AirwayGraph.from_json(a.graph) if a.graph else None

    cfg = BronchoTrackConfig(use_graph=not a.no_graph and graph is not None, use_lc=a.lc)
    cfg.tracker.use_kf = not a.no_kf
    cfg.use_geometry = a.geometry
    cfg.fusion.sigma, cfg.fusion.w_geo = a.geo_sigma, a.geo_weight
    cfg.use_motion = a.motion
    cfg.motion.speed, cfg.motion.max_factor = a.speed, a.speed_max_factor
    vfps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) if os.path.isfile(a.video) else 0
    cfg.motion.fps = a.fps or vfps or 30.0
    cfg.export_particles = 300 if (a.map and a.motion) else 0
    cfg.tracker.high_thresh = cfg.tracker.new_track_thresh = a.high_thresh
    cfg.tracker.det_thresh = a.conf
    cfg.association.init_roll = np.radians(a.init_roll)
    cfg.association.mirror = a.mirror
    cfg.association.truncate = a.truncate
    matcher = None
    if a.lc:
        from bronchotrack.loop_closure import build_matcher
        matcher = build_matcher(a.lc_matcher, a.device)
    bt = BronchoTrack(detector, graph, reid, cfg, matcher)
    if a.init_loc and bt.assoc is not None:
        bt.assoc.force_init(a.init_loc, np.radians(a.init_roll))

    amap = AirwayMap(graph) if (a.map and graph is not None) else None
    fps = a.fps or (cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) if os.path.isfile(a.video) else 0) or 15
    os.makedirs(a.out_dir, exist_ok=True)
    mot = open(os.path.join(a.out_dir, "tracks.txt"), "w")
    lab = csv.writer(open(os.path.join(a.out_dir, "tracks_labels.csv"), "w", newline=""))
    lab.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "score", "label", "level", "prob",
                  "assoc_label", "assoc_prob", "ratio_label", "ratio_prob"])
    loc = csv.writer(open(os.path.join(a.out_dir, "location.csv"), "w", newline=""))
    loc.writerow(["frame", "location", "roll_deg", "loop_closed", "ms", "loc_prob", "assoc_location",
                  "ratio_err_assoc", "ratio_err_fused", "evidence_location"])
    writer = None
    t0 = time.perf_counter()
    n = 0
    for fr in frames_from(a.video):
        ts = time.perf_counter()
        r = bt.process(fr)
        ms = 1e3 * (time.perf_counter() - ts)
        n += 1
        for tr in r.tracks:
            if not tr.activated:
                continue
            x1, y1, x2, y2 = tr.box
            mot.write(f"{r.t + 1},{tr.track_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{tr.score:.3f},-1,-1,-1\n")
            lab.writerow([r.t + 1, tr.track_id, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                          f"{tr.score:.3f}", tr.label or "", tr.level,
                          "" if tr.prob is None else f"{tr.prob:.3f}",
                          (tr.assoc or ("", None))[0] or "", "" if tr.assoc is None else f"{tr.assoc[1]:.3f}",
                          (tr.ratio or ("", None))[0] or "", "" if tr.ratio is None else f"{tr.ratio[1]:.3f}"])
        fmt = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
        loc.writerow([r.t + 1, r.location or "", f"{np.degrees(r.roll):.1f}", int(r.loop_closed), f"{ms:.1f}",
                      fmt(r.loc_prob), r.assoc_location or "", fmt(r.ratio_err_assoc), fmt(r.ratio_err_fused),
                      r.evidence_location or ""])
        vis = draw_with_map(fr, r, amap) if amap is not None else draw(fr, r)
        if not a.no_video:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(os.path.join(a.out_dir, "overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(vis)
        if a.show:
            cv2.imshow("BronchoTrack", vis)
            if cv2.waitKey(1) == 27:
                break
    if writer is not None:
        writer.release()
    dt = time.perf_counter() - t0
    print(f"{n} frames in {dt:.1f}s -> {n / max(dt, 1e-9):.1f} FPS; results in {a.out_dir}/")


if __name__ == "__main__":
    main()
