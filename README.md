# BronchoTrack — re-implementation

Re-implementation of **Tian et al., "BronchoTrack: Airway Lumen Tracking for Branch-Level
Bronchoscopic Localization", IEEE TMI 44(3):1321–1333, 2025** (doi:10.1109/TMI.2024.3493170).
No official code or weights were used; everything is built from the paper text.

```
bronchotrack/
  airway_graph.py   Sec. III      semantic airway graph M, standard frame, Fig. 1(d) labels, g^k(.)
  detector.py       Sec. III-A    YOLOv7 lumen detector (+ ultralytics and replay back-ends)
  kalman.py         Sec. III-B-1  7-D constant-velocity KF [xc,yc,h,a,dxc,dyc,dh], Eq. (2)
  reid.py           Sec. III-B-2  ResNet50 Re-ID, EMA Eq. (3), cosine cost Eq. (4)
  tracker.py        Sec. III-B    two-stage matching, fused cost Eq. (5), label gating, 50-frame buffer
  subgraph.py       Sec. III-C    lumen subgraph S^t from box inclusion (Fig. 3), pruning
  association.py    Sec. III-C    Algorithms 1 & 2, Eqs. (6)–(9), gallery G
  loop_closure.py   Sec. III-D    BronchoTrack-LC: LoFTR keyframe loop closure (eta=100, lambda=1)
  pipeline.py       Fig. 1        per-frame pipeline + Fig. 4-style overlay
  metrics.py        Sec. IV-C     MOTA, FP, FN, IDs, IDF1, HOTA, Loc Acc, label AP, AD/t-test/Wilcoxon
  synthetic.py      (testing)     synthetic airway tree + virtual bronchoscopy with GT
tools/
  voxelize_mesh.py    closed airway surface mesh (.vtk/.stl/.obj) -> binary mask (.npz)
  build_graph.py      segmentation mask / VTK centre line -> airway graph JSON (+ CT transform)
  cache_detections.py run the detector once, save MOT detections for fast re-runs
  make_reid_crops.py  labelled frames -> Re-ID crop dataset (one class per lumen)
  train_reid.py       ResNet50 softmax training, 128x128 crops
  run.py              run BronchoTrack / BronchoTrack-LC on a video
  evaluate.py         Table III/IV-style metrics + significance test
  demo_synthetic.py   end-to-end check and Table V-style ablation on synthetic data
tests/                20 unit / integration tests (python -m pytest -q tests)
```

## Workflow

```bash
pip install -r requirements.txt

# 1. Airway graph from the pre-operative segmentation (Sec. III)
python tools/build_graph.py --mask airway_seg.nii.gz --out airway.json --preview airway.png \
    --patient-right -1 0 0          # LPS (SimpleITK/DICOM); use 1 0 0 for RAS / 3D Slicer
#    or from a 3D Slicer centre line:  --vtk CenterlineModel.vtk

# 2. Detector (Sec. III-A): YOLOv7, one class "lumen", 256x256 input
git clone https://github.com/WongKinYiu/yolov7
python yolov7/train.py --img 256 --data lumen.yaml --cfg yolov7/cfg/training/yolov7.yaml \
    --weights yolov7.pt --hyp yolov7/data/hyp.scratch.p5.yaml --batch 32 --epochs 300
#    (any ultralytics checkpoint also works with --detector ultralytics)

# 3. Re-ID (Sec. IV-A): crops of the same lumen = one class, ResNet50 + softmax
python tools/make_reid_crops.py --seq p01 frames/p01 ann/p01.csv --out reid_crops
python tools/train_reid.py --data reid_crops --out reid_resnet50.pth

# 4. Run (BronchoTrack; add --lc for BronchoTrack-LC)
python tools/run.py --video p01.mp4 --graph airway.json --detector yolov7 \
    --weights runs/train/exp/weights/best.pt --yolov7-repo yolov7 \
    --reid-weights reid_resnet50.pth --nms-iou 0.6 --lc --out-dir out/p01

# 5. Evaluate against GT (frame,id,x1,y1,x2,y2,label  and  frame,location)
python tools/evaluate.py --graph airway.json --seq gt/p01_boxes.csv gt/p01_loc.csv out/p01
```

From a surface mesh (e.g. a 3D Slicer segment export) instead of a mask:
`python tools/voxelize_mesh.py ModelV3.vtk mask.npz 0.5` then `build_graph.py --mask mask.npz`.
`run.py --map` adds an airway-map inset (current branch red, labelled lumens green).

Outputs of `run.py`: `tracks.txt` (MOTChallenge), `tracks_labels.csv` (box, track, score,
branch label, hierarchy level), `location.csv` (branch-level location, roll, loop flag, ms),
`overlay.mp4` (labels `[branch]-[track]-[conf]` as in Fig. 4).

Ablations of Table V: `--no-kf`, `--reid none`, `--no-graph`, `--lc`.

## Parameters taken from the paper

| Parameter | Value | Where |
|---|---|---|
| detection threshold | 0.1 | Sec. IV-A |
| NMS IoU | 0.6 patient / 0.7 porcine | Sec. IV-A |
| input size (detector / Re-ID) | 256² / 128² | Sec. IV-A |
| EMA momentum α | 0.9 | Eq. (3) |
| cost weight λ | 0.5 | Eq. (5) |
| stage-1 matching threshold | 0.4 | Sec. IV-A |
| stage-2 threshold | 0.7 if stage 1 matched, else 0.9 | Sec. IV-A |
| lost-track buffer | 50 frames | Sec. III-B |
| label gating | > 3 generations from previous location | Sec. III-B |
| loop closure η / λ | 100 matches / 1 most recent record | Sec. III-D |

## Where the paper is under-specified — choices made here

These are the places a reader of the paper has to decide something. Each is a config field, so
you can report or ablate it in the thesis.

1. **High/low confidence split** – not given; ByteTrack's 0.5 (`TrackerConfig.high_thresh`).
2. **Stage-2 candidates** – "unmatched tracklets" is read as ByteTrack does it: only tracklets
   tracked in the previous frame; lost tracklets are recovered by appearance (stage 1) only.
3. **Roll sign** – Eqs. (6)–(7) use `arccos`, which loses the rotation direction; the signed
   `atan2` equivalent is used.
4. **Tangent-plane basis** – the paper does not define the 2-D axes after projecting onto a
   branch's tangent plane. The trachea uses the projected standard x-axis (so Eq. (6) is
   measured against e_x); every other branch inherits that basis by parallel transport down the
   tree, so a scope that advances without twisting keeps one roll value, which is what Eq. (8)
   accumulates. `AirwayGraph.transport_frames = False` gives the naive "project e_x everywhere".
5. **Two oldest tracklets (Eq. 7)** – a nested parent/child pair has almost the same centre, so
   the oldest *non-nested* pair is used; roll steps above 30°/frame are rejected
   (`max_roll_step`) as physically implausible (guards against ID switches).
6. **Observed children ĉh(l)** – likelihood decreasing with intersection angle: the n̂ most
   aligned children, n̂ = number of lumens to label. Children are truncated at 10 mm
   (`truncate`).
7. **Graph matching** – "unknown transformation" handled by centring and scaling both point sets
   before the Hungarian step (direction from the parent box centre when a single lumen).
8. **Label refinement** ("refined based on contextual information and anatomical constraints",
   Fig. 1 caption) – each labelled lumen implies a location through Eq. (9); the track-age
   weighted majority wins and infeasible/inconsistent/duplicate labels are cleared before the
   intra-frame propagation of Algorithm 1 (`refine`).
9. **Recovery when every tracklet is lost** – primary lumens are re-associated around the last
   location; if a primary lumen that filled ≥30 % of the frame just vanished, the scope is taken
   to have entered it (`entry_area`). This is an addition, not in the paper.
10. **Box hierarchy** – parent = smallest box containing ≥80 % of the child; a parent with a
    single child of IoU ≥ 0.6 is a duplicate and the less reliable box is pruned.
11. **LC re-association** – labels of the matched keyframe are transferred through a RANSAC
    similarity estimated from the LoFTR matches; the roll follows its rotation.
12. **Initialisation** – the carina pair must be held by the same two tracklets for 5
    consecutive frames (`init_stable_frames`); a single-frame ID switch otherwise initialises
    the whole run with swapped labels.
13. **Labels** – Fig. 1(d) binary-string scheme; children ordered by angle in the parent's
    tangent plane. Anatomical names (Trachea/RMB/LMB) are kept in `Branch.name`.

## Verification (synthetic)

`tools/demo_synthetic.py` renders six virtual bronchoscopies with forward and retraction moves
(Fig. 4 style) through random 5-generation trees, with GT boxes following the Fig. 3 rule. The
detector is replaced by jittered GT boxes (3 % dropped), so FPS excludes detector/Re-ID time.

| Method (6 seqs) | MOTA | IDF1 | HOTA | Loc Acc |
|---|---|---|---|---|
| BronchoTrack, simulated trained Re-ID | 85.8 | 73.7 | 72.4 | 78.3 % |
| BronchoTrack, colour-histogram Re-ID | 85.3 | 65.9 | 65.8 | 51.6 % |
| w/o Re-ID | 82.3 | 63.9 | 63.9 | 54.7 % |

These numbers only show that the pipeline works end to end; they are **not** comparable to the
paper's patient/porcine results. The simulator's frames are feature-poor, so LoFTR/SIFT loop
closure never reaches η = 100 there (the LC mechanics are covered by a unit test instead).

## Not reproducible from the paper alone

The patient/porcine data, the trained YOLOv7 and Re-ID weights, and the exact airway
segmentation network [26] are not public. LoFTR weights are downloaded by kornia on first use;
offline, `--lc-matcher sift` is used as a fallback.

## Extension: roll-corrected diameter:distance ratio, fused with the association

`bronchotrack/geometry.py`, enabled with `run.py --geometry` (`BronchoTrackConfig.use_geometry`).

1. Each tracked box is straightened with the current roll estimate. The lumen opening is
   modelled as an ellipse (w', h'); its axis-aligned box after a roll θ satisfies
   W² = w'²cos²θ + h'²sin²θ, H² = w'²sin²θ + h'²cos²θ, which is inverted for (w', h').
   (A rectangle model is available, but it wrongly shrinks round lumens under roll.)
2. The longest straightened edge is the apparent diameter D. For two lumens seen together,
   the observed ratio is D_i / d_ij, with d_ij the distance between their box centres.
3. The actual ratio comes from the airway graph: 2 r_a / s_ab (branch radius from the CT mask;
   separation of the two branch entrances, 1.5 radii in, on the parent's tangent plane).
4. For every sibling group, the association's labels form the prior (confidence 0.55 → 0.95
   with tracklet age) and the ratio fit a log-normal likelihood (σ = 0.55, fitted on synthetic
   ground truth). Hypotheses cover the association's parent branch, its parent and its
   children. The posterior gives fused labels, a probability per lumen (`p=` on each box) and
   a location probability (probability-weighted Eq. (9) vote averaged over voting lumens).
   `location.csv` also keeps the association-only location and the ratio errors.

### Original vs fused (ModelV3 videos, synthetic benchmark)

Both variants are identical outside the frames listed below. No full ground truth exists for
the videos; the frames where the variants disagree were checked by rendering the ModelV3 mesh
inside each candidate branch and comparing with the video (`tools/gt_from_renders.py`).

| | Video 1, frames 724–853 (truth 010→0100) | Video 2, frames 983–1030 (truth 000) | Synthetic (6 seqs, Loc Acc) |
|---|---|---|---|
| Original | 12 % exact, mean error 0.88 generations | 0 % exact, 001 (2 steps off) | 78.3 % |
| Fused (default, w = 1) | identical to original | 0 % exact, 00 (1 step off) | 78.1 % |
| Fused, ratio weight 3 | 39 % exact, mean error 0.96 | 0 % exact, 00 (1 step off) | 77.0 % |

* The cue is weak in a self-similar tree: the ratio is scale-free, so sibling pairs one
  generation up or down fit almost as well. On synthetic ground truth the true labels give the
  best ratio fit in only 33 % of sibling groups.
* The probabilities are not calibrated: frames that are wrong still show p ≈ 0.93–0.97,
  because the association prior dominates. Calibrating them needs per-frame ground truth.
