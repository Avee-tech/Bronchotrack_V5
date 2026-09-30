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
3. The actual ratio comes from the airway graph. From the bifurcation, go 1.1 × the parent's
   radius along each child's centre line to points p_a, p_b. D_a is the lumen diameter *at that
   point*: twice the in-plane distance from p_a to the nearest wall, on the cross-section of the
   CT mask perpendicular to the centre line (`AirwayGraph.measure_diameters`, run by
   `build_graph.py --mask`; the area-equivalent diameter is stored as `diam_area`).
   s_ab = |p_a − p_b| projected on the parent's tangent plane; R_a = D_a / s_ab.
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
* Actual-ratio rule (1.1 × parent radius, diameter at the point) vs the first version
  (1.5 × own radius, median branch diameter), on sibling pairs labelled by the association:
  Spearman ρ between observed and actual ratio 0.26 / 0.50 (videos 1 / 2) vs −0.03 / −0.15;
  median |log(obs/actual)| 0.28 / 0.18 vs 0.22 / 0.24; observed ratios are 12–27 % below the
  actual ones (the diameter at 1.1 × parent radius is still slightly inflated by the junction).
  Localisation results in the table are unchanged at w = 1; at w = 3 video 1 drops to
  16 % exact (mean error 1.57) and synthetic to 76.3 %.
* The probabilities are not calibrated: frames that are wrong still show p ≈ 0.93–0.97,
  because the association prior dominates. Calibrating them needs per-frame ground truth.

## Paper-protocol test on ModelV3 (virtual bronchoscopy with exact ground truth)

`tools/make_vb_benchmark.py` renders runs through the ModelV3 mesh in the style of 3D Slicer's
virtual endoscopy (60° FOV, circular field) with known camera poses: forward insertion,
retraction to a parent branch and re-insertion elsewhere (as in the paper's patient data).
Ground truth per frame: branch-level location, lumen boxes by the Fig. 3 rule (visible
openings only, depth-buffer test), identities and branch labels. `tools/eval_benchmark.py`
scores every variant with the paper's metrics; `tools/report_benchmark.py` prints the tables.
Six runs, 3591 frames (seeds 0; run03 seed 19). Tables and ground truth:
`results/ModelV3/benchmark/`.

| Condition | Loc Acc original | Loc Acc + ratio fusion | p (paired) | MOTA / IDF1 / HOTA |
|---|---|---|---|---|
| `best.pt` detector, histogram Re-ID | 32.8 % | 33.2 % | 1.00 | −17.0 / 25.4 / 20.0 |
| GT boxes (3 % jitter, 5 % dropped), histogram Re-ID | 26.1 % | 31.5 % | 0.11 | 92.3 / 63.1 / 65.1 |
| GT boxes, simulated trained Re-ID | 27.2 % | 27.4 % | 0.50 | 92.4 / 63.6 / 66.0 |

Loop closure changed nothing (no loop reached η = 100 matches). Tracking metrics are identical
with and without fusion (fusion only relabels). Main failure modes, visible in every condition:
* at a branch entry the entered lumen fills the view and is not boxed while a sibling is still
  visible at the side; with one primary lumen Eq. (9) places the scope *in that sibling*, and
  later lumens inherit the wrong subtree (the ratio cue needs ≥ 2 siblings, so cannot help);
* `best.pt` finds ~38 % of the rendered openings at IoU 0.3 (trained on other data and a
  different box convention), so near bifurcations one sibling is often missed.
The fused location probability ranks frames (accuracy 1 % for p < 0.6 up to 43 % for p ≥ 0.95)
but is over-confident.

## Version 3: speed-constrained motion model (`bronchotrack/motion.py`)

`run.py --geometry --motion [--speed 7 --speed-max-factor 2]`. Versions: **v1** = the paper's
method, **v2** = v1 + ratio fusion (`--geometry`), **v3** = v2 + motion model.

The scope tip is tracked along the airway centre lines with a particle filter:
* **State**: branch, arc length s and signed velocity v (+ insertion, − retraction) per particle.
* **Predict**: v changes gradually (random walk, about one average speed per second), clipped
  to ±v_max = 2 × average speed; s += v. Past the end of a branch a particle continues into one of
  the children, past its start back into the parent. The reachable region grows at most
  v_max per second, so the location cannot jump across generations instantly.
* **Update**: the fused labels' probability-weighted Eq. (9) votes. Evidence for a child also
  supports the last ~20 mm of its parent (and vice versa), which both handles the Eq. (9)
  ambiguity at a bifurcation and lets the belief cross it. A likelihood floor (0.05) keeps
  single wrong frames from moving it.
* **Output**: the branch with the largest posterior mass and that mass as its probability
  (`p=` on the top line; `evidence:` shows the per-frame Eq. (9) location before the model).
  The particles are drawn on the airway map (orange).
* **Feedback**: fusion hypotheses are weighted by the probability that their location is
  reachable within 1 s; the location is fed back to the association (gating, recovery); when
  the carina is first recognised the tip is placed within 40 mm of it; a relabel of a tracked
  lumen competes with the label it had kept (confidence grows with label age, not track age).

Average speed: 7 mm/s, measured from the render-registered camera paths of the ModelV3 videos
(≈ 230 mm in 29–36 s; short bursts up to 10–17 mm/s).

| ModelV3 render benchmark (ground truth, 6 runs) | v1 | v2 | v3 | p v3 vs v1 |
|---|---|---|---|---|
| Loc Acc, `best.pt` detector | 32.8 % | 33.2 % | **36.5 %** | 0.084 |
| Loc Acc, ground-truth boxes | 26.1 % | 31.5 % | **37.5 %** | 0.054 |
| Synthetic trees (6 seqs, a bifurcation every ~0.7 s) | 78.3 % | 78.1 % | 75.9 % | |

v3 is equal or better on all 12 benchmark run/condition pairs. On the videos it removes
short-lived switches (location changes 16 → 12 on ModelV3_1, 17 → 10 on testvideo3; stays
under 10 frames 6 → 2 and 8 → 2). It does not undo a wrong label that the association keeps for
many frames (testvideo3 from frame 150). Tables: `results/ModelV3/benchmark/tables_v3_*.txt`.

## From 3D Slicer to the airway graph (`tools/slicer_to_airway.py`)

One command turns what you export from 3D Slicer into the graph `tools/run.py --graph` needs:

```bash
python tools/slicer_to_airway.py ModelV3.vtk                              # surface model
python tools/slicer_to_airway.py Segmentation.seg.nrrd --segment airway   # Segment Editor output
python tools/slicer_to_airway.py CenterlineModel.vtk                      # Extract Centerline output
python tools/run.py --video case.mp4 --graph ModelV3_airway/airway.json ...
```

| Slicer export | How to get it in Slicer | Handled as |
|---|---|---|
| Surface model `.vtk/.vtp/.stl/.obj/.ply` | Segmentations → Export to files, or Models → Save | voxelised at 0.5 mm |
| Segmentation `.seg.nrrd`, label map `.nrrd/.nii(.gz)` | Segment Editor → Save | the segment named `*airway*` (or `--segment`) |
| Centre-line model `.vtk/.vtp` with `Radius` | Extract Centerline → Centerline model → Save | rebuilt as a tube (0.3 mm voxels) |

Coordinates are read in the file's own system: Slicer ≥ 4.11 records `SPACE=LPS` (or RAS) in
models and `space` in NRRD; override with `--space`. Voxel direction matrices (flipped/oblique
axes) are honoured. Steps: mask → 3-D skeleton → branches (leaf spurs < 4 mm pruned) →
trachea from the most superior point, right/left main bronchus from the patient's right → the
paper's standard frame → lumen diameter along every centre line → labels (`--labels anatomical`:
trachea, R, L, R1, R2, R11 …, default; `numeric`: the paper's 0, 00, 01 …).

Outputs in `<input>_airway/`: `airway.json` (for the pipeline), `airway_nodes.json` (thesis
format, original Slicer coordinates, radius per centre-line point), `airway_preview.png`, and
`airway_centerlines.vtk` + `airway_labels.mrk.json` — drag both into Slicer on top of the model
to check the branches and labels. The tool prints trachea and main-bronchus lengths/diameters and
warns about likely problems (left/right swapped, trachea cut short, gaps).

Checked on ModelV3: the surface model, a RAS `.seg.nrrd` with flipped axes and an Extract
Centerline-style model (overlapping root-to-leaf paths) all give the same 21 branches and labels
(end points identical for model/segmentation, median 0.6 mm apart for the centre line).
Example output: `results/ModelV3/slicer_converted/`.
