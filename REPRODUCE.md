# Reproducing BronchoTrack V5

This file lists what each stage reads and writes, how the reported numbers were produced,
and the exact commands to regenerate them. Everything below was re-run on the current code.
The rendered frames, airway graph, detections and the benchmark tables (v1/v2/v3, detector
and ground-truth boxes) were reproduced **exactly** from `ModelV3.vtk` (see "Verification").

## 1. Pipeline and data contract

| stage | input | output | tool |
|---|---|---|---|
| airway model | CT segmented in 3D Slicer, exported as a surface model (`.vtk/.vtp/.stl/.obj/.ply`), a segmentation (`.seg.nrrd/.nrrd/.nii`) or an Extract Centerline model (`.vtk/.vtp` with a `Radius` array). Coordinates LPS or RAS, read from the file header (`--space` to override) | the same file | 3D Slicer |
| airway graph | the Slicer export | `airway.json` (format below) | `tools/slicer_to_airway.py` |
| video | video file, image folder (`00001.png`, …) or camera index; BGR frames, any size | – | – |
| lumen detection | frames + YOLO weights (one class, lumen) | per frame: boxes + scores; cached as MOT txt `frame,-1,x,y,w,h,score` | `bronchotrack/detector.py`, `tools/cache_detections.py` |
| Re-ID | frame + boxes | one embedding per box | `bronchotrack/reid.py` |
| tracking | boxes, scores, embeddings | tracklets (MOT txt `tracks.txt`) | `bronchotrack/tracker.py` |
| association + localisation | tracklets + `airway.json` | per-frame branch (`location.csv`), per-box label (`tracks_labels.csv`) | `bronchotrack/association.py`, `geometry.py` (v2), `motion.py` (v3) |
| evaluation | `location.csv` / tracks + ground truth (`gt_loc.csv`, `gt_boxes.csv`) | Loc Acc, MOTA, IDF1, HOTA, AP, tests | `tools/eval_benchmark.py`, `tools/evaluate.py` |

**`airway.json`**: `{"root": "0", "ct_frame": {"origin": [3], "R": [3x3]}, "branches": [ ... ]}`, one
entry per branch: `label`, `parent`, `children`, `start`, `end`, `points` (N×3 centre line, mm, in
the paper's standard frame), `radius`, `generation`, `name` (`Trachea`/`RMB`/`LMB`), `diam_s`, `diam`
(lumen diameter along the centre line), `diam_area`. `ct_frame` maps standard-frame points back to the
original Slicer (LPS/RAS) coordinates. A thesis-format graph (`{"nodes": [...]}` with per-point radii)
can be converted with `tools/import_graph_json.py`.

**Ground truth (benchmark only)**: `gt_loc.csv` = `frame,location`; `gt_boxes.csv` =
`frame,id,x1,y1,x2,y2,label,level`; `poses.csv` = camera pose per frame.

## 2. What was used for the reported results

**Airway model.** `ModelV3.vtk` is a closed airway **surface mesh** exported directly from 3D Slicer
(`3D Slicer output. SPACE=LPS`, 28,072 points). Nothing was drawn or edited by hand:

1. voxelised at 0.5 mm, smoothed with a Gaussian of 1 voxel;
2. skeletonised by 3-D thinning (Lee et al. 1994, scikit-image); split into branches at bifurcations;
   leaf spurs shorter than 4 mm removed;
3. trachea = the terminal branch with the largest volume (`build_graph.py`) or the one reaching the
   most superior point (`slicer_to_airway.py`); both pick the same branch here. Right/left main
   bronchus from the patient-right direction of LPS (−x);
4. moved into the paper's standard frame (y along the trachea, x from left to right main bronchus);
5. labels assigned **automatically**: the paper's binary scheme (`0` trachea, `00` LMB, `01` RMB,
   children of every branch ordered by their angle in the parent's tangent plane); with
   `--labels anatomical` the same tree is named trachea/R/L/R1/…; no manual correction;
6. lumen diameter measured every 0.5 mm on the cross-section perpendicular to the centre line
   (largest inscribed circle).

Result: `results/ModelV3/airway_v3.json` (21 branches, 4 generations). Either command regenerates it
bit-for-bit:

```bash
python tools/slicer_to_airway.py ModelV3.vtk --labels numeric --out-dir airway      # one step
# or, as originally run:
python tools/voxelize_mesh.py ModelV3.vtk mask_v3.npz 0.5
python tools/build_graph.py --mask mask_v3.npz --out airway_v3.json --patient-right -1 0 0 --min-branch 4
```

**Benchmark frames and ground truth.** The 6 benchmark runs (3,591 frames) are virtual-bronchoscopy
renders of `ModelV3.vtk` made with VTK by `tools/make_vb_benchmark.py` (seed 0): 512×512 px,
60° field of view, a light at the camera with distance attenuation, camera moving along the smoothed
centre lines at 0.5 mm/frame (30 fps = 15 mm/s), looking ahead 16 frames, roll random walk
0.4°/frame. Each run picks two targets (generation ≥ 3, radius ≥ 1.4 mm, ≥ 3 branches apart) and
goes forward to the first, retracts, and re-inserts to the second.

The ground truth is **known by construction**, not annotated:

- `gt_loc.csv`: the path is generated as (branch, arc length) samples, so the branch of every frame is
  exact;
- `gt_boxes.csv`: each lumen opening is a circle on the centre line at 1.1 × parent radius into the
  branch, with the measured local diameter, projected into the camera; kept only when the depth
  buffer shows it is not occluded and it is at least 10 px and at most 90 % of the frame; a parent box
  encloses its visible children (the paper's Fig. 3 rule); `id` = one identity per branch.

**Detections.** `best.pt` with confidence 0.1, NMS IoU 0.6, input 640 (Ultralytics letterbox), on CPU.
The cached files are in `results/ModelV3/benchmark/run0*/dets.txt`.

**"Ground-truth boxes" condition.** Not the raw boxes: each GT box is dropped with probability 0.05,
each corner coordinate jittered by N(0, 0.03 × box width/height), and given a score drawn from
N(0.8, 0.1) clipped to [0.35, 0.99]; random seed 0 for every run (`eval_benchmark.oracle_detections`,
`--oracle-dets`).

**Re-ID.** All reported numbers use the **colour-histogram embedder** (`--reid hist`: HSV histogram,
8×8×4 bins, 64×64 crops, square-rooted, L2-normalised). No lumen-specific ResNet50 was trained.
`tables_vb_results_oracle_reid.txt` is a separate experiment with a *simulated* trained Re-ID
(embeddings clustered by true lumen identity, noise 0.3). Note: `tools/run.py --reid auto` (its
default) uses ImageNet ResNet50 features; pass `--reid hist` to match the reported results.

**Lumen detector (`best.pt`).** The reported results use the checkpoint with MD5
`482dd64f5b8998ba6de8ea3d03c37ebe` (check with `md5sum best.pt`). It records: Ultralytics 8.4.117, **YOLO11s**
architecture (`train_args.model = yolo11s.pt`, run name `bronchotrack_yolov12`), 9.4 M parameters,
one class `lumen`, 150 epochs (early-stopping patience 50), batch 4, input 640, seed 0, default
Ultralytics augmentation (HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, horizontal flip 0.5,
mosaic 1.0, closed for the last 10 epochs), dataset `split_dataset_merged/data.yaml`; best validation
mAP50 0.636, mAP50-95 0.303, precision 0.598, recall 0.647. The code's default detector wrapper
(`--detector yolov12`) loads any Ultralytics checkpoint; **all reported results use this YOLO11s
checkpoint.** <!-- TODO (author): dataset source, number of train/val/test images, annotation tool and rule -->

**Tracking / association parameters.** Detection threshold 0.1; high/low confidence split 0.3;
EMA α 0.9; cost weight λ 0.5; gates 0.4 / 0.7 (0.9); lost-track buffer 50 frames; label gating
3 generations; child truncation 10 mm; ratio fusion σ 0.55; motion model speed 0.5 mm/frame × 30 fps
on the benchmark (`--speed 0.5`, mm/frame, in `eval_benchmark.py`). All other values are the
defaults in `BronchoTrackConfig`.

**Recorded videos.** `ModelV3_1.mp4`, `ModelV3_2.mp4` and `testvideo3.mp4` have **no ground truth**.
Only the number of location switches is reported for them. `tools/gt_from_renders.py` is an
experimental registration to renders and is not reliable enough to be used as ground truth.

## 3. Commands for the reported benchmark

```bash
# 1. airway graph (or use results/ModelV3/airway_v3.json)
python tools/slicer_to_airway.py ModelV3.vtk --labels numeric --out-dir airway

# 2. render the 6 runs: frames + ground truth (about 10 min on CPU)
python tools/make_vb_benchmark.py --mesh ModelV3.vtk --graph results/ModelV3/airway_v3.json --out vb_bench --runs 6 --seed 0

# 3a. with the detector: copy the shipped detections into the run folders
#     (or omit --dets-file and pass --weights best.pt to run the detector; gives the same file)
for r in 1 2 3 4 5 6; do cp results/ModelV3/benchmark/run0$r/dets.txt vb_bench/run0$r/; done
python tools/eval_benchmark.py --bench vb_bench --dets-file dets.txt --reid hist \
    --variants "BronchoTrack" "BronchoTrack + ratio fusion" "v3: ratio fusion + motion model" --out res_det
# 3b. with jittered ground-truth boxes
python tools/eval_benchmark.py --bench vb_bench --oracle-dets --reid hist \
    --variants "BronchoTrack" "BronchoTrack + ratio fusion" "v3: ratio fusion + motion model" --out res_gt
# 4. tables
python tools/report_benchmark.py res_det/results.json
python tools/report_benchmark.py res_gt/results.json
```

Expected (Loc Acc over all frames):

| condition | v1 | v2 | v3 | v3 vs v1 |
|---|---|---|---|---|
| detector (`dets.txt`) | 32.83 % | 33.17 % | 36.48 % | p = 0.084 (paired t) |
| jittered GT boxes | 26.09 % | 31.50 % | 37.51 % | p = 0.054 (paired t) |

Per run, detector, v1: 52.5 / 45.4 / 69.1 / 12.7 / 14.7 / 24.2 %. Full tables:
`results/ModelV3/benchmark/tables_v3_det.txt`, `tables_v3_oracle.txt`.

## 4. Verification (done on the current code)

- `make_vb_benchmark.py` re-render of run 1: `gt_loc.csv`, `gt_boxes.csv`, `poses.csv` identical,
  all 474 frames pixel-identical.
- `slicer_to_airway.py --labels numeric` and the original two-step command both reproduce
  `airway_v3.json` exactly (same labels, centre lines and diameters).
- `best.pt` re-run on the frames gives detections identical to the shipped `dets.txt`.
- Steps 3–4 reproduce both rows of the table above exactly.

Environment: Python 3.11, torch 2.14 (CPU), Ultralytics 8.4.165, VTK 9.7.1 (offscreen EGL),
NumPy 2.4, SciPy 1.17, scikit-image 0.26, OpenCV 4.13.

## 5. Running a new case

```bash
python tools/slicer_to_airway.py patient_airway.seg.nrrd --out-dir airway   # check airway/airway_preview.png
python tools/run.py --video case.mp4 --graph airway/airway.json --weights best.pt --reid hist \
    --geometry --motion --map --out-dir out/case
```

The scope must start in the trachea: the association initialises when it sees the carina
(two main-bronchus lumens held for 5 frames). Use `--init-loc <label>` to start elsewhere.
