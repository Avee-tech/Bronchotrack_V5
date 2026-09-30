# BronchoTrack V5

Branch-level bronchoscope localisation, re-implemented from Tian et al., *"BronchoTrack: Airway Lumen
Tracking for Branch-Level Bronchoscopic Localization"* (IEEE TMI 2025). This version adds a YOLOv12
lumen detector and two new cues: ratio fusion and a motion model.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

**1. Build the airway graph.** Export the airway from 3D Slicer as one of:
- a surface model (`.vtk` / `.stl`)
- a segmentation (`.seg.nrrd`)
- an Extract Centerline model

Then run:

```powershell
python tools\slicer_to_airway.py ModelV3.vtk --out-dir airway
```

If you already have a thesis-format `airway_graph.json`, import it instead:

```powershell
python tools\import_graph_json.py airway_graph.json --out airway\airway.json
```

**2. Run on a video.** Use the YOLOv12 lumen detector (`best.pt`, one class `lumen`):

```powershell
python tools\run.py --video ModelV3_1.mp4 --graph airway\airway.json --weights best.pt `
    --geometry --motion --map --device cpu --out-dir out\ModelV3_1
```

The flags choose the version:

| flags | version |
|---|---|
| *(none)* | v1: original BronchoTrack |
| `--geometry` | v2: v1 + ratio fusion |
| `--geometry --motion` | v3: v2 + motion model (recommended) |

Other options:
- `--speed 7`: average scope speed in mm/s.
- `--lc`: loop closure.
- `--detector yolov7 --yolov7-repo <path>`: use a YOLOv7 checkpoint.

**Outputs** in `--out-dir`:
- `overlay.mp4`: boxes with labels and probabilities, and the airway map on the right.
- `location.csv`: the branch for each frame.
- `tracks_labels.csv`

**Tests:** `python -m pytest -q tests`

## Improvements over the original BronchoTrack

- **YOLOv12 detector.** Replaces the paper's YOLOv7 as the default lumen detector.
- **v2: ratio fusion.**
  - Each lumen box is straightened using the roll estimate. Its longest edge gives the lumen diameter.
  - The observed diameter : distance ratio between lumens is compared with the ratio from the airway graph. The graph ratio uses the diameter measured 1.1 × parent radius into each child branch.
  - The resulting likelihood is fused with the original graph association.
  - Each box shows three probabilities: association (A), ratio (R) and fused (F).
- **v3: motion model.**
  - A particle filter follows the scope along the airway (branch, depth and speed), using the average insertion speed.
  - Branches the scope cannot reach within about 1 s are suppressed, so the location no longer jumps several generations at once.

**Localisation accuracy** on the rendered ModelV3 benchmark (6 runs, 3591 frames):

| detections | v1 | v2 | v3 |
|---|---|---|---|
| `best.pt` | 32.8 % | 33.2 % | **36.5 %** |
| ground-truth boxes | 26.1 % | 31.5 % | **37.5 %** |

**Location switches** on the real videos:

| video | v1 | v3 |
|---|---|---|
| ModelV3_1 | 16 | 12 |
| testvideo3 | 17 | 10 |
