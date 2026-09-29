#!/usr/bin/env python3
"""Build the semantic airway graph M from a pre-operative airway model.

Inputs (one of):
  --mask   binary airway segmentation (.nii/.nii.gz/.nrrd/.mha via SimpleITK, or .npy)
  --vtk    centre-line poly-data (.vtk/.vtp, e.g. 3D Slicer "Extract Centerline")
  --polylines  .npz with arrays line_0, line_1, ... (N_i x 3)

The graph is transformed into the paper's standard frame and labelled as in
Fig. 1(d). Output: JSON consumed by tools/run.py, plus an optional PNG preview.

Example
-------
python tools/build_graph.py --mask airway_seg.nii.gz --out airway.json --preview airway.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402


def load_mask(path):
    if path.endswith(".npy"):
        return np.load(path), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    if path.endswith(".npz"):  # tools/voxelize_mesh.py output
        z = np.load(path)
        return z["mask"], tuple(z["spacing"]), tuple(z["origin"])
    import SimpleITK as sitk

    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)  # -> (x, y, z) index order
    return arr > 0, img.GetSpacing(), img.GetOrigin()


def preview(g: AirwayGraph, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    for b in g.branches.values():
        p = b.points if b.points is not None else np.stack([b.start, b.end])
        ax.plot(p[:, 0], -p[:, 1], lw=max(0.6, 3.0 - 0.45 * b.generation), color="tab:blue")
        if b.generation <= 3:
            m = b.point_at(b.length / 2)
            ax.text(m[0], -m[1], b.label, fontsize=7, color="tab:red")
    ax.set_aspect("equal")
    ax.set_xlabel("x (standard frame)")
    ax.set_ylabel("-y (trachea direction)")
    ax.set_title(f"Airway graph: {len(g)} branches, {g.max_generation()} generations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mask")
    src.add_argument("--vtk")
    src.add_argument("--polylines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-hint", type=float, nargs=3, help="point near the top of the trachea")
    ap.add_argument("--patient-right", type=float, nargs=3,
                    help="vector to the patient's right in input coords (RAS/Slicer: 1 0 0; LPS/DICOM/SimpleITK: -1 0 0). Default: the more vertical main bronchus is taken as right")
    ap.add_argument("--min-branch", type=float, default=3.0, help="prune leaf spurs shorter than this (mm)")
    ap.add_argument("--merge-tol", type=float, default=1.0)
    ap.add_argument("--max-generation", type=int, default=None, help="drop branches deeper than this")
    ap.add_argument("--preview")
    a = ap.parse_args()

    kw = dict(root_hint=None if a.root_hint is None else np.array(a.root_hint),
              patient_right=None if a.patient_right is None else np.array(a.patient_right))
    if a.mask:
        m, sp, org = load_mask(a.mask)
        g = AirwayGraph.from_mask(m, sp, org, min_branch_length=a.min_branch, **kw)
        g.measure_diameters(m, sp, org)  # local lumen diameter along every centre line
    elif a.vtk:
        g = AirwayGraph.from_vtk(a.vtk, merge_tol=a.merge_tol, min_branch_length=a.min_branch, **kw)
    else:
        z = np.load(a.polylines)
        g = AirwayGraph.from_polylines([z[k] for k in sorted(z.files)], merge_tol=a.merge_tol,
                                       min_branch_length=a.min_branch, **kw)
    if a.max_generation is not None:
        for lab in [l for l in g.labels() if g.generation(l) > a.max_generation]:
            g.branches.pop(lab, None)
        for b in g.branches.values():
            b.children = [c for c in b.children if c in g.branches]
    g.to_json(a.out)
    names = {b.name: b.label for b in g.branches.values() if b.name in ("Trachea", "RMB", "LMB")}
    print(f"{len(g)} branches, max generation {g.max_generation()}  {names}  -> {a.out}")
    if a.preview:
        preview(g, a.preview)


if __name__ == "__main__":
    main()
