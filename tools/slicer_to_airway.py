#!/usr/bin/env python3
"""3D Slicer model  ->  airway graph for BronchoTrack.

One command from whatever you exported from 3D Slicer to the airway graph the
pipeline needs (``tools/run.py --graph``):

    python tools/slicer_to_airway.py ModelV3.vtk                     # surface model
    python tools/slicer_to_airway.py Segmentation.seg.nrrd --segment airway
    python tools/slicer_to_airway.py CenterlineModel.vtk             # Extract Centerline output

Input type is detected from the file (surface model / segmentation or label map /
centre-line model with a Radius array). Coordinates are read in the file's own
system (Slicer records LPS or RAS; override with --space).

Steps
  1. input -> binary airway mask (voxelised surface, chosen segment, or tube around
     the centre lines)
  2. 3-D skeleton -> branches split at bifurcations, short leaf spurs pruned
  3. trachea = branch at the most superior point; right / left main bronchus from
     the patient's right direction of the coordinate system
  4. paper's standard frame (y along the trachea, x from left to right main bronchus)
  5. lumen diameter measured along every centre line (cross-sections of the mask)
  6. labels: thesis style (trachea, R, L, R1, R2, R11 ...) or the paper's (0, 00, 01 ...)

Outputs (in --out-dir, default <input name>_airway/):
  airway.json              the graph for tools/run.py --graph
  airway_nodes.json        same tree in the thesis format, in the original Slicer coordinates
                           (label, parent, generation, centerline, radius per point, ...)
  airway_preview.png       2-D view of the labelled tree
  airway_centerlines.vtk   labelled centre lines  } drag both into Slicer on top of the
  airway_labels.mrk.json   branch-label points    } model to check the result
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.slicer_io import (centerline_to_volume, detect_space, guess_kind,  # noqa: E402
                                    labelmap_to_volume, mesh_to_volume)


# --------------------------------------------------------------------------- #
def build(path, kind="auto", segment=None, label=None, space=None, spacing=None, min_branch=4.0,
          max_generation=None, labels="anatomical", smooth=1.0, log=print):
    kind = guess_kind(path) if kind == "auto" else kind
    space = space or detect_space(path)
    if spacing is None:  # thin tubes rebuilt from centre lines need a finer grid
        spacing = 0.3 if kind == "centerline" else 0.5
    if kind == "model":
        vol = mesh_to_volume(path, spacing, space=space)
        log(f"surface model, {space}: voxelised at {spacing} mm -> {vol.mask.shape}, "
            f"{vol.mask.sum() * spacing ** 3 / 1000:.1f} cm3")
    elif kind == "segmentation":
        vol, used = labelmap_to_volume(path, segment, label, space)
        log(f"segmentation, {vol.space}: {used}, voxel {np.round(vol.spacing, 3).tolist()} mm, "
            f"{vol.mask.sum() * np.prod(vol.spacing) / 1000:.1f} cm3")
    elif kind == "centerline":
        vol = centerline_to_volume(path, spacing, space=space)
        log(f"centre-line model, {space}: rebuilt as a tube mask {vol.mask.shape}")
    else:
        raise ValueError(kind)
    if vol.mask.sum() == 0:
        raise ValueError("the airway mask is empty")
    g = AirwayGraph.from_mask(vol.mask, vol.spacing, vol.origin, root_hint=vol.top_point(),
                              patient_right=vol.patient_right, min_branch_length=min_branch,
                              smooth_sigma=smooth, direction=vol.direction)
    g.measure_diameters(vol.mask, vol.spacing, vol.origin, direction=vol.direction)
    if max_generation is not None:
        for lab in [l for l in g.labels() if g.generation(l) > max_generation]:
            g.branches.pop(lab, None)
        for b in g.branches.values():
            b.children = [c for c in b.children if c in g.branches]
        g.invalidate()
    if labels == "anatomical":
        g.relabel_anatomical()
    g.space = vol.space
    return g


def check(g: AirwayGraph, log=print):
    """Sanity checks; returns a list of warnings."""
    warn = []
    tr = g[g.root]
    mains = g.children(g.root)
    byname = {g[c].name: c for c in mains}
    r, l = byname.get("RMB"), byname.get("LMB")
    d = lambda lab: 2 * (g[lab].radius or 0)  # noqa: E731
    log(f"{len(g)} branches, {g.max_generation()} generations; trachea {tr.length:.0f} mm, "
        f"diameter {d(g.root):.1f} mm")
    if r and l:
        log(f"right main bronchus '{r}': {g[r].length:.0f} mm, diameter {d(r):.1f} mm, "
            f"{np.degrees(g.intersection_angle(g.root, r)):.0f} deg from the trachea")
        log(f"left main bronchus  '{l}': {g[l].length:.0f} mm, diameter {d(l):.1f} mm, "
            f"{np.degrees(g.intersection_angle(g.root, l)):.0f} deg from the trachea")
        if g[r].length > g[l].length and d(r) < d(l):
            warn.append("the 'right' main bronchus is longer and narrower than the left: "
                        "left/right may be swapped - check --space (LPS/RAS)")
    if len(mains) != 2:
        warn.append(f"the trachea has {len(mains)} children (expected 2): check the model / --min-branch")
    if tr.length < 40:
        warn.append(f"trachea only {tr.length:.0f} mm: is the top of the trachea in the model?")
    single = [x for x in g.labels() if len(g.children(x)) == 1]
    if single:
        warn.append(f"{len(single)} branch(es) with a single child (e.g. {single[:3]}); fine, but check "
                    "the model for gaps")
    for w in warn:
        log("WARNING: " + w)
    return warn


# --------------------------------------------------------------------------- #
def to_nodes(g: AirwayGraph) -> dict:
    """Thesis-format tree in the original (Slicer) coordinates."""
    nodes = []
    for lab in sorted(g.labels(), key=lambda x: (g.generation(x), x)):
        b = g[lab]
        pts = b.points if b.points is not None else np.stack([b.start, b.end])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        rad = [float(b.diameter_at(s) or 2 * (b.radius or 0)) / 2 for s in cum]
        ct = g.to_ct(pts)
        nodes.append({"label": lab, "generation": g.generation(lab), "parent": b.parent,
                      "start": ct[0].tolist(), "end": ct[-1].tolist(), "centerline": ct.tolist(),
                      "radius": rad, "length_mm": round(float(cum[-1]), 2),
                      "median_radius_mm": round(float(np.median(rad)), 3), "name": b.name})
    return {"coordinate_system": getattr(g, "space", "LPS"), "nodes": nodes}


def write_slicer_checks(g: AirwayGraph, out_vtk: str, out_mrk: str):
    import vtk
    space = getattr(g, "space", "LPS")
    pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    ids, names = vtk.vtkIntArray(), vtk.vtkStringArray()
    ids.SetName("BranchIndex")
    names.SetName("BranchLabel")
    labels = sorted(g.labels(), key=lambda x: (g.generation(x), x))
    ctrl = []
    for k, lab in enumerate(labels):
        b = g[lab]
        p = g.to_ct(b.points if b.points is not None else np.stack([b.start, b.end]))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(p))
        for j, q in enumerate(p):
            line.GetPointIds().SetId(j, pts.InsertNextPoint(*q))
        lines.InsertNextCell(line)
        ids.InsertNextValue(k)
        names.InsertNextValue(lab)
        mid = g.to_ct(b.point_at(b.length / 2))
        ctrl.append({"id": str(k + 1), "label": lab, "position": [float(x) for x in mid],
                     "selected": True, "locked": True, "visibility": True, "positionStatus": "defined"})
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    pd.GetCellData().AddArray(ids)
    pd.GetCellData().AddArray(names)
    pd.GetCellData().SetActiveScalars("BranchIndex")
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(out_vtk)
    w.SetInputData(pd)
    w.SetHeader(f"3D Slicer output. SPACE={space}")  # Slicer reads the coordinate system from here
    if hasattr(w, "SetFileVersion"):
        w.SetFileVersion(42)  # legacy 4.2 format, readable by Slicer 4.x and 5.x
    w.Write()
    mrk = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/"
                      "Resources/Schema/markups-schema-v1.0.3.json#",
           "markups": [{"type": "Fiducial", "coordinateSystem": space, "coordinateUnits": "mm",
                        "locked": True, "labelFormat": "%N-%d", "controlPoints": ctrl,
                        "display": {"visibility": True, "glyphScale": 1.5, "textScale": 3.0,
                                    "pointLabelsVisibility": True}}]}
    json.dump(mrk, open(out_mrk, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Slicer export: .vtk/.vtp/.stl/.obj/.ply model, .seg.nrrd/.nrrd/.nii "
                                  "segmentation, or Extract Centerline 'Centerline model' (.vtk/.vtp)")
    ap.add_argument("--kind", default="auto", choices=["auto", "model", "segmentation", "centerline"])
    ap.add_argument("--segment", help="segment name in a .seg.nrrd (default: one named *airway*, else the first)")
    ap.add_argument("--label", type=int, help="label value in a label map (default: all non-zero)")
    ap.add_argument("--space", choices=["LPS", "RAS"], help="override the coordinate system read from the file")
    ap.add_argument("--spacing", type=float, help="voxel size (mm) for models (default 0.5) / centre lines (0.3)")
    ap.add_argument("--min-branch", type=float, default=4.0, help="prune leaf spurs shorter than this (mm)")
    ap.add_argument("--max-generation", type=int, help="drop branches deeper than this")
    ap.add_argument("--smooth", type=float, default=1.0, help="mask smoothing (voxels) before skeletonising")
    ap.add_argument("--labels", default="anatomical", choices=["anatomical", "numeric"],
                    help="anatomical: trachea/R/L/R1...; numeric: the paper's 0/00/01...")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    stem = os.path.basename(a.input)
    for ext in (".seg.nrrd", ".nii.gz", ".mrk.json"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    else:
        stem = os.path.splitext(stem)[0]
    out = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.input)), stem + "_airway")
    os.makedirs(out, exist_ok=True)

    g = build(a.input, a.kind, a.segment, a.label, a.space, a.spacing, a.min_branch, a.max_generation,
              a.labels, a.smooth)
    check(g)
    g.to_json(os.path.join(out, "airway.json"))
    json.dump(to_nodes(g), open(os.path.join(out, "airway_nodes.json"), "w"), indent=1)
    write_slicer_checks(g, os.path.join(out, "airway_centerlines.vtk"), os.path.join(out, "airway_labels.mrk.json"))
    from build_graph import preview
    preview(g, os.path.join(out, "airway_preview.png"))
    print(f"-> {out}/airway.json   (use: tools/run.py --graph {os.path.join(out, 'airway.json')})")


if __name__ == "__main__":
    main()
