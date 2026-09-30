"""Readers for 3D Slicer exports and conversion to a binary airway mask.

Every input is turned into a binary mask on a voxel grid (``Volume``) in the file's
own patient coordinate system (LPS or RAS), which ``AirwayGraph.from_mask``
skeletonises and ``measure_diameters`` samples for the lumen diameters.

Supported Slicer outputs
------------------------
* **Surface models** – ``.vtk`` / ``.vtp`` / ``.stl`` / ``.obj`` / ``.ply`` (segment
  "Export to files" / "Export visible segments to models", Models module, or a mesh
  from any other tool). Voxelised with a polygon-to-stencil filter.
* **Segmentations / label maps** – ``.seg.nrrd`` (Segment Editor), ``.nrrd`` label map,
  ``.nii`` / ``.nii.gz``. For ``.seg.nrrd`` the segment is chosen by name.
* **Centre lines** – the "Centerline model" of Extract Centerline (poly lines with a
  point array ``Radius``; VMTK's ``MaximumInscribedSphereRadius`` also works), as
  ``.vtk`` / ``.vtp``. Rebuilt as a tube mask (a sphere of the local radius at every
  centre-line point), which handles overlapping, merged or branch-split centre lines
  alike.

Coordinate systems
------------------
Slicer ≥ 4.11 writes models in LPS and records it in the file (``SPACE=LPS`` in the
header of .vtk/.stl/.obj, ``space`` in .nrrd, ``coordinateSystem`` in .mrk.json);
older files are RAS. ``detect_space`` reads this; it can be overridden.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Volume:
    """Binary mask with its voxel-to-patient mapping:  p = origin + direction @ (index * spacing)."""

    mask: np.ndarray            # bool, indexed (i, j, k)
    spacing: np.ndarray         # (3,) mm
    origin: np.ndarray          # (3,) mm
    direction: np.ndarray       # (3, 3), columns = unit index axes in patient space
    space: str                  # "LPS" or "RAS"

    def world(self, idx: np.ndarray) -> np.ndarray:
        return (np.asarray(idx, float) * self.spacing) @ self.direction.T + self.origin

    @property
    def patient_right(self) -> np.ndarray:
        """Unit vector to the patient's right in this volume's coordinates."""
        return np.array([-1.0, 0.0, 0.0]) if self.space == "LPS" else np.array([1.0, 0.0, 0.0])

    def top_point(self) -> np.ndarray:
        """The most superior (+S) lumen voxel: the upper end of the trachea."""
        idx = np.argwhere(self.mask)
        w = self.world(idx)
        return w[int(np.argmax(w[:, 2]))]


# --------------------------------------------------------------------------- #
# coordinate system
# --------------------------------------------------------------------------- #
def detect_space(path: str, default: str = "LPS") -> str:
    """LPS / RAS as recorded by Slicer in the file (``default`` if not recorded)."""
    low = path.lower()
    try:
        if low.endswith((".nrrd", ".nhdr")):
            import nrrd
            sp = str(nrrd.read_header(path).get("space", "")).lower()
            if "right-anterior" in sp or sp == "ras":
                return "RAS"
            if "left-posterior" in sp or sp == "lps":
                return "LPS"
            return default
        if low.endswith((".nii", ".nii.gz")):
            return "LPS"  # read through SimpleITK, which always returns LPS
        if low.endswith(".json"):
            d = json.load(open(path))
            cs = d.get("markups", [{}])[0].get("coordinateSystem", default)
            return "RAS" if str(cs).upper() == "RAS" else "LPS"
        with open(path, "rb") as f:
            head = f.read(2048).decode("latin-1", "ignore")
        m = re.search(r"SPACE\s*=\s*(RAS|LPS)", head, re.I)
        if m:
            return m.group(1).upper()
        if low.endswith(".vtk") and "3D Slicer" in head:
            return "RAS"  # Slicer < 4.11 wrote models in RAS without a tag
    except Exception:
        pass
    return default


# --------------------------------------------------------------------------- #
# surface models
# --------------------------------------------------------------------------- #
def read_polydata(path: str):
    import vtk

    low = path.lower()
    if low.endswith(".vtp"):
        r = vtk.vtkXMLPolyDataReader()
    elif low.endswith(".stl"):
        r = vtk.vtkSTLReader()
    elif low.endswith(".obj"):
        r = vtk.vtkOBJReader()
    elif low.endswith(".ply"):
        r = vtk.vtkPLYReader()
    else:
        r = vtk.vtkPolyDataReader()
    r.SetFileName(path)
    r.Update()
    pd = r.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise ValueError(f"no geometry read from {path}")
    return pd


def mesh_to_volume(path: str, spacing: float = 0.5, pad: float = 3.0, space: Optional[str] = None) -> Volume:
    """Voxelise a closed airway surface (inside = 1)."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    pd = read_polydata(path)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    pd = clean.GetOutput()
    if pd.GetNumberOfPolys() == 0:
        raise ValueError(f"{path} has no surface polygons (is it a centre line? use --centerline)")
    b = np.array(pd.GetBounds()).reshape(3, 2)
    origin = b[:, 0] - pad
    dims = np.ceil((b[:, 1] + pad - origin) / spacing).astype(int) + 1
    img = vtk.vtkImageData()
    img.SetSpacing([spacing] * 3)
    img.SetOrigin(origin.tolist())
    img.SetDimensions(dims.tolist())
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(1)
    st = vtk.vtkPolyDataToImageStencil()
    st.SetInputData(pd)
    st.SetOutputOrigin(origin.tolist())
    st.SetOutputSpacing([spacing] * 3)
    st.SetOutputWholeExtent(img.GetExtent())
    st.Update()
    cut = vtk.vtkImageStencil()
    cut.SetInputData(img)
    cut.SetStencilConnection(st.GetOutputPort())
    cut.ReverseStencilOff()
    cut.SetBackgroundValue(0)
    cut.Update()
    arr = vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1]).transpose(2, 1, 0)
    mask = arr > 0
    if mask.sum() == 0:
        raise ValueError(f"voxelising {path} gave an empty mask (surface not closed?)")
    return Volume(mask, np.full(3, float(spacing)), origin, np.eye(3), space or detect_space(path))


# --------------------------------------------------------------------------- #
# segmentations / label maps
# --------------------------------------------------------------------------- #
def _nrrd_segments(header) -> List[dict]:
    segs = []
    k = 0
    while f"Segment{k}_ID" in header or f"Segment{k}_Name" in header:
        segs.append({"index": k, "name": header.get(f"Segment{k}_Name", f"Segment_{k}"),
                     "label": int(header.get(f"Segment{k}_LabelValue", k + 1)),
                     "layer": int(header.get(f"Segment{k}_Layer", 0))})
        k += 1
    return segs


def labelmap_to_volume(path: str, segment: Optional[str] = None, label: Optional[int] = None,
                       space: Optional[str] = None) -> Tuple[Volume, str]:
    """Binary mask of one segment / label. Returns (volume, description of what was used)."""
    low = path.lower()
    if low.endswith((".nrrd", ".nhdr")):
        import nrrd
        data, h = nrrd.read(path)
        D = np.asarray(h.get("space directions"), dtype=object)
        rows = [np.asarray(r, float) for r in D if r is not None and not (isinstance(r, str))
                and np.all(np.isfinite(np.asarray(r, float)))]
        Dm = np.stack(rows, 0).T  # columns = index axes (with spacing)
        spacing = np.linalg.norm(Dm, axis=0)
        direction = Dm / spacing
        origin = np.asarray(h.get("space origin", [0, 0, 0]), float)
        if data.ndim == 4:  # (layers, i, j, k) in multi-layer .seg.nrrd
            layer_axis = [i for i, r in enumerate(D) if r is None or isinstance(r, str)
                          or not np.all(np.isfinite(np.asarray(r, float)))]
            data = np.moveaxis(data, layer_axis[0] if layer_axis else 0, 0)
        segs = _nrrd_segments(h)
        used = ""
        if segs:
            pick = None
            if segment:
                pick = next((s for s in segs if segment.lower() in s["name"].lower()), None)
                if pick is None:
                    raise ValueError(f"segment '{segment}' not found; segments: {[s['name'] for s in segs]}")
            if pick is None:
                pick = next((s for s in segs if "airway" in s["name"].lower()), segs[0])
            layer = data[pick["layer"]] if data.ndim == 4 else data
            mask = layer == pick["label"]
            used = f"segment '{pick['name']}' (label {pick['label']}, layer {pick['layer']})"
        else:
            mask = (data == label) if label is not None else (data > 0)
            used = f"label {label}" if label is not None else "all non-zero voxels"
        sp_name = space or detect_space(path)
        return Volume(mask.astype(bool), spacing, origin, direction, sp_name), used
    # NIfTI / MHA / anything SimpleITK reads (returned in LPS)
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    mask = (arr == label) if label is not None else (arr > 0)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return (Volume(mask.astype(bool), np.asarray(img.GetSpacing(), float), np.asarray(img.GetOrigin(), float),
                   direction, "LPS"), f"label {label}" if label is not None else "all non-zero voxels")


# --------------------------------------------------------------------------- #
# centre lines (Extract Centerline)
# --------------------------------------------------------------------------- #
RADIUS_ARRAYS = ("Radius", "MaximumInscribedSphereRadius", "radius")


def read_centerline(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Poly lines and their per-point radii from a centre-line model."""
    from vtk.util.numpy_support import vtk_to_numpy
    import vtk

    pd = read_polydata(path)
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    rad = None
    for name in RADIUS_ARRAYS:
        a = pd.GetPointData().GetArray(name)
        if a is not None:
            rad = vtk_to_numpy(a).astype(float).reshape(-1)
            break
    if rad is None:
        raise ValueError(f"{path}: no radius array ({', '.join(RADIUS_ARRAYS)}) - export the "
                         "'Centerline model' from Extract Centerline, or pass the airway model instead")
    lines, radii = [], []
    cells = pd.GetLines()
    cells.InitTraversal()
    ids = vtk.vtkIdList()
    while cells.GetNextCell(ids):
        k = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if len(k) >= 2:
            lines.append(pts[k])
            radii.append(rad[k])
    if not lines:
        raise ValueError(f"{path}: no poly lines")
    return lines, radii


def centerline_to_volume(path: str, spacing: float = 0.5, step: Optional[float] = None,
                         space: Optional[str] = None) -> Volume:
    """Tube mask: every voxel within the local radius of the nearest centre-line point."""
    from scipy.ndimage import distance_transform_edt

    lines, radii = read_centerline(path)
    step = step or spacing / 2
    P, R = [], []
    for p, r in zip(lines, radii):  # resample densely so the spheres overlap
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        s = np.arange(0, cum[-1] + 1e-9, step)
        P.append(np.stack([np.interp(s, cum, p[:, j]) for j in range(3)], 1))
        R.append(np.interp(s, cum, r))
    P, R = np.concatenate(P), np.maximum(np.concatenate(R), spacing)
    pad = R.max() + 3
    origin = P.min(0) - pad
    dims = np.ceil((P.max(0) + pad - origin) / spacing).astype(int) + 1
    seeds = np.ones(dims, bool)
    rad_img = np.zeros(dims, np.float32)
    ij = np.round((P - origin) / spacing).astype(int)
    seeds[ij[:, 0], ij[:, 1], ij[:, 2]] = False
    rad_img[ij[:, 0], ij[:, 1], ij[:, 2]] = np.maximum(rad_img[ij[:, 0], ij[:, 1], ij[:, 2]], R)
    dist, ind = distance_transform_edt(seeds, sampling=spacing, return_indices=True)
    mask = dist <= rad_img[ind[0], ind[1], ind[2]]
    return Volume(mask, np.full(3, float(spacing)), origin, np.eye(3), space or detect_space(path))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
MESH_EXT = (".vtk", ".vtp", ".stl", ".obj", ".ply")
LABEL_EXT = (".nrrd", ".nhdr", ".nii", ".nii.gz", ".mha", ".mhd")


def guess_kind(path: str) -> str:
    low = path.lower()
    if low.endswith(LABEL_EXT):
        return "segmentation"
    if low.endswith(MESH_EXT):
        if low.endswith((".vtk", ".vtp")):
            try:
                pd = read_polydata(path)
                if pd.GetNumberOfPolys() == 0 and pd.GetNumberOfLines() > 0:
                    return "centerline"
            except Exception:
                pass
        return "model"
    raise ValueError(f"unsupported file type: {os.path.basename(path)}")
