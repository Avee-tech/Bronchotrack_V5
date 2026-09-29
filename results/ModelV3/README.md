# ModelV3 run (virtual bronchoscopy, 3D Slicer renders)

- `airway_v3.json` / `airway_v3.png` – airway graph built from `ModelV3.vtk` (surface mesh):
  `tools/voxelize_mesh.py ModelV3.vtk mask_v3.npz 0.5` then
  `tools/build_graph.py --mask mask_v3.npz --out airway_v3.json --patient-right -1 0 0 --min-branch 4`
  (now also stores the lumen diameter profile along every centre line)
  (21 branches, 4 generations; `00` = LMB, `01` = RMB; `ct_frame` maps back to LPS coordinates).
- `dets_ModelV3_*.txt` – cached `best.pt` detections (conf 0.1, NMS 0.6, imgsz 640), from
  `tools/cache_detections.py`.
- `location_*.csv`, `tracks_labels_*.csv` – BronchoTrack output:
  `tools/run.py --video ModelV3_1.mp4 --graph airway_v3.json --detector mot --weights dets_ModelV3_1.txt --reid hist --high-thresh 0.3 --map`
- No ground truth yet. `tools/gt_from_renders.py` (+ `render_lib.py`) is an experimental
  render-registration pseudo-GT; it was not reliable for video 1 (camera leaves the centre line).
