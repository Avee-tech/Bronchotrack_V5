# testvideo3 run

- `airway_graph.json` – graph exported by the thesis pipeline (labels trachea / L / R / L1 ...).
- `airway_bt.json` – converted with `tools/import_graph_json.py` (standard frame, per-point radii
  as the diameter profile, L = LMB, R = RMB). `airway_bt.png` preview.
- `dets_testvideo3.txt` – `best.pt` detections (conf 0.1, NMS 0.6, imgsz 640).
- `location_*.csv`, `tracks_labels_*.csv` – original and `--geometry` runs:
  `tools/run.py --video testvideo3.mp4 --graph airway_bt.json --detector mot --weights dets_testvideo3.txt --reid hist --high-thresh 0.3 --map [--geometry]`
- Both variants give identical locations. Frames ~75-84 are a render glitch (camera clips
  through the mesh). At frame 128 a spurious large detection around the left lumen becomes a
  parent box labelled L, the lumen inside it is relabelled L1, and at frame 150 the right lumen
  is relabelled L2 as its sibling: from then on the output is most likely offset by one generation.
  No ground truth for this video.
