# Grayscale vs colour input (same `best.pt`)

`--grayscale` converts every frame to grey before the detector, Re-ID and overlay.

- `detection_rendered_val.json`: `tools/compare_detectors.py` on 352 held-out rendered frames (runs 03 and 06, 829 boxes).
- `benchmark_localisation.json`: `tools/eval_benchmark.py [--grayscale]` on the 6-run ModelV3 benchmark (3591 frames), v1, v2 and v3.
- Real videos: `tools/cache_detections.py [--grayscale]`, then `tools/run.py --detector mot ... [--grayscale]`.
