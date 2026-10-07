#!/usr/bin/env python3
"""Learn a doctor's speed profile from past runs (version 4).

Each run is a location CSV: tools/run.py's location.csv, or a reference path
(frame,location), e.g. annotated or EM-tracked. Every complete pass through a branch gives
one speed observation (branch length / time in the branch), which updates the doctor's
profile and the population profile (bronchotrack/speed_model.py).

python tools/learn_speed.py --graph airway.json --operator dr_lee --fps 30 \
    out/case01/location.csv out/case02/location.csv
python tools/learn_speed.py --graph airway.json --operator dr_lee --show      # print the profile
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.speed_model import (SpeedProfiles, branch_features, extract_traversals,  # noqa: E402
                                      read_location_csv, traversal_data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="location CSVs, one per run")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--operator", help="doctor's name (default: population profile only)")
    ap.add_argument("--profile-dir", default="speed_profiles")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=3.5, help="prior average speed for a new profile, mm/s")
    ap.add_argument("--min-dur", type=float, default=0.3, help="ignore location flicker shorter than this (s)")
    ap.add_argument("--show", action="store_true", help="print the profile and per-branch speeds")
    a = ap.parse_args()
    g = AirwayGraph.from_json(a.graph)
    prof = SpeedProfiles(a.profile_dir, a.speed)
    for path in a.runs:
        fr, loc, pr = read_location_csv(path)
        trav = extract_traversals(g, loc, a.fps, pr, fr, a.min_dur)
        if not trav:
            print(f"{path}: no complete branch traversal")
            continue
        prof.learn(a.operator, *traversal_data(g, trav))
        sp = ", ".join(f"{t.branch}{'<' if t.retract else '>'}{t.speed:.1f}" for t in trav)
        print(f"{path}: {len(trav)} traversals (mm/s): {sp}")
    m = prof.get(a.operator)
    print(m.describe())
    if a.show or a.runs:
        print(f"{'branch':8s} {'gen':>3s} {'diam':>5s} {'angle':>5s} {'cplx':>5s} {'adv mm/s':>9s} {'ret mm/s':>9s}")
        for lab in sorted(g.labels(), key=lambda x: (g.generation(x), x)):
            x = branch_features(g, lab)
            print(f"{lab:8s} {g.generation(lab):3d} {5 * np.exp(x[1]):5.1f} {np.degrees(x[3]):5.0f} {x[4]:5.2f} "
                  f"{m.speed(x):9.1f} {m.speed(branch_features(g, lab, True)):9.1f}")


if __name__ == "__main__":
    main()
