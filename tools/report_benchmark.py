#!/usr/bin/env python3
"""Tables in the paper's layout from eval_benchmark.py results.json.

python tools/report_benchmark.py vb_results/results.json [--baseline BronchoTrack]
"""

import json
import sys
from collections import defaultdict

import numpy as np
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronchotrack.metrics import compare_accuracy  # noqa: E402


def main(path, baseline="BronchoTrack"):
    R = json.load(open(path))
    methods = list(R)
    runs = sorted(next(iter(R.values())).keys())
    print(f"{len(runs)} trajectories, {sum(R[methods[0]][r]['n'] for r in runs)} frames\n")

    # ---- Table III: tracking (mean over trajectories), FPS
    keys = ["MOTA", "IDF1", "HOTA", "FP", "FN", "IDs", "FPS"]
    print("Tracking (mean over trajectories)".ljust(34) + "".join(k.rjust(9) for k in keys))
    for m in methods:
        vals = [np.mean([R[m][r][k] for r in runs]) for k in keys]
        print(m.ljust(34) + "".join(f"{v:9.2f}" for v in vals))

    # ---- Table IV: localisation
    print("\nLocalisation".ljust(34) + "LocAcc(all frames)  mean per-traj  gen.err  tree dist   AP    p vs " + baseline)
    base = [R[baseline][r]["LocAcc"] for r in runs]
    for m in methods:
        corr = np.concatenate([R[m][r]["correct"] for r in runs])
        per = [R[m][r]["LocAcc"] for r in runs]
        ge = np.concatenate([R[m][r]["gen_err"] for r in runs]) if runs else []
        td = np.mean([R[m][r]["tree_dist"] for r in runs])
        ap_ = np.mean([R[m][r]["AP"] for r in runs])
        p = compare_accuracy(per, base) if m != baseline else {"p": float("nan"), "test": ""}
        print(m.ljust(34) + f"{100 * corr.mean():10.2f} %      {np.mean(per):8.2f}   {np.mean(ge):+7.2f}  "
              f"{td:8.2f}  {ap_:6.2f}   {p['p']:.3f} {p['test']}")

    # ---- per trajectory
    print("\nPer-trajectory Loc Acc".ljust(34) + "".join(r.rjust(8) for r in runs))
    for m in methods:
        print(m.ljust(34) + "".join(f"{R[m][r]['LocAcc']:8.1f}" for r in runs))

    # ---- per branch (baseline vs every method with 'fusion' in its name)
    pb = defaultdict(lambda: defaultdict(list))
    for m in methods:
        for r in runs:
            for b, acc in R[m][r]["per_branch"].items():
                pb[m][b].append(acc)
    branches = sorted({b for m in methods for b in pb[m]}, key=lambda x: (len(x), x))
    print("\nPer-branch Loc Acc (mean over trajectories visiting it)")
    print("branch".ljust(8) + "".join(m.replace("BronchoTrack", "BT")[:22].rjust(24) for m in methods))
    for b in branches:
        print(b.ljust(8) + "".join((f"{np.mean(pb[m][b]):24.1f}" if pb[m][b] else " " * 24) for m in methods))

    # ---- probability calibration of fused variants
    for m in methods:
        probs = [(p, c) for r in runs for p, c in zip(R[m][r]["loc_prob"], R[m][r]["correct"]) if p is not None]
        if probs:
            P = np.array(probs, float)
            ok, bad = P[P[:, 1] == 1, 0], P[P[:, 1] == 0, 0]
            print(f"\n{m}: location probability mean {P[:, 0].mean():.2f}; when correct {ok.mean():.2f} "
                  f"(n={len(ok)}), when wrong {bad.mean() if len(bad) else float('nan'):.2f} (n={len(bad)})")
            for lo, hi in [(0, .6), (.6, .8), (.8, .95), (.95, 1.01)]:
                sel = (P[:, 0] >= lo) & (P[:, 0] < hi)
                if sel.any():
                    print(f"   p in [{lo:.2f},{min(hi, 1):.2f}): {sel.sum():5d} frames, accuracy {100 * P[sel, 1].mean():.1f} %")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = sys.argv[sys.argv.index("--baseline") + 1] if "--baseline" in sys.argv else "BronchoTrack"
    main(args[0], base)
