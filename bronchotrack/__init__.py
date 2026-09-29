"""BronchoTrack: airway lumen tracking for branch-level bronchoscopic localization.

Re-implementation of Tian et al., IEEE TMI 44(3):1321-1333, 2025
(doi:10.1109/TMI.2024.3493170).

Modules map onto the paper as follows

    airway_graph.py   Sec. III (preamble)  standard airway graph M, labels, g^k(.)
    detector.py       Sec. III-A           YOLOv7 lumen detector (+ ultralytics / GT backends)
    kalman.py         Sec. III-B-1         7-D constant-velocity Kalman filter, Eq. (2)
    reid.py           Sec. III-B-2         ResNet50 Re-ID embeddings, EMA Eq. (3), Eq. (4)
    tracker.py        Sec. III-B           two-stage matching, Eq. (5), gating, 50-frame buffer
    subgraph.py       Sec. III-C / IV-A    lumen subgraph S^t from box inclusion
    association.py    Sec. III-C           Algorithms 1 & 2, Eqs. (6)-(9), gallery
    loop_closure.py   Sec. III-D           BronchoTrack-LC (LoFTR keyframe loop closure)
    pipeline.py       Fig. 1               the full per-frame pipeline
    metrics.py        Sec. IV-C            MOTA/IDF1/HOTA/FP/FN/IDs, Loc Acc, AP, stats tests
"""

__version__ = "1.0.0"

from .airway_graph import AirwayGraph, Branch  # noqa: F401
from .pipeline import BronchoTrack, BronchoTrackConfig  # noqa: F401
