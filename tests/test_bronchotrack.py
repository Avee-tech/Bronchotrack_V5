"""Unit / integration tests.  Run:  python -m pytest -q tests"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bronchotrack.airway_graph import AirwayGraph  # noqa: E402
from bronchotrack.association import AirwayAssociator, rotate  # noqa: E402
from bronchotrack.detector import AnnotationDetector, nms  # noqa: E402
from bronchotrack.kalman import KalmanFilter7, xyha_to_xyxy, xyxy_to_xyha  # noqa: E402
from bronchotrack.metrics import compare_accuracy, hota, idf1, clear_mot, label_ap  # noqa: E402
from bronchotrack.pipeline import BronchoTrack, BronchoTrackConfig  # noqa: E402
from bronchotrack.reid import appearance_cost, ema_update  # noqa: E402
from bronchotrack.subgraph import build_subgraph  # noqa: E402
from bronchotrack.synthetic import OracleEmbedder, make_tree, simulate  # noqa: E402
from bronchotrack.tracker import LumenTracker, TrackerConfig  # noqa: E402


# ------------------------------------------------------------------ airway graph
def test_standard_frame_and_labels():
    g = make_tree(4, seed=3)
    tr = g["0"]
    assert np.allclose(tr.direction, [0, 1, 0], atol=1e-6)          # y along the trachea
    assert np.allclose(tr.end, 0, atol=1e-6)                        # origin at the carina
    assert g["00"].name == "RMB" and g["00"].end[0] > g["01"].end[0]  # x: left -> right main
    assert abs(g["00"].end[2] - g["01"].end[2]) < 1e-6              # both ends in the x-y plane
    for lab, b in g.branches.items():
        for c in b.children:
            assert c.startswith(lab) and len(c) == len(lab) + 1   # Fig. 1(d) labels
    assert g.ancestor("0110", 2) == "01" and g.ancestor("0", 3) == "0"
    assert g.ancestor_strict("01", 2) is None
    assert g.tree_distance("000", "011") == 4


def test_json_roundtrip(tmp_path):
    g = make_tree(3, seed=1)
    p = tmp_path / "g.json"
    g.to_json(str(p))
    h = AirwayGraph.from_json(str(p))
    assert set(h.labels()) == set(g.labels())
    assert np.allclose(h["011"].end, g["011"].end)


def test_transported_frames_are_orthonormal():
    g = make_tree(4, seed=2)
    for lab in g.labels():
        e1, e2, d = g.frame(lab)
        M = np.stack([e1, e2, d])
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.cross(e1, e2), d, atol=1e-6)


def _voxelize(g, spacing=1.0, pad=10):
    pts = np.concatenate([b.points for b in g.branches.values()])
    lo = pts.min(0) - pad
    shape = np.ceil((pts.max(0) + pad - lo) / spacing).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1) * spacing + lo
    mask = np.zeros(shape, bool)
    for b in g.branches.values():
        a, c = b.start, b.end
        v = c - a
        t = np.clip(((grid - a) @ v) / (v @ v), 0, 1)
        d = np.linalg.norm(grid - (a + t[..., None] * v), axis=-1)
        mask |= d <= max(1.5, 0.6 * b.radius)
    return mask, lo


def test_graph_from_mask_skeleton():
    g = make_tree(2, seed=5)  # 1 + 2 + 4 branches
    mask, lo = _voxelize(g)
    top = g["0"].start
    h = AirwayGraph.from_mask(mask, (1, 1, 1), lo, root_hint=top, min_branch_length=4.0)
    assert len(h) == len(g)
    assert h.max_generation() == 2
    assert np.allclose(h["0"].direction, [0, 1, 0], atol=1e-6)
    assert abs(h["0"].length - g["0"].length) < 5          # trachea recovered end to end
    assert h["00"].name == "RMB" and h["01"].name == "LMB"


# ------------------------------------------------------------------ detection / KF
def test_nms_and_box_conversions():
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 70]], float)
    keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.6)
    assert list(keep) == [0, 2]
    assert np.allclose(xyha_to_xyxy(xyxy_to_xyha(b)), b)


def test_kalman_constant_velocity():
    kf = KalmanFilter7()
    m, P = kf.initiate(np.array([10, 10, 20, 1.0]))
    for t in range(1, 30):
        m, P = kf.predict(m, P)
        m, P = kf.update(m, P, np.array([10 + 3 * t, 10 - 2 * t, 20 + 0.5 * t, 1.0]))
    assert abs(m[4] - 3) < 0.2 and abs(m[5] + 2) < 0.2 and abs(m[6] - 0.5) < 0.1


def test_reid_ema_and_cost():
    e = ema_update(None, np.array([1.0, 0, 0]))
    e = ema_update(e, np.array([0, 1.0, 0]), 0.9)
    assert e[0] > e[1] > 0 and np.isclose(np.linalg.norm(e), 1)
    c = appearance_cost(np.eye(3)[:2], np.eye(3))
    assert np.allclose(c, [[0, 1, 1], [1, 0, 1]])


# ------------------------------------------------------------------ tracker
def test_tracker_keeps_identities_and_buffer():
    trk = LumenTracker(TrackerConfig(use_reid=False))
    ids = None
    for t in range(20):
        boxes = np.array([[10 + 2 * t, 10, 40 + 2 * t, 40], [100 - t, 80, 130 - t, 110]], float)
        trk.update(boxes, np.array([0.9, 0.8]), None, t)
        cur = sorted((tr.det_index, tr.track_id) for tr in trk.active())
        if t >= 1:
            ids = ids or cur
            assert cur == ids
    for t in range(20, 20 + 52):
        trk.update(np.zeros((0, 4)), np.zeros(0), None, t)
    assert not trk.tracks  # removed after the 50-frame buffer


def test_tracker_low_score_second_stage():
    trk = LumenTracker(TrackerConfig(use_reid=False))
    trk.update(np.array([[0, 0, 20, 20]], float), np.array([0.9]), None, 0)
    trk.update(np.array([[1, 0, 21, 20]], float), np.array([0.2]), None, 1)  # low score still tracked
    assert len(trk.active()) == 1 and trk.active()[0].track_id == 1


# ------------------------------------------------------------------ subgraph
def test_subgraph_hierarchy_and_pruning():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 40, 40], [60, 10, 90, 40],
                      [150, 0, 200, 50], [152, 2, 199, 49]], float)
    S = build_subgraph(boxes, np.array([0.9, 0.8, 0.8, 0.9, 0.5]))
    assert S.nodes[1].parent == 0 and S.nodes[2].parent == 0 and S.nodes[1].level == 2
    assert 4 not in S.nodes and S.nodes[3].level == 1  # duplicate single child pruned
    assert len(S.primary()) == 2


# ------------------------------------------------------------------ association
def _carina_view(g, roll, size=256, scale=6.0):
    """Two main-bronchus boxes placed where the 2-D graph says, rotated by roll."""
    g2 = g.projected_children("0", 10.0)
    boxes = []
    for lab in ("00", "01"):
        c = np.array([size / 2, size / 2]) + scale * rotate(g2[lab], roll)
        boxes.append([c[0] - 15, c[1] - 15, c[0] + 15, c[1] + 15])
    return np.array(boxes)


@pytest.mark.parametrize("roll_deg", [0, 25, -40])
def test_eq6_initialisation_and_labels(roll_deg):
    g = make_tree(3, seed=4)
    A = AirwayAssociator(g)
    A.cfg.init_roll = np.radians(roll_deg)  # prior close to the truth
    A.cfg.init_stable_frames = 1
    boxes = _carina_view(g, np.radians(roll_deg))
    S = build_subgraph(boxes, np.ones(2), track_ids=[1, 2], track_ages=[5, 5])
    loc = A.step(S, 0, (256, 256))
    assert loc == "0"
    assert S.nodes[0].label == "00" and S.nodes[1].label == "01"
    assert abs(np.degrees(A.roll) - roll_deg) < 1e-6


def test_eq7_8_roll_propagation():
    g = make_tree(3, seed=4)
    A = AirwayAssociator(g)
    A.cfg.init_stable_frames = 1
    S = build_subgraph(_carina_view(g, 0.0), np.ones(2), track_ids=[1, 2], track_ages=[5, 5])
    A.step(S, 0, (256, 256))
    for t, r in enumerate([5, 10, 15, 20], start=1):  # scope rolls 5 deg per frame
        S = build_subgraph(_carina_view(g, np.radians(r)), np.ones(2), track_ids=[1, 2], track_ages=[5 + t] * 2)
        for n, lab in zip(S, ("00", "01")):
            n.label = lab
        A.step(S, t, (256, 256))
        assert abs(np.degrees(A.roll) - r) < 1e-6


def test_intra_frame_child_labelling_and_eq9():
    g = make_tree(3, seed=4)
    A = AirwayAssociator(g)
    A.force_init("0", 0.0)
    # scope in "01" looking at its bifurcation: primary "01" box containing its two children
    g2 = g.projected_children("01", 10.0)
    kids = []
    for lab in sorted(g2):
        c = np.array([128, 128]) + 4 * g2[lab]
        kids.append([c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8])
    kids = np.array(kids)
    parent = np.r_[kids[:, :2].min(0) - 5, kids[:, 2:].max(0) + 5]
    S = build_subgraph(np.vstack([parent, kids]), np.ones(3), track_ids=[1, 2, 3], track_ages=[9, 1, 1])
    S.nodes[0].label = "01"  # inherited from its tracklet
    loc = A.step(S, 1, (256, 256))
    assert [S.nodes[i].label for i in (1, 2)] == sorted(g2)
    assert loc == "01"  # n_p = 1 -> g^{k-1}(l)


# ------------------------------------------------------------------ metrics
def test_metrics_perfect_tracking():
    gt = {f: [(1, np.array([f, 0, f + 10, 10.0])), (2, np.array([50, f, 60, f + 10.0]))] for f in range(20)}
    pr = {f: [(7, b), (9, c)] for f, ((_, b), (_, c)) in gt.items()}
    assert clear_mot(gt, pr)["MOTA"] == 100 and clear_mot(gt, pr)["IDs"] == 0
    assert idf1(gt, pr)["IDF1"] == 100
    assert abs(hota(gt, pr)["HOTA"] - 100) < 1e-6
    gtl = {f: [(i, b, str(i)) for i, b in v] for f, v in gt.items()}
    prl = {f: [(i, b, 0.9, str(1 if i == 7 else 2)) for i, b in v] for f, v in pr.items()}
    assert label_ap(gtl, prl)["mAP"] == 100


def test_metrics_id_switch():
    gt = {f: [(1, np.array([0, 0, 10, 10.0]))] for f in range(10)}
    pr = {f: [(1 if f < 5 else 2, np.array([0, 0, 10, 10.0]))] for f in range(10)}
    m = clear_mot(gt, pr)
    assert m["IDs"] == 1 and abs(m["MOTA"] - 90) < 1e-9
    assert idf1(gt, pr)["IDF1"] < 100


def test_stats():
    r = compare_accuracy([80, 82, 79, 85, 90, 77], [50, 55, 52, 49, 60, 51])
    assert r["p"] < 0.01


# ------------------------------------------------------------------ loop closure
def test_loop_closure_reassociates():
    from bronchotrack.association import GalleryRecord
    from bronchotrack.loop_closure import LoopClosure, SIFTMatcher

    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((256, 256, 3)) * 255).astype(np.uint8), (3, 3), 0)
    g = make_tree(3, seed=4)
    A = AirwayAssociator(g)
    A.force_init("0", 0.0)
    A.gallery["01"] = GalleryRecord("01", 5, 0.0, {1: np.array([60.0, 60]), 2: np.array([180.0, 70])},
                                    {}, {1: "010", 2: "011"}, 2, keyframe=img, updated=5)
    A.gallery["00"] = GalleryRecord("00", 9, 0.0, {}, {}, {}, 2, keyframe=None, updated=9)
    A.loc = "00"  # (wrongly) believes it entered a new branch
    shifted = np.roll(img, (4, 6), axis=(0, 1))
    S = build_subgraph(np.array([[50, 50, 80, 80], [172, 60, 202, 90]], float), np.ones(2), track_ids=[11, 12],
                       track_ages=[0, 0])
    res = LoopClosure(SIFTMatcher(), eta=100, lam=2).search(A, S, shifted, 10)
    assert res.detected and res.record_label == "01" and A.loc == "01"
    assert [n.label for n in S] == ["010", "011"]


# ------------------------------------------------------------------ end to end
def test_pipeline_end_to_end_synthetic():
    g = make_tree(4, seed=100)
    frames = simulate(g, ["0110", "0100"], seed=100)
    table = {i: [(*b, 0.88) for _, b, _, _ in f.gt] for i, f in enumerate(frames)}
    bt = BronchoTrack(AnnotationDetector(table), g, OracleEmbedder(frames), BronchoTrackConfig())
    hits = [bt.process(f.image).location == f.location for f in frames]
    assert np.mean(hits) > 0.6


def test_initialisation_waits_for_stable_pair():
    g = make_tree(3, seed=4)
    A = AirwayAssociator(g)  # default: 5 stable frames
    for t in range(5):
        S = build_subgraph(_carina_view(g, 0.0), np.ones(2), track_ids=[1, 2], track_ages=[t, t])
        A.step(S, t, (256, 256))
        assert A.initialized == (t == 4)


# ------------------------------------------------------------------ geometry extension
@pytest.mark.parametrize("theta_deg", [0, 15, 30, -20, 70])
def test_derotate_box_recovers_rectangle_and_ellipse(theta_deg):
    from bronchotrack.geometry import derotate_box
    w, h, th = 40.0, 20.0, np.radians(theta_deg)
    c, s = abs(np.cos(th)), abs(np.sin(th))
    rw, rh = derotate_box(w * c + h * s, w * s + h * c, th, model="rect")
    assert abs(rw - w) < 1e-6 and abs(rh - h) < 1e-6
    # ellipse with semi-axes 20, 10: axis-aligned box of the rotated ellipse
    W, H = 2 * np.sqrt(20 ** 2 * c * c + 10 ** 2 * s * s), 2 * np.sqrt(20 ** 2 * s * s + 10 ** 2 * c * c)
    ew, eh = derotate_box(W, H, th)
    assert abs(ew - 40) < 1e-6 and abs(eh - 20) < 1e-6


def test_derotate_near_45_and_circles():
    from bronchotrack.geometry import derotate_box
    rw, rh = derotate_box(30.0, 30.0, np.radians(45), model="rect")
    assert rw == rh and 20 < rw < 22  # square of side 30/sqrt(2)
    for deg in (0, 20, 45, 70):       # a round lumen's box does not change with roll
        assert np.allclose(derotate_box(30.0, 30.0, np.radians(deg)), (30.0, 30.0))


def test_ratio_model_and_fusion_prefers_geometric_fit():
    from bronchotrack.geometry import GeometricFusion, RatioModel
    g = AirwayGraph.from_json(os.path.join(os.path.dirname(__file__), "..", "results", "ModelV3", "airway_v3.json"))
    M = RatioModel(g)
    ra, rb = M.ratio("010", "011")  # bronchus intermedius vs right upper lobe: 4.5 vs 2.5 mm
    assert np.allclose(M.ratio("011", "010"), (rb, ra)) and ra > rb
    # two sibling lumens (not yet labelled) whose apparent sizes match 010 / 011
    d = 200.0
    c1, c2 = np.array([150.0, 150]), np.array([150.0 + d, 150])
    D1, D2 = ra * d, rb * d
    boxes = np.array([np.r_[c1 - D1 / 2, c1 + D1 / 2], np.r_[c2 - D2 / 2, c2 + D2 / 2]])
    S = build_subgraph(boxes, np.ones(2), track_ids=[1, 2], track_ages=[0, 0])
    res = GeometricFusion(g).fuse(S, "01", 0.0)
    assert res.labels[0][0] == "010" and res.labels[1][0] == "011"
    # the ratio alone is scale-free, so sibling pairs of other generations stay plausible
    assert res.labels[0][1] > 0.1 and res.location == "01"
    # the same boxes rolled by 30 deg: the straightened longest edge keeps the decision
    S2 = build_subgraph(boxes, np.ones(2), track_ids=[1, 2], track_ages=[0, 0])
    res2 = GeometricFusion(g).fuse(S2, "01", np.radians(30))
    assert res2.labels[0][0] == "010"
