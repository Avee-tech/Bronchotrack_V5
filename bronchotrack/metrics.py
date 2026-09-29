"""Evaluation metrics (paper Sec. IV-C).

Tracking: MOTA (Eq. 10), FP, FN, IDs (CLEAR MOT, ref [43]), IDF1 (ref [44]),
HOTA (ref [45], implemented following TrackEval).
Localisation: frame-level localisation accuracy, per-branch accuracy, signed
generation error (Fig. 6c / 7d), tree distance.
Identification: AP of detecting *and* labelling each visible lumen (area under
the precision-recall curve).
Statistics: Anderson-Darling normality test -> paired t-test or Wilcoxon.

Ground truth / predictions per frame are lists of ``(id, box_xyxy[, score, label])``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

from .kalman import iou_matrix

Frame = List[Tuple]  # (id, box, ...)


# --------------------------------------------------------------------------- #
# CLEAR MOT + IDF1
# --------------------------------------------------------------------------- #
def clear_mot(gt: Dict[int, Frame], pr: Dict[int, Frame], iou_thr: float = 0.5) -> Dict[str, float]:
    frames = sorted(set(gt) | set(pr))
    n_gt = n_pr = tp = fp = fn = ids = 0
    last_match: Dict = {}                   # gt id -> pred id (last matched)
    for f in frames:
        g, p = gt.get(f, []), pr.get(f, [])
        n_gt += len(g)
        n_pr += len(p)
        if not g or not p:
            fp += len(p)
            fn += len(g)
            continue
        iou = iou_matrix(np.array([x[1] for x in g]), np.array([x[1] for x in p]))
        cost = 1 - iou
        # keep previous correspondences when still valid (CLEAR MOT)
        gid = [x[0] for x in g]
        pid = [x[0] for x in p]
        for i, gi in enumerate(gid):
            if gi in last_match and last_match[gi] in pid:
                j = pid.index(last_match[gi])
                if iou[i, j] >= iou_thr:
                    cost[i, :] += 10
                    cost[:, j] += 10
                    cost[i, j] = -1
        cost[iou < iou_thr] = 1e6
        r, c = linear_sum_assignment(cost)
        m = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thr]
        tp += len(m)
        fp += len(p) - len(m)
        fn += len(g) - len(m)
        for i, j in m:
            if gid[i] in last_match and last_match[gid[i]] != pid[j]:
                ids += 1
            last_match[gid[i]] = pid[j]
    mota = 1 - (fn + fp + ids) / max(n_gt, 1)
    return {"MOTA": 100 * mota, "FP": fp, "FN": fn, "IDs": ids, "TP": tp, "GT": n_gt,
            "FP_rate": fp / max(n_pr, 1), "FN_rate": fn / max(n_gt, 1)}


def idf1(gt: Dict[int, Frame], pr: Dict[int, Frame], iou_thr: float = 0.5) -> Dict[str, float]:
    """Global min-cost bipartite matching of trajectories (Ristani et al.)."""
    gt_ids = sorted({x[0] for fr in gt.values() for x in fr})
    pr_ids = sorted({x[0] for fr in pr.values() for x in fr})
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: i for i, p in enumerate(pr_ids)}
    overlap = np.zeros((len(gt_ids), len(pr_ids)))
    g_len = np.zeros(len(gt_ids))
    p_len = np.zeros(len(pr_ids))
    for f in set(gt) | set(pr):
        g, p = gt.get(f, []), pr.get(f, [])
        for x in g:
            g_len[gi[x[0]]] += 1
        for x in p:
            p_len[pi[x[0]]] += 1
        if g and p:
            iou = iou_matrix(np.array([x[1] for x in g]), np.array([x[1] for x in p]))
            for a, b in zip(*np.where(iou >= iou_thr)):
                overlap[gi[g[a][0]], pi[p[b][0]]] += 1
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0}
    # cost = FN + FP for pairing trajectories
    cost = g_len[:, None] + p_len[None, :] - 2 * overlap
    r, c = linear_sum_assignment(cost - g_len[:, None] - p_len[None, :])
    idtp = overlap[r, c].sum()
    idfn, idfp = g_len.sum() - idtp, p_len.sum() - idtp
    return {"IDF1": 100 * 2 * idtp / max(2 * idtp + idfp + idfn, 1),
            "IDP": 100 * idtp / max(idtp + idfp, 1), "IDR": 100 * idtp / max(idtp + idfn, 1)}


# --------------------------------------------------------------------------- #
# HOTA (Luiten et al. 2021, TrackEval implementation)
# --------------------------------------------------------------------------- #
def hota(gt: Dict[int, Frame], pr: Dict[int, Frame]) -> Dict[str, float]:
    alphas = np.arange(0.05, 0.99, 0.05)
    frames = sorted(set(gt) | set(pr))
    gt_ids = sorted({x[0] for fr in gt.values() for x in fr})
    pr_ids = sorted({x[0] for fr in pr.values() for x in fr})
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: i for i, p in enumerate(pr_ids)}
    nG, nP = len(gt_ids), len(pr_ids)
    if nG == 0 or nP == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "LocA": 0.0}
    pot = np.zeros((nG, nP))
    g_cnt, p_cnt = np.zeros((nG, 1)), np.zeros((1, nP))
    sims = {}
    for f in frames:
        g, p = gt.get(f, []), pr.get(f, [])
        gidx = np.array([gi[x[0]] for x in g], int)
        pidx = np.array([pi[x[0]] for x in p], int)
        if len(g) and len(p):
            s = iou_matrix(np.array([x[1] for x in g]), np.array([x[1] for x in p]))
            denom = s.sum(0, keepdims=True) + s.sum(1, keepdims=True) - s
            sim_iou = np.where(denom > np.finfo(float).eps, s / np.maximum(denom, np.finfo(float).eps), 0)
            pot[gidx[:, None], pidx[None, :]] += sim_iou
        else:
            s = np.zeros((len(g), len(p)))
        g_cnt[gidx] += 1
        p_cnt[0, pidx] += 1
        sims[f] = (gidx, pidx, s)
    glob = pot / (g_cnt + p_cnt - pot)
    A = len(alphas)
    TP, FN, FP, LOC = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    match_counts = [np.zeros((nG, nP)) for _ in alphas]
    for f in frames:
        gidx, pidx, s = sims[f]
        if len(gidx) == 0:
            FP += len(pidx)
            continue
        if len(pidx) == 0:
            FN += len(gidx)
            continue
        score = glob[gidx[:, None], pidx[None, :]] * s
        r, c = linear_sum_assignment(-score)
        for a, al in enumerate(alphas):
            ok = s[r, c] >= al - np.finfo(float).eps
            rr, cc = r[ok], c[ok]
            n = len(rr)
            TP[a] += n
            FN[a] += len(gidx) - n
            FP[a] += len(pidx) - n
            if n:
                LOC[a] += s[rr, cc].sum()
                match_counts[a][gidx[rr], pidx[cc]] += 1
    AssA = np.zeros(A)
    for a in range(A):
        mc = match_counts[a]
        ass_a = mc / np.maximum(1, g_cnt + p_cnt - mc)
        AssA[a] = (mc * ass_a).sum() / max(1, TP[a])
    DetA = TP / np.maximum(1, TP + FN + FP)
    H = np.sqrt(DetA * AssA)
    LocA = np.maximum(1e-10, LOC) / np.maximum(1e-10, TP)
    return {"HOTA": 100 * H.mean(), "DetA": 100 * DetA.mean(), "AssA": 100 * AssA.mean(),
            "LocA": 100 * LocA.mean()}


def tracking_metrics(gt, pr) -> Dict[str, float]:
    out = clear_mot(gt, pr)
    out.update(idf1(gt, pr))
    out.update(hota(gt, pr))
    return out


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
def localization_metrics(gt_loc: Sequence[Optional[str]], pr_loc: Sequence[Optional[str]], graph=None
                         ) -> Dict:
    pairs = [(g, p) for g, p in zip(gt_loc, pr_loc) if g is not None]
    acc = np.mean([g == p for g, p in pairs]) if pairs else 0.0
    per = defaultdict(list)
    for g, p in pairs:
        per[g].append(g == p)
    out = {"LocAcc": 100 * acc, "per_branch": {k: 100 * float(np.mean(v)) for k, v in per.items()},
           "n_frames": len(pairs)}
    if graph is not None:
        err = [graph.generation(p) - graph.generation(g) for g, p in pairs if p in graph and g in graph]
        dist = [graph.tree_distance(g, p) for g, p in pairs if p in graph and g in graph]
        out.update({"gen_error_mean": float(np.mean(err)) if err else 0.0,
                    "gen_error": err, "tree_dist_mean": float(np.mean(dist)) if dist else 0.0})
    return out


# --------------------------------------------------------------------------- #
# Label-aware AP (identifying each visible lumen as its branch)
# --------------------------------------------------------------------------- #
def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.r_[0.0, prec, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def label_ap(gt: Dict[int, List[Tuple]], pr: Dict[int, List[Tuple]], iou_thr: float = 0.5
             ) -> Dict:
    """gt frames: (id, box, label); pr frames: (id, box, score, label).
    A prediction is a TP if it overlaps (IoU>=thr) an unmatched GT box with the same label."""
    by_label_gt = defaultdict(int)
    dets = defaultdict(list)  # label -> [(score, tp)]
    for f in set(gt) | set(pr):
        g = gt.get(f, [])
        for x in g:
            by_label_gt[x[2]] += 1
        p = sorted(pr.get(f, []), key=lambda x: -x[2])
        used = set()
        for x in p:
            lab = x[3]
            if lab is None:
                continue
            cands = [(i, y) for i, y in enumerate(g) if y[2] == lab and i not in used]
            best, bi = 0.0, -1
            for i, y in cands:
                v = iou_matrix(np.array([x[1]]), np.array([y[1]]))[0, 0]
                if v > best:
                    best, bi = v, i
            tp = best >= iou_thr
            if tp:
                used.add(bi)
            dets[lab].append((x[2], tp))
    aps = {}
    for lab, n in by_label_gt.items():
        d = sorted(dets.get(lab, []), key=lambda z: -z[0])
        if not d:
            aps[lab] = 0.0
            continue
        tp = np.cumsum([z[1] for z in d])
        fp = np.cumsum([not z[1] for z in d])
        aps[lab] = voc_ap(tp / n, tp / np.maximum(tp + fp, 1e-9))
    return {"mAP": 100 * float(np.mean(list(aps.values()))) if aps else 0.0,
            "AP": {k: 100 * v for k, v in aps.items()}}


# --------------------------------------------------------------------------- #
# Statistics (Sec. IV-C)
# --------------------------------------------------------------------------- #
def compare_accuracy(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Anderson-Darling normality on the paired differences; paired t-test if
    normal (5 % level) else Wilcoxon signed-rank."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"test": "none (identical)", "p": 1.0, "normal": None}
    normal = True
    if len(d) >= 3 and np.std(d) > 0:
        try:  # SciPy >= 1.17: p-value interface
            ad = stats.anderson(d, dist="norm", method="interpolate")
            normal = ad.pvalue > 0.05
        except TypeError:
            ad = stats.anderson(d, dist="norm")
            crit = ad.critical_values[list(ad.significance_level).index(5.0)]
            normal = ad.statistic < crit
    if normal:
        r = stats.ttest_rel(a, b)
        return {"test": "paired t-test", "p": float(r.pvalue), "normal": True}
    r = stats.wilcoxon(a, b)
    return {"test": "wilcoxon", "p": float(r.pvalue), "normal": False}
