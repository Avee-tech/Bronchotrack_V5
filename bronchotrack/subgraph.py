"""Lumen subgraph S^t (paper Sec. III-C and Sec. IV-A, Fig. 3).

Lumen boxes follow the annotation rule of Fig. 3: a child (or independent)
lumen's box bounds its entry, a parent lumen's box is the minimum area enclosing
all its child lumens. Hence the observed hierarchy is recovered from box
*inclusion*: a box's parent is the smallest box that (nearly) contains it.

"We construct a local airway subgraph based on the overlap of detection boxes,
pruning redundant parent branches and isolated child branches with large IoU
with their parents."  (Sec. IV-A)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .kalman import iou_matrix


@dataclass(eq=False)
class LumenNode:
    det: int                       # detection index in the frame
    box: np.ndarray
    score: float
    parent: Optional[int] = None   # det index of the parent lumen
    children: List[int] = field(default_factory=list)
    parent_candidates: List[int] = field(default_factory=list)  # all containing boxes, smallest first
    level: int = 1                 # 1 = primary lumen
    track_id: Optional[int] = None
    track_age: int = -1
    label: Optional[str] = None
    label_age: int = -1            # frames the lumen had kept its previous label (-1: unknown)
    prev_label: Optional[str] = None  # the lumen's label before this frame's association

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))


class LumenSubgraph:
    def __init__(self, nodes: Dict[int, LumenNode]):
        self.nodes = nodes

    def reparent(self, d: int, p: Optional[int]):
        n = self.nodes[d]
        if n.parent is not None and n.parent in self.nodes:
            self.nodes[n.parent].children.remove(d)
        n.parent = p
        if p is not None:
            self.nodes[p].children.append(d)
        self.relevel()

    def relevel(self):
        for n in self.nodes.values():
            k, cur = 1, n.parent
            while cur is not None:
                k, cur = k + 1, self.nodes[cur].parent
            n.level = k

    def roots(self) -> List[LumenNode]:
        return [n for n in self.nodes.values() if n.parent is None]

    def primary(self) -> List[LumenNode]:
        return [n for n in self.nodes.values() if n.level == 1]

    def siblings(self, d: int) -> List[LumenNode]:
        n = self.nodes[d]
        if n.parent is None:
            return [m for m in self.roots() if m.det != d]
        return [self.nodes[c] for c in self.nodes[n.parent].children if c != d]

    def children(self, d: int) -> List[LumenNode]:
        return [self.nodes[c] for c in self.nodes[d].children]

    def __iter__(self):
        return iter(self.nodes.values())

    def __len__(self):
        return len(self.nodes)


def containment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,m) fraction of box a_i's area inside box b_j."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / (aa[:, None] + 1e-9)


def build_subgraph(boxes: np.ndarray, scores: np.ndarray, contain_thr: float = 0.8,
                   prune_iou: float = 0.6, track_ids: Optional[List[Optional[int]]] = None,
                   track_ages: Optional[List[int]] = None, dets: Optional[List[int]] = None) -> LumenSubgraph:
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    n = len(boxes)
    dets = list(range(n)) if dets is None else list(dets)
    nodes = {dets[i]: LumenNode(dets[i], boxes[i], float(scores[i]),
                                track_id=None if track_ids is None else track_ids[i],
                                track_age=-1 if track_ages is None else track_ages[i]) for i in range(n)}
    if n == 0:
        return LumenSubgraph(nodes)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    C = containment(boxes, boxes)
    for i in range(n):
        cands = [j for j in range(n) if j != i and C[i, j] >= contain_thr and area[j] > area[i]]
        if cands:
            cands.sort(key=lambda j: area[j])
            nodes[dets[i]].parent = dets[cands[0]]
            nodes[dets[i]].parent_candidates = [dets[j] for j in cands]
    for nd in nodes.values():
        if nd.parent is not None:
            nodes[nd.parent].children.append(nd.det)

    # ---- pruning: a parent with a single child of large IoU is a duplicate lumen
    iou = iou_matrix(boxes, boxes)
    pos = {d: i for i, d in enumerate(dets)}
    changed = True
    while changed:
        changed = False
        for nd in list(nodes.values()):
            if len(nd.children) != 1:
                continue
            ch = nodes[nd.children[0]]
            if iou[pos[nd.det], pos[ch.det]] < prune_iou:
                continue
            # keep the more confident (older, if tracked) of the two boxes
            keep_parent = (nd.track_age, nd.score) >= (ch.track_age, ch.score)
            drop, keep = (ch, nd) if keep_parent else (nd, ch)
            if keep_parent:  # re-attach the child's children to the parent
                nd.children = list(ch.children)
                for c in ch.children:
                    nodes[c].parent = nd.det
            else:            # the child takes the parent's place
                ch.parent = nd.parent
                if nd.parent is not None:
                    pc = nodes[nd.parent].children
                    pc[pc.index(nd.det)] = ch.det
            del nodes[drop.det]
            for m in nodes.values():
                m.parent_candidates = [c for c in m.parent_candidates if c != drop.det]
            changed = True
            break

    # ---- levels
    def depth(d):
        k, cur = 1, nodes[d].parent
        while cur is not None:
            k, cur = k + 1, nodes[cur].parent
        return k

    for d in nodes:
        nodes[d].level = depth(d)
    return LumenSubgraph(nodes)
