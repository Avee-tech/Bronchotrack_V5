"""Pseudo ground truth for virtual-bronchoscopy videos: register every frame to
renders of the same mesh along the airway centre line, then decode a smooth path
through the tree with Viterbi. Output: frame,location,branch_s,score."""
import sys, csv, numpy as np, cv2
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from render_lib import Renderer
from bronchotrack.airway_graph import AirwayGraph

def smooth(p, k=7):
    if len(p) < k: return p
    ker = np.ones(k) / k
    q = np.stack([np.convolve(np.pad(p[:, i], (k//2, k//2), mode='edge'), ker, 'valid') for i in range(3)], 1)
    return q

def samples(g, step=1.0):
    S = []  # (label, s, pos_std, dir_std)
    for lab, b in g.branches.items():
        pts = smooth(b.points)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]; L = cum[-1]
        for s in np.arange(0, L + 1e-6, step):
            p = np.array([np.interp(s, cum, pts[:, i]) for i in range(3)])
            q = np.array([np.interp(min(s + 6, L), cum, pts[:, i]) for i in range(3)])
            if np.linalg.norm(q - p) < 1e-3: q = pts[-1] + b.direction
            d = (q - p) / (np.linalg.norm(q - p) + 1e-9)
            S.append((lab, s, p, d))
    return S

SZ = 64
yy, xx = np.mgrid[0:SZ, 0:SZ]; MASK = ((xx - SZ/2 + .5)**2 + (yy - SZ/2 + .5)**2) <= (SZ/2 - 1)**2
def feat(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(cv2.resize(g, (SZ, SZ), interpolation=cv2.INTER_AREA), (5, 5), 0)
    v = np.log1p(g[MASK]); v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-9)

def fov_circle(video):
    """Centre and diameter of the circular endoscope field (saturated pixels over several frames)."""
    c = cv2.VideoCapture(video); n = int(c.get(7)); acc = None
    for i in np.linspace(0, n - 1, 12).astype(int):
        c.set(1, i); ok, f = c.read()
        if not ok: continue
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); m = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 10)
        acc = m.astype(int) if acc is None else acc + m
    ys, xs = np.where(acc >= 3)
    return (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, xs.max() - xs.min() + 1

def crop_circle(fr, cx, cy, d, out=160):
    """Square crop of side d centred on the field of view (zero padding outside the frame)."""
    M = np.float32([[out / d, 0, out / 2 - cx * out / d], [0, out / d, out / 2 - cy * out / d]])
    return cv2.warpAffine(fr, M, (out, out), borderValue=(0, 0, 0))

def main(mesh, graph, video, out, fov=60, step=1.0, nrot=24, max_move=2.5, lam=0.02):
    g = AirwayGraph.from_json(graph)
    S = samples(g, step)
    R = Renderer(mesh, 160, fov, 0.002)
    up_ct = np.array([0, -1, 0.])  # anterior in LPS
    F = []
    for lab, s, p, d in S:
        pc = g.to_ct(p); dc = g.to_ct(p + d) - pc
        up = up_ct - np.dot(up_ct, dc) * dc
        if np.linalg.norm(up) < 1e-3: up = np.array([0, 0, 1.])
        F.append(feat(R.render(pc, dc, up / np.linalg.norm(up))))
    F = np.stack(F)                                         # (N, D)
    # video features at nrot in-plane rotations (camera roll)
    cx, cy, dia = fov_circle(video)
    cap = cv2.VideoCapture(video); V = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        c = crop_circle(fr, cx, cy, dia)
        rots = [feat(cv2.warpAffine(c, cv2.getRotationMatrix2D((80, 80), 360 * k / nrot, 1), (160, 160))) for k in range(nrot)]
        V.append(np.stack(rots))
    V = np.stack(V)                                         # (T, nrot, D)
    sim3 = np.einsum('trd,nd->tnr', V, F)                   # (T, N, R) similarity per position and roll
    # sample graph (along branches + across bifurcations) -> geodesic distances
    N = len(S); A = lil_matrix((N, N)); idx = {}
    for i, (lab, s, p, d) in enumerate(S): idx.setdefault(lab, []).append(i)
    for lab, ii in idx.items():
        for a, b in zip(ii[:-1], ii[1:]): A[a, b] = A[b, a] = step
        par = g[lab].parent
        if par is not None:
            j = idx[par][-1]; A[ii[0], j] = A[j, ii[0]] = np.linalg.norm(S[ii[0]][2] - S[j][2]) + 1e-3
    D = shortest_path(A.tocsr(), directed=False)
    trans = np.where(D <= max_move, -lam * D, -np.inf)       # log-transition over position
    # Viterbi over (position, roll): position moves <= max_move mm, roll <= 1 step per frame
    T, _, Rn = sim3.shape; ll = sim3 * 20
    dp = ll[0].copy()
    bad = [i for i, x in enumerate(S) if x[0] != g.root or x[1] > 25]
    dp[bad] = -np.inf                                       # start in the upper trachea
    bpos = np.zeros((T, N, Rn), np.int32); brot = np.zeros((T, N, Rn), np.int8)
    for t in range(1, T):
        M = np.empty((N, Rn)); Mi = np.empty((N, Rn), np.int32)
        for r in range(Rn):
            cand = dp[:, r][:, None] + trans              # (from, to)
            Mi[:, r] = cand.argmax(0); M[:, r] = cand.max(0)
        best = np.full((N, Rn), -np.inf); bp = np.zeros((N, Rn), np.int32); br = np.zeros((N, Rn), np.int8)
        for dr in (-1, 0, 1):
            Ms = np.roll(M, dr, axis=1) - 0.05 * abs(dr)    # from roll r-dr to r
            Mis = np.roll(Mi, dr, axis=1)
            better = Ms > best
            best = np.where(better, Ms, best); bp = np.where(better, Mis, bp); br = np.where(better, dr, br)
        bpos[t], brot[t] = bp, br
        dp = best + ll[t]
    j, r = np.unravel_index(int(dp.argmax()), dp.shape)
    path = [(j, r)]
    for t in range(T - 1, 0, -1):
        pj, dr = bpos[t, j, r], brot[t, j, r]
        j, r = pj, (r - dr) % Rn
        path.append((j, r))
    path = path[::-1]
    sim = sim3.max(2)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['frame', 'location', 'branch_s', 'roll_deg', 'score'])
        for t, (i, r) in enumerate(path):
            w.writerow([t + 1, S[i][0], f"{S[i][1]:.1f}", 360 * r / Rn, f"{sim3[t, i, r]:.3f}"])
    print(out, 'frames', T, 'samples', N, 'mean sim on path', np.mean([sim3[t, i, r] for t, (i, r) in enumerate(path)]).round(3),
          'mean best sim', sim.max(1).mean().round(3))

if __name__ == '__main__':
    main(*sys.argv[1:5])
