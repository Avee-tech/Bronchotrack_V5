"""Constant-velocity Kalman filter in the image plane (paper Sec. III-B-1).

State (7-D, exactly as in the paper):
    x = [xc, yc, h, a, dxc, dyc, dh]
where (xc, yc) is the box centre, h its height and a = w/h its aspect ratio;
(dxc, dyc) and dh are the velocities of the centre and of the height. The
aspect ratio is modelled as constant. Measurements are z = [xc, yc, h, a].

Noise is scaled with the box height as in DeepSORT/ByteTrack (refs [32],[34]).
"""

from __future__ import annotations

import numpy as np

# chi-square 0.95 quantiles (for optional Mahalanobis gating)
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


class KalmanFilter7:
    ndim_z = 4
    ndim_x = 7

    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160,
                 std_aspect: float = 1e-2, std_aspect_process: float = 1e-5, dt: float = 1.0):
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = dt
        self.F = F
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.wp, self.wv = std_weight_position, std_weight_velocity
        self.sa, self.sap = std_aspect, std_aspect_process

    # ------------------------------------------------------------------
    def initiate(self, z: np.ndarray):
        mean = np.r_[z, 0.0, 0.0, 0.0]
        h = z[2]
        std = [2 * self.wp * h, 2 * self.wp * h, 2 * self.wp * h, self.sa,
               10 * self.wv * h, 10 * self.wv * h, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[2]
        std = [self.wp * h, self.wp * h, self.wp * h, self.sap,
               self.wv * h, self.wv * h, self.wv * h]
        Q = np.diag(np.square(std))
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        return mean, cov

    def project(self, mean, cov):
        h = mean[2]
        R = np.diag(np.square([self.wp * h, self.wp * h, self.wp * h, self.sa]))
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(self, mean, cov, z):
        pm, pc = self.project(mean, cov)
        K = np.linalg.solve(pc, (cov @ self.H.T).T).T
        mean = mean + K @ (z - pm)
        cov = cov - K @ pc @ K.T
        return mean, cov

    def gating_distance(self, mean, cov, zs):
        pm, pc = self.project(mean, cov)
        L = np.linalg.cholesky(pc)
        d = (zs - pm).T
        y = np.linalg.solve(L, d)
        return (y * y).sum(0)


# ---------------------------------------------------------------------- helpers
def xyxy_to_xyha(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, h, w / np.maximum(h, 1e-6)], -1)


def xyha_to_xyxy(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    h = s[..., 2]
    w = s[..., 3] * h
    return np.stack([s[..., 0] - w / 2, s[..., 1] - h / 2, s[..., 0] + w / 2, s[..., 1] + h / 2], -1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)
