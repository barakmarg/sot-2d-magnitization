"""Loss and error metrics between predicted and true unit vectors."""
from __future__ import annotations

import numpy as np


def model_loss(net, X, Y, W):
    """3D unit-vector loss with the phi part weighted by W (X: N,C,H,W ; Y: N,3 ; W: N)"""
    Yp = net(X)
    Yp = Yp / (Yp.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
    return dir_loss(Yp.T, Y.T, W).sum() / Y.shape[0]


def dir_loss(Yp, Y, W):
    """per-sample loss between unit vectors Yp and Y (3, N); works for numpy and torch.
    The squared distance splits exactly into a theta part and a phi part:
      |Yp - Y|^2 = 2(1 - cos dtheta) + 2 sin(theta_p) sin(theta_t) (1 - cos dphi)
    The phi part (already zero at the poles) is multiplied by W; W = 1 is the plain 3D loss."""
    rp = (Yp[0] ** 2 + Yp[1] ** 2 + 1e-12) ** 0.5       # sin(theta) of the prediction
    rt = (Y[0] ** 2 + Y[1] ** 2) ** 0.5                 # sin(theta) of the truth
    Lth = 1 - Yp[2] * Y[2] - rp * rt                    # 1 - cos(dtheta)
    Lphi = rp * rt - Yp[0] * Y[0] - Yp[1] * Y[1]        # sin sin (1 - cos dphi)
    return 2 * (Lth + W * Lphi)


def phi_weight(Y, T):
    """weight of the phi error vs the TRUE polar angle: W = 1 - exp(-(|theta - c|/w)^p),
    T.phi_supp = [c, w, p]. Default [90, 5, 4]: W = 0 at 90 deg, 0.03 at 88, 0.63 at 85,
    0.98 at 83, 1 below 80 deg. w = 0 switches the suppression off (plain 3D loss)."""
    c, w, p = T.phi_supp
    if w <= 0:
        return np.ones(Y.shape[1], np.float32)
    th = np.degrees(np.arccos(np.clip(Y[2], -1, 1)))
    return (1 - np.exp(-(np.abs(th - c) / w) ** p)).astype(np.float32)


def loss_angle(Yp, Y, W):
    """the angle (deg) whose plain 3D loss equals the weighted loss: the error the training sees"""
    return np.degrees(np.arccos(np.clip(1 - dir_loss(np.asarray(Yp, float), np.asarray(Y, float), W) / 2, -1, 1)))


def angle_err(Yp, Y):
    """angle (deg) between predicted and true directions"""
    return np.degrees(np.arccos(np.clip((np.asarray(Yp) * np.asarray(Y)).sum(axis=0), -1, 1)))

