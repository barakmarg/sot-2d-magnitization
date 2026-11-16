"""Inference: batched prediction, test-time augmentation, printed examples."""
from __future__ import annotations

import numpy as np

from sot.model.augment import apply_sym
from sot.model.loss import angle_err
from sot.model.network import _torch


def predict_dirs(net, X, bs=64):
    """predicted unit vectors (3, N) for inputs X (N, C, n, n), in batches"""
    torch = _torch()
    dev = next(net.parameters()).device
    net.eval()
    N = X.shape[0]
    Yp = np.zeros((3, N), np.float32)
    with torch.no_grad():
        for s in range(0, N, bs):
            Xb = torch.from_numpy(np.ascontiguousarray(X[s:s + bs], np.float32)).to(dev)
            Yp[:, s:s + bs] = net(Xb).cpu().numpy().T
    return Yp / np.sqrt((Yp ** 2).sum(axis=0))


def predict_dirs_tta(net, X, bs=64):
    """prediction averaged over the 8 symmetric copies of each map (4 rotations x mirror)"""
    N = X.shape[0]
    Yp = np.zeros((3, N), np.float32)
    for k in range(4):
        for mir in (0, 1):
            Xt, _ = apply_sym(X, np.zeros((3, N), np.float32), k, mir)
            Q = predict_dirs(net, Xt, bs)
            if mir:
                Q[0] = -Q[0]                              # undo the mirror
            for r in range(k):
                Q = np.stack([Q[1], -Q[0], Q[2]])         # undo the rotations
            Yp = Yp + Q
    return Yp / np.sqrt((Yp ** 2).sum(axis=0))


def show_examples(net, X, Y, k):
    Yp = predict_dirs(net, X[:k])
    for i in range(k):
        tt = np.degrees(np.arccos(Y[2, i]))
        pt = np.degrees(np.arctan2(Y[1, i], Y[0, i]))
        tp = np.degrees(np.arccos(Yp[2, i]))
        pp = np.degrees(np.arctan2(Yp[1, i], Yp[0, i]))
        print(f'example {i + 1}: true theta {tt:5.1f} phi {pt:6.1f} | predicted theta {tp:5.1f} phi {pp:6.1f} '
              f'| error {angle_err(Yp[:, i:i + 1], Y[:, i:i + 1])[0]:.1f} deg')

