"""Symmetry augmentation (90 deg rotations, mirror) and the known-phi input channels."""
from __future__ import annotations

import numpy as np

from sot.common import rng


def apply_sym(X, Y, k, mir):
    """rotate the whole sample by k*90 deg, then mirror x if mir. X: (N, C, n, n), Y: (3, N).
    The moment Y and the known-direction channels (index 6 = cos phi, 7 = sin phi) transform with it."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    for r in range(k):
        X = np.flip(np.swapaxes(X, 2, 3), 3)          # rotate the maps by +90 deg
        Y = np.stack([-Y[1], Y[0], Y[2]])             # (mx,my) -> (-my,mx)
        if X.shape[1] >= 8:
            X = X.copy()
            c7, c8 = X[:, 6].copy(), X[:, 7].copy()
            X[:, 6], X[:, 7] = -c8, c7
    if mir:
        X = np.flip(X, 3)                             # x -> -x, mx -> -mx
        Y = Y.copy()
        Y[0] = -Y[0]
        if X.shape[1] >= 8:
            X = X.copy()
            X[:, 6] = -X[:, 6]
    return np.ascontiguousarray(X), Y


def augment(X, Y):
    """random symmetry of the physics for a training batch"""
    return apply_sym(X, Y, rng.randi([0, 3]), rng.rand() < 0.5)


def add_phi_channels(X, Y):
    """the in-plane direction of the moment as two constant channels (cos phi, sin phi)"""
    ph = np.arctan2(Y[1], Y[0])
    N, C, n, m = X.shape
    c = np.broadcast_to(np.cos(ph)[:, None, None, None], (N, 1, n, m))
    s = np.broadcast_to(np.sin(ph)[:, None, None, None], (N, 1, n, m))
    return np.concatenate([X, c, s], axis=1).astype(np.float32)

