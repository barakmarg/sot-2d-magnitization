"""Shared building blocks: MATLAB-compatible random numbers (one global `rng`),
MATLAB-like math helpers, the Struct container and direction vectors."""
from __future__ import annotations

import numpy as np
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator


EPS = np.finfo(float).eps


class RNG:
    """Random draws with MATLAB semantics (rand, randn, randi, randperm).

    Normal mode uses numpy.  Replay mode reads draws recorded from MATLAB/Octave
    (one stream per function, values in column-major order), so the simulator
    produces exactly the same sample as the .m code."""

    def __init__(self, seed=None):
        self.g = np.random.default_rng(seed)
        self.streams = None

    def seed(self, seed):
        self.g = np.random.default_rng(seed)
        self.streams = None

    def replay(self, streams):
        """streams: dict name -> 1D array, for 'rand', 'randn', 'randi', 'randperm'."""
        self.streams = {k: np.asarray(v, float).ravel() for k, v in streams.items()}
        self.pos = {k: 0 for k in self.streams}

    def _take(self, name, shape):
        n = int(np.prod(shape)) if shape else 1
        s = self.streams[name]
        p = self.pos[name]
        if p + n > s.size:
            raise RuntimeError(f'replay stream {name!r} exhausted')
        self.pos[name] = p + n
        v = s[p:p + n]
        return float(v[0]) if not shape else v.reshape(shape, order='F').copy()

    def rand(self, *shape):
        if self.streams is not None:
            return self._take('rand', shape)
        return self.g.random() if not shape else self.g.random(shape)

    def randn(self, *shape):
        if self.streams is not None:
            return self._take('randn', shape)
        return self.g.standard_normal() if not shape else self.g.standard_normal(shape)

    def randi(self, r, *shape):
        """MATLAB randi: r = imax -> 1..imax, r = [imin imax] -> imin..imax."""
        r = np.atleast_1d(r)
        lo, hi = (1, int(r[0])) if r.size == 1 else (int(r[0]), int(r[1]))
        if self.streams is not None:
            v = self._take('randi', shape)
            return int(v) if not shape else v.astype(int)
        return int(self.g.integers(lo, hi + 1)) if not shape else self.g.integers(lo, hi + 1, shape)

    def randperm(self, n):
        """random permutation of 1..n (MATLAB, 1-based)"""
        if self.streams is not None:
            return self._take('randperm', (n,)).astype(int)
        return self.g.permutation(n) + 1


rng = RNG(0)


def mround(x):
    """MATLAB round: halves away from zero"""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def sind(x):
    x = np.asarray(x, float)
    y = np.sin(np.deg2rad(np.mod(x, 360.0)))
    y = np.where(np.mod(x, 180.0) == 0, 0.0, y)
    return y if y.ndim else float(y)


def cosd(x):
    x = np.asarray(x, float)
    y = np.cos(np.deg2rad(np.mod(x, 360.0)))
    y = np.where(np.mod(x - 90.0, 180.0) == 0, 0.0, y)
    return y if y.ndim else float(y)


def std1(a):
    """MATLAB std (normalized by N-1) over all elements"""
    return float(np.std(np.asarray(a, float), ddof=1))


def inpolygon(xq, yq, xv, yv):
    """points inside (or on the edge of) the polygon (xv, yv), like MATLAB inpolygon"""
    xq = np.asarray(xq, float)
    yq = np.asarray(yq, float)
    xv = np.asarray(xv, float).ravel()
    yv = np.asarray(yv, float).ravel()
    inside = np.zeros(xq.shape, bool)
    on = np.zeros(xq.shape, bool)
    n = xv.size
    for i in range(n):
        x1, y1 = xv[i], yv[i]
        x2, y2 = xv[(i + 1) % n], yv[(i + 1) % n]
        crosses = (y1 > yq) != (y2 > yq)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (x2 - x1) * (yq - y1) / (y2 - y1) + x1
        inside ^= crosses & (xq < xi)
        # on the edge
        cr = (x2 - x1) * (yq - y1) - (y2 - y1) * (xq - x1)
        on |= (np.abs(cr) <= 1e-12 * max(abs(x2 - x1) + abs(y2 - y1), 1e-300)) & \
              (xq >= min(x1, x2)) & (xq <= max(x1, x2)) & (yq >= min(y1, y2)) & (yq <= max(y1, y2))
    return inside | on


def interp2(x, y, Z, xq, yq, fill=np.nan):
    """MATLAB interp2(x, y, Z, xq, yq, 'linear', fill)"""
    f = RegularGridInterpolator((np.asarray(y, float), np.asarray(x, float)), np.asarray(Z, float),
                                method='linear', bounds_error=False, fill_value=fill)
    xq = np.asarray(xq, float)
    pts = np.stack([np.asarray(yq, float).ravel(), xq.ravel()], axis=-1)
    return f(pts).reshape(xq.shape)


class Struct(dict):
    """dict with attribute access, to read like the MATLAB structs"""
    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v

    def __deepcopy__(self, memo):
        return Struct({k: deepcopy(v, memo) for k, v in self.items()})


def dir_vec(theta, phi):
    """unit vector of polar angle theta (from +z) and azimuth phi (from +x), degrees"""
    return np.array([sind(theta) * cosd(phi), sind(theta) * sind(phi), cosd(theta)])

