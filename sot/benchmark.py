"""The paper's two-map method (reconstruct m from the field-along-z map, then fit
the tilted map), used as the benchmark for the networks."""
from __future__ import annotations

import math
import numpy as np
from scipy.signal import convolve2d
from scipy.special import j1

from sot.common import EPS, Struct, cosd, mround, sind, std1
from sot.data.physics import kgrid


def two_map_method(S, h, d, phi):
    """The paper's procedure: reconstruct m from the field-along-z map, tilt it towards the
    KNOWN in-plane direction phi, fit Bz_tilt ~ a*F_z + b*F_inplane + c0 + cx*x + cy*y."""
    dx = S.x[1] - S.x[0]
    supp = support_from_mask(S.mask_input, dx)
    m = reconstruct_mz(np.asarray(S.Bz_zero, float), supp, dx, h, d)
    F1, F2 = fields_from_m(m, dx, h, d, phi)
    ny, nx = np.shape(S.Bz_tilt)
    XX, YY = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    A = np.column_stack([F1.ravel('F'), F2.ravel('F'), np.ones(ny * nx), XX.ravel('F'), YY.ravel('F')])
    c = np.linalg.lstsq(A, np.asarray(S.Bz_tilt, float).ravel('F'), rcond=None)[0]
    th = math.atan2(c[1], c[0])
    return np.array([math.sin(th) * cosd(phi), math.sin(th) * sind(phi), math.cos(th)])


def support_from_mask(mask, dx):
    """where m may be non-zero: the known mask, widened by 150 nm for its misregistration"""
    r = max(1, int(mround(0.15 / dx)))
    u, w = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    return convolve2d((np.asarray(mask) > 0.3).astype(float), (u ** 2 + w ** 2 <= r ** 2).astype(float),
                      mode='same') > 0.5


def map_kernels(Ny, Nx, dx, h, d, phi):
    """Fourier kernels (normalized to max 1) for a moment along z and in-plane along phi"""
    KX, KY = kgrid(Ny, Nx, dx)
    K = np.sqrt(KX ** 2 + KY ** 2)
    T = np.ones_like(K)
    q = K * d / 2
    nz = q > 0
    T[nz] = 2 * j1(q[nz]) / q[nz]
    E = np.exp(-K * h) * T
    Kz = E * K
    sc = Kz.max()
    Kz = Kz / sc
    Kp = E * (-1j) * (KX * cosd(phi) + KY * sind(phi)) / sc
    return Kz, Kp


def reconstruct_mz(B, supp, dx, h, d):
    """Regularized least squares with CGLS:
    min || field(m) + offset + plane - B ||^2 + lam*||grad m||^2,  m = 0 outside supp.
    Vectors are in MATLAB column-major order."""
    ny, nx = B.shape
    C = Struct(ny=ny, nx=nx, Ny=2 * ny, Nx=2 * nx)
    C.ry = slice(ny // 2, ny // 2 + ny)
    C.rx = slice(nx // 2, nx // 2 + nx)
    C.Kz = map_kernels(C.Ny, C.Nx, dx, h, d, 0)[0]
    C.sidx = np.flatnonzero(supp.ravel('F'))
    C.ns = C.sidx.size
    XX, YY = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    C.Bk = np.column_stack([np.ones(ny * nx), XX.ravel('F'), YY.ravel('F')]) / math.sqrt(ny * nx)
    C.sl = math.sqrt(0.03)
    B = (B - np.median(B)) / max(std1(B), EPS)
    xv = np.zeros(C.ns + 3)
    r = np.concatenate([B.ravel('F'), np.zeros(ny * (nx - 1) + (ny - 1) * nx)])
    s = cgls_AT(r, C)
    p = s.copy()
    gam = s @ s
    for it in range(150):
        q = cgls_A(p, C)
        al = gam / max(q @ q, EPS)
        xv = xv + al * p
        r = r - al * q
        s = cgls_AT(r, C)
        gn = s @ s
        p = s + (gn / max(gam, EPS)) * p
        gam = gn
    m = np.zeros(ny * nx)
    m[C.sidx] = xv[:C.ns]
    return m.reshape((ny, nx), order='F')


def cgls_A(xv, C):
    """forward operator: [field of m + offset + plane ; smoothness terms]"""
    Mf = np.zeros(C.ny * C.nx)
    Mf[C.sidx] = xv[:C.ns]
    Mf = Mf.reshape((C.ny, C.nx), order='F')
    Mp = np.zeros((C.Ny, C.Nx))
    Mp[C.ry, C.rx] = Mf
    F = np.real(np.fft.ifft2(C.Kz * np.fft.fft2(Mp)))
    dd = F[C.ry, C.rx] + (C.Bk @ xv[C.ns:]).reshape((C.ny, C.nx), order='F')
    gx = Mf[:, 1:] - Mf[:, :-1]
    gy = Mf[1:, :] - Mf[:-1, :]
    return np.concatenate([dd.ravel('F'), C.sl * gx.ravel('F'), C.sl * gy.ravel('F')])


def cgls_AT(r, C):
    """transpose of cgls_A"""
    n1 = C.ny * C.nx
    n2 = C.ny * (C.nx - 1)
    Rd = r[:n1].reshape((C.ny, C.nx), order='F')
    Gx = r[n1:n1 + n2].reshape((C.ny, C.nx - 1), order='F')
    Gy = r[n1 + n2:].reshape((C.ny - 1, C.nx), order='F')
    Rp = np.zeros((C.Ny, C.Nx))
    Rp[C.ry, C.rx] = Rd
    gm = np.real(np.fft.ifft2(C.Kz * np.fft.fft2(Rp)))[C.ry, C.rx]
    Ar = np.zeros((C.ny, C.nx))
    Ar[:, :-1] -= Gx
    Ar[:, 1:] += Gx
    Ar[:-1, :] -= Gy
    Ar[1:, :] += Gy
    gm = gm + C.sl * Ar
    return np.concatenate([gm.ravel('F')[C.sidx], C.Bk.T @ Rd.ravel('F')])


def fields_from_m(m, dx, h, d, phi):
    """field of m pointing along z (F1) and in-plane along phi (F2), same normalization"""
    ny, nx = m.shape
    Ny, Nx = 2 * ny, 2 * nx
    ry = slice(ny // 2, ny // 2 + ny)
    rx = slice(nx // 2, nx // 2 + nx)
    Kz, Kp = map_kernels(Ny, Nx, dx, h, d, phi)
    Mp = np.zeros((Ny, Nx))
    Mp[ry, rx] = m
    Mk = np.fft.fft2(Mp)
    return np.real(np.fft.ifft2(Kz * Mk))[ry, rx], np.real(np.fft.ifft2(Kp * Mk))[ry, rx]

