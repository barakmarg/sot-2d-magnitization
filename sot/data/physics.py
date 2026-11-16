"""Forward model: stray field Bz at the SQUID-on-tip from a magnetization map."""
from __future__ import annotations

import numpy as np
from scipy.special import j1


def sot_forward(mx, my, mz, dx, h, d_tip, t=0.0):
    """Out-of-plane stray field of a magnetic film, as seen by a SOT (nT).
    Bz(k) = (mu0/2) exp(-k h) [k mz - i kx mx - i ky my] (1-exp(-k t))/(k t) * 2 J1(k d/2)/(k d/2)
    The grid is treated as periodic, so the caller must zero-pad."""
    mu0 = 4 * np.pi * 1e-7
    muB = 9.2740100783e-24
    conv = muB / 1e-4
    ny, nx = np.shape(mz)
    KX, KY = kgrid(ny, nx, dx)
    K = np.sqrt(KX ** 2 + KY ** 2)
    B = np.exp(-K * h) * (K * np.fft.fft2(mz) - 1j * KX * np.fft.fft2(mx) - 1j * KY * np.fft.fft2(my))
    if t > 0:
        q = K * t
        F = np.ones_like(q)
        nz = q > 0
        F[nz] = (1 - np.exp(-q[nz])) / q[nz]
        B = B * F
    if d_tip > 0:
        q = K * d_tip / 2
        T = np.ones_like(q)
        nz = q > 0
        T[nz] = 2 * j1(q[nz]) / q[nz]
        B = B * T
    return np.real(np.fft.ifft2(B)) * (mu0 / 2) * conv * 1e6 * 1e9


def kgrid(ny, nx, dx):
    """angular wavevector grids (1/um) matching fft2 ordering"""
    kx = 2 * np.pi * np.fft.ifftshift(np.arange(nx) - nx // 2) / (nx * dx)
    ky = 2 * np.pi * np.fft.ifftshift(np.arange(ny) - ny // 2) / (ny * dx)
    return np.meshgrid(kx, ky)

