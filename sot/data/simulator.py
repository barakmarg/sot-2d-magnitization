"""One random synthetic sample (outline, bubbles, magnetization, stray field,
noise, background), plus a preview figure and physics checks."""
from __future__ import annotations

import math
import numpy as np

from sot.common import EPS, Struct, cosd, inpolygon, interp2, mround, rng, sind, std1
from sot.data.physics import kgrid, sot_forward


def sot_make_sample(P):
    """One random synthetic sample (see the MATLAB function of the same name)."""
    # ---------- 1. outline, the scan frame, and the pixel size ----------
    partial = rng.rand() < P.p_partial
    fov_min = np.array(P.fov_min, float)
    fov_max = np.array(P.fov_max, float)
    for attempt in range(100):
        polys, L = make_outline(P)
        sc = 1
        if partial:
            sc = U(P.partial_scale)
            polys = [p * sc for p in polys]
        allv = np.vstack(polys)
        bmin = allv.min(axis=0) - 0.1
        bmax = allv.max(axis=0) + 0.1
        need = bmax - bmin + 2 * P.frame_margin + 2 * P.frame_jitter
        if partial or np.all(need <= fov_max):
            break
    if partial:
        fov = fov_min + (np.minimum(fov_max, np.maximum(fov_min, need)) - fov_min) * rng.rand(1, 2).ravel()
        gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 40) * fov[0], np.linspace(-0.5, 0.5, 40) * fov[1])
        partial = False
        for tries in range(300):
            ctr = bmin + (bmax - bmin) * rng.rand(1, 2).ravel()
            inn = np.zeros(gx.shape, bool)
            for p in polys:
                inn |= inpolygon(gx + ctr[0], gy + ctr[1], p[:, 0], p[:, 1])
            whole = np.all(bmin >= ctr - fov / 2) and np.all(bmax <= ctr + fov / 2)
            if (not whole) and P.partial_cover[0] <= inn.mean() <= P.partial_cover[1]:
                partial = True
                break
    if not partial:
        fov = np.maximum(fov_min, need)
        ctr = (bmin + bmax) / 2 + P.frame_jitter * (2 * rng.rand(1, 2).ravel() - 1)
    L.partial = float(partial)
    L.scale = sc
    L.max_pixels = U(P.max_pixels)
    cp = np.cumsum(P.dx_out_prob) / np.sum(P.dx_out_prob)
    dxo = P.dx_out[int(np.argmax(rng.rand() <= cp))]
    dxo = max(dxo, math.sqrt(np.prod(fov) / L.max_pixels))
    dxo = P.dx_sim * math.ceil(dxo / P.dx_sim - 1e-9)   # whole number of simulation pixels
    L.dx_out = dxo
    fov = np.ceil(fov / (2 * dxo)) * 2 * dxo             # whole number of output pixel pairs
    polys = [p - ctr for p in polys]                       # frame centre = origin
    L.fov_x = fov[0]
    L.fov_y = fov[1]
    L.n_pixels = float(mround(np.prod(fov) / dxo ** 2))
    L.scan_hours = L.n_pixels * P.pixel_time / 3600

    # ---------- 2. padded simulation grid ----------
    dx = P.dx_sim
    half = np.maximum(fov / 2, np.max(np.abs(np.vstack([bmin, bmax]) - ctr), axis=0) + 0.5)
    nxF = int(mround(fov[0] / dx))
    nyF = int(mround(fov[1] / dx))
    nx = int(P.pad * 2 * math.ceil(half[0] / dx))
    ny = int(P.pad * 2 * math.ceil(half[1] / dx))
    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    y = (np.arange(ny) - ny / 2 + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    ix = slice((nx - nxF) // 2, (nx - nxF) // 2 + nxF)
    iy = slice((ny - nyF) // 2, (ny - nyF) // 2 + nyF)

    # ---------- 3. mask with rough soft edges, and bubbles ----------
    mask, L = rasterize(X, Y, dx, polys, P, L)
    L.area = float(mask.sum() * dx ** 2)
    L.fov_cover = float(np.mean(mask[iy, ix] > 0.5))
    bfac, bub = make_bubbles(X, Y, x, y, mask, 2 * half, L.area, P)

    # ---------- 4. switching participation ----------
    L.switch_offset = U(P.switch_offset)
    L.switch_disorder = U(P.switch_disorder)
    L.switch_lc = U(P.switch_lc)
    a = L.switch_offset + L.switch_disorder * smooth2d(ny, nx, dx, L.switch_lc)
    part = np.sqrt(np.maximum(1 - a ** 2, 0))
    base = mask * bfac * part

    # ---------- 5. magnetization components ----------
    L.sigma0 = UL(P.sigma0)
    L.n_comp = 1 + float(rng.rand() < P.p_two_comp)
    L.c1_theta = U(P.theta_deg)
    L.c1_phi = U(P.phi_deg)
    L.c1_ratio = 1.0
    L.c1_orbital = 0.0
    L.c2_theta = np.nan
    L.c2_phi = np.nan
    L.c2_ratio = np.nan
    L.c2_orbital = np.nan
    if L.n_comp == 2:
        L.c2_orbital = float(rng.rand() < P.p_orbital)
        if L.c2_orbital:
            L.c2_theta = 180.0 * (rng.rand() < 0.5)
            L.c2_phi = 0.0
        else:
            L.c2_theta = U(P.theta_deg)
            L.c2_phi = U(P.phi_deg)
        L.c2_ratio = UL(P.comp2_ratio)
    mx = np.zeros((ny, nx))
    my = np.zeros((ny, nx))
    mz = np.zeros((ny, nx))
    mz0 = np.zeros((ny, nx))
    for c in range(1, int(L.n_comp) + 1):
        th, ph = L[f'c{c}_theta'], L[f'c{c}_phi']
        rat, orb = L[f'c{c}_ratio'], L[f'c{c}_orbital']
        tstd = U(P.moment_tex_std)
        tex = np.maximum(1 + tstd * smooth2d(ny, nx, dx, U(P.moment_tex_lc)), 0)
        scm = L.sigma0 * rat * base * tex
        mx = mx + scm * sind(th) * cosd(ph)
        my = my + scm * sind(th) * sind(ph)
        mz = mz + scm * cosd(th)
        mz0 = mz0 + (scm * cosd(th) if orb else scm)

    # ---------- 6. stray field at the SOT ----------
    L.thickness = UL(P.thickness)
    L.h = U(P.h)
    L.d_tip = U(P.d_tip)
    Z = np.zeros((ny, nx))
    Bt = sot_forward(mx, my, mz, dx, L.h, L.d_tip, L.thickness)
    Bo = sot_forward(-mx, -my, mz, dx, L.h, L.d_tip, L.thickness)
    B0 = sot_forward(Z, Z, mz0, dx, L.h, L.d_tip, L.thickness)

    # ---------- 7. extra source layer below the sample ----------
    L.extra = float(rng.rand() < P.p_extra)
    L.extra_type = np.nan
    L.extra_depth = np.nan
    L.extra_ratio = np.nan
    Be = Z
    if L.extra:
        ae = 360 * rng.rand()
        ox = (2 * rng.rand() - 1) * fov[0] / 2
        oy = (2 * rng.rand() - 1) * fov[1] / 2
        u = -(X - ox) * sind(ae) + (Y - oy) * cosd(ae)
        if rng.rand() < P.extra_halfplane:
            L.extra_type = 1.0
            me = (u < 0).astype(float)
        else:
            L.extra_type = 2.0
            me = (np.abs(u) < U(P.extra_W) / 2).astype(float)
        win = (np.abs(X) < x.max() - 1) & (np.abs(Y) < y.max() - 1)
        me = soften(me * win, dx, 0.05)
        L.extra_depth = U(P.extra_depth)
        L.extra_ratio = UL(P.extra_ratio) * float(np.sign(rng.randn()))
        te = 1 + 0.1 * smooth2d(ny, nx, dx, U(P.extra_tex_lc))
        Be = sot_forward(Z, Z, L.sigma0 * L.extra_ratio * me * te, dx,
                         L.h + L.thickness + L.extra_depth, L.d_tip, 0)
        Bt = Bt + Be
        Bo = Bo + Be
        B0 = B0 + Be

    # ---------- 8. crop to the frame and bin to output pixels ----------
    f = int(mround(L.dx_out / dx))
    Bt = binavg(Bt[iy, ix], f)
    Bo = binavg(Bo[iy, ix], f)
    B0 = binavg(B0[iy, ix], f)
    Be = binavg(Be[iy, ix], f)
    mvec = np.stack([binavg(mx[iy, ix], f), binavg(my[iy, ix], f), binavg(mz[iy, ix], f)], axis=2)
    mk = binavg(mask[iy, ix], f)

    # the mask as the experimentalist knows it: shifted and blurred
    KX, KY = kgrid(ny, nx, dx)
    L.mask_shift_x = P.mask_shift_std * rng.randn()
    L.mask_shift_y = P.mask_shift_std * rng.randn()
    L.mask_blur = U(P.mask_blur)
    Mk = np.real(np.fft.ifft2(np.fft.fft2(mask) * np.exp(-1j * (KX * L.mask_shift_x + KY * L.mask_shift_y))
                              * np.exp(-(KX ** 2 + KY ** 2) * L.mask_blur ** 2 / 2)))
    mki = np.clip(binavg(Mk[iy, ix], f), 0, 1)

    # ---------- 9. noise and background ----------
    pk = float(np.max(np.abs(B0)))
    L.noise_frac = U(P.noise_frac)
    if P.real_noise is None or np.size(P.real_noise) == 0:
        sig_n = L.noise_frac * pk
        Nt = make_noise(Bt.shape, sig_n, P)
        No = make_noise(Bo.shape, sig_n, P)
        N0 = make_noise(B0.shape, sig_n, P)
    else:
        Nt = real_noise_patch(Bt.shape, P.real_noise)
        No = real_noise_patch(Bo.shape, P.real_noise)
        N0 = real_noise_patch(B0.shape, P.real_noise)
        L.noise_frac = std1(Nt) / pk
    L.noise_rms_nT = std1(Nt)
    Nt = Nt + make_background(Bt.shape, pk, P)
    No = No + make_background(Bo.shape, pk, P)
    N0 = N0 + make_background(B0.shape, pk, P)

    # ---------- 10. output ----------
    S = Struct()
    S.Bz_tilt = Bt + Nt
    S.Bz_opp = Bo + No
    S.Bz_zero = B0 + N0
    S.Bz_tilt_clean = Bt
    S.Bz_opp_clean = Bo
    S.Bz_zero_clean = B0
    S.Bz_extra = Be
    S.mvec = mvec
    S.mask = mk
    S.mask_input = mki
    S.x = (np.arange(mk.shape[1]) - mk.shape[1] / 2 + 0.5) * L.dx_out
    S.y = (np.arange(mk.shape[0]) - mk.shape[0] / 2 + 0.5) * L.dx_out
    L.n_bubbles = float(bub.shape[0])
    S.bubbles = bub
    S.labels = L
    return S


def make_outline(P):
    """random sample outline (+ arms) as polygons in um, centred near the origin"""
    L = Struct()
    cp = np.cumsum(P.shape_probs) / np.sum(P.shape_probs)
    t = int(np.argmax(rng.rand() <= cp)) + 1
    L.shape_type = float(t)          # 1 strip, 2 trapezoid, 3 polygon, 4 blob
    L.W = np.nan
    L.length = np.nan
    L.trap_angle = np.nan
    L.n_vertices = np.nan
    if t == 1:                       # strip
        L.W = U(P.W)
        L.length = U(P.strip_length)
        px = np.array([-1, 1, 1, -1]) * L.W / 2
        py = np.array([-1, -1, 1, 1]) * L.length / 2
    elif t == 2:                     # trapezoid: one straight side, one slanted side
        L.W = U(P.W)
        L.length = U(P.strip_length)
        L.trap_angle = U(P.trap_angle)
        ta = math.tan(math.radians(L.trap_angle))
        yt = L.length / 2
        yb = max(-yt, -(L.W - 0.5) / ta)
        px = np.array([-L.W / 2, L.W / 2 + yb * ta, L.W / 2 + yt * ta, -L.W / 2])
        py = np.array([yb, yb, yt, yt])
        if rng.rand() < 0.5:
            px = -px
    elif t == 3:                     # polygon: random corners, can be concave
        K = rng.randi(P.poly_vertices)
        L.n_vertices = float(K)
        ang = np.arange(K) * 2 * np.pi / K + (rng.rand(1, K).ravel() - 0.5) * 0.8 * 2 * np.pi / K
        rad = P.poly_radius[0] + (P.poly_radius[1] - P.poly_radius[0]) * rng.rand(1, K).ravel()
        px = rad * np.cos(ang) * U(P.poly_aspect)
        py = rad * np.sin(ang)
    else:                            # blob: smooth amorphous outline
        ang = np.linspace(0, 2 * np.pi, 121)[:-1]
        R0 = U(P.poly_radius)
        R = R0 * np.ones_like(ang)
        for n in range(2, 6):
            amp = R0 * P.blob_harm * rng.rand()
            R = R + amp * np.cos(n * ang + 2 * np.pi * rng.rand())
        px = R * np.cos(ang) * U(P.poly_aspect)
        py = R * np.sin(ang)
    L.rot = U(P.rot_deg)
    qx = px * cosd(L.rot) - py * sind(L.rot)
    qy = px * sind(L.rot) + py * cosd(L.rot)
    polys = [np.column_stack([qx, qy])]
    # arms: rectangles starting at a random point inside the sample, pointing outwards
    L.n_arms = 0.0
    if rng.rand() < P.arm_prob:
        lo = polys[0].min(axis=0)
        hi = polys[0].max(axis=0)
        for q in range(rng.randi(P.n_arms)):
            for tries in range(200):
                p0 = lo + (hi - lo) * rng.rand(1, 2).ravel()
                if inpolygon(p0[0], p0[1], qx, qy):
                    break
            al = 360 * rng.rand()
            aw = U(P.arm_width)
            alen = U(P.arm_length)
            R = np.array([[cosd(al), -sind(al)], [sind(al), cosd(al)]])
            rect = (R @ np.array([[0, alen, alen, 0], [-aw / 2, -aw / 2, aw / 2, aw / 2]])).T + p0
            polys.append(rect)
            L.n_arms += 1
    return polys, L


def rasterize(X, Y, dx, polys, P, L):
    """fill the polygons on the grid, with rough edges (coordinate warping) and soft edges"""
    ny, nx = X.shape
    L.edge_rough = U(P.edge_rough_amp)
    Xw = X + L.edge_rough * smooth2d(ny, nx, dx, P.edge_rough_lc)
    Yw = Y + L.edge_rough * smooth2d(ny, nx, dx, P.edge_rough_lc)
    inn = np.zeros((ny, nx))
    for p in polys:
        inn = np.maximum(inn, inpolygon(Xw, Yw, p[:, 0], p[:, 1]).astype(float))
    return soften(inn, dx, P.edge_width), L


def soften(A, dx, w):
    """Gaussian blur of width w (um), clipped to [0,1]"""
    ny, nx = A.shape
    KX, KY = kgrid(ny, nx, dx)
    return np.clip(np.real(np.fft.ifft2(np.fft.fft2(A) * np.exp(-(KX ** 2 + KY ** 2) * w ** 2 / 2))), 0, 1)


def make_bubbles(X, Y, x, y, mask, fov, area, P):
    """elliptical regions where the magnetic phase is destroyed"""
    ny, nx = X.shape
    nb = int(min(mround(U(P.bubble_density) * area), P.n_bubbles_cap))
    bub = np.zeros((0, 6))           # [xb yb r aspect angle supp]
    bfac = np.ones((ny, nx))
    ang = np.linspace(0, 2 * np.pi, 17)
    tries = 0
    while bub.shape[0] < nb and tries < 50 * max(nb, 1):
        tries += 1
        r = U(P.bubble_r)
        ar = U(P.bubble_aspect)
        xb = (2 * rng.rand() - 1) * fov[0] / 2
        yb = (2 * rng.rand() - 1) * fov[1] / 2
        rc = r * math.sqrt(ar) + P.bubble_gap
        xq = np.concatenate([[xb], xb + rc * np.cos(ang)])
        yq = np.concatenate([[yb], yb + rc * np.sin(ang)])
        if np.min(mask_at(mask, x, y, xq, yq)) < 0.95:
            continue
        ok = True
        for q in range(bub.shape[0]):
            if math.hypot(xb - bub[q, 0], yb - bub[q, 1]) < \
                    r * math.sqrt(ar) + bub[q, 2] * math.sqrt(bub[q, 3]) + P.bubble_gap:
                ok = False
                break
        if not ok:
            continue
        phi = np.pi * rng.rand()
        s = U(P.bubble_supp)
        bub = np.vstack([bub, [xb, yb, r, ar, phi, s]])
        a = r * math.sqrt(ar)
        b = r / math.sqrt(ar)
        u = (X - xb) * math.cos(phi) + (Y - yb) * math.sin(phi)
        v = -(X - xb) * math.sin(phi) + (Y - yb) * math.cos(phi)
        dist = (np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1) * r
        bfac = bfac * (s + (1 - s) * 0.5 * (1 + np.tanh(dist / P.bubble_edge)))
    return bfac, bub


def U(r):
    """uniform draw in [r[0] r[1]]; a scalar is returned unchanged"""
    if np.size(r) == 1:
        return float(np.asarray(r).ravel()[0])
    return r[0] + (r[1] - r[0]) * rng.rand()


def UL(r):
    """log-uniform draw in [r[0] r[1]]"""
    if np.size(r) == 1:
        return float(np.asarray(r).ravel()[0])
    return math.exp(math.log(r[0]) + (math.log(r[1]) - math.log(r[0])) * rng.rand())


def smooth2d(ny, nx, dx, lc):
    """2D Gaussian random field, zero mean, unit std, correlation length lc"""
    KX, KY = kgrid(ny, nx, dx)
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.randn(ny, nx)) * np.exp(-((KX ** 2 + KY ** 2) * (lc / 2) ** 2) / 2)))
    return (g - g.mean()) / std1(g)


def binavg(A, f):
    """average f x f blocks"""
    ny, nx = A.shape
    my, mx = ny // f, nx // f
    return A[:f * my, :f * mx].reshape(my, f, mx, f).mean(axis=(1, 3))


def mask_at(mask, x, y, xq, yq):
    """mask value at points (xq,yq), nearest pixel; points outside the grid count as 0"""
    j = mround((np.asarray(xq) - x[0]) / (x[1] - x[0])).astype(int)
    i = mround((np.asarray(yq) - y[0]) / (y[1] - y[0])).astype(int)
    v = np.zeros(np.shape(xq))
    ok = (i >= 0) & (i < len(y)) & (j >= 0) & (j < len(x))
    v[ok] = mask[i[ok], j[ok]]
    return v


def make_noise(sz, sig, P):
    """white + per-line offsets (with glitches) + slow drift; fast axis = x"""
    ny, nx = sz
    ww = U(P.w_white)
    wl = min(U(P.w_line), 1 - ww - 0.02)
    wd = 1 - ww - wl
    Nw = rng.randn(ny, nx)
    lo = rng.randn(ny, 1).ravel()
    g = rng.rand(ny, 1).ravel() < P.glitch_prob
    if g.any():
        lo[g] = lo[g] + P.glitch_amp * np.sign(rng.randn(int(g.sum()), 1).ravel())
    lo = (lo - lo.mean()) / std1(lo)
    dr = np.cumsum(rng.randn(ny, 1).ravel())
    dr = dr - np.linspace(dr[0], dr[-1], ny)
    dr = (dr - dr.mean()) / max(std1(dr), EPS)
    return sig * (math.sqrt(ww) * Nw + math.sqrt(wl) * lo[:, None] + math.sqrt(wd) * dr[:, None]
                  * np.ones((1, nx)))


def real_noise_patch(sz, bank):
    """random crop of a real noise map (nT); flips keep the scan-line structure"""
    bank = np.asarray(bank, float)
    if bank.ndim == 2:
        bank = bank[:, :, None]
    A = bank[:, :, rng.randi(bank.shape[2]) - 1]
    if rng.rand() < 0.5:
        A = np.fliplr(A)
    if rng.rand() < 0.5:
        A = np.flipud(A)
    A = np.tile(A, (max(math.ceil(sz[0] / A.shape[0]), 1), max(math.ceil(sz[1] / A.shape[1]), 1)))
    i0 = rng.randi(A.shape[0] - sz[0] + 1) - 1
    j0 = rng.randi(A.shape[1] - sz[1] + 1) - 1
    N = A[i0:i0 + sz[0], j0:j0 + sz[1]]
    return N - N.mean()


def make_background(sz, pk, P):
    """random constant offset + linear plane; slope = change across the whole FOV"""
    YY, XX = np.meshgrid(np.linspace(-0.5, 0.5, sz[0]), np.linspace(-0.5, 0.5, sz[1]), indexing='ij')
    a = rng.randn()
    b = rng.randn()
    c = rng.randn()
    return pk * (P.bg_offset * a + P.bg_slope * b * XX + P.bg_slope * c * YY)


def sot_preview(S, show=True, save=None):
    """row 1: true magnetization; row 2: field along z and tilted; row 3: opposite tilt + cut"""
    import matplotlib.pyplot as plt
    L = S.labels
    fig = plt.figure(figsize=(14, 9.8))
    mag = np.sqrt((S.mvec ** 2).sum(axis=2))
    mm = mag.max()
    c = max(np.abs(S.Bz_zero_clean).max(), np.abs(S.Bz_tilt_clean).max(), np.abs(S.Bz_opp_clean).max())
    ph = L.c1_phi
    smax = min(S.x[-1] / max(abs(cosd(ph)), 1e-9), S.y[-1] / max(abs(sind(ph)), 1e-9))
    s = np.linspace(-smax, smax, 400)
    xq = s * cosd(ph)
    yq = s * sind(ph)
    pan = [(mag / mm, '|m| / max', (0, 1)),
           (S.mvec[:, :, 0] / mm, 'm_x / max', (-1, 1)),
           (S.mvec[:, :, 1] / mm, 'm_y / max', (-1, 1)),
           (S.mvec[:, :, 2] / mm, 'm_z / max', (-1, 1)),
           (S.Bz_zero_clean, 'clean B_z, field along z (nT)', (-c, c)),
           (S.Bz_zero, 'noisy B_z, field along z (nT)', (-c, c)),
           (S.Bz_tilt_clean, 'clean B_z, tilted (nT)', (-c, c)),
           (S.Bz_tilt, 'noisy B_z, tilted (nT)', (-c, c)),
           (S.Bz_opp_clean, 'clean B_z, opposite tilt (nT)', (-c, c)),
           (S.Bz_opp, 'noisy B_z, opposite tilt (nT)', (-c, c))]
    ext = [S.x[0], S.x[-1], S.y[0], S.y[-1]]
    for p, (img, ttl, cl) in enumerate(pan):
        ax = fig.add_subplot(3, 4, p + 1)
        im = ax.imshow(img, extent=ext, origin='lower', cmap='jet', vmin=cl[0], vmax=cl[1])
        fig.colorbar(im, ax=ax)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('x (um)')
        if p in (0, 4, 8):
            ax.set_ylabel('y (um)')
        if p == 0:
            ax.contour(S.x, S.y, S.mask_input, [0.5], colors='w', linewidths=1)
            ax.plot(xq, yq, 'w--', lw=1)
    ax = fig.add_subplot(3, 4, (11, 12))
    v0 = interp2(S.x, S.y, S.Bz_zero_clean, xq, yq)
    vt = interp2(S.x, S.y, S.Bz_tilt_clean, xq, yq)
    vo = interp2(S.x, S.y, S.Bz_opp_clean, xq, yq)
    vm = interp2(S.x, S.y, S.mask, xq, yq)
    ax.fill_between(s, vm * c, color=(0.85, 0.85, 0.85), lw=0, label='sample (grey)')
    ax.plot(s, v0, 'k', s, vt, 'r', s, vo, 'b', lw=1.3)
    ax.grid(True)
    ax.set_xlim(s[0], s[-1])
    ax.set_xlabel('position along the dashed line (um)')
    ax.set_ylabel('B_z (nT)')
    ax.legend(['sample (grey)', 'field along z', 'tilted', 'opposite tilt'], loc='center left',
              bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title(f'clean cut along the in-plane direction of comp 1 (phi = {ph:.0f} deg)', fontsize=9)
    txt = (f'shape {L.shape_type:.0f}, {L.n_arms:.0f} arms, {L.n_bubbles:.0f} bubbles, frame {L.fov_x:.1f} x '
           f'{L.fov_y:.1f} um, pixel {1e3 * L.dx_out:.0f} nm ({L.n_pixels:.0f} px, {L.scan_hours:.1f} h scan), '
           f't = {1e3 * L.thickness:.1f} nm, h = {1e3 * L.h:.0f} nm, tip = {1e3 * L.d_tip:.0f} nm, '
           f'noise = {100 * L.noise_frac:.0f}%  |  comp 1: theta = {L.c1_theta:.0f}, phi = {L.c1_phi:.0f} '
           f'(opposite: phi = {np.mod(L.c1_phi + 360, 360) - 180:.0f})')
    if L.n_comp == 2:
        ostr = ' (orbital)' if L.c2_orbital else ''
        txt += f'  |  comp 2: theta = {L.c2_theta:.0f}, phi = {L.c2_phi:.0f}, ratio {L.c2_ratio:.2f}{ostr}'
    if L.extra:
        txt += f'  |  extra layer {1e3 * L.extra_depth:.0f} nm below, ratio {L.extra_ratio:.2f}'
    fig.suptitle(txt, fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        fig.savefig(save, dpi=90)
    if show:
        plt.show()
    return fig


def sot_run_checks(show=True):
    """physics checks of the simulator; returns the numbers it prints"""
    out = {}
    # ---- check 1: FFT vs direct dipole sum ----
    dx, n, h = 0.05, 240, 0.3
    x = (np.arange(n) - n / 2 + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    src = (np.abs(X) < 1.5) & (np.abs(Y) < 1.0)
    rng.seed(1)
    mz = 1e12 * src * (1 + 0.3 * rng.randn(n, n))
    mx = 0.7e12 * src * (1 + 0.3 * rng.randn(n, n))
    my = -0.4e12 * src
    Bfft = sot_forward(mx, my, mz, dx, h, 0)
    mu0 = 4 * np.pi * 1e-7
    c = 9.2740100783e-24 / 1e-4
    k = src.T.ravel()                 # column-major order, as MATLAB find
    xs = X.T.ravel()[k] * 1e-6
    ys = Y.T.ravel()[k] * 1e-6
    dA = (dx * 1e-6) ** 2
    mxs = mx.T.ravel()[k] * c * dA
    mys = np.broadcast_to(my, X.shape).T.ravel()[k] * c * dA
    mzs = mz.T.ravel()[k] * c * dA
    ip = int(round(n / 2)) + np.arange(-50, 51, 10)       # 1-based column numbers
    Bdir = np.zeros(ip.size)
    for t, col in enumerate(ip):
        rx = X[n // 2 - 1, col - 1] * 1e-6 - xs
        ry = Y[n // 2 - 1, col - 1] * 1e-6 - ys
        rz = h * 1e-6
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        mdr = mxs * rx + mys * ry + mzs * rz
        Bdir[t] = mu0 / (4 * np.pi) * np.sum(3 * mdr * rz / r ** 5 - mzs / r ** 3) * 1e9
    out['check1'] = 100 * np.max(np.abs(Bfft[n // 2 - 1, ip - 1] - Bdir)) / np.max(np.abs(Bdir))
    print(f"Check 1: FFT vs direct dipole sum, max rel. diff = {out['check1']:.2f}%")

    # ---- check 2: paper Fig. 3d-f ----
    import matplotlib.pyplot as plt
    dx, nx, ny = 0.025, 600, 400
    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    strip = np.tile((np.abs(x) < 1.3).astype(float), (ny, 1)) * 0.63e12
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    out['check2'] = []
    for t, th in enumerate([0, 45, -45]):
        B = sot_forward(strip * sind(th), 0 * strip, strip * cosd(th), dx, 0.2, 0)
        b = B[ny // 2 - 1, :]
        imax, imin = int(np.argmax(b)), int(np.argmin(b))
        out['check2'].append([b[imax], x[imax], b[imin], x[imin]])
        print(f'Check 2: theta_s = {th:+3d} deg: max {b[imax]:+5.1f} nT at x = {x[imax]:+5.2f} um, '
              f'min {b[imin]:+5.1f} nT at x = {x[imin]:+5.2f} um')
        ax = axs[t]
        ax.plot(x, b, 'b', lw=1.5)
        ax.plot([-1.3, -1.3], [-55, 60], 'k--', [1.3, 1.3], [-55, 60], 'k--')
        ax.set_xlim(-2.7, 2.7)
        ax.set_ylim(-55, 60)
        ax.grid(True)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('B_z (nT)')
        ax.set_title(f'theta_s = {th} deg')
    out['check2'] = np.array(out['check2'])

    # ---- check 3: lock-in amplitude of a blinking magnet ----
    a = 0.4
    ph = np.linspace(0, 2 * np.pi, 200001)[:-1]
    on = (np.sin(ph) > a).astype(float)
    b1 = 2 * np.mean(on * np.sin(ph))
    out['check3'] = [b1, 2 / np.pi * np.sqrt(1 - a ** 2)]
    print(f'Check 3: simulated lock-in {b1:.4f} vs formula (2/pi)sqrt(1-a^2) = {out["check3"][1]:.4f}')

    # ---- check 4: thick-film kernel vs a stack of 40 thin sheets ----
    t = 0.02
    Bthick = sot_forward(strip * sind(45), 0 * strip, strip * cosd(45), dx, 0.15, 0, t)
    Bstack = 0 * Bthick
    for s in range(1, 41):
        Bstack = Bstack + sot_forward(strip * sind(45) / 40, 0 * strip, strip * cosd(45) / 40, dx,
                                      0.15 + (s - 0.5) * t / 40, 0)
    out['check4'] = 100 * np.max(np.abs(Bthick - Bstack)) / np.max(np.abs(Bstack))
    print(f"Check 4: thick film (20 nm) vs 40 stacked sheets, max rel. diff = {out['check4']:.3f}%")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out

