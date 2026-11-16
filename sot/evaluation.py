"""Error analysis: summary tables per group and residual histograms."""
from __future__ import annotations

import math
import numpy as np

from sot.common import mround, sind
from sot.model.loss import angle_err


def report_errors(Yps, names, Y, L, show=True):
    import matplotlib.pyplot as plt
    nm = len(Yps)
    n = Y.shape[1]
    tag = [chr(ord('A') + j) for j in range(nm)]
    tY = np.degrees(np.arccos(np.clip(Y[2], -1, 1)))
    E = [angle_err(Yp, Y) for Yp in Yps]
    Pt = [np.abs(np.degrees(np.arccos(np.clip(Yp[2], -1, 1))) - tY) for Yp in Yps]
    print(f'\n================ RESULTS on {n} test maps never used in training ================')
    for j in range(nm):
        print(f'  [{tag[j]}] {names[j]}')
    print(f"\n{'':38s}" + ''.join(f'{"[" + t + "]":>11s}' for t in tag))

    def row(lab, vals):
        print(f'{lab:38s}' + ''.join(f'{v:>11s}' for v in vals))
    row('median error, full 3D direction', [f'{np.median(e):.1f} deg' for e in E])
    row('median error, polar angle theta only', [f'{np.median(e):.1f} deg' for e in Pt])
    row('maps within 5 deg (3D)', [f'{(e < 5).sum()}' for e in E])
    row('maps within 10 deg (3D)', [f'{(e < 10).sum()}' for e in E])
    row('90% of maps are better than', [f'{prctile_(e, 90):.1f} deg' for e in E])
    row('worst map', [f'{e.max():.1f} deg' for e in E])

    print('\nmedian 3D error (deg) per group:')
    shp = ['strip', 'trapezoid', 'polygon', 'blob']
    print_groups('noise level', E, tag, ['5-10%', '10-15%', '15-20%', '20-25%'],
                 bins_(L.noise_frac, [0.05, 0.10, 0.15, 0.20, 0.25]))
    print_groups('polar angle theta', E, tag, ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180'],
                 bins_(L.c1_theta, np.arange(0, 181, 30)))
    nbub = L.n_bubbles
    print_groups('bubbles', E, tag, ['none', '1-3', '4 or more'],
                 [nbub == 0, (nbub >= 1) & (nbub <= 3), nbub >= 4])
    print_groups('extra source layer (gate)', E, tag, ['no', 'yes'], [L.extra == 0, L.extra == 1])
    print_groups('sample shape', E, tag, shp, [L.shape_type == k for k in (1, 2, 3, 4)])
    pz = mround(1e3 * L.dx_out)
    print_groups('pixel size', E, tag, ['50 nm', '75 nm', '100 nm'], [pz == 50, pz == 75, pz == 100])
    print_groups('arms', E, tag, ['no arms', 'with arms'], [L.n_arms == 0, L.n_arms > 0])

    w = np.argsort(-E[0], kind='stable')
    print(f'\n10 worst test maps of [A]:\n  {"map":>5s}' + ''.join(f'{"[" + t + "]":>8s}' for t in tag)
          + f'{"noise":>7s} {"theta":>6s} {"phi":>6s} {"bubbles":>8s} {"gate":>5s} {"shape":>10s} {"pixel":>7s}')
    for k in w[:min(10, n)]:
        print(f'  {k + 1:5d}' + ''.join(f'{E[j][k]:7.1f}d' for j in range(nm))
              + f'{100 * L.noise_frac[k]:6.0f}% {L.c1_theta[k]:6.0f} {L.c1_phi[k]:6.0f} {int(L.n_bubbles[k]):8d} '
              f'{int(L.extra[k]):5d} {shp[int(L.shape_type[k]) - 1]:>10s} {1e3 * L.dx_out[k]:4.0f} nm')

    col = [(0.85, 0.1, 0.1), (1, 0.55, 0), (0.1, 0.6, 0.1), (0.1, 0.2, 0.9), (0.5, 0, 0.6)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for j in range(nm):
        a1.plot(np.sort(E[j]), np.arange(1, n + 1) / n * 100, color=col[j % 5], lw=1.5,
                label=f'[{tag[j]}] {names[j]}')
    a1.grid(True)
    a1.set_xlim(0, 15)
    a1.set_xlabel('error (deg)')
    a1.set_ylabel('% of test maps with a smaller error')
    a1.legend(loc='lower right', fontsize=8)
    a1.set_title('how many maps are within a given error')
    for j in sorted({0, nm - 1}):
        a2.plot(100 * L.noise_frac, E[j], '.', color=col[j % 5], label=f'[{tag[j]}]')
    a2.grid(True)
    a2.set_xlabel('noise (% of peak field)')
    a2.set_ylabel('error (deg)')
    a2.set_ylim(0, 15)
    a2.legend()
    a2.set_title('error vs noise')
    fig.tight_layout()
    return E


def print_groups(name, E, tag, labels, idx):
    print(f'  {name:26s} {"maps":>5s}' + ''.join(f'{"[" + t + "]":>8s}' for t in tag))
    for g, lab in enumerate(labels):
        k = np.asarray(idx[g]).ravel().astype(bool)
        if not k.any():
            continue
        print(f'    {lab:24s} {k.sum():5d}' + ''.join(f'{np.median(e[k]):8.1f}' for e in E))


def bins_(v, edges):
    v = np.asarray(v)
    out = []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        out.append((v >= edges[i]) & ((v < edges[i + 1]) | (last & (v <= edges[-1]))))
    return out


def prctile_(v, q):
    v = np.sort(np.ravel(v))
    return v[max(1, math.ceil(q / 100 * v.size)) - 1]


def plot_residuals(Yps, names, Y, ttl):
    import matplotlib.pyplot as plt
    nm = len(Yps)
    tT = np.degrees(np.arccos(np.clip(Y[2], -1, 1)))
    pT = np.degrees(np.arctan2(Y[1], Y[0]))
    lab = ['dtheta = predicted - true (deg)', 'dphi * sin(theta) (deg)', '3D error (deg)']
    edges = [np.arange(-15, 15.01, 0.5), np.arange(-15, 15.01, 0.5), np.arange(0, 20.01, 0.5)]
    fig, axs = plt.subplots(nm, 3, figsize=(13, 3 * nm), squeeze=False)
    for j, Yp in enumerate(Yps):
        tP = np.degrees(np.arccos(np.clip(Yp[2], -1, 1)))
        pP = np.degrees(np.arctan2(Yp[1], Yp[0]))
        dphi = np.mod(pP - pT + 180, 360) - 180
        R = [tP - tT, dphi * sind(tT), angle_err(Yp, Y)]
        for c in range(3):
            ax = axs[j, c]
            r = np.asarray(R[c], float).ravel()
            e = edges[c]
            cnt = histcounts_(r, e)
            ax.bar((e[:-1] + e[1:]) / 2, cnt, width=e[1] - e[0], color=(0.55, 0.65, 0.85))
            q = quantile_(r, [0.25, 0.5, 0.75])
            top = cnt.max() * 1.15 + 1
            ax.axvspan(q[0], q[2], color=(1, 0.6, 0), alpha=0.25, lw=0)
            ax.axvline(q[0], color='r', ls='--', lw=1)
            ax.axvline(q[2], color='r', ls='--', lw=1)
            ax.axvline(q[1], color='r', lw=2)
            ax.set_xlim(e[0], e[-1])
            ax.set_ylim(0, top)
            ax.grid(True)
            ax.set_xlabel(lab[c])
            if c == 0:
                ax.set_ylabel(f'{names[j]}\nnumber of maps', fontsize=8)
            ax.set_title(f'median {q[1]:.2f} | IQR {q[2] - q[0]:.2f}  [{q[0]:.2f}, {q[2]:.2f}]', fontsize=9)
            nout = int(((r < e[0]) | (r > e[-1])).sum())
            if nout:
                ax.text(e[-1], 0.93 * top, f'{nout} maps outside the plot  ', ha='right', fontsize=8)
    fig.suptitle(f'{ttl}   |   solid red: median   |   shaded band, dashed red: interquartile range', fontsize=10)
    fig.tight_layout()
    return fig


def histcounts_(v, e):
    c = np.zeros(len(e) - 1)
    for i in range(len(e) - 1):
        if i < len(e) - 2:
            c[i] = np.sum((v >= e[i]) & (v < e[i + 1]))
        else:
            c[i] = np.sum((v >= e[i]) & (v <= e[-1]))
    return c


def quantile_(v, p):
    v = np.sort(np.ravel(v))
    n = v.size
    pos = 1 + (n - 1) * np.asarray(p, float)
    lo = np.floor(pos).astype(int)
    hi = np.ceil(pos).astype(int)
    return v[lo - 1] + (pos - lo) * (v[hi - 1] - v[lo - 1])

