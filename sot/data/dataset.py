"""Datasets: full-sample .mat export, the fixed grid the network sees, training
files (X, Y) and test sets (with the two-map benchmark answer and all labels)."""
from __future__ import annotations

import numpy as np
import os
import time
from copy import deepcopy

from sot.benchmark import two_map_method
from sot.common import EPS, Struct, dir_vec, interp2, rng, std1
from sot.data.params import sot_default_params
from sot.data.simulator import sot_make_sample


IMAGE_FIELDS = ['Bz_tilt', 'Bz_opp', 'Bz_zero', 'Bz_tilt_clean', 'Bz_opp_clean', 'Bz_zero_clean',
                'Bz_extra', 'mvec', 'mask', 'mask_input']


def sot_generate_dataset(N, outFile=None, seed=0, P=None):
    """N random samples; frames differ in size, so every image field is a list.
    Saved as a MATLAB .mat file (cell arrays), readable by MATLAB and scipy.io.loadmat."""
    if P is None:
        P = sot_default_params()
    rng.seed(seed)
    D = Struct({f: [] for f in IMAGE_FIELDS})
    D.x, D.y, D.bubbles, D.params = [], [], [], P
    D.labels = Struct()
    for i in range(N):
        S = sot_make_sample(P)
        if i == 0:
            for k in S.labels:
                D.labels[k] = np.zeros(N)
        for f in IMAGE_FIELDS:
            D[f].append(S[f].astype(np.float32))
        for k in D.labels:
            D.labels[k][i] = S.labels[k]
        D.x.append(S.x)
        D.y.append(S.y)
        D.bubbles.append(S.bubbles)
        if (i + 1) % 50 == 0:
            print(f'{i + 1} / {N}')
    if outFile:
        save_dataset_mat(D, outFile)
        print(f'saved {N} samples to {outFile}')
    return D


def _cell(lst):
    c = np.empty((len(lst), 1), dtype=object)
    for i, v in enumerate(lst):
        c[i, 0] = v
    return c


def save_dataset_mat(D, outFile):
    from scipy.io import savemat
    out = {f: _cell(D[f]) for f in IMAGE_FIELDS + ['x', 'y', 'bubbles']}
    out['labels'] = {k: v.reshape(-1, 1) for k, v in D.labels.items()}
    out['params'] = {k: (np.zeros((0, 0)) if v is None else np.asarray(v, float)) for k, v in D.params.items()}
    savemat(outFile, out, do_compression=True)


def grid_params():
    """fixed physical grid the network sees: 128 x 128 pixels of 80 nm (10.24 um square)"""
    G = Struct()
    G.n = 128
    G.dx = 0.08
    G.x = (np.arange(G.n) - G.n / 2 + 0.5) * G.dx
    G.nch = 6           # Bz, known mask, valid frame, h, tip, pixel size
    G.h_err = 0.015
    G.tip_err = 0.010
    G.disp_every = 100
    return G


def make_training_set(N, seed, P, G):
    """N simulated samples -> network inputs X (N, nch, n, n) and unit-vector labels Y (3, N)"""
    rng.seed(seed)
    X = np.zeros((N, G.nch, G.n, G.n), np.float32)
    Y = np.zeros((3, N), np.float32)
    XG, YG = np.meshgrid(G.x, G.x)
    t0 = time.time()
    for i in range(N):
        S = sot_make_sample(P)
        X[i] = to_network_input(S, XG, YG, G)
        L = S.labels
        Y[:, i] = dir_vec(L.c1_theta, L.c1_phi)
        if (i + 1) % G.disp_every == 0 or i == N - 1:
            el = time.time() - t0
            print(f'generated {i + 1} of {N} samples | {el / 60:.1f} min elapsed, '
                  f'~{el / (i + 1) * (N - i - 1) / 60:.1f} min left')
    return X, Y


def to_network_input(S, XG, YG, G):
    """resample one sample onto the fixed grid and stack the input channels -> (nch, n, n)"""
    L = S.labels
    valid = ((np.abs(XG) <= S.x[-1] + 1e-6) & (np.abs(YG) <= S.y[-1] + 1e-6)).astype(float)
    v = valid > 0
    B = interp2(S.x, S.y, S.Bz_tilt, XG, YG, 0.0)
    M = interp2(S.x, S.y, S.mask_input, XG, YG, 0.0)
    B = B - np.median(B[v])
    B = B / max(std1(B[v]), EPS)
    B[~v] = 0
    h = L.h + G.h_err * rng.randn()
    d = L.d_tip + G.tip_err * rng.randn()
    on = np.ones((G.n, G.n))
    return np.stack([B, M, valid, on * (h - 0.185) / 0.035, on * (d - 0.195) / 0.025,
                     on * (L.dx_out - 0.075) / 0.025]).astype(np.float32)


def _read_xy(f, name):
    """read Xtr/Xva/Ytr/Yva from a Python .h5 file or a MATLAB v7.3 .mat file.
    Returns X as (N, C, n, n) and Y as (3, N)."""
    import h5py
    with h5py.File(f, 'r') as h:
        a = h[name][()]
        matlab = 'MATLAB_class' in h[name].attrs
    if matlab:   # MATLAB stores column-major: X(i,j,c,k) -> h5 [k, c, j, i]
        a = np.swapaxes(a, -1, -2) if name.startswith('X') else a.T
    return a.astype(np.float32)


def load_training_data(files):
    Xtr, Ytr, Xva, Yva = [], [], [], []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found: create it first (--mode make_data).")
        print(f'loading {f}')
        Xtr.append(_read_xy(f, 'Xtr'))
        Ytr.append(_read_xy(f, 'Ytr'))
        Xva.append(_read_xy(f, 'Xva'))
        Yva.append(_read_xy(f, 'Yva'))
    Xtr, Xva = np.concatenate(Xtr), np.concatenate(Xva)
    Ytr, Yva = np.concatenate(Ytr, axis=1), np.concatenate(Yva, axis=1)
    print(f'training on {Xtr.shape[0]} maps, validating on {Xva.shape[0]} maps')
    return Xtr, Ytr, Xva, Yva


def make_data_file(D, P, G):
    import h5py
    if os.path.exists(D.file):
        print(f'{D.file} already exists, nothing to do')
        return
    P = deepcopy(P)
    P.p_partial = D.p_partial
    print(f'--- generating {D.file}: {D.n_train} training + {D.n_val} validation maps, '
          f'{100 * D.p_partial:.0f}% partial frames ---')
    Xtr, Ytr = make_training_set(D.n_train, D.seed, P, G)
    Xva, Yva = make_training_set(D.n_val, D.seed + 5000, P, G)
    with h5py.File(D.file, 'w') as h:
        for k, v in dict(Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva).items():
            h.create_dataset(k, data=v, compression='gzip', compression_opts=1)
    print(f'saved {D.file}')


def get_test_set(file, N, seed, P, G):
    if os.path.exists(file):
        print(f'loading {file}')
        Z = np.load(file, allow_pickle=False)
        TS = Struct(X=Z['X'], Y=Z['Y'], Ypaper=Z['Ypaper'])
        TS.labels = Struct({k[7:]: Z[k] for k in Z.files if k.startswith('labels_')})
        return TS
    TS = make_test_set(N, seed, P, G)
    np.savez_compressed(file, X=TS.X, Y=TS.Y, Ypaper=TS.Ypaper, **{'labels_' + k: v for k, v in TS.labels.items()})
    return TS


def make_test_set(N, seed, P, G):
    rng.seed(seed)
    TS = Struct()
    TS.X = np.zeros((N, G.nch, G.n, G.n), np.float32)
    TS.Y = np.zeros((3, N), np.float32)
    TS.Ypaper = np.zeros((3, N), np.float32)
    TS.labels = Struct()
    XG, YG = np.meshgrid(G.x, G.x)
    print("--- generating test set (and running the paper's two-map method on it) ---")
    t0 = time.time()
    for i in range(N):
        S = sot_make_sample(P)
        L = S.labels
        TS.X[i] = to_network_input(S, XG, YG, G)
        TS.Y[:, i] = dir_vec(L.c1_theta, L.c1_phi)
        hb = L.h + G.h_err * rng.randn()
        db = L.d_tip + G.tip_err * rng.randn()
        TS.Ypaper[:, i] = two_map_method(S, hb, db, L.c1_phi)
        if i == 0:
            for k in L:
                TS.labels[k] = np.zeros(N)
        for k in TS.labels:
            TS.labels[k][i] = L[k]
        if (i + 1) % G.disp_every == 0 or i == N - 1:
            el = time.time() - t0
            print(f'test map {i + 1} of {N} done | {el / 60:.1f} min elapsed, ~{el / (i + 1) * (N - i - 1) / 60:.1f} min left')
    return TS

