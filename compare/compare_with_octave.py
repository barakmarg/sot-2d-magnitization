"""Compare the Python port (sot_simulator.py) with the MATLAB/Octave code.

Run from the repository root, after compare/octave_reference.m has written compare/ref/:
    .venv/bin/python compare/compare_with_octave.py [--n-stat 200]

1) replay test: for every reference case the Python simulator consumes exactly the random
   numbers Octave drew, and every output (maps, magnetization, masks, labels, bubbles,
   network input, two-map answer) is compared with Octave's.
2) physics checks: the numbers printed by sot_run_checks in both languages.
3) statistics: a dataset generated with each language's OWN random numbers; label
   distributions (two-sample Kolmogorov-Smirnov) and map statistics are compared.
"""
import argparse
import glob
import io
import os
import re
import sys
import contextlib

import numpy as np
from scipy.io import loadmat
from scipy.stats import ks_2samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sot_simulator as sim  # noqa: E402

REF = os.path.join(ROOT, 'compare', 'ref')
ARRAYS = ['Bz_tilt', 'Bz_opp', 'Bz_zero', 'Bz_tilt_clean', 'Bz_opp_clean', 'Bz_zero_clean',
          'Bz_extra', 'mvec', 'mask', 'mask_input', 'x', 'y']


def mat_struct(o):
    """scipy mat_struct -> dict"""
    return {k: getattr(o, k) for k in o._fieldnames}


def params_for(case):
    """the parameter overrides of a case (same table as octave_reference.m)"""
    table = {
        'baseline_a': dict(p_two_comp=0), 'baseline_b': dict(p_two_comp=0), 'baseline_c': dict(p_two_comp=0),
        'strip_bubbles': dict(shape_probs=[1, 0, 0, 0], bubble_density=[0.1, 0.15], arm_prob=1),
        'trapezoid': dict(shape_probs=[0, 1, 0, 0], p_extra=1, extra_halfplane=1),
        'polygon_2comp': dict(shape_probs=[0, 0, 1, 0], p_two_comp=1, p_orbital=0),
        'blob_orbital': dict(shape_probs=[0, 0, 0, 1], p_two_comp=1, p_orbital=1),
        'gate_stripe': dict(p_extra=1, extra_halfplane=0, arm_prob=1),
        'partial_a': dict(p_partial=1), 'partial_b': dict(p_partial=1, p_extra=1, p_two_comp=1),
        'real_noise': dict(), 'thick_film': dict(thickness=0.02, bubble_density=[0.15, 0.15]),
    }
    return table[case]


def rel_err(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def replay_test(tol=1e-9, tol_single=1e-6):
    files = sorted(glob.glob(os.path.join(REF, 'case_*.mat')))
    if not files:
        sys.exit('no reference cases: run compare/octave_reference.m first')
    print('=' * 100)
    print('1) REPLAY TEST: same random numbers in Python as in Octave -> outputs must agree')
    print('=' * 100)
    worst_all, ok_all = 0.0, True
    rows = []
    for f in files:
        case = re.match(r'case_\d+_(.*)\.mat', os.path.basename(f)).group(1)
        R = loadmat(f, squeeze_me=True, struct_as_record=False)
        So = mat_struct(R['S'])
        Lo = mat_struct(So['labels'])
        st = mat_struct(R['streams'])
        P = sim.sot_default_params()
        for k, v in params_for(case).items():
            P[k] = v
        if case == 'real_noise':
            P.real_noise = np.asarray(R['bank'], float)
        sim.rng.replay({'rand': st['rand'], 'randn': st['randn'], 'randi': st['randi']})
        S = sim.sot_make_sample(P)
        G = sim.grid_params()
        XG, YG = np.meshgrid(G.x, G.x)
        Xi = sim.to_network_input(S, XG, YG, G)
        used = {k: sim.rng.pos[k] for k in sim.rng.pos}
        total = {k: sim.rng.streams[k].size for k in sim.rng.streams}
        Yp = sim.two_map_method(S, float(R['hb']), float(R['db']), S.labels.c1_phi)

        errs = {}
        for k in ARRAYS:
            errs[k] = rel_err(S[k], So[k])
        errs['bubbles'] = rel_err(S.bubbles, np.atleast_2d(So['bubbles']).reshape(-1, 6) if np.size(So['bubbles']) else np.zeros((0, 6)))
        lab = [rel_err(S.labels[k], Lo[k]) if not (np.isnan(S.labels[k]) and np.isnan(Lo[k])) else 0.0
               for k in Lo]
        errs['labels'] = max(lab)
        missing = set(Lo) ^ set(S.labels)
        # network input: (C, n, n) in Python, (n, n, C) in MATLAB
        errs['net_input'] = rel_err(Xi, np.moveaxis(np.asarray(R['Xi'], float), 2, 0))
        errs['two_map_dir'] = float(np.degrees(np.arccos(np.clip(np.dot(Yp, R['Ypaper']), -1, 1))))
        streams_ok = used == total
        # the network input is single precision (as in MATLAB): last-bit flips ~1e-8 are expected
        worst = max(v for k, v in errs.items() if k not in ('two_map_dir', 'net_input'))
        ok = (worst < tol and errs['net_input'] < tol_single and errs['two_map_dir'] < 1e-4
              and streams_ok and not missing)
        ok_all &= ok
        worst_all = max(worst_all, worst)
        rows.append((case, S.Bz_tilt.shape, worst, errs['net_input'], errs['two_map_dir'], streams_ok, ok))
        if not ok:
            print(f'  {case}: FAILED', {k: f'{v:.2e}' for k, v in errs.items()}, 'streams', used, total,
                  'label mismatch' if missing else '')
    print(f"  {'case':16s} {'map size':>10s} {'maps+labels':>12s} {'net input':>11s} "
          f"{'two-map dir':>12s} {'all draws used':>15s}  result")
    print(f"  {'':16s} {'':>10s} {'(rel. diff)':>12s} {'(float32)':>11s} {'(angle)':>12s}")
    for case, shp, w, ni, tm, so, ok in rows:
        print(f'  {case:16s} {str(shp[0]) + "x" + str(shp[1]):>10s} {w:12.1e} {ni:11.1e} {tm:10.1e}deg '
              f'{str(so):>15s}  {"PASS" if ok else "FAIL"}')
    print(f'  -> all maps and labels agree to a relative {worst_all:.1e} (tolerance {tol:.0e}), '
          f'float32 network inputs to {tol_single:.0e}: {"PASS" if ok_all else "FAIL"}\n')
    return ok_all


def checks_test():
    print('=' * 100)
    print('2) PHYSICS CHECKS (sot_run_checks)')
    print('=' * 100)
    import matplotlib
    matplotlib.use('Agg')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.sot_run_checks(show=False)
    py = buf.getvalue().strip().splitlines()
    oc = open(os.path.join(REF, 'checks.txt')).read().strip().splitlines()
    ok = True
    for a, b in zip(oc, py):
        same = a.strip() == b.strip()
        # check 1 uses random source textures; numpy and Octave draw different ones
        if not same and a.startswith('Check 1'):
            va = float(re.findall(r'= ([\d.]+)%', a)[0])
            vb = float(re.findall(r'= ([\d.]+)%', b)[0])
            same = abs(va - vb) < 0.2 and vb < 1
            tag = 'close (different random texture)' if same else 'DIFFERENT'
        else:
            tag = 'identical' if same else 'DIFFERENT'
        ok &= same
        print(f'  Octave: {a}\n  Python: {b}\n          -> {tag}')
    print(f'  -> {"PASS" if ok else "FAIL"}\n')
    return ok


def stats_test(n):
    print('=' * 100)
    print(f'3) STATISTICS: {n} samples, each language with its OWN random numbers')
    print('=' * 100)
    fo = os.path.join(REF, 'octave_dataset.mat')
    if not os.path.exists(fo):
        print('  (skipped: compare/ref/octave_dataset.mat not found)\n')
        return True
    Do = loadmat(fo, squeeze_me=True, struct_as_record=False)
    Lo = mat_struct(Do['labels'])
    with contextlib.redirect_stdout(io.StringIO()):
        Dp = sim.sot_generate_dataset(n, os.path.join(REF, 'python_dataset.mat'), 7, sim.sot_default_params())
    Lp = Dp.labels

    def mapstats(D, key):
        out = []
        for i in range(len(D[key])):
            a = np.asarray(D[key][i], float)
            out.append(a)
        return out
    feats = {}
    for name, D in (('octave', Do), ('python', Dp)):
        Bt = mapstats(D, 'Bz_tilt_clean')
        B0 = mapstats(D, 'Bz_zero_clean')
        mk = mapstats(D, 'mask')
        N = mapstats(D, 'Bz_tilt')
        feats[name] = {
            'log10 peak |Bz| (field along z)': np.log10([np.abs(b).max() for b in B0]),
            'peak |Bz tilt| / peak |Bz zero|': [np.abs(a).max() / np.abs(b).max() for a, b in zip(Bt, B0)],
            'mask: covered fraction of frame': [m.mean() for m in mk],
            'noise rms / peak (measured)': [np.std(n_ - c, ddof=1) / np.abs(b).max()
                                            for n_, c, b in zip(N, Bt, B0)],
            'pixels per map': [b.size for b in Bt],
        }
    ok = True
    print(f"  {'quantity':38s} {'Octave mean':>12s} {'Python mean':>12s} {'KS p-value':>11s}")
    keys = ['shape_type', 'n_arms', 'n_bubbles', 'dx_out', 'fov_x', 'fov_y', 'area', 'fov_cover', 'n_comp',
            'c1_theta', 'c1_phi', 'sigma0', 'thickness', 'h', 'd_tip', 'extra', 'noise_frac', 'mask_blur']
    pvals = []
    for k in keys:
        a = np.asarray(Lo[k], float)
        b = np.asarray(Lp[k], float)
        if k == 'sigma0':
            a, b, k = np.log10(a), np.log10(b), 'log10 sigma0'
        p = ks_2samp(a, b).pvalue
        pvals.append(p)
        print(f'  {"label " + k:38s} {a.mean():12.4g} {b.mean():12.4g} {p:11.3f}')
    for k in feats['octave']:
        a = np.asarray(feats['octave'][k], float)
        b = np.asarray(feats['python'][k], float)
        p = ks_2samp(a, b).pvalue
        pvals.append(p)
        print(f'  {k:38s} {a.mean():12.4g} {b.mean():12.4g} {p:11.3f}')
    pvals = np.array(pvals)
    # with ~25 tests, a few p < 0.05 happen by chance; a real difference gives p ~ 0
    nlow = int((pvals < 0.01).sum())
    ok = nlow <= 1
    print(f'  -> {nlow} of {pvals.size} distributions differ at p < 0.01 (expected by chance: '
          f'{0.01 * pvals.size:.2f}): {"PASS" if ok else "FAIL"}\n')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-stat', type=int, default=200)
    a = ap.parse_args()
    r1 = replay_test()
    r2 = checks_test()
    r3 = stats_test(a.n_stat)
    print('OVERALL:', 'PASS' if (r1 and r2 and r3) else 'FAIL')
    sys.exit(0 if (r1 and r2 and r3) else 1)
