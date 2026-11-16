"""SOT simulator + CNN command line (Python port of create_2D_Magnetization_map.m).

Usage:
    python -m sot --mode preview
    python -m sot --mode checks
    python -m sot --mode dataset --n 100 --out sot_train.mat
    python -m sot --mode make_data
    python -m sot --mode train --net-type one_map
    python -m sot --mode evaluate
"""
from __future__ import annotations

import argparse
import numpy as np
from copy import deepcopy

from sot.common import Struct, rng
from sot.data.dataset import get_test_set, grid_params, load_training_data, make_data_file, sot_generate_dataset
from sot.data.params import sot_default_params
from sot.data.simulator import sot_make_sample, sot_preview, sot_run_checks
from sot.evaluation import plot_residuals, report_errors
from sot.model.augment import add_phi_channels
from sot.model.network import build_cnn, load_best_network
from sot.model.predict import predict_dirs_tta, show_examples
from sot.model.train import train_cnn, train_settings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='evaluate',
                    choices=['make_data', 'evaluate', 'finetune', 'train', 'checks', 'preview', 'dataset'])
    ap.add_argument('--net-type', default='one_map', choices=['one_map', 'one_map_phi'])
    ap.add_argument('--data-files', nargs='+', default=['sot_trainset_20K.h5', 'sot_trainset_partial_10K.h5'])
    ap.add_argument('--new-file', default='sot_trainset_partial_10K.h5')
    ap.add_argument('--n-train', type=int, default=10000)
    ap.add_argument('--n-val', type=int, default=500)
    ap.add_argument('--p-partial', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--n-test-partial', type=int, default=300)
    ap.add_argument('--n', type=int, default=100, help='samples for --mode dataset')
    ap.add_argument('--out', default='sot_train.mat', help='output file for --mode dataset')
    ap.add_argument('--no-show', action='store_true', help='do not open figure windows')
    a = ap.parse_args(argv)
    if a.no_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    P = sot_default_params()
    P.p_two_comp = 0             # baseline: one magnetization direction per sample
    G = grid_params()

    if a.mode == 'checks':
        sot_run_checks(show=not a.no_show)
    elif a.mode == 'preview':
        seeds = np.random.default_rng().integers(1, 2 ** 31 - 1, 3)
        for s in seeds:
            rng.seed(int(s))
            sot_preview(sot_make_sample(P), show=False, save=f'preview_seed_{s}.png' if a.no_show else None)
            print(f'preview seed = {s}')
        if not a.no_show:
            plt.show()
    elif a.mode == 'dataset':
        sot_generate_dataset(a.n, a.out, a.seed, P)
    elif a.mode == 'make_data':
        D = Struct(file=a.new_file, n_train=a.n_train, n_val=a.n_val, p_partial=a.p_partial, seed=a.seed)
        make_data_file(D, P, G)
    elif a.mode == 'evaluate':
        tests = [('sot_testset.npz', a.n_test, 3, 0, 'whole sample in the frame'),
                 ('sot_testset_partial.npz', a.n_test_partial, 13, 1, 'only part of the sample in the frame')]
        net1 = load_best_network('one_map', False)
        net2 = load_best_network('one_map_phi', False)
        for t, (file, N, seed, pp, desc) in enumerate(tests):
            Pt = deepcopy(P)
            Pt.p_partial = pp
            TS = get_test_set(file, N, seed, Pt, G)
            print(f'\n\n######## TEST SET {t + 1}: {desc} ########')
            Yps, names = [], []
            if net1 is not None:
                Yps.append(predict_dirs_tta(net1, TS.X))
                names.append('network, 1 map, averaged')
            if net2 is not None:
                Yps.append(predict_dirs_tta(net2, add_phi_channels(TS.X, TS.Y)))
                names.append('network, 1 map + known phi, averaged')
            Yps.append(TS.Ypaper)
            names.append('paper, 2 maps + known phi')
            report_errors(Yps, names, TS.Y, TS.labels)
            plot_residuals(Yps, names, TS.Y, desc)
            if a.no_show:
                plt.savefig(f'residuals_test{t + 1}.png', dpi=90)
        if not a.no_show:
            plt.show()
    elif a.mode in ('train', 'finetune'):
        Xtr, Ytr, Xva, Yva = load_training_data(a.data_files)
        T = train_settings(a.mode, a.net_type)
        if a.mode == 'train':
            Gn = deepcopy(G)
            if T.use_phi:
                Gn.nch = G.nch + 2
            net = build_cnn(Gn)
        else:
            net = load_best_network(a.net_type, True)
        net = train_cnn(net, Xtr, Ytr, Xva, Yva, T)
        if T.use_phi:
            Xva = add_phi_channels(Xva, Yva)
        show_examples(net, Xva, Yva, 8)



if __name__ == '__main__':
    main()
