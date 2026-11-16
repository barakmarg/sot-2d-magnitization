"""Training: settings and the Adam / cosine-schedule training loop."""
from __future__ import annotations

import math
import numpy as np
import os
import time

from sot.common import Struct, rng
from sot.model.augment import add_phi_channels, augment
from sot.model.loss import angle_err, loss_angle, model_loss, phi_weight
from sot.model.network import _device, _torch, best_file_of, save_net
from sot.model.predict import predict_dirs


def train_settings(mode, type_):
    T = Struct()
    T.batch = 64
    T.disp_every = 50
    T.use_phi = type_ == 'one_map_phi'
    T.best_file = best_file_of(type_)
    T.ckpt_file = T.best_file.replace('_best', f'_ckpt_{mode}')
    T.phi_supp = [90, 5, 4]       # suppress the phi error near theta = 90 deg: [centre width power] (deg), width 0 = off
    if mode == 'train':
        T.epochs, T.lr, T.lr_min, T.check_start = 30, 1e-3, 1e-5, False
    else:
        T.epochs, T.lr, T.lr_min, T.check_start = 15, 5e-4, 1e-6, True
    return T


def train_cnn(net, Xtr, Ytr, Xva, Yva, T):
    """Adam with a cosine learning-rate schedule; checkpoint every epoch (rerun resumes),
    best network so far kept in T.best_file."""
    torch = _torch()
    dev = _device()
    print(f'training on {dev}')
    net = net.to(dev)
    N = Xtr.shape[0]
    nb = N // T.batch
    if T.use_phi:
        Xva = add_phi_channels(Xva, Yva)
    opt = torch.optim.Adam(net.parameters(), lr=T.lr, betas=(0.9, 0.999), eps=1e-8)
    ep0, best = 1, np.inf
    if os.path.exists(T.best_file):
        best = torch.load(T.best_file, map_location='cpu', weights_only=False)['best']
    if os.path.exists(T.ckpt_file):
        Q = torch.load(T.ckpt_file, map_location='cpu', weights_only=False)
        net.load_state_dict(Q['state'])
        opt.load_state_dict(Q['opt'])
        ep0 = Q['ep'] + 1
        print(f"resuming from {T.ckpt_file} after epoch {Q['ep']} of {T.epochs}")
    elif T.check_start:
        err = loss_angle(predict_dirs(net, Xva, T.batch), Yva, phi_weight(Yva, T))
        print(f'starting network: validation median error {np.median(err):.1f} deg (phi suppressed near 90 deg)')
        if np.median(err) < best:
            best = float(np.median(err))
            save_net(T.best_file, net, best=best, ep=0)
    t0 = time.time()
    for ep in range(ep0, T.epochs + 1):
        lr = T.lr_min + 0.5 * (T.lr - T.lr_min) * (1 + math.cos(math.pi * (ep - 1) / max(T.epochs - 1, 1)))
        for gr in opt.param_groups:
            gr['lr'] = lr
        perm = rng.randperm(N) - 1
        net.train()
        for b in range(nb):
            ii = perm[b * T.batch:(b + 1) * T.batch]
            Xb = Xtr[ii]
            if T.use_phi:
                Xb = add_phi_channels(Xb, Ytr[:, ii])
            Xb, Yb = augment(Xb, Ytr[:, ii])
            W = torch.from_numpy(phi_weight(Yb, T)).to(dev)     # weight of the phi error per sample
            Xb = torch.from_numpy(np.ascontiguousarray(Xb, np.float32)).to(dev)
            Yb = torch.from_numpy(np.ascontiguousarray(Yb.T, np.float32)).to(dev)
            opt.zero_grad()
            loss = model_loss(net, Xb, Yb, W)
            loss.backward()
            opt.step()
            if (b + 1) % T.disp_every == 0 or b == nb - 1:
                print(f'epoch {ep:2d}/{T.epochs} | trained on {(b + 1) * T.batch:6d} of {N} samples this epoch '
                      f'| loss {loss.item():.4f} | lr {lr:.1e} | {(time.time() - t0) / 60:.1f} min')
        Ypv = predict_dirs(net, Xva, T.batch)
        e3d = angle_err(Ypv, Yva)                            # plain 3D error, for information
        err = loss_angle(Ypv, Yva, phi_weight(Yva, T))       # the error the loss sees: used for 'best'
        nv = err.size
        print(f'epoch {ep:2d} VALIDATION: within 5 deg: {(err < 5).sum()} of {nv} ({100 * np.mean(err < 5):.1f}%) | '
              f'within 10 deg: {(err < 10).sum()} of {nv} ({100 * np.mean(err < 10):.1f}%) | '
              f'median error {np.median(err):.1f} deg (plain 3D: {np.median(e3d):.1f} deg)')
        if np.median(err) < best:
            best = float(np.median(err))
            save_net(T.best_file, net, best=best, ep=ep)
            print(f'   -> best network so far, saved to {T.best_file}')
        torch.save({'state': net.state_dict(), 'opt': opt.state_dict(), 'ep': ep, 'nch': net.nch}, T.ckpt_file)
    print(f'training finished: best validation median error {best:.1f} deg (network in {T.best_file})')
    return net

