"""The CNN: architecture, device selection, saving/loading, best-network files."""
from __future__ import annotations

import os

from sot.data.dataset import grid_params


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_cnn(G):
    """VGG-style CNN: 4 stages of (conv-BN-ReLU) x2 + pooling, global pooling, 2 dense layers.
    Initialization as in MATLAB (Glorot weights, zero bias)."""
    torch = _torch()
    nn = torch.nn
    w = [32, 64, 128, 256]
    layers, cin = [], G.nch
    for s in w:
        for r in range(2):
            layers += [nn.Conv2d(cin, s, 3, padding=1), nn.BatchNorm2d(s), nn.ReLU()]
            cin = s
        layers.append(nn.MaxPool2d(2, 2))
    layers += [nn.Conv2d(cin, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
               nn.AdaptiveAvgPool2d(1), nn.Flatten(),
               nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3)]   # raw (mx,my,mz)
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    net.nch = G.nch
    print(f'network: {len(layers)} layers, {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M learnable parameters')
    return net


def save_net(f, net, **extra):
    torch = _torch()
    torch.save({'state': net.state_dict(), 'nch': net.nch, **extra}, f)


def load_net(f):
    torch = _torch()
    Q = torch.load(f, map_location='cpu', weights_only=False)
    G = grid_params()
    G.nch = Q['nch']
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        net = build_cnn(G)
    net.load_state_dict(Q['state'])
    return net, Q


def best_file_of(type_):
    if type_ == 'one_map':
        return 'sot_cnn_best.pt'
    if type_ == 'one_map_phi':
        return 'sot_cnn_phi_best.pt'
    raise ValueError("NET_TYPE must be 'one_map' or 'one_map_phi'.")


def load_best_network(type_, required):
    f = best_file_of(type_)
    if os.path.exists(f):
        net, Q = load_net(f)
        net = net.to(_device())
        print(f"{type_} network: using {f} (validation median error {Q['best']:.1f} deg)")
        return net
    if required:
        raise FileNotFoundError(f"No saved {type_} network found: train it first (--mode train).")
    print(f'{type_} network: not trained yet (--mode train --net-type {type_})')
    return None

