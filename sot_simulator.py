"""Compatibility entry point. The code now lives in the sot/ package:

    sot/data/    data generation (parameters, forward model, simulator, datasets)
    sot/model/   the network (architecture, loss, augmentation, training, prediction)
    sot/benchmark.py, sot/evaluation.py, sot/cli.py

This module re-exports every function under its old name (sim.rng is the same shared
generator), so `import sot_simulator as sim` and `python sot_simulator.py --mode ...`
keep working. New code can call `python -m sot --mode ...` directly.
"""
import importlib as _importlib

for _m in ('common', 'data.params', 'data.physics', 'data.simulator', 'data.dataset', 'benchmark',
           'model.augment', 'model.network', 'model.loss', 'model.train', 'model.predict',
           'evaluation', 'cli'):
    _mod = _importlib.import_module('sot.' + _m)
    globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('__')})
del _m, _mod

if __name__ == '__main__':
    main()  # noqa: F821  (re-exported from sot.cli)
