# sot-2d-magnitization

Synthetic SQUID-on-tip (SOT) maps of a tilted 2D magnetization, a CNN that reads the
magnetization direction from one tilted map, and the paper's two-map method as a benchmark.
Units: lengths in um, fields in nT, moment density in uB/cm^2.

| file | what |
|---|---|
| `create_2D_Magnetization_map.m` | MATLAB original (v7; older versions are kept commented out at the top). Set `MODE` at the top and run. |
| `run_in_octave.m` | runs the `.m` script in GNU Octave (Octave needs functions defined before use) |
| `sot/` | Python port: same functions, same names, same order of random draws (layout below) |
| `sot_simulator.py` | compatibility entry point: re-exports everything in `sot/` under the old names |
| `compare/` | Octave-vs-Python comparison (`comparison_report.txt` is the latest result) |

### Python package layout
```
sot/
  common.py          shared random generator `rng` (MATLAB draw order), MATLAB-like helpers, Struct, dir_vec
  data/              DATA GENERATION
    params.py          sot_default_params: every tunable simulator parameter
    physics.py         sot_forward: stray field at the SOT from (mx, my, mz); kgrid
    simulator.py       sot_make_sample (+ outline, bubbles, noise, background), sot_preview, sot_run_checks
    dataset.py         .mat dataset export, grid_params / to_network_input, training files, test sets
  model/             MODEL
    network.py         build_cnn, save/load, best-network files
    loss.py            model_loss, dir_loss, phi_weight, loss_angle, angle_err
    augment.py         symmetry augmentation, known-phi input channels
    train.py           train_settings, train_cnn
    predict.py         predict_dirs, predict_dirs_tta (8-fold symmetry averaging), show_examples
  benchmark.py       the paper's two-map method
  evaluation.py      error tables and residual plots
  cli.py             command line (`python -m sot`)
```
`sot.data` never imports `sot.model`. The model reads only `grid_params` from the data side,
because that function defines the input grid the network expects.

## MATLAB / Octave
- MATLAB: run `create_2D_Magnetization_map.m`. The networks (`train`, `finetune`, and the network
  part of `evaluate`) need the Deep Learning Toolbox; everything else runs without it.
- Octave (`brew install octave`): `octave --eval run_in_octave` (simulator, data, checks and the
  paper's method; no networks).

## Python
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m sot --mode checks          # physics checks
.venv/bin/python -m sot --mode preview         # 3 random samples
.venv/bin/python -m sot --mode dataset --n 1000 --out sot_train.mat   # MATLAB-readable
.venv/bin/python -m sot --mode make_data       # network training file (.h5)
.venv/bin/python -m sot --mode train --net-type one_map
.venv/bin/python -m sot --mode evaluate        # networks vs the paper's two-map method
```
`python sot_simulator.py --mode ...` still works the same way.
`--mode train` also reads training files made by MATLAB (`-v7.3` .mat), e.g.
`--data-files sot_trainset_20K.mat`. Networks are saved as PyTorch `.pt` files; networks
trained in MATLAB cannot be loaded in Python (and vice versa).

## Comparison
```bash
octave --no-gui --eval "addpath('compare'); octave_reference('compare/ref', 200)"
.venv/bin/python compare/compare_with_octave.py
```
1. **Replay**: Octave's random draws are recorded (`compare/rng_shadow/`) and replayed in Python, so
   both produce the same sample; 12 cases cover every branch (shapes, arms, bubbles, two
   components, gate layer, partial frames, real noise). All maps and labels agree to ~1e-12.
2. **Physics checks** print identical numbers.
3. **Statistics**: 200 samples each, with each language's own random numbers; all label and
   map distributions agree (Kolmogorov-Smirnov).
