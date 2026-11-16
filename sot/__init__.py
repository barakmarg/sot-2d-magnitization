"""SOT-2D magnetization: synthetic SQUID-on-tip maps and a CNN that reads the magnetization
direction from one tilted map. Units: lengths in um, fields in nT, moment density in uB/cm^2.

Components
    sot.data        data generation: parameters, forward model, sample simulator, datasets
    sot.model       the network: architecture, loss, augmentation, training, prediction
    sot.benchmark   the paper's two-map method
    sot.evaluation  error tables and residual plots
    sot.cli         command line (python -m sot --mode ...)

All random draws go through the single shared generator sot.common.rng, in the same order as
the MATLAB code, so seeds and Octave replay (compare/) give identical samples.
"""
