"""All tunable parameters of the synthetic SOT-map simulator."""
from __future__ import annotations

from sot.common import Struct


def sot_default_params():
    """All tunable parameters. [min max] = drawn per sample (uniform, or
    log-uniform where marked "log")."""
    P = Struct()
    # ---- grids and frame ----
    P.dx_sim = 0.025              # simulation pixel (um)
    P.dx_out = [0.05, 0.075, 0.10]  # pixel sizes an experimentalist would pick (um)
    P.dx_out_prob = [0.5, 0.3, 0.2]  # ... and how often
    P.max_pixels = [10000, 25000]  # scan-time budget: pixels per map
    P.pixel_time = 1.2            # integration time per pixel (s), as in the paper
    P.fov_max = [9, 10]           # largest frame [x y] anyone would scan (um)
    P.fov_min = [5.0, 7.0]        # smallest frame [x y] (um); it grows to fit the sample
    P.frame_margin = 1.0          # minimum empty space between sample and frame edge (um)
    P.frame_jitter = 0.5          # random shift of the sample inside the frame (um)
    P.pad = 2                     # padded grid = pad * simulated area (avoids FFT wrap-around)
    P.p_partial = 0               # probability that the frame shows only PART of the sample
    P.partial_cover = [0.2, 0.85]  # ... then the sample covers this fraction of the frame
    P.partial_scale = [1, 2]      # ... and the sample may be this much larger than usual
    # ---- sample outline ----
    P.shape_probs = [0.2, 0.3, 0.3, 0.2]  # strip, trapezoid, polygon, blob
    P.rot_deg = [-180, 180]
    P.W = [1.5, 3.0]
    P.strip_length = [3, 6]
    P.trap_angle = [5, 30]
    P.poly_vertices = [3, 8]
    P.poly_radius = [1.5, 3.0]
    P.poly_aspect = [0.3, 1.0]
    P.blob_harm = 0.15
    P.edge_rough_amp = [0.02, 0.04]
    P.edge_rough_lc = 0.3
    P.edge_width = 0.02
    # ---- arms ----
    P.arm_prob = 0.4
    P.n_arms = [1, 3]
    P.arm_width = [0.4, 1.2]
    P.arm_length = [1, 2.5]
    # ---- film ----
    P.thickness = [0.0003, 0.02]  # (um), log
    # ---- bubbles ----
    P.bubble_density = [0, 0.15]
    P.n_bubbles_cap = 30
    P.bubble_r = [0.25, 0.45]
    P.bubble_aspect = [1.0, 1.4]
    P.bubble_supp = [0.0, 0.2]
    P.bubble_edge = 0.04
    P.bubble_gap = 0.10
    # ---- switching participation ----
    P.switch_offset = [-0.5, 0.5]
    P.switch_disorder = [0.1, 0.6]
    P.switch_lc = [0.2, 1.0]
    # ---- static moment-density texture ----
    P.moment_tex_std = [0, 0.15]
    P.moment_tex_lc = [0.3, 1.0]
    # ---- magnetization components ----
    P.sigma0 = [1e11, 1e16]       # log
    P.theta_deg = [0, 180]
    P.phi_deg = [-180, 180]
    P.p_two_comp = 0.4
    P.p_orbital = 0.7
    P.comp2_ratio = [0.1, 3]      # log
    # ---- extra source layer below the sample ----
    P.p_extra = 0.3
    P.extra_halfplane = 0.5
    P.extra_depth = [0.01, 0.05]
    P.extra_ratio = [0.05, 0.5]   # log, random sign
    P.extra_W = [3, 10]
    P.extra_tex_lc = [1, 3]
    # ---- SOT ----
    P.h = [0.15, 0.22]
    P.d_tip = [0.17, 0.22]
    # ---- noise ----
    P.noise_frac = [0.05, 0.25]
    P.w_white = [0.5, 0.8]
    P.w_line = [0.15, 0.4]
    P.glitch_prob = 0.03
    P.glitch_amp = 3
    P.real_noise = None           # optional: real noise maps (nT), ny x nx x K
    # ---- imperfections of the real measurement ----
    P.mask_shift_std = 0.08
    P.mask_blur = [0, 0.08]
    P.bg_offset = 0.10
    P.bg_slope = 0.10
    return P

