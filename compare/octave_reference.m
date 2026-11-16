function octave_reference(outdir, n_stat)
% OCTAVE_REFERENCE  Reference outputs of create_2D_Magnetization_map.m, for the Python port.
%   Run from the repository root:
%     octave --no-gui --eval "addpath('compare'); octave_reference('compare/ref', 200)"
%   1) replay cases: one sample per case, with every random draw recorded (rng_shadow/),
%      plus its network input and the paper's two-map answer
%   2) a dataset of n_stat samples from sot_generate_dataset (for statistics)
if nargin < 1, outdir = fullfile('compare', 'ref'); end
if nargin < 2, n_stat = 200; end
if ~exist(outdir, 'dir'), mkdir(outdir); end
run_in_octave('create_2D_Magnetization_map.m', true);     % define the functions only
here = fileparts(mfilename('fullpath'));

% name, seed, parameter overrides  (cover every branch of sot_make_sample)
cases = {
  'baseline_a',      1, {'p_two_comp', 0}
  'baseline_b',      2, {'p_two_comp', 0}
  'baseline_c',      3, {'p_two_comp', 0}
  'strip_bubbles',   4, {'shape_probs', [1 0 0 0], 'bubble_density', [0.1 0.15], 'arm_prob', 1}
  'trapezoid',       5, {'shape_probs', [0 1 0 0], 'p_extra', 1, 'extra_halfplane', 1}
  'polygon_2comp',   6, {'shape_probs', [0 0 1 0], 'p_two_comp', 1, 'p_orbital', 0}
  'blob_orbital',    7, {'shape_probs', [0 0 0 1], 'p_two_comp', 1, 'p_orbital', 1}
  'gate_stripe',     8, {'p_extra', 1, 'extra_halfplane', 0, 'arm_prob', 1}
  'partial_a',       9, {'p_partial', 1}
  'partial_b',      10, {'p_partial', 1, 'p_extra', 1, 'p_two_comp', 1}
  'real_noise',     11, {'real_noise', 'bank'}
  'thick_film',     12, {'thickness', 0.02, 'bubble_density', [0.15 0.15]}
};
global SOT_RNG_LOG
addpath(fullfile(here, 'rng_shadow'));
G = grid_params();  [XG, YG] = meshgrid(G.x, G.x);
for c = 1:size(cases, 1)
    name = cases{c,1};  seed = cases{c,2};  ov = cases{c,3};
    P = sot_default_params();
    bank = [];
    for q = 1:2:numel(ov)
        if strcmp(ov{q}, 'real_noise')
            SOT_RNG_LOG = [];  rng(1000 + seed);  bank = 3*randn(70, 50, 3);
            P.real_noise = bank;
        else
            P.(ov{q}) = ov{q+1};
        end
    end
    SOT_RNG_LOG = struct('rand', {{}}, 'randn', {{}}, 'randi', {{}});
    rng(seed);
    t0 = tic;
    S  = sot_make_sample(P);
    Xi = to_network_input(S, XG, YG, G);
    hb = S.labels.h + 0.010;  db = S.labels.d_tip - 0.005;        % fixed "believed" values
    Ypaper = two_map_method(S, hb, db, S.labels.c1_phi);
    streams.rand  = vertcat(SOT_RNG_LOG.rand{:});
    streams.randn = vertcat(SOT_RNG_LOG.randn{:});
    streams.randi = vertcat(SOT_RNG_LOG.randi{:});
    if isempty(streams.randi), streams.randi = zeros(0, 1); end
    SOT_RNG_LOG = [];
    overrides = ov;  %#ok<NASGU>
    save(fullfile(outdir, sprintf('case_%02d_%s.mat', c, name)), ...
         'S', 'Xi', 'Ypaper', 'hb', 'db', 'streams', 'bank', 'seed', '-v7');
    fprintf('case %2d %-14s  %4d x %4d px  (%.1f s)\n', c, name, size(S.Bz_tilt,1), size(S.Bz_tilt,2), toc(t0));
end
rmpath(fullfile(here, 'rng_shadow'));

% physics checks (numbers only)
set(0, 'defaultfigurevisible', 'off');
txt = evalc('sot_run_checks();');
fid = fopen(fullfile(outdir, 'checks.txt'), 'w');  fprintf(fid, '%s', txt);  fclose(fid);
close all

% statistics: a dataset generated with Octave's own random numbers
if n_stat > 0
    t0 = tic;
    sot_generate_dataset(n_stat, fullfile(outdir, 'octave_dataset.mat'), 7, sot_default_params());
    fprintf('dataset: %d samples in %.1f min\n', n_stat, toc(t0)/60);
end
end
