% %% SOT_SIMULATOR  Synthetic SQUID-on-tip maps of a tilted 2D magnetization
% % Paste this whole file as  sot_simulator.m  and press Run (MATLAB R2021b+).
% % Units: lengths in um, fields in nT, moment density in uB/cm^2.
% clear; close all; clc;
% P = sot_default_params();
% 
% %% 1) physics checks: FFT vs dipole sum, and reproduction of paper Fig. 3d-f
% sot_run_checks();
% 
% %% 2) look at random samples
% for seed = [3 21]
%     rng(seed);
%     S = sot_make_sample(P);
%     sot_preview(S);
% end
% 
% %% 3) generate a dataset (uncomment when the maps look right)
% % D = sot_generate_dataset(1000, 'sot_train.mat', 1, P);
% 
% %% ======================= local functions =======================
% 
% function P = sot_default_params()
% % SOT_DEFAULT_PARAMS  All tunable parameters of the synthetic SOT-map simulator.
% % Lengths in micrometres, fields in nT, moment density in uB/cm^2.
% % A parameter given as [min max] is drawn uniformly per sample.
% 
% % ---- grids ----
% P.dx_sim   = 0.025;          % simulation pixel (um)
% P.dx_out   = 0.050;          % output pixel, as in the paper (um)
% P.fov      = [5.0 7.0];      % field of view [x y] (um), x = across the strip
% P.pad      = 4;              % padded grid = pad * fov (avoids FFT wrap-around)
% 
% % ---- sample shape ----
% P.shape_probs    = [0.2 0.3 0.3 0.2]; % probability of: strip, trapezoid, polygon, blob
% P.rot_deg        = [-180 180];  % in-plane rotation of the whole sample (deg)
% P.center_offset  = [-1.5 1.5];  % sample centre offset from FOV centre, per axis (um)
% P.W              = [1.5 3.5];   % strip / trapezoid width at the sample centre (um)
% P.trap_angle     = [5 30];      % trapezoid: angle of the slanted side (deg)
% P.poly_vertices  = [3 8];       % polygon: number of corners
% P.poly_radius    = [3 7];       % polygon / blob: centre-to-boundary distance (um)
% P.poly_aspect    = [0.3 1.0];   % polygon / blob: squeeze along one axis (elongation)
% P.blob_harm      = 0.15;        % blob: max relative amplitude of each boundary harmonic
% P.fov_cover      = [0.2 0.85];  % accepted fraction of the FOV covered by the sample
% P.edge_rough_amp = [0.02 0.04]; % edge wiggle std (um)
% P.edge_rough_lc  = 0.3;         % edge wiggle correlation length (um)
% P.edge_width     = 0.02;        % Gaussian softening of the edge (um)
% 
% % ---- imperfections of the real measurement ----
% P.mask_shift_std = 0.08;        % misregistration of the known mask, std per axis (um)
% P.mask_blur      = [0 0.08];    % extra blur of the known mask (um)
% P.bg_offset      = 0.10;        % background offset std, fraction of peak |Bz|
% P.bg_slope       = 0.10;        % background plane: std of its change across the FOV, fraction of peak |Bz|
% 
% % ---- bubbles ----
% P.n_bubbles_max  = 5;        % number drawn uniformly from 0..max
% P.bubble_r       = [0.25 0.45]; % mean radius (um)
% P.bubble_margin  = 0.2;      % bubble centres may lie this far outside the FOV (um)
% P.bubble_aspect  = [1.0 1.4];   % ellipse aspect ratio
% P.bubble_supp    = [0.0 0.2];   % remaining m inside a bubble (fraction)
% P.bubble_edge    = 0.04;     % tanh half-width of bubble edge (um), ~80 nm total
% P.bubble_gap     = 0.10;     % min gap bubble-bubble and bubble-edge (um)
% 
% % ---- amplitude texture ----
% P.tex_lc   = [0.3 1.0];      % correlation length of the Gaussian random field (um)
% P.tex_std  = [0.05 0.20];    % relative std of the amplitude texture
% 
% % ---- magnetization ----
% P.sigma0   = 0.63e12 * [0.7 1.3]; % moment density (uB/cm^2), Fig. 3d value +-30%
% P.theta_s  = [-90 90];       % spin tilt (deg), from z towards +x
% 
% % ---- SOT ----
% P.h        = [0.15 0.22];    % graphene-to-SQUID-loop distance (um)
% P.d_tip    = [0.17 0.22];    % effective SQUID loop diameter (um), paper: 0.194
% 
% % ---- noise ----
% P.noise_frac   = [0.05 0.25];% total noise RMS / peak |Bz| of the theta=0 map
% P.w_white      = [0.5 0.8];  % fraction of noise variance that is white
% P.w_line       = [0.15 0.4]; % fraction that is per-line offsets (rest = drift)
% P.glitch_prob  = 0.03;       % probability a scan line is a glitch line
% P.glitch_amp   = 3;          % glitch offset in units of the line-noise std
% end
% 
% 
% function Bz = sot_forward(mx, my, mz, dx, h, d_tip)
% % SOT_FORWARD  Out-of-plane stray field of a thin magnetic sheet, as seen by a SOT.
% %   mx,my,mz : moment density maps (uB/cm^2), rows = y, columns = x
% %   dx       : pixel size (um)
% %   h        : height of the SQUID loop above the sheet (um)
% %   d_tip    : effective SQUID loop diameter (um); 0 = point sensor
% %   Bz       : field in nT, same grid as the input
% %
% % Fourier-space Biot-Savart (equivalent to the real-space dipole integral
% % in the paper's Methods):
% %   Bz(k) = (mu0/2) exp(-k h) [ k mz(k) - i kx mx(k) - i ky my(k) ]
% % followed by averaging over the SQUID loop (a disk of diameter d_tip):
% %   Bz(k) -> Bz(k) * 2 J1(k d/2) / (k d/2)
% % The grid is treated as periodic, so the caller must zero-pad.
% 
% mu0   = 4*pi*1e-7;           % T m / A
% muB   = 9.2740100783e-24;    % A m^2
% conv  = muB / 1e-4;          % uB/cm^2  ->  A  (moment per area, SI)
% 
% [ny, nx] = size(mz);
% kx = 2*pi*ifftshift((0:nx-1) - floor(nx/2)) / (nx*dx);   % 1/um
% ky = 2*pi*ifftshift((0:ny-1) - floor(ny/2)) / (ny*dx);
% [KX, KY] = meshgrid(kx, ky);
% K = sqrt(KX.^2 + KY.^2);
% 
% Mx = fft2(mx); My = fft2(my); Mz = fft2(mz);
% B  = exp(-K*h) .* (K.*Mz - 1i*KX.*Mx - 1i*KY.*My);
% 
% if d_tip > 0
%     q = K * d_tip/2;
%     T = ones(size(q));
%     nz = q > 0;
%     T(nz) = 2*besselj(1, q(nz)) ./ q(nz);
%     B = B .* T;
% end
% 
% % units: k in 1/um -> *1e6 to 1/m ; T -> nT : *1e9
% Bz = real(ifft2(B)) * (mu0/2) * conv * 1e6 * 1e9;
% end
% 
% 
% function S = sot_make_sample(P)
% % SOT_MAKE_SAMPLE  One random synthetic sample: magnetization, clean and noisy
% % SOT maps at the random tilt theta_s, and a partner pair at theta = 0.
% %   S = sot_make_sample(sot_default_params())
% 
% % ---------- 1. padded simulation grid (um), centred on the FOV ----------
% dx  = P.dx_sim;
% nxF = round(P.fov(1)/dx);  nyF = round(P.fov(2)/dx);   % FOV size in sim pixels
% nx  = P.pad*nxF;           ny  = P.pad*nyF;
% x   = ((0:nx-1) - nx/2 + 0.5) * dx;
% y   = ((0:ny-1) - ny/2 + 0.5) * dx;
% [X, Y] = meshgrid(x, y);
% ix  = (nx-nxF)/2 + (1:nxF);  iy = (ny-nyF)/2 + (1:nyF); % FOV indices
% 
% % ---------- 2. sample shape: random type, rotation, rough edges ----------
% [mask, L] = make_shape(X, Y, dx, ix, iy, P);
% 
% % ---------- 3. bubbles ----------
% nb  = randi([0 P.n_bubbles_max]);
% bub = zeros(0,6);                         % [xb yb r aspect angle supp]
% bfac = ones(ny, nx);
% xr  = P.fov(1)/2 + P.bubble_margin;       % bubble centres within FOV (+ margin)
% yr  = P.fov(2)/2 + P.bubble_margin;
% tries = 0;
% while size(bub,1) < nb && tries < 500
%     tries = tries + 1;
%     r   = U(P.bubble_r);  ar = U(P.bubble_aspect);
%     xb  = (2*rand-1)*xr;
%     yb  = (2*rand-1)*yr;
%     % the whole bubble plus a gap must lie inside the sample
%     ang = linspace(0, 2*pi, 17);
%     rc  = r*sqrt(ar) + P.bubble_gap;
%     if min(mask_at(mask, x, y, [xb, xb + rc*cos(ang)], [yb, yb + rc*sin(ang)])) < 0.95
%         continue;
%     end
%     ok = true;
%     for q = 1:size(bub,1)
%         if hypot(xb-bub(q,1), yb-bub(q,2)) < r*sqrt(ar) + bub(q,3)*sqrt(bub(q,4)) + P.bubble_gap
%             ok = false; break;
%         end
%     end
%     if ~ok, continue; end
%     phi = pi*rand;  s = U(P.bubble_supp);
%     bub(end+1,:) = [xb yb r ar phi s]; %#ok<AGROW>
%     a = r*sqrt(ar);  b = r/sqrt(ar);
%     u =  (X-xb)*cos(phi) + (Y-yb)*sin(phi);
%     v = -(X-xb)*sin(phi) + (Y-yb)*cos(phi);
%     rho = sqrt((u/a).^2 + (v/b).^2);
%     dist = (rho - 1)*r;                    % approx. distance from bubble boundary (um)
%     bfac = bfac .* (s + (1-s)*0.5*(1 + tanh(dist/P.bubble_edge)));
% end
% 
% % ---------- 4. amplitude texture (Gaussian random field) ----------
% L.tex_lc  = U(P.tex_lc);  L.tex_std = U(P.tex_std);
% grf  = smooth2d(ny, nx, dx, L.tex_lc);
% amp  = max(1 + L.tex_std*grf, 0);
% 
% L.sigma0 = U(P.sigma0);
% sigma    = L.sigma0 * mask .* bfac .* amp;        % moment density (uB/cm^2)
% 
% % ---------- 5-7. tilt, Biot-Savart, SOT blur ----------
% L.theta_s = U(P.theta_s);
% L.h       = U(P.h);
% L.d_tip   = U(P.d_tip);
% th = L.theta_s*pi/180;
% Bt = sot_forward(sigma*sin(th), 0*sigma, sigma*cos(th), dx, L.h, L.d_tip);
% B0 = sot_forward(0*sigma,       0*sigma, sigma,         dx, L.h, L.d_tip);
% 
% % ---------- 8. crop to FOV and bin to output pixels ----------
% f   = round(P.dx_out/dx);
% Bt  = binavg(Bt(iy,ix), f);
% B0  = binavg(B0(iy,ix), f);
% m   = binavg(sigma(iy,ix), f);
% mk  = binavg(mask(iy,ix), f);
% 
% % ---------- 8b. the mask as the experimentalist knows it: shifted and blurred ----------
% [KX, KY] = kgrid(ny, nx, dx);
% L.mask_shift_x = P.mask_shift_std*randn;
% L.mask_shift_y = P.mask_shift_std*randn;
% L.mask_blur    = U(P.mask_blur);
% Mk = real(ifft2( fft2(mask) .* exp(-1i*(KX*L.mask_shift_x + KY*L.mask_shift_y)) ...
%                             .* exp(-(KX.^2 + KY.^2)*L.mask_blur^2/2) ));
% S.mask_input = min(max(binavg(Mk(iy,ix), f), 0), 1);   % network input channel
% 
% % ---------- 9. noise (same instrument noise level for both maps) ----------
% L.noise_frac = U(P.noise_frac);
% sig_n = L.noise_frac * max(abs(B0(:)));          % total noise RMS (nT)
% L.noise_rms_nT = sig_n;
% Nt = make_noise(size(Bt), sig_n, P);
% N0 = make_noise(size(B0), sig_n, P);
% 
% % ---------- 9b. scan background: offset + plane, different for each scan ----------
% pk = max(abs(B0(:)));
% Nt = Nt + make_background(size(Bt), pk, P);
% N0 = N0 + make_background(size(B0), pk, P);
% 
% % ---------- 10. output ----------
% S.Bz_tilt       = Bt + Nt;     % network input
% S.Bz_zero       = B0 + N0;     % partner map for the two-map benchmark
% S.Bz_tilt_clean = Bt;
% S.Bz_zero_clean = B0;
% S.m             = m;           % true moment density (uB/cm^2)
% S.mask          = mk;
% S.x = ((0:size(m,2)-1) - size(m,2)/2 + 0.5)*P.dx_out;
% S.y = ((0:size(m,1)-1) - size(m,1)/2 + 0.5)*P.dx_out;
% L.n_bubbles = size(bub,1);
% S.bubbles   = bub;
% S.labels    = L;
% end
% 
% % ======================= helper functions =======================
% function v = U(r)
% % uniform draw in [r(1) r(2)]; a scalar is returned unchanged
% if numel(r) == 1, v = r; else, v = r(1) + (r(2)-r(1))*rand; end
% end
% 
% function s = smooth1d(n, dx, lc)
% % 1D Gaussian random process, zero mean, unit std, correlation length lc
% k = 2*pi*ifftshift((0:n-1) - floor(n/2))'/(n*dx);
% s = real(ifft(fft(randn(n,1)) .* exp(-(k*lc/2).^2/2)));
% s = (s - mean(s))/std(s);
% end
% 
% function g = smooth2d(ny, nx, dx, lc)
% % 2D Gaussian random field, zero mean, unit std, correlation length lc
% kx = 2*pi*ifftshift((0:nx-1) - floor(nx/2))/(nx*dx);
% ky = 2*pi*ifftshift((0:ny-1) - floor(ny/2))/(ny*dx);
% [KX, KY] = meshgrid(kx, ky);
% g = real(ifft2(fft2(randn(ny,nx)) .* exp(-((KX.^2+KY.^2)*(lc/2)^2)/2)));
% g = (g - mean(g(:)))/std(g(:));
% end
% 
% function B = binavg(A, f)
% % average f x f blocks
% [ny, nx] = size(A);
% A = A(1:f*floor(ny/f), 1:f*floor(nx/f));
% B = reshape(mean(reshape(A, f, []), 1), floor(ny/f), []);        % bin rows
% B = reshape(mean(reshape(B', f, []), 1), floor(nx/f), [])';      % bin columns
% end
% 
% function [mask, L] = make_shape(X, Y, dx, ix, iy, P)
% % random sample outline on the padded grid; redrawn until it crosses the FOV
% [ny, nx] = size(X);
% [KX, KY] = kgrid(ny, nx, dx);
% Rlim  = min(max(abs(X(:))), max(abs(Y(:)))) - 1 - max(abs(P.center_offset)); % stay inside padded grid
% types = {'strip','trapezoid','polygon','blob'};
% cp    = cumsum(P.shape_probs)/sum(P.shape_probs);
% for attempt = 1:50
%     t = find(rand <= cp, 1);
%     L.shape_type = t;                     % 1 strip, 2 trapezoid, 3 polygon, 4 blob
%     L.W = NaN; L.trap_angle = NaN; L.n_vertices = NaN;
%     switch types{t}
%         case 'strip'
%             L.W = U(P.W);
%             px = [-1 1 1 -1]*L.W/2;  py = [-1 -1 1 1]*Rlim;
%         case 'trapezoid'                  % one straight side, one slanted side
%             L.W = U(P.W);  L.trap_angle = U(P.trap_angle);
%             ta = tand(L.trap_angle);
%             yb = max(-Rlim, -(L.W - 0.5)/ta);           % keep the width >= 0.5 um
%             px = [-L.W/2, L.W/2 + yb*ta, L.W/2 + Rlim*ta, -L.W/2];
%             py = [yb yb Rlim Rlim];
%             if rand < 0.5, px = -px; end                % slanted side left or right
%         case 'polygon'                    % random corners, can be concave
%             K   = randi(P.poly_vertices);  L.n_vertices = K;
%             ang = (0:K-1)*2*pi/K + (rand(1,K)-0.5)*0.8*2*pi/K;
%             rad = min(P.poly_radius(1) + diff(P.poly_radius)*rand(1,K), Rlim);
%             px  = rad.*cos(ang)*U(P.poly_aspect);  py = rad.*sin(ang);
%         case 'blob'                       % smooth amorphous outline
%             ang = linspace(0, 2*pi, 121);  ang(end) = [];
%             R0  = U(P.poly_radius);  R = R0*ones(size(ang));
%             for n = 2:5
%                 R = R + R0*P.blob_harm*rand*cos(n*ang + 2*pi*rand);
%             end
%             R   = min(R, Rlim);
%             px  = R.*cos(ang)*U(P.poly_aspect);  py = R.*sin(ang);
%     end
%     L.rot = U(P.rot_deg);
%     L.cx  = U(P.center_offset);  L.cy = U(P.center_offset);
%     qx = px*cosd(L.rot) - py*sind(L.rot) + L.cx;
%     qy = px*sind(L.rot) + py*cosd(L.rot) + L.cy;
%     % rough edges: warp the coordinates with a smooth random field
%     L.edge_rough = U(P.edge_rough_amp);
%     Xw = X + L.edge_rough*smooth2d(ny, nx, dx, P.edge_rough_lc);
%     Yw = Y + L.edge_rough*smooth2d(ny, nx, dx, P.edge_rough_lc);
%     in = double(inpolygon(Xw, Yw, qx, qy));
%     in = in .* ((X - L.cx).^2 + (Y - L.cy).^2 < Rlim^2);  % finite sample, no FFT wrap-around
%     cover = mean(mean(in(iy, ix)));
%     if cover >= P.fov_cover(1) && cover <= P.fov_cover(2), break; end
% end
% L.fov_cover = cover;
% mask = real(ifft2(fft2(in) .* exp(-(KX.^2 + KY.^2)*P.edge_width^2/2)));  % soft edge
% mask = min(max(mask, 0), 1);
% end
% 
% function v = mask_at(mask, x, y, xq, yq)
% % mask value at points (xq,yq), nearest pixel; points outside the grid count as 0
% j  = round((xq - x(1))/(x(2) - x(1))) + 1;
% i  = round((yq - y(1))/(y(2) - y(1))) + 1;
% v  = zeros(size(xq));
% ok = i >= 1 & i <= numel(y) & j >= 1 & j <= numel(x);
% v(ok) = mask(sub2ind(size(mask), i(ok), j(ok)));
% end
% 
% function [KX, KY] = kgrid(ny, nx, dx)
% % angular wavevector grids (1/um) matching fft2 ordering
% kx = 2*pi*ifftshift((0:nx-1) - floor(nx/2))/(nx*dx);
% ky = 2*pi*ifftshift((0:ny-1) - floor(ny/2))/(ny*dx);
% [KX, KY] = meshgrid(kx, ky);
% end
% 
% function G = make_background(sz, pk, P)
% % random constant offset + linear plane; slope = change across the whole FOV
% [YY, XX] = ndgrid(linspace(-0.5, 0.5, sz(1)), linspace(-0.5, 0.5, sz(2)));
% G = pk*( P.bg_offset*randn + P.bg_slope*randn*XX + P.bg_slope*randn*YY );
% end
% function N = make_noise(sz, sig, P)
% % white + per-line offsets (with glitches) + slow drift; fast axis = x
% ny = sz(1); nx = sz(2);
% ww = U(P.w_white);
% wl = min(U(P.w_line), 1 - ww - 0.02);
% wd = 1 - ww - wl;
% 
% Nw = randn(ny, nx);
% 
% lo = randn(ny, 1);
% g  = rand(ny,1) < P.glitch_prob;
% lo(g) = lo(g) + P.glitch_amp*sign(randn(nnz(g),1));
% lo = (lo - mean(lo))/std(lo);
% 
% dr = cumsum(randn(ny,1));
% dr = dr - linspace(dr(1), dr(end), ny)';         % remove net trend
% dr = (dr - mean(dr))/max(std(dr), eps);
% 
% N = sig*( sqrt(ww)*Nw + sqrt(wl)*repmat(lo,1,nx) + sqrt(wd)*repmat(dr,1,nx) );
% end
% 
% 
% function fig = sot_preview(S)
% % SOT_PREVIEW  Plot one sample: true m, clean and noisy maps at theta=0 and theta_s,
% % and y-averaged line cuts.
% L = S.labels;
% fig = figure('Color','w','Position',[50 50 1500 520]);
% cm = jet(256);
% clim = max(abs(S.Bz_zero_clean(:)));
% 
% panels = {S.m/1e12, 'true m (10^{12} \mu_B/cm^2)', [0 max(S.m(:))/1e12];
%           S.Bz_zero_clean, 'clean B_z, \theta = 0 (nT)', [-clim clim];
%           S.Bz_zero, 'noisy B_z, \theta = 0 (nT)', [-clim clim];
%           S.Bz_tilt_clean, sprintf('clean B_z, \\theta_s = %.1f\\circ (nT)', L.theta_s), [-clim clim];
%           S.Bz_tilt, sprintf('noisy B_z, \\theta_s = %.1f\\circ (nT)', L.theta_s), [-clim clim]};
% for p = 1:5
%     subplot(1,6,p);
%     imagesc(S.x, S.y, panels{p,1}, panels{p,3}); axis image; set(gca,'YDir','normal');
%     colormap(gca, cm); colorbar('southoutside');
%     title(panels{p,2}); xlabel('x (\mum)');
%     if p == 1, ylabel('y (\mum)'); end
% end
% 
% subplot(1,6,6);
% plot(S.x, mean(S.Bz_zero,1), 'b', S.x, mean(S.Bz_tilt,1), 'r', ...
%      S.x, mean(S.Bz_zero_clean,1), 'b--', S.x, mean(S.Bz_tilt_clean,1), 'r--');
% grid on; xlabel('x (\mum)'); ylabel('<B_z>_y (nT)');
% legend('noisy \theta=0','noisy \theta_s','clean \theta=0','clean \theta_s','Location','southoutside');
% title(sprintf('h=%.0f nm, d=%.0f nm, noise=%.0f%%, %d bubbles', ...
%       1e3*L.h, 1e3*L.d_tip, 100*L.noise_frac, L.n_bubbles));
% end
% 
% 
% function sot_run_checks()
% % SOT_RUN_CHECKS  Two sanity checks of the forward model.
% %  (1) FFT Biot-Savart vs. brute-force real-space dipole sum (paper's Methods formula)
% %  (2) Reproduce Fig. 3d-f: uniform 2.6 um strip, 0.63e12 uB/cm^2, h = 200 nm
% 
% %% ---- check 1: FFT vs direct dipole sum ----
% dx = 0.05; n = 240; h = 0.3;                         % 12 x 12 um grid
% x = ((0:n-1) - n/2 + 0.5)*dx;  [X,Y] = meshgrid(x,x);
% src = (abs(X) < 1.5) & (abs(Y) < 1.0);               % 3 x 2 um source patch
% rng(1);
% mz = 1e12*src.*(1 + 0.3*randn(n));  mx = 0.7e12*src.*(1 + 0.3*randn(n));  my = -0.4e12*src;
% Bfft = sot_forward(mx, my, mz, dx, h, 0);
% 
% mu0 = 4*pi*1e-7; c = 9.2740100783e-24/1e-4;          % uB/cm^2 -> A
% k = find(src); xs = X(k)*1e-6; ys = Y(k)*1e-6; dA = (dx*1e-6)^2;
% mxs = mx(k)*c*dA; mys = my(k)*c*dA; mzs = mz(k)*c*dA;  % point moments (A m^2)
% ip = round(n/2) + (-50:10:50);                         % test points on a line y=0
% Bdir = zeros(size(ip));
% for t = 1:numel(ip)
%     rx = X(n/2, ip(t))*1e-6 - xs; ry = Y(n/2, ip(t))*1e-6 - ys; rz = h*1e-6;
%     r  = sqrt(rx.^2 + ry.^2 + rz.^2);
%     mdr = mxs.*rx + mys.*ry + mzs.*rz;
%     Bdir(t) = mu0/(4*pi)*sum(3*mdr.*rz./r.^5 - mzs./r.^3)*1e9;   % nT
% end
% Bf = Bfft(n/2, ip);
% fprintf('Check 1: FFT vs direct dipole sum, max rel. diff = %.2f%%\n', ...
%         100*max(abs(Bf - Bdir))/max(abs(Bdir)));
% 
% %% ---- check 2: Fig. 3d-f ----
% dx = 0.025; nx = 600; ny = 400;
% x = ((0:nx-1) - nx/2 + 0.5)*dx;
% strip = repmat(double(abs(x) < 1.3), ny, 1) * 0.63e12;   % infinite along y (periodic)
% figure('Color','w','Position',[50 50 1200 300]);
% th = [0 45 -45];
% for t = 1:3
%     a = th(t)*pi/180;
%     B = sot_forward(strip*sin(a), 0*strip, strip*cos(a), dx, 0.2, 0);
%     b = B(ny/2, :);
%     [bmax, imax] = max(b); [bmin, imin] = min(b);
%     fprintf('Check 2: theta_s = %+3d deg: max %+5.1f nT at x = %+5.2f um, min %+5.1f nT at x = %+5.2f um\n', ...
%             th(t), bmax, x(imax), bmin, x(imin));
%     subplot(1,3,t); plot(x, b, 'b', 'LineWidth', 1.5); hold on;
%     plot([-1.3 -1.3], [-55 60], 'k--', [1.3 1.3], [-55 60], 'k--');
%     xlim([-2.7 2.7]); ylim([-55 60]); grid on;
%     xlabel('x (\mum)'); ylabel('B_z (nT)'); title(sprintf('\\theta_s = %d\\circ', th(t)));
% end
% end
% 
% 
% function D = sot_generate_dataset(N, outFile, seed, P)
% % SOT_GENERATE_DATASET  Generate N random samples and save them to a .mat file.
% %   D = sot_generate_dataset(1000, 'sot_train.mat', 1)
% % Saved arrays (ny x nx x N, single precision):
% %   Bz_tilt, Bz_zero              noisy maps (input / benchmark partner)
% %   Bz_tilt_clean, Bz_zero_clean  noise-free maps
% %   m, mask                       true moment density and sample mask
% % plus a table-like struct 'labels' (one field per parameter, N x 1 each)
% % and a cell array 'bubbles' ([xb yb r aspect angle supp] per bubble).
% 
% if nargin < 3 || isempty(seed), seed = 0; end
% if nargin < 4 || isempty(P),    P = sot_default_params(); end
% rng(seed);
% 
% S = sot_make_sample(P);
% [ny, nx] = size(S.m);
% f = {'Bz_tilt','Bz_zero','Bz_tilt_clean','Bz_zero_clean','m','mask','mask_input'};
% for q = 1:numel(f), D.(f{q}) = zeros(ny, nx, N, 'single'); end
% ln = fieldnames(S.labels);
% for q = 1:numel(ln), D.labels.(ln{q}) = zeros(N,1); end
% D.bubbles = cell(N,1);
% D.x = S.x; D.y = S.y; D.params = P;
% 
% for i = 1:N
%     if i > 1, S = sot_make_sample(P); end
%     for q = 1:numel(f), D.(f{q})(:,:,i) = single(S.(f{q})); end
%     for q = 1:numel(ln), D.labels.(ln{q})(i) = S.labels.(ln{q}); end
%     D.bubbles{i} = S.bubbles;
%     if mod(i, 50) == 0, fprintf('%d / %d\n', i, N); end
% end
% 
% if nargin >= 2 && ~isempty(outFile)
%     if exist('OCTAVE_VERSION', 'builtin'), save(outFile, '-struct', 'D', '-v7');
%     else, save(outFile, '-struct', 'D', '-v7.3'); end
%     fprintf('saved %d samples to %s\n', N, outFile);
% end
% end

% %% SOT_SIMULATOR v2  Synthetic SQUID-on-tip maps of a thin magnetic film
% % Paste this whole file as  sot_simulator.m  and press Run (MATLAB R2021b+).
% % Units: lengths in um, fields in nT, moment density in uB/cm^2.
% clear; close all; clc;
% P = sot_default_params();
% 
% %% 1) checks: Biot-Savart, paper Fig. 3d-f, lock-in formula, film thickness
% sot_run_checks();
% 
% %% 2) look at random samples (new ones every run)
% rng('shuffle');                       % start the generator from the clock
% seeds = randi(2^31-1, 1, 3);          % three fresh random seeds
% for seed = seeds
%     rng(seed);
%     S = sot_make_sample(P);
%     sot_preview(S);
%     fprintf('preview seed = %d\n', seed);   % note it down to reproduce this sample
% end
% 
% %% 3) generate a dataset (uncomment when the maps look right)
% % D = sot_generate_dataset(1000, 'sot_train.mat', 1, P);
% 
% %% ======================= local functions =======================
% 
% function P = sot_default_params()
% % All tunable parameters. [min max] = drawn per sample (uniform, or
% % log-uniform where marked "log").
% 
% % ---- grids ----
% P.dx_sim   = 0.025;          % simulation pixel (um)
% P.dx_out   = 0.050;          % output pixel, as in the paper (um)
% P.fov      = [5.0 7.0];      % field of view [x y] (um)
% P.pad      = 4;              % padded grid = pad * fov (avoids FFT wrap-around)
% 
% % ---- sample outline ----
% P.shape_probs    = [0.2 0.3 0.3 0.2]; % probability of: strip, trapezoid, polygon, blob
% P.rot_deg        = [-180 180];  % in-plane rotation of the whole sample (deg)
% P.center_offset  = [-1.5 1.5];  % sample centre offset from FOV centre, per axis (um)
% P.W              = [1.5 3.5];   % strip / trapezoid width (um)
% P.trap_angle     = [5 30];      % trapezoid: angle of the slanted side (deg)
% P.poly_vertices  = [3 8];       % polygon: number of corners
% P.poly_radius    = [3 7];       % polygon / blob: centre-to-boundary distance (um)
% P.poly_aspect    = [0.3 1.0];   % polygon / blob: squeeze along one axis
% P.blob_harm      = 0.15;        % blob: max relative amplitude of each boundary harmonic
% P.fov_cover      = [0.2 0.85];  % accepted fraction of the FOV covered by the sample
% P.edge_rough_amp = [0.02 0.04]; % edge wiggle std (um)
% P.edge_rough_lc  = 0.3;         % edge wiggle correlation length (um)
% P.edge_width     = 0.02;        % Gaussian softening of the edge (um)
% 
% % ---- arms (Hall-bar arms made of the same film) ----
% P.arm_prob   = 0.4;          % probability that a sample has arms
% P.n_arms     = [1 3];        % number of arms
% P.arm_width  = [0.4 1.2];    % (um)
% P.arm_length = [2 6];        % (um)
% 
% % ---- film ----
% P.thickness  = [0.0003 0.02];% film thickness (um), log: 0.3 nm .. 20 nm
% 
% % ---- bubbles ----
% P.n_bubbles_max  = 5;        % number drawn uniformly from 0..max
% P.bubble_r       = [0.25 0.45]; % mean radius (um)
% P.bubble_margin  = 0.2;      % bubble centres may lie this far outside the FOV (um)
% P.bubble_aspect  = [1.0 1.4];% ellipse aspect ratio
% P.bubble_supp    = [0.0 0.2];% remaining m inside a bubble (fraction)
% P.bubble_edge    = 0.04;     % tanh half-width of bubble edge (um)
% P.bubble_gap     = 0.10;     % min gap bubble-bubble and bubble-edge (um)
% 
% % ---- switching participation (physics-based amplitude) ----
% % a(x,y) = (n_c(x,y) - n0)/n_ac : local transition density relative to the
% % parked density, in units of the a.c. wobble. Lock-in amplitude ~ sqrt(1-a^2).
% P.switch_offset   = [-0.5 0.5]; % mean of a
% P.switch_disorder = [0.1 0.6];  % std of a (disorder in the local transition point)
% P.switch_lc       = [0.2 1.0];  % correlation length of the disorder (um)
% 
% % ---- static moment-density texture (independent per component) ----
% P.moment_tex_std = [0 0.15];    % relative std
% P.moment_tex_lc  = [0.3 1.0];   % correlation length (um)
% 
% % ---- magnetization components ----
% P.sigma0      = [1e11 1e16]; % moment density of component 1 (uB/cm^2), log
% P.theta_deg   = [0 180];     % polar angle from +z (deg)
% P.phi_deg     = [-180 180];  % azimuth from +x (deg)
% P.p_two_comp  = 0.4;         % probability of a second component
% P.p_orbital   = 0.7;         % ... which is z-locked (orbital-like) with this probability
% P.comp2_ratio = [0.1 3];     % amplitude of component 2 relative to 1, log
% 
% % ---- extra source layer below the sample (e.g. graphite gate) ----
% P.p_extra      = 0.3;        % probability
% P.extra_depth  = [0.01 0.05];% distance below the sample (um)
% P.extra_ratio  = [0.05 0.5]; % amplitude relative to sigma0, log, random sign
% P.extra_W      = [3 10];     % width of its outline (um)
% P.extra_tex_lc = [1 3];      % correlation length of its 10% texture (um)
% 
% % ---- SOT ----
% P.h        = [0.15 0.22];    % SQUID-loop height above the film's top surface (um)
% P.d_tip    = [0.17 0.22];    % effective SQUID loop diameter (um), paper: 0.194
% 
% % ---- noise ----
% P.noise_frac   = [0.05 0.25];% total noise RMS / peak |Bz| of the field-along-z map
% P.w_white      = [0.5 0.8];  % fraction of noise variance that is white
% P.w_line       = [0.15 0.4]; % fraction that is per-line offsets (rest = drift)
% P.glitch_prob  = 0.03;       % probability a scan line is a glitch line
% P.glitch_amp   = 3;          % glitch offset in units of the line-noise std
% P.real_noise   = [];         % optional: real noise maps (nT), ny x nx x K, replaces synthetic noise
% 
% % ---- imperfections of the real measurement ----
% P.mask_shift_std = 0.08;     % misregistration of the known mask, std per axis (um)
% P.mask_blur      = [0 0.08]; % extra blur of the known mask (um)
% P.bg_offset      = 0.10;     % background offset std, fraction of peak |Bz|
% P.bg_slope       = 0.10;     % background plane: std of its change across the FOV, fraction of peak
% end
% 
% 
% function Bz = sot_forward(mx, my, mz, dx, h, d_tip, t)
% % SOT_FORWARD  Out-of-plane stray field of a magnetic film, as seen by a SOT.
% %   mx,my,mz : moment per area, integrated over the thickness (uB/cm^2)
% %   dx       : pixel size (um)
% %   h        : height of the SQUID loop above the film's top surface (um)
% %   d_tip    : effective SQUID loop diameter (um); 0 = point sensor
% %   t        : film thickness (um); 0 or omitted = infinitely thin sheet
% %   Bz       : field in nT
% % Fourier-space Biot-Savart:
% %   Bz(k) = (mu0/2) exp(-k h) [ k mz - i kx mx - i ky my ] * (1-exp(-k t))/(k t)
% % then averaged over the SQUID loop:  * 2 J1(k d/2)/(k d/2).
% % The grid is treated as periodic, so the caller must zero-pad.
% if nargin < 7, t = 0; end
% mu0  = 4*pi*1e-7;            % T m / A
% muB  = 9.2740100783e-24;     % A m^2
% conv = muB / 1e-4;           % uB/cm^2 -> A
% 
% [ny, nx] = size(mz);
% [KX, KY] = kgrid(ny, nx, dx);
% K = sqrt(KX.^2 + KY.^2);
% 
% B = exp(-K*h) .* (K.*fft2(mz) - 1i*KX.*fft2(mx) - 1i*KY.*fft2(my));
% if t > 0                                       % average over the film thickness
%     q = K*t;  F = ones(size(q));  nz = q > 0;
%     F(nz) = (1 - exp(-q(nz))) ./ q(nz);
%     B = B .* F;
% end
% if d_tip > 0                                   % average over the SQUID loop
%     q = K*d_tip/2;  T = ones(size(q));  nz = q > 0;
%     T(nz) = 2*besselj(1, q(nz)) ./ q(nz);
%     B = B .* T;
% end
% Bz = real(ifft2(B)) * (mu0/2) * conv * 1e6 * 1e9;   % k in 1/um -> 1/m ; T -> nT
% end
% 
% 
% function S = sot_make_sample(P)
% % SOT_MAKE_SAMPLE  One random synthetic sample.
% 
% % ---------- 1. padded simulation grid (um), centred on the FOV ----------
% dx  = P.dx_sim;
% nxF = round(P.fov(1)/dx);  nyF = round(P.fov(2)/dx);
% nx  = P.pad*nxF;           ny  = P.pad*nyF;
% x   = ((0:nx-1) - nx/2 + 0.5) * dx;
% y   = ((0:ny-1) - ny/2 + 0.5) * dx;
% [X, Y] = meshgrid(x, y);
% ix  = (nx-nxF)/2 + (1:nxF);  iy = (ny-nyF)/2 + (1:nyF);
% 
% % ---------- 2. sample outline (+ arms) and bubbles ----------
% [mask, L] = make_shape(X, Y, dx, ix, iy, P);
% [bfac, bub] = make_bubbles(X, Y, x, y, mask, P);
% 
% % ---------- 3. switching participation: which spots blink, and how strongly ----------
% L.switch_offset   = U(P.switch_offset);
% L.switch_disorder = U(P.switch_disorder);
% L.switch_lc       = U(P.switch_lc);
% a    = L.switch_offset + L.switch_disorder*smooth2d(ny, nx, dx, L.switch_lc);
% part = sqrt(max(1 - a.^2, 0));          % lock-in amplitude / its maximum (2/pi) M
% base = mask .* bfac .* part;
% 
% % ---------- 4. magnetization components ----------
% L.sigma0 = UL(P.sigma0);
% L.n_comp = 1 + (rand < P.p_two_comp);
% L.c1_theta = U(P.theta_deg);  L.c1_phi = U(P.phi_deg);  L.c1_ratio = 1;  L.c1_orbital = 0;
% L.c2_theta = NaN;  L.c2_phi = NaN;  L.c2_ratio = NaN;  L.c2_orbital = NaN;
% if L.n_comp == 2
%     L.c2_orbital = double(rand < P.p_orbital);
%     if L.c2_orbital, L.c2_theta = 180*(rand < 0.5);  L.c2_phi = 0;     % z-locked, either sign
%     else,            L.c2_theta = U(P.theta_deg);    L.c2_phi = U(P.phi_deg);
%     end
%     L.c2_ratio = UL(P.comp2_ratio);
% end
% mx = zeros(ny, nx);  my = mx;  mz = mx;  mz0 = mx;
% for c = 1:L.n_comp
%     th  = L.(sprintf('c%d_theta', c));  ph  = L.(sprintf('c%d_phi', c));
%     rat = L.(sprintf('c%d_ratio', c));  orb = L.(sprintf('c%d_orbital', c));
%     tex = max(1 + U(P.moment_tex_std)*smooth2d(ny, nx, dx, U(P.moment_tex_lc)), 0);
%     sc  = L.sigma0 * rat * base .* tex;            % this component's moment density
%     mx  = mx + sc*sind(th)*cosd(ph);
%     my  = my + sc*sind(th)*sind(ph);
%     mz  = mz + sc*cosd(th);
%     % partner map (applied field along +z): spins point up, orbital parts keep their sign
%     if orb, mz0 = mz0 + sc*cosd(th); else, mz0 = mz0 + sc; end
% end
% 
% % ---------- 5. stray field at the SOT ----------
% L.thickness = UL(P.thickness);
% L.h         = U(P.h);
% L.d_tip     = U(P.d_tip);
% Z  = zeros(ny, nx);
% Bt = sot_forward(mx, my, mz,  dx, L.h, L.d_tip, L.thickness);
% B0 = sot_forward(Z,  Z,  mz0, dx, L.h, L.d_tip, L.thickness);
% 
% % ---------- 6. extra source layer below the sample ----------
% L.extra = double(rand < P.p_extra);  L.extra_depth = NaN;  L.extra_ratio = NaN;
% Be = Z;
% if L.extra
%     Pe = P;  Pe.shape_probs = [0.5 0.5 0 0];  Pe.W = P.extra_W;
%     Pe.fov_cover = [0.3 1.0];  Pe.arm_prob = 0;
%     me = make_shape(X, Y, dx, ix, iy, Pe);
%     L.extra_depth = U(P.extra_depth);
%     L.extra_ratio = UL(P.extra_ratio) * sign(randn);
%     te = 1 + 0.1*smooth2d(ny, nx, dx, U(P.extra_tex_lc));
%     Be = sot_forward(Z, Z, L.sigma0*L.extra_ratio*me.*te, dx, ...
%                      L.h + L.thickness + L.extra_depth, L.d_tip, 0);
%     Bt = Bt + Be;  B0 = B0 + Be;
% end
% 
% % ---------- 7. crop to FOV and bin to output pixels ----------
% f   = round(P.dx_out/dx);
% Bt  = binavg(Bt(iy,ix), f);
% B0  = binavg(B0(iy,ix), f);
% Be  = binavg(Be(iy,ix), f);
% mvec = cat(3, binavg(mx(iy,ix), f), binavg(my(iy,ix), f), binavg(mz(iy,ix), f));
% mk  = binavg(mask(iy,ix), f);
% 
% % the mask as the experimentalist knows it: shifted and blurred
% [KX, KY] = kgrid(ny, nx, dx);
% L.mask_shift_x = P.mask_shift_std*randn;
% L.mask_shift_y = P.mask_shift_std*randn;
% L.mask_blur    = U(P.mask_blur);
% Mk = real(ifft2( fft2(mask) .* exp(-1i*(KX*L.mask_shift_x + KY*L.mask_shift_y)) ...
%                             .* exp(-(KX.^2 + KY.^2)*L.mask_blur^2/2) ));
% mki = min(max(binavg(Mk(iy,ix), f), 0), 1);
% 
% % ---------- 8. noise and background ----------
% pk = max(abs(B0(:)));
% L.noise_frac = U(P.noise_frac);
% if isempty(P.real_noise)
%     sig_n = L.noise_frac * pk;
%     Nt = make_noise(size(Bt), sig_n, P);
%     N0 = make_noise(size(B0), sig_n, P);
% else
%     Nt = real_noise_patch(size(Bt), P.real_noise);
%     N0 = real_noise_patch(size(B0), P.real_noise);
%     L.noise_frac = std(Nt(:))/pk;
% end
% L.noise_rms_nT = std(Nt(:));
% Nt = Nt + make_background(size(Bt), pk, P);
% N0 = N0 + make_background(size(B0), pk, P);
% 
% % ---------- 9. output ----------
% S.Bz_tilt       = Bt + Nt;     % network input: field at the sample's magnetization direction
% S.Bz_zero       = B0 + N0;     % partner map (field along z) for the two-map benchmark
% S.Bz_tilt_clean = Bt;
% S.Bz_zero_clean = B0;
% S.Bz_extra      = Be;          % clean contribution of the extra layer
% S.mvec          = mvec;        % true moment density vector (mx,my,mz), uB/cm^2
% S.mask          = mk;          % true outline
% S.mask_input    = mki;         % outline as known to the experimentalist
% S.x = ((0:size(mk,2)-1) - size(mk,2)/2 + 0.5)*P.dx_out;
% S.y = ((0:size(mk,1)-1) - size(mk,1)/2 + 0.5)*P.dx_out;
% L.n_bubbles = size(bub,1);
% S.bubbles   = bub;
% S.labels    = L;
% end
% 
% % ======================= building blocks =======================
% function [mask, L] = make_shape(X, Y, dx, ix, iy, P)
% % random sample outline (+ optional arms); redrawn until it crosses the FOV
% [ny, nx] = size(X);
% [KX, KY] = kgrid(ny, nx, dx);
% Rlim  = min(max(abs(X(:))), max(abs(Y(:)))) - 1 - max(abs(P.center_offset));
% types = {'strip','trapezoid','polygon','blob'};
% cp    = cumsum(P.shape_probs)/sum(P.shape_probs);
% for attempt = 1:50
%     t = find(rand <= cp, 1);
%     L.shape_type = t;                     % 1 strip, 2 trapezoid, 3 polygon, 4 blob
%     L.W = NaN; L.trap_angle = NaN; L.n_vertices = NaN;
%     switch types{t}
%         case 'strip'
%             L.W = U(P.W);
%             px = [-1 1 1 -1]*L.W/2;  py = [-1 -1 1 1]*Rlim;
%         case 'trapezoid'                  % one straight side, one slanted side
%             L.W = U(P.W);  L.trap_angle = U(P.trap_angle);
%             ta = tand(L.trap_angle);
%             yb = max(-Rlim, -(L.W - 0.5)/ta);
%             px = [-L.W/2, L.W/2 + yb*ta, L.W/2 + Rlim*ta, -L.W/2];
%             py = [yb yb Rlim Rlim];
%             if rand < 0.5, px = -px; end
%         case 'polygon'                    % random corners, can be concave
%             K   = randi(P.poly_vertices);  L.n_vertices = K;
%             ang = (0:K-1)*2*pi/K + (rand(1,K)-0.5)*0.8*2*pi/K;
%             rad = min(P.poly_radius(1) + diff(P.poly_radius)*rand(1,K), Rlim);
%             px  = rad.*cos(ang)*U(P.poly_aspect);  py = rad.*sin(ang);
%         case 'blob'                       % smooth amorphous outline
%             ang = linspace(0, 2*pi, 121);  ang(end) = [];
%             R0  = U(P.poly_radius);  R = R0*ones(size(ang));
%             for n = 2:5
%                 R = R + R0*P.blob_harm*rand*cos(n*ang + 2*pi*rand);
%             end
%             R   = min(R, Rlim);
%             px  = R.*cos(ang)*U(P.poly_aspect);  py = R.*sin(ang);
%     end
%     L.rot = U(P.rot_deg);
%     L.cx  = U(P.center_offset);  L.cy = U(P.center_offset);
%     qx = px*cosd(L.rot) - py*sind(L.rot) + L.cx;
%     qy = px*sind(L.rot) + py*cosd(L.rot) + L.cy;
%     L.edge_rough = U(P.edge_rough_amp);
%     Xw = X + L.edge_rough*smooth2d(ny, nx, dx, P.edge_rough_lc);
%     Yw = Y + L.edge_rough*smooth2d(ny, nx, dx, P.edge_rough_lc);
%     in = double(inpolygon(Xw, Yw, qx, qy));
%     % arms: rectangles starting inside the sample (in the FOV) and pointing outwards
%     L.n_arms = 0;
%     if rand < P.arm_prob
%         for q = 1:randi(P.n_arms)
%             cand = find(in(iy, ix) > 0);
%             if isempty(cand), break; end
%             [ii, jj] = ind2sub([numel(iy) numel(ix)], cand(randi(numel(cand))));
%             x0 = X(iy(ii), ix(jj));  y0 = Y(iy(ii), ix(jj));
%             al = 2*pi*rand;  aw = U(P.arm_width);  alen = U(P.arm_length);
%             u  =  (Xw - x0)*cos(al) + (Yw - y0)*sin(al);
%             v  = -(Xw - x0)*sin(al) + (Yw - y0)*cos(al);
%             in = max(in, double(u >= 0 & u <= alen & abs(v) <= aw/2));
%             L.n_arms = L.n_arms + 1;
%         end
%     end
%     in = in .* ((X - L.cx).^2 + (Y - L.cy).^2 < Rlim^2);   % finite sample
%     cover = mean(mean(in(iy, ix)));
%     if cover >= P.fov_cover(1) && cover <= P.fov_cover(2), break; end
% end
% L.fov_cover = cover;
% mask = real(ifft2(fft2(in) .* exp(-(KX.^2 + KY.^2)*P.edge_width^2/2)));
% mask = min(max(mask, 0), 1);
% end
% 
% function [bfac, bub] = make_bubbles(X, Y, x, y, mask, P)
% % elliptical regions where the magnetic phase is destroyed
% [ny, nx] = size(X);
% nb   = randi([0 P.n_bubbles_max]);
% bub  = zeros(0,6);                        % [xb yb r aspect angle supp]
% bfac = ones(ny, nx);
% xr   = P.fov(1)/2 + P.bubble_margin;
% yr   = P.fov(2)/2 + P.bubble_margin;
% ang  = linspace(0, 2*pi, 17);
% tries = 0;
% while size(bub,1) < nb && tries < 500
%     tries = tries + 1;
%     r  = U(P.bubble_r);  ar = U(P.bubble_aspect);
%     xb = (2*rand-1)*xr;  yb = (2*rand-1)*yr;
%     rc = r*sqrt(ar) + P.bubble_gap;       % the whole bubble + gap must be inside the sample
%     if min(mask_at(mask, x, y, [xb, xb + rc*cos(ang)], [yb, yb + rc*sin(ang)])) < 0.95
%         continue;
%     end
%     ok = true;
%     for q = 1:size(bub,1)
%         if hypot(xb-bub(q,1), yb-bub(q,2)) < r*sqrt(ar) + bub(q,3)*sqrt(bub(q,4)) + P.bubble_gap
%             ok = false; break;
%         end
%     end
%     if ~ok, continue; end
%     phi = pi*rand;  s = U(P.bubble_supp);
%     bub(end+1,:) = [xb yb r ar phi s]; %#ok<AGROW>
%     a = r*sqrt(ar);  b = r/sqrt(ar);
%     u =  (X-xb)*cos(phi) + (Y-yb)*sin(phi);
%     v = -(X-xb)*sin(phi) + (Y-yb)*cos(phi);
%     dist = (sqrt((u/a).^2 + (v/b).^2) - 1)*r;
%     bfac = bfac .* (s + (1-s)*0.5*(1 + tanh(dist/P.bubble_edge)));
% end
% end
% 
% % ======================= helper functions =======================
% function v = U(r)
% % uniform draw in [r(1) r(2)]; a scalar is returned unchanged
% if numel(r) == 1, v = r; else, v = r(1) + (r(2)-r(1))*rand; end
% end
% 
% function v = UL(r)
% % log-uniform draw in [r(1) r(2)]
% if numel(r) == 1, v = r; else, v = exp(log(r(1)) + (log(r(2))-log(r(1)))*rand); end
% end
% 
% function g = smooth2d(ny, nx, dx, lc)
% % 2D Gaussian random field, zero mean, unit std, correlation length lc
% [KX, KY] = kgrid(ny, nx, dx);
% g = real(ifft2(fft2(randn(ny,nx)) .* exp(-((KX.^2+KY.^2)*(lc/2)^2)/2)));
% g = (g - mean(g(:)))/std(g(:));
% end
% 
% function [KX, KY] = kgrid(ny, nx, dx)
% % angular wavevector grids (1/um) matching fft2 ordering
% kx = 2*pi*ifftshift((0:nx-1) - floor(nx/2))/(nx*dx);
% ky = 2*pi*ifftshift((0:ny-1) - floor(ny/2))/(ny*dx);
% [KX, KY] = meshgrid(kx, ky);
% end
% 
% function B = binavg(A, f)
% % average f x f blocks
% [ny, nx] = size(A);
% A = A(1:f*floor(ny/f), 1:f*floor(nx/f));
% B = reshape(mean(reshape(A, f, []), 1), floor(ny/f), []);
% B = reshape(mean(reshape(B', f, []), 1), floor(nx/f), [])';
% end
% 
% function v = mask_at(mask, x, y, xq, yq)
% % mask value at points (xq,yq), nearest pixel; points outside the grid count as 0
% j  = round((xq - x(1))/(x(2) - x(1))) + 1;
% i  = round((yq - y(1))/(y(2) - y(1))) + 1;
% v  = zeros(size(xq));
% ok = i >= 1 & i <= numel(y) & j >= 1 & j <= numel(x);
% v(ok) = mask(sub2ind(size(mask), i(ok), j(ok)));
% end
% 
% function N = make_noise(sz, sig, P)
% % white + per-line offsets (with glitches) + slow drift; fast axis = x
% ny = sz(1); nx = sz(2);
% ww = U(P.w_white);
% wl = min(U(P.w_line), 1 - ww - 0.02);
% wd = 1 - ww - wl;
% Nw = randn(ny, nx);
% lo = randn(ny, 1);
% g  = rand(ny,1) < P.glitch_prob;
% lo(g) = lo(g) + P.glitch_amp*sign(randn(nnz(g),1));
% lo = (lo - mean(lo))/std(lo);
% dr = cumsum(randn(ny,1));
% dr = dr - linspace(dr(1), dr(end), ny)';
% dr = (dr - mean(dr))/max(std(dr), eps);
% N = sig*( sqrt(ww)*Nw + sqrt(wl)*repmat(lo,1,nx) + sqrt(wd)*repmat(dr,1,nx) );
% end
% 
% function N = real_noise_patch(sz, bank)
% % random crop of a real noise map (nT); flips keep the scan-line structure
% A = bank(:,:,randi(size(bank,3)));
% if rand < 0.5, A = fliplr(A); end
% if rand < 0.5, A = flipud(A); end
% i0 = randi(size(A,1) - sz(1) + 1);  j0 = randi(size(A,2) - sz(2) + 1);
% N  = A(i0:i0+sz(1)-1, j0:j0+sz(2)-1);
% N  = N - mean(N(:));
% end
% 
% function G = make_background(sz, pk, P)
% % random constant offset + linear plane; slope = change across the whole FOV
% [YY, XX] = ndgrid(linspace(-0.5, 0.5, sz(1)), linspace(-0.5, 0.5, sz(2)));
% G = pk*( P.bg_offset*randn + P.bg_slope*randn*XX + P.bg_slope*randn*YY );
% end
% 
% % ======================= display, checks, dataset =======================
% function fig = sot_preview(S)
% % top row: true magnetization; bottom row: SOT maps
% L   = S.labels;
% fig = figure('Color','w','Position',[40 40 1500 760]);
% mag = sqrt(sum(S.mvec.^2, 3));  mm = max(mag(:));
% c   = max(abs(S.Bz_zero_clean(:)));
% pan = {mag/mm,             '|m| / max',                    [0 1];
%        S.mvec(:,:,1)/mm,   'm_x / max',                    [-1 1];
%        S.mvec(:,:,2)/mm,   'm_y / max',                    [-1 1];
%        S.mvec(:,:,3)/mm,   'm_z / max',                    [-1 1];
%        S.Bz_zero_clean,    'clean B_z, field along z (nT)', [-c c];
%        S.Bz_zero,          'noisy B_z, field along z (nT)', [-c c];
%        S.Bz_tilt_clean,    'clean B_z, tilted (nT)',        [-c c];
%        S.Bz_tilt,          'noisy B_z, tilted (nT)',        [-c c]};
% for p = 1:8
%     subplot(2,4,p);
%     imagesc(S.x, S.y, pan{p,1}, pan{p,3}); axis image; set(gca,'YDir','normal');
%     colormap(gca, jet(256)); colorbar; title(pan{p,2}); xlabel('x (\mum)');
%     if p == 1 || p == 5, ylabel('y (\mum)'); end
%     if p == 1, hold on; contour(S.x, S.y, S.mask_input, [0.5 0.5], 'w', 'LineWidth', 1); end
% end
% txt = sprintf(['shape %d, %d arms, %d bubbles, t = %.1f nm, h = %.0f nm, tip = %.0f nm, noise = %.0f%%   |   ' ...
%                'comp 1: \\theta = %.0f\\circ, \\phi = %.0f\\circ'], L.shape_type, L.n_arms, L.n_bubbles, ...
%                1e3*L.thickness, 1e3*L.h, 1e3*L.d_tip, 100*L.noise_frac, L.c1_theta, L.c1_phi);
% if L.n_comp == 2
%     if L.c2_orbital, ostr = ' (orbital)'; else, ostr = ''; end
%     txt = [txt sprintf('   |   comp 2: \\theta = %.0f\\circ, \\phi = %.0f\\circ, ratio %.2f%s', ...
%            L.c2_theta, L.c2_phi, L.c2_ratio, ostr)];
% end
% if L.extra, txt = [txt sprintf('   |   extra layer %.0f nm below, ratio %.2f', 1e3*L.extra_depth, L.extra_ratio)]; end
% annotation('textbox', [0 0.95 1 0.05], 'String', txt, 'EdgeColor', 'none', ...
%            'HorizontalAlignment', 'center', 'FontSize', 10);
% end
% 
% function sot_run_checks()
% %% ---- check 1: FFT vs direct dipole sum ----
% dx = 0.05; n = 240; h = 0.3;
% x = ((0:n-1) - n/2 + 0.5)*dx;  [X,Y] = meshgrid(x,x);
% src = (abs(X) < 1.5) & (abs(Y) < 1.0);
% rng(1);
% mz = 1e12*src.*(1 + 0.3*randn(n));  mx = 0.7e12*src.*(1 + 0.3*randn(n));  my = -0.4e12*src;
% Bfft = sot_forward(mx, my, mz, dx, h, 0);
% mu0 = 4*pi*1e-7; c = 9.2740100783e-24/1e-4;
% k = find(src); xs = X(k)*1e-6; ys = Y(k)*1e-6; dA = (dx*1e-6)^2;
% mxs = mx(k)*c*dA; mys = my(k)*c*dA; mzs = mz(k)*c*dA;
% ip = round(n/2) + (-50:10:50);
% Bdir = zeros(size(ip));
% for t = 1:numel(ip)
%     rx = X(n/2, ip(t))*1e-6 - xs; ry = Y(n/2, ip(t))*1e-6 - ys; rz = h*1e-6;
%     r  = sqrt(rx.^2 + ry.^2 + rz.^2);
%     mdr = mxs.*rx + mys.*ry + mzs.*rz;
%     Bdir(t) = mu0/(4*pi)*sum(3*mdr.*rz./r.^5 - mzs./r.^3)*1e9;
% end
% fprintf('Check 1: FFT vs direct dipole sum, max rel. diff = %.2f%%\n', ...
%         100*max(abs(Bfft(n/2, ip) - Bdir))/max(abs(Bdir)));
% 
% %% ---- check 2: paper Fig. 3d-f ----
% dx = 0.025; nx = 600; ny = 400;
% x = ((0:nx-1) - nx/2 + 0.5)*dx;
% strip = repmat(double(abs(x) < 1.3), ny, 1) * 0.63e12;
% figure('Color','w','Position',[50 50 1200 300]);
% th = [0 45 -45];
% for t = 1:3
%     B = sot_forward(strip*sind(th(t)), 0*strip, strip*cosd(th(t)), dx, 0.2, 0);
%     b = B(ny/2, :);
%     [bmax, imax] = max(b); [bmin, imin] = min(b);
%     fprintf('Check 2: theta_s = %+3d deg: max %+5.1f nT at x = %+5.2f um, min %+5.1f nT at x = %+5.2f um\n', ...
%             th(t), bmax, x(imax), bmin, x(imin));
%     subplot(1,3,t); plot(x, b, 'b', 'LineWidth', 1.5); hold on;
%     plot([-1.3 -1.3], [-55 60], 'k--', [1.3 1.3], [-55 60], 'k--');
%     xlim([-2.7 2.7]); ylim([-55 60]); grid on;
%     xlabel('x (\mum)'); ylabel('B_z (nT)'); title(sprintf('\\theta_s = %d\\circ', th(t)));
% end
% 
% %% ---- check 3: lock-in amplitude of a blinking magnet ----
% a  = 0.4;                                   % transition sits 0.4 n_ac above the parked density
% ph = linspace(0, 2*pi, 200001);  ph(end) = [];
% on = double(sin(ph) > a);                   % magnet "on" while the wobble is past the transition
% b1 = 2*mean(on .* sin(ph));                 % first harmonic = what the lock-in reports
% fprintf('Check 3: simulated lock-in %.4f vs formula (2/pi)sqrt(1-a^2) = %.4f\n', b1, 2/pi*sqrt(1-a^2));
% 
% %% ---- check 4: thick-film kernel vs a stack of 40 thin sheets ----
% t = 0.02;  Bthick = sot_forward(strip*sind(45), 0*strip, strip*cosd(45), dx, 0.15, 0, t);
% Bstack = 0*Bthick;
% for s = 1:40
%     Bstack = Bstack + sot_forward(strip*sind(45)/40, 0*strip, strip*cosd(45)/40, dx, 0.15 + (s-0.5)*t/40, 0);
% end
% fprintf('Check 4: thick film (20 nm) vs 40 stacked sheets, max rel. diff = %.3f%%\n', ...
%         100*max(abs(Bthick(:) - Bstack(:)))/max(abs(Bstack(:))));
% end
% 
% function D = sot_generate_dataset(N, outFile, seed, P)
% % SOT_GENERATE_DATASET  N random samples, saved to a .mat file.
% %   Image fields are stored as [size of one sample, N], single precision.
% if nargin < 3 || isempty(seed), seed = 0; end
% if nargin < 4 || isempty(P),    P = sot_default_params(); end
% rng(seed);
% S = sot_make_sample(P);
% f = {'Bz_tilt','Bz_zero','Bz_tilt_clean','Bz_zero_clean','Bz_extra','mvec','mask','mask_input'};
% for q = 1:numel(f)
%     sz.(f{q}) = size(S.(f{q}));
%     D.(f{q})  = zeros(numel(S.(f{q})), N, 'single');
% end
% ln = fieldnames(S.labels);
% for q = 1:numel(ln), D.labels.(ln{q}) = zeros(N,1); end
% D.bubbles = cell(N,1);
% D.x = S.x; D.y = S.y; D.params = P;
% for i = 1:N
%     if i > 1, S = sot_make_sample(P); end
%     for q = 1:numel(f), D.(f{q})(:,i) = single(S.(f{q})(:)); end
%     for q = 1:numel(ln), D.labels.(ln{q})(i) = S.labels.(ln{q}); end
%     D.bubbles{i} = S.bubbles;
%     if mod(i, 50) == 0, fprintf('%d / %d\n', i, N); end
% end
% for q = 1:numel(f), D.(f{q}) = reshape(D.(f{q}), [sz.(f{q}) N]); end
% if nargin >= 2 && ~isempty(outFile)
%     if exist('OCTAVE_VERSION', 'builtin'), save(outFile, '-struct', 'D', '-v7');
%     else, save(outFile, '-struct', 'D', '-v7.3'); end
%     fprintf('saved %d samples to %s\n', N, outFile);
% end
% end

%% SOT_SIMULATOR v7: simulator + CNNs + comparison with the paper's two-map method
% The networks need the Deep Learning Toolbox (MATLAB only); the simulator, the data
% generation and the paper's two-map method also run without it, and in GNU Octave.
% Units: lengths in um, fields in nT, moment density in uB/cm^2.
clear; close all; clc;
P = sot_default_params();
P.p_two_comp = 0;            % baseline: one magnetization direction per sample
G = grid_params();

%% 0) what to run
MODE     = 'evaluate';      % 'make_data' : create the new training file described in NEW_DATA
                             % 'evaluate'  : compare all saved networks with the paper's two-map method
                             % 'finetune'  : continue training a saved network with small steps
                             % 'train'     : train a new network from scratch
NET_TYPE = 'one_map';        % which network 'train'/'finetune' works on:
                             %   'one_map'     : one tilted map only (in-plane direction unknown)
                             %   'one_map_phi' : one tilted map + the known in-plane direction

% training data: ALL files listed here are used together
DATA_FILES = {'sot_trainset_20K.mat', 'sot_trainset_partial_10K.mat'};

% what MODE = 'make_data' creates (for a cluster: one job per file, each with its own
% file name and seed, then list all the files in DATA_FILES)
NEW_DATA.file      = 'sot_trainset_partial_10K.mat';
NEW_DATA.n_train   = 10000;
NEW_DATA.n_val     = 500;
NEW_DATA.p_partial = 0.5;    % half of the new maps show only part of the sample
NEW_DATA.seed      = 11;     % different from all other data sets

% test sets (generated once, then reused)
TEST_FILE         = 'sot_testset.mat';          % whole sample in the frame
N_TEST            = 1000;
TEST_FILE_PARTIAL = 'sot_testset_partial.mat';  % only part of the sample in the frame
N_TEST_PARTIAL    = 300;

RUN_CHECKS    = false;       % physics checks of the simulator
SHOW_PREVIEWS = false;       % plot a few random samples
if RUN_CHECKS, sot_run_checks(); end
if SHOW_PREVIEWS
    rng('shuffle');
    for seed = randi(2^31-1, 1, 3)
        rng(seed);  sot_preview(sot_make_sample(P));  fprintf('preview seed = %d\n', seed);
    end
end

%% 1) run
switch MODE
    case 'make_data'
        make_data_file(NEW_DATA, P, G);
    case 'evaluate'
        tests = {TEST_FILE,         N_TEST,          3, 0, 'whole sample in the frame';
                 TEST_FILE_PARTIAL, N_TEST_PARTIAL, 13, 1, 'only part of the sample in the frame'};
        net1 = load_best_network('one_map', false);   % [] if not trained yet: then only
        net2 = load_best_network('one_map_phi', false);%    the paper's method is evaluated
        for t = 1:size(tests, 1)
            Pt = P;  Pt.p_partial = tests{t,4};
            TS = get_test_set(tests{t,1}, tests{t,2}, tests{t,3}, Pt, G);
            fprintf('\n\n######## TEST SET %d: %s ########\n', t, tests{t,5});
            Yps = {};  names = {};
            if ~isempty(net1)
                Yps{end+1} = predict_dirs_tta(net1, TS.X);  names{end+1} = 'network, 1 map, averaged';
            end
            if ~isempty(net2)
                Yps{end+1} = predict_dirs_tta(net2, add_phi_channels(TS.X, TS.Y));
                names{end+1} = 'network, 1 map + known phi, averaged';
            end
            Yps{end+1} = TS.Ypaper;  names{end+1} = 'paper, 2 maps + known phi';
            report_errors(Yps, names, TS.Y, TS.labels);
            set(gcf, 'Name', tests{t,5});
            plot_residuals(Yps, names, TS.Y, tests{t,5});
        end
    case {'train', 'finetune'}
        require_dl_toolbox();
        [Xtr, Ytr, Xva, Yva] = load_training_data(DATA_FILES);
        T = train_settings(MODE, NET_TYPE);
        if strcmp(MODE, 'train')
            Gn = G;  if T.use_phi, Gn.nch = G.nch + 2; end     % two extra input channels
            net = build_cnn(Gn);
        else
            net = load_best_network(NET_TYPE, true);
        end
        net = train_cnn(net, Xtr, Ytr, Xva, Yva, T);
        if T.use_phi, Xva = add_phi_channels(Xva, Yva); end
        show_examples(net, Xva, Yva, 8);
    otherwise
        error('MODE must be ''make_data'', ''evaluate'', ''finetune'' or ''train''.');
end

%% ======================= local functions =======================

function P = sot_default_params()
% All tunable parameters. [min max] = drawn per sample (uniform, or
% log-uniform where marked "log").

% ---- grids and frame ----
P.dx_sim       = 0.025;      % simulation pixel (um)
P.dx_out       = [0.05 0.075 0.10]; % pixel sizes an experimentalist would pick (um)
P.dx_out_prob  = [0.5 0.3 0.2];     % ... and how often
P.max_pixels   = [10000 25000]; % scan-time budget: pixels per map (1.2 s each -> ~3.3-8.3 h)
P.pixel_time   = 1.2;        % integration time per pixel (s), as in the paper
P.fov_max      = [9 10];     % largest frame [x y] anyone would scan (um)
P.fov_min      = [5.0 7.0];  % smallest frame [x y] (um); it grows to fit the sample
P.frame_margin = 1.0;        % minimum empty space between sample and frame edge (um)
P.frame_jitter = 0.5;        % random shift of the sample inside the frame (um)
P.pad          = 2;          % padded grid = pad * simulated area (avoids FFT wrap-around)
P.p_partial    = 0;          % probability that the frame shows only PART of the sample
P.partial_cover = [0.2 0.85];% ... then the sample covers this fraction of the frame
P.partial_scale = [1 2];     % ... and the sample may be this much larger than usual

% ---- sample outline ----
P.shape_probs    = [0.2 0.3 0.3 0.2]; % probability of: strip, trapezoid, polygon, blob
P.rot_deg        = [-180 180];  % in-plane rotation of the whole sample (deg)
P.W              = [1.5 3.0];   % strip / trapezoid width (um)
P.strip_length   = [3 6];      % strip / trapezoid length (um)
P.trap_angle     = [5 30];      % trapezoid: angle of the slanted side (deg)
P.poly_vertices  = [3 8];       % polygon: number of corners
P.poly_radius    = [1.5 3.0];     % polygon / blob: centre-to-boundary distance (um)
P.poly_aspect    = [0.3 1.0];   % polygon / blob: squeeze along one axis
P.blob_harm      = 0.15;        % blob: max relative amplitude of each boundary harmonic
P.edge_rough_amp = [0.02 0.04]; % edge wiggle std (um)
P.edge_rough_lc  = 0.3;         % edge wiggle correlation length (um)
P.edge_width     = 0.02;        % Gaussian softening of the edge (um)

% ---- arms (Hall-bar arms made of the same film) ----
P.arm_prob   = 0.4;          % probability that a sample has arms
P.n_arms     = [1 3];        % number of arms
P.arm_width  = [0.4 1.2];    % (um)
P.arm_length = [1 2.5];        % (um)

% ---- film ----
P.thickness  = [0.0003 0.02];% film thickness (um), log: 0.3 nm .. 20 nm

% ---- bubbles ----
P.bubble_density = [0 0.15]; % bubbles per um^2 of sample
P.n_bubbles_cap  = 30;       % at most this many
P.bubble_r       = [0.25 0.45]; % mean radius (um)
P.bubble_aspect  = [1.0 1.4];% ellipse aspect ratio
P.bubble_supp    = [0.0 0.2];% remaining m inside a bubble (fraction)
P.bubble_edge    = 0.04;     % tanh half-width of bubble edge (um)
P.bubble_gap     = 0.10;     % min gap bubble-bubble and bubble-edge (um)

% ---- switching participation (physics-based amplitude) ----
% a(x,y) = (n_c(x,y) - n0)/n_ac : local transition density relative to the
% parked density, in units of the a.c. wobble. Lock-in amplitude ~ sqrt(1-a^2).
P.switch_offset   = [-0.5 0.5]; % mean of a
P.switch_disorder = [0.1 0.6];  % std of a (disorder in the local transition point)
P.switch_lc       = [0.2 1.0];  % correlation length of the disorder (um)

% ---- static moment-density texture (independent per component) ----
P.moment_tex_std = [0 0.15];    % relative std
P.moment_tex_lc  = [0.3 1.0];   % correlation length (um)

% ---- magnetization components ----
P.sigma0      = [1e11 1e16]; % moment density of component 1 (uB/cm^2), log
P.theta_deg   = [0 180];     % polar angle from +z (deg)
P.phi_deg     = [-180 180];  % azimuth from +x (deg)
P.p_two_comp  = 0.4;         % probability of a second component
P.p_orbital   = 0.7;         % ... which is z-locked (orbital-like) with this probability
P.comp2_ratio = [0.1 3];     % amplitude of component 2 relative to 1, log

% ---- extra source layer below the sample (e.g. graphite gate) ----
P.p_extra         = 0.3;     % probability
P.extra_halfplane = 0.5;     % probability that it is one edge crossing the frame (else a stripe)
P.extra_depth     = [0.01 0.05]; % distance below the sample (um)
P.extra_ratio     = [0.05 0.5];  % amplitude relative to sigma0, log, random sign
P.extra_W         = [3 10];  % stripe width (um)
P.extra_tex_lc    = [1 3];   % correlation length of its 10% texture (um)

% ---- SOT ----
P.h        = [0.15 0.22];    % SQUID-loop height above the film's top surface (um)
P.d_tip    = [0.17 0.22];    % effective SQUID loop diameter (um), paper: 0.194

% ---- noise ----
P.noise_frac   = [0.05 0.25];% total noise RMS / peak |Bz| of the field-along-z map
P.w_white      = [0.5 0.8];  % fraction of noise variance that is white
P.w_line       = [0.15 0.4]; % fraction that is per-line offsets (rest = drift)
P.glitch_prob  = 0.03;       % probability a scan line is a glitch line
P.glitch_amp   = 3;          % glitch offset in units of the line-noise std
P.real_noise   = [];         % optional: real noise maps (nT), ny x nx x K, replaces synthetic noise

% ---- imperfections of the real measurement ----
P.mask_shift_std = 0.08;     % misregistration of the known mask, std per axis (um)
P.mask_blur      = [0 0.08]; % extra blur of the known mask (um)
P.bg_offset      = 0.10;     % background offset std, fraction of peak |Bz|
P.bg_slope       = 0.10;     % background plane: std of its change across the frame, fraction of peak
end

function Bz = sot_forward(mx, my, mz, dx, h, d_tip, t)
% SOT_FORWARD  Out-of-plane stray field of a magnetic film, as seen by a SOT.
%   mx,my,mz : moment per area, integrated over the thickness (uB/cm^2)
%   dx       : pixel size (um)
%   h        : height of the SQUID loop above the film's top surface (um)
%   d_tip    : effective SQUID loop diameter (um); 0 = point sensor
%   t        : film thickness (um); 0 or omitted = infinitely thin sheet
%   Bz       : field in nT
% Fourier-space Biot-Savart:
%   Bz(k) = (mu0/2) exp(-k h) [ k mz - i kx mx - i ky my ] * (1-exp(-k t))/(k t)
% then averaged over the SQUID loop:  * 2 J1(k d/2)/(k d/2).
% The grid is treated as periodic, so the caller must zero-pad.
if nargin < 7, t = 0; end
mu0  = 4*pi*1e-7;            % T m / A
muB  = 9.2740100783e-24;     % A m^2
conv = muB / 1e-4;           % uB/cm^2 -> A

[ny, nx] = size(mz);
[KX, KY] = kgrid(ny, nx, dx);
K = sqrt(KX.^2 + KY.^2);

B = exp(-K*h) .* (K.*fft2(mz) - 1i*KX.*fft2(mx) - 1i*KY.*fft2(my));
if t > 0                                       % average over the film thickness
    q = K*t;  F = ones(size(q));  nz = q > 0;
    F(nz) = (1 - exp(-q(nz))) ./ q(nz);
    B = B .* F;
end
if d_tip > 0                                   % average over the SQUID loop
    q = K*d_tip/2;  T = ones(size(q));  nz = q > 0;
    T(nz) = 2*besselj(1, q(nz)) ./ q(nz);
    B = B .* T;
end
Bz = real(ifft2(B)) * (mu0/2) * conv * 1e6 * 1e9;   % k in 1/um -> 1/m ; T -> nT
end

function S = sot_make_sample(P)
% SOT_MAKE_SAMPLE  One random synthetic sample, fully inside its scan frame.
% ---------- 1. outline, the scan frame, and the pixel size ----------
partial = rand < P.p_partial;              % frame shows only part of the sample?
for attempt = 1:100                        % redraw outlines that would need too large a scan
    [polys, L] = make_outline(P);
    sc = 1;
    if partial                             % partial scans: the sample may be larger
        sc = U(P.partial_scale);
        for q = 1:numel(polys), polys{q} = polys{q}*sc; end
    end
    allv = vertcat(polys{:});
    bmin = min(allv, [], 1) - 0.1;         % 0.1 um extra for the edge roughness
    bmax = max(allv, [], 1) + 0.1;
    need = bmax - bmin + 2*P.frame_margin + 2*P.frame_jitter;
    if partial || all(need <= P.fov_max), break; end
end
if partial
    % a frame between the smallest frame and the sample size, placed so that it shows
    % part of the sample (covering partial_cover of the frame) but not all of it
    fov = P.fov_min + (min(P.fov_max, max(P.fov_min, need)) - P.fov_min).*rand(1, 2);
    [gx, gy] = meshgrid(linspace(-0.5, 0.5, 40)*fov(1), linspace(-0.5, 0.5, 40)*fov(2));
    partial = false;
    for tries = 1:300
        ctr = bmin + (bmax - bmin).*rand(1, 2);
        in  = false(size(gx));
        for q = 1:numel(polys)
            in = in | inpolygon(gx + ctr(1), gy + ctr(2), polys{q}(:,1), polys{q}(:,2));
        end
        whole = all(bmin >= ctr - fov/2) && all(bmax <= ctr + fov/2);
        if ~whole && mean(in(:)) >= P.partial_cover(1) && mean(in(:)) <= P.partial_cover(2)
            partial = true;  break;
        end
    end
end
if ~partial                                % whole sample in view, with a margin
    fov = max(P.fov_min, need);
    ctr = (bmin + bmax)/2 + P.frame_jitter*(2*rand(1,2) - 1);
end
L.partial = double(partial);  L.scale = sc;
% pixel size: the experimentalist's choice, made coarser if the scan-time budget requires it
L.max_pixels = U(P.max_pixels);
dxo = P.dx_out(find(rand <= cumsum(P.dx_out_prob)/sum(P.dx_out_prob), 1));
dxo = max(dxo, sqrt(prod(fov)/L.max_pixels));
dxo = P.dx_sim*ceil(dxo/P.dx_sim - 1e-9);  % whole number of simulation pixels
L.dx_out = dxo;
fov  = ceil(fov/(2*dxo))*2*dxo;            % whole number of output pixel pairs
for q = 1:numel(polys), polys{q} = polys{q} - ctr; end     % frame centre = origin
L.fov_x = fov(1);  L.fov_y = fov(2);
L.n_pixels   = round(prod(fov)/dxo^2);
L.scan_hours = L.n_pixels*P.pixel_time/3600;

% ---------- 2. padded simulation grid: covers the frame AND the whole sample ----------
dx   = P.dx_sim;
half = max(fov/2, max(abs([bmin; bmax] - ctr), [], 1) + 0.5);   % half-size of the simulated area
nxF  = round(fov(1)/dx);  nyF = round(fov(2)/dx);
nx   = P.pad*2*ceil(half(1)/dx);  ny = P.pad*2*ceil(half(2)/dx);
x    = ((0:nx-1) - nx/2 + 0.5) * dx;
y    = ((0:ny-1) - ny/2 + 0.5) * dx;
[X, Y] = meshgrid(x, y);
ix  = (nx-nxF)/2 + (1:nxF);  iy = (ny-nyF)/2 + (1:nyF);

% ---------- 3. mask with rough soft edges, and bubbles ----------
[mask, L] = rasterize(X, Y, dx, polys, P, L);
L.area = sum(mask(:))*dx^2;
L.fov_cover = mean(mean(mask(iy, ix) > 0.5));     % fraction of the frame covered by the sample
[bfac, bub] = make_bubbles(X, Y, x, y, mask, 2*half, L.area, P);

% ---------- 4. switching participation: which spots blink, and how strongly ----------
L.switch_offset   = U(P.switch_offset);
L.switch_disorder = U(P.switch_disorder);
L.switch_lc       = U(P.switch_lc);
a    = L.switch_offset + L.switch_disorder*smooth2d(ny, nx, dx, L.switch_lc);
part = sqrt(max(1 - a.^2, 0));          % lock-in amplitude / its maximum (2/pi) M
base = mask .* bfac .* part;

% ---------- 5. magnetization components ----------
L.sigma0 = UL(P.sigma0);
L.n_comp = 1 + (rand < P.p_two_comp);
L.c1_theta = U(P.theta_deg);  L.c1_phi = U(P.phi_deg);  L.c1_ratio = 1;  L.c1_orbital = 0;
L.c2_theta = NaN;  L.c2_phi = NaN;  L.c2_ratio = NaN;  L.c2_orbital = NaN;
if L.n_comp == 2
    L.c2_orbital = double(rand < P.p_orbital);
    if L.c2_orbital, L.c2_theta = 180*(rand < 0.5);  L.c2_phi = 0;     % z-locked, either sign
    else,            L.c2_theta = U(P.theta_deg);    L.c2_phi = U(P.phi_deg);
    end
    L.c2_ratio = UL(P.comp2_ratio);
end
mx = zeros(ny, nx);  my = mx;  mz = mx;  mz0 = mx;
for c = 1:L.n_comp
    th  = L.(sprintf('c%d_theta', c));  ph  = L.(sprintf('c%d_phi', c));
    rat = L.(sprintf('c%d_ratio', c));  orb = L.(sprintf('c%d_orbital', c));
    tex = max(1 + U(P.moment_tex_std)*smooth2d(ny, nx, dx, U(P.moment_tex_lc)), 0);
    sc  = L.sigma0 * rat * base .* tex;            % this component's moment density
    mx  = mx + sc*sind(th)*cosd(ph);
    my  = my + sc*sind(th)*sind(ph);
    mz  = mz + sc*cosd(th);
    % partner map (applied field along +z): spins point up, orbital parts keep their sign
    if orb, mz0 = mz0 + sc*cosd(th); else, mz0 = mz0 + sc; end
end

% ---------- 6. stray field at the SOT: tilted, opposite tilt, field along z ----------
% opposite tilt = in-plane part reversed, out-of-plane part kept (like theta_B -> -theta_B)
L.thickness = UL(P.thickness);
L.h         = U(P.h);
L.d_tip     = U(P.d_tip);
Z  = zeros(ny, nx);
Bt = sot_forward( mx,  my, mz,  dx, L.h, L.d_tip, L.thickness);
Bo = sot_forward(-mx, -my, mz,  dx, L.h, L.d_tip, L.thickness);
B0 = sot_forward( Z,   Z,  mz0, dx, L.h, L.d_tip, L.thickness);

% ---------- 7. extra source layer below the sample ----------
L.extra = double(rand < P.p_extra);  L.extra_type = NaN;  L.extra_depth = NaN;  L.extra_ratio = NaN;
Be = Z;
if L.extra
    ae = 360*rand;  ox = (2*rand-1)*fov(1)/2;  oy = (2*rand-1)*fov(2)/2;
    u  = -(X - ox)*sind(ae) + (Y - oy)*cosd(ae);     % distance from a line through the frame
    if rand < P.extra_halfplane
        L.extra_type = 1;  me = double(u < 0);         % one gate edge crossing the frame
    else
        L.extra_type = 2;  me = double(abs(u) < U(P.extra_W)/2);   % a gate stripe
    end
    win = (abs(X) < max(x) - 1) & (abs(Y) < max(y) - 1);           % stay inside the padded grid
    me  = soften(me .* win, dx, 0.05);
    L.extra_depth = U(P.extra_depth);
    L.extra_ratio = UL(P.extra_ratio) * sign(randn);
    te = 1 + 0.1*smooth2d(ny, nx, dx, U(P.extra_tex_lc));
    Be = sot_forward(Z, Z, L.sigma0*L.extra_ratio*me.*te, dx, ...
                     L.h + L.thickness + L.extra_depth, L.d_tip, 0);
    Bt = Bt + Be;  Bo = Bo + Be;  B0 = B0 + Be;
end

% ---------- 8. crop to the frame and bin to output pixels ----------
f   = round(L.dx_out/dx);
Bt  = binavg(Bt(iy,ix), f);
Bo  = binavg(Bo(iy,ix), f);
B0  = binavg(B0(iy,ix), f);
Be  = binavg(Be(iy,ix), f);
mvec = cat(3, binavg(mx(iy,ix), f), binavg(my(iy,ix), f), binavg(mz(iy,ix), f));
mk  = binavg(mask(iy,ix), f);

% the mask as the experimentalist knows it: shifted and blurred
[KX, KY] = kgrid(ny, nx, dx);
L.mask_shift_x = P.mask_shift_std*randn;
L.mask_shift_y = P.mask_shift_std*randn;
L.mask_blur    = U(P.mask_blur);
Mk = real(ifft2( fft2(mask) .* exp(-1i*(KX*L.mask_shift_x + KY*L.mask_shift_y)) ...
                            .* exp(-(KX.^2 + KY.^2)*L.mask_blur^2/2) ));
mki = min(max(binavg(Mk(iy,ix), f), 0), 1);

% ---------- 9. noise and background (independent for each scan) ----------
pk = max(abs(B0(:)));
L.noise_frac = U(P.noise_frac);
if isempty(P.real_noise)
    sig_n = L.noise_frac * pk;
    Nt = make_noise(size(Bt), sig_n, P);
    No = make_noise(size(Bo), sig_n, P);
    N0 = make_noise(size(B0), sig_n, P);
else
    Nt = real_noise_patch(size(Bt), P.real_noise);
    No = real_noise_patch(size(Bo), P.real_noise);
    N0 = real_noise_patch(size(B0), P.real_noise);
    L.noise_frac = std(Nt(:))/pk;
end
L.noise_rms_nT = std(Nt(:));
Nt = Nt + make_background(size(Bt), pk, P);
No = No + make_background(size(Bo), pk, P);
N0 = N0 + make_background(size(B0), pk, P);

% ---------- 10. output ----------
S.Bz_tilt       = Bt + Nt;     % network input: field at the sample's magnetization direction
S.Bz_opp        = Bo + No;     % same sample, opposite tilt (in-plane part reversed)
S.Bz_zero       = B0 + N0;     % partner map (field along z) for the two-map benchmark
S.Bz_tilt_clean = Bt;
S.Bz_opp_clean  = Bo;
S.Bz_zero_clean = B0;
S.Bz_extra      = Be;          % clean contribution of the extra layer
S.mvec          = mvec;        % true moment density vector (mx,my,mz), uB/cm^2
S.mask          = mk;          % true outline
S.mask_input    = mki;         % outline as known to the experimentalist
S.x = ((0:size(mk,2)-1) - size(mk,2)/2 + 0.5)*L.dx_out;
S.y = ((0:size(mk,1)-1) - size(mk,1)/2 + 0.5)*L.dx_out;
L.n_bubbles = size(bub,1);
S.bubbles   = bub;
S.labels    = L;
end

% ======================= building blocks =======================
function [polys, L] = make_outline(P)
% random sample outline (+ arms) as polygons in um, centred near the origin
types = {'strip','trapezoid','polygon','blob'};
cp    = cumsum(P.shape_probs)/sum(P.shape_probs);
t     = find(rand <= cp, 1);
L.shape_type = t;                     % 1 strip, 2 trapezoid, 3 polygon, 4 blob
L.W = NaN; L.length = NaN; L.trap_angle = NaN; L.n_vertices = NaN;
switch types{t}
    case 'strip'
        L.W = U(P.W);  L.length = U(P.strip_length);
        px = [-1 1 1 -1]*L.W/2;  py = [-1 -1 1 1]*L.length/2;
    case 'trapezoid'                  % one straight side, one slanted side
        L.W = U(P.W);  L.length = U(P.strip_length);  L.trap_angle = U(P.trap_angle);
        ta = tand(L.trap_angle);  yt = L.length/2;
        yb = max(-yt, -(L.W - 0.5)/ta);               % keep the width >= 0.5 um
        px = [-L.W/2, L.W/2 + yb*ta, L.W/2 + yt*ta, -L.W/2];
        py = [yb yb yt yt];
        if rand < 0.5, px = -px; end
    case 'polygon'                    % random corners, can be concave
        K   = randi(P.poly_vertices);  L.n_vertices = K;
        ang = (0:K-1)*2*pi/K + (rand(1,K)-0.5)*0.8*2*pi/K;
        rad = P.poly_radius(1) + diff(P.poly_radius)*rand(1,K);
        px  = rad.*cos(ang)*U(P.poly_aspect);  py = rad.*sin(ang);
    case 'blob'                       % smooth amorphous outline
        ang = linspace(0, 2*pi, 121);  ang(end) = [];
        R0  = U(P.poly_radius);  R = R0*ones(size(ang));
        for n = 2:5
            R = R + R0*P.blob_harm*rand*cos(n*ang + 2*pi*rand);
        end
        px  = R.*cos(ang)*U(P.poly_aspect);  py = R.*sin(ang);
end
L.rot = U(P.rot_deg);
qx = px*cosd(L.rot) - py*sind(L.rot);
qy = px*sind(L.rot) + py*cosd(L.rot);
polys = {[qx(:) qy(:)]};
% arms: rectangles starting at a random point inside the sample, pointing outwards
L.n_arms = 0;
if rand < P.arm_prob
    lo = min(polys{1}, [], 1);  hi = max(polys{1}, [], 1);
    for q = 1:randi(P.n_arms)
        for tries = 1:200                       % random point inside the outline
            p0 = lo + (hi - lo).*rand(1,2);
            if inpolygon(p0(1), p0(2), qx, qy), break; end
        end
        al = 360*rand;  aw = U(P.arm_width);  alen = U(P.arm_length);
        R  = [cosd(al) -sind(al); sind(al) cosd(al)];
        rect = (R*[0 alen alen 0; -aw/2 -aw/2 aw/2 aw/2])' + p0;
        polys{end+1} = rect; %#ok<AGROW>
        L.n_arms = L.n_arms + 1;
    end
end
end

function [mask, L] = rasterize(X, Y, dx, polys, P, L)
% fill the polygons on the grid, with rough edges (coordinate warping) and soft edges
[ny, nx] = size(X);
L.edge_rough = U(P.edge_rough_amp);
Xw = X + L.edge_rough*smooth2d(ny, nx, dx, P.edge_rough_lc);
Yw = Y + L.edge_rough*smooth2d(ny, nx, dx, P.edge_rough_lc);
in = zeros(ny, nx);
for q = 1:numel(polys)
    in = max(in, double(inpolygon(Xw, Yw, polys{q}(:,1), polys{q}(:,2))));
end
mask = soften(in, dx, P.edge_width);
end

function A = soften(A, dx, w)
% Gaussian blur of width w (um), clipped to [0,1]
[ny, nx] = size(A);
[KX, KY] = kgrid(ny, nx, dx);
A = min(max(real(ifft2(fft2(A) .* exp(-(KX.^2 + KY.^2)*w^2/2))), 0), 1);
end

function [bfac, bub] = make_bubbles(X, Y, x, y, mask, fov, area, P)
% elliptical regions where the magnetic phase is destroyed
[ny, nx] = size(X);
nb   = min(round(U(P.bubble_density)*area), P.n_bubbles_cap);
bub  = zeros(0,6);                        % [xb yb r aspect angle supp]
bfac = ones(ny, nx);
ang  = linspace(0, 2*pi, 17);
tries = 0;
while size(bub,1) < nb && tries < 50*max(nb,1)
    tries = tries + 1;
    r  = U(P.bubble_r);  ar = U(P.bubble_aspect);
    xb = (2*rand-1)*fov(1)/2;  yb = (2*rand-1)*fov(2)/2;
    rc = r*sqrt(ar) + P.bubble_gap;       % the whole bubble + gap must be inside the sample
    if min(mask_at(mask, x, y, [xb, xb + rc*cos(ang)], [yb, yb + rc*sin(ang)])) < 0.95
        continue;
    end
    ok = true;
    for q = 1:size(bub,1)
        if hypot(xb-bub(q,1), yb-bub(q,2)) < r*sqrt(ar) + bub(q,3)*sqrt(bub(q,4)) + P.bubble_gap
            ok = false; break;
        end
    end
    if ~ok, continue; end
    phi = pi*rand;  s = U(P.bubble_supp);
    bub(end+1,:) = [xb yb r ar phi s]; %#ok<AGROW>
    a = r*sqrt(ar);  b = r/sqrt(ar);
    u =  (X-xb)*cos(phi) + (Y-yb)*sin(phi);
    v = -(X-xb)*sin(phi) + (Y-yb)*cos(phi);
    dist = (sqrt((u/a).^2 + (v/b).^2) - 1)*r;
    bfac = bfac .* (s + (1-s)*0.5*(1 + tanh(dist/P.bubble_edge)));
end
end

% ======================= helper functions =======================
function v = U(r)
% uniform draw in [r(1) r(2)]; a scalar is returned unchanged
if numel(r) == 1, v = r; else, v = r(1) + (r(2)-r(1))*rand; end
end

function v = UL(r)
% log-uniform draw in [r(1) r(2)]
if numel(r) == 1, v = r; else, v = exp(log(r(1)) + (log(r(2))-log(r(1)))*rand); end
end

function g = smooth2d(ny, nx, dx, lc)
% 2D Gaussian random field, zero mean, unit std, correlation length lc
[KX, KY] = kgrid(ny, nx, dx);
g = real(ifft2(fft2(randn(ny,nx)) .* exp(-((KX.^2+KY.^2)*(lc/2)^2)/2)));
g = (g - mean(g(:)))/std(g(:));
end

function [KX, KY] = kgrid(ny, nx, dx)
% angular wavevector grids (1/um) matching fft2 ordering
kx = 2*pi*ifftshift((0:nx-1) - floor(nx/2))/(nx*dx);
ky = 2*pi*ifftshift((0:ny-1) - floor(ny/2))/(ny*dx);
[KX, KY] = meshgrid(kx, ky);
end

function B = binavg(A, f)
% average f x f blocks
[ny, nx] = size(A);
A = A(1:f*floor(ny/f), 1:f*floor(nx/f));
B = reshape(mean(reshape(A, f, []), 1), floor(ny/f), []);
B = reshape(mean(reshape(B', f, []), 1), floor(nx/f), [])';
end

function v = mask_at(mask, x, y, xq, yq)
% mask value at points (xq,yq), nearest pixel; points outside the grid count as 0
j  = round((xq - x(1))/(x(2) - x(1))) + 1;
i  = round((yq - y(1))/(y(2) - y(1))) + 1;
v  = zeros(size(xq));
ok = i >= 1 & i <= numel(y) & j >= 1 & j <= numel(x);
v(ok) = mask(sub2ind(size(mask), i(ok), j(ok)));
end

function N = make_noise(sz, sig, P)
% white + per-line offsets (with glitches) + slow drift; fast axis = x
ny = sz(1); nx = sz(2);
ww = U(P.w_white);
wl = min(U(P.w_line), 1 - ww - 0.02);
wd = 1 - ww - wl;
Nw = randn(ny, nx);
lo = randn(ny, 1);
g  = rand(ny,1) < P.glitch_prob;
lo(g) = lo(g) + P.glitch_amp*sign(randn(nnz(g),1));
lo = (lo - mean(lo))/std(lo);
dr = cumsum(randn(ny,1));
dr = dr - linspace(dr(1), dr(end), ny)';
dr = (dr - mean(dr))/max(std(dr), eps);
N = sig*( sqrt(ww)*Nw + sqrt(wl)*repmat(lo,1,nx) + sqrt(wd)*repmat(dr,1,nx) );
end

function N = real_noise_patch(sz, bank)
% random crop of a real noise map (nT); flips keep the scan-line structure;
% a map smaller than the frame is tiled
A = bank(:,:,randi(size(bank,3)));
if rand < 0.5, A = fliplr(A); end
if rand < 0.5, A = flipud(A); end
A  = repmat(A, max(ceil(sz ./ size(A)), 1));
i0 = randi(size(A,1) - sz(1) + 1);  j0 = randi(size(A,2) - sz(2) + 1);
N  = A(i0:i0+sz(1)-1, j0:j0+sz(2)-1);
N  = N - mean(N(:));
end

function G = make_background(sz, pk, P)
% random constant offset + linear plane; slope = change across the whole FOV
[YY, XX] = ndgrid(linspace(-0.5, 0.5, sz(1)), linspace(-0.5, 0.5, sz(2)));
G = pk*( P.bg_offset*randn + P.bg_slope*randn*XX + P.bg_slope*randn*YY );
end

% ======================= display, checks, dataset =======================
function fig = sot_preview(S)
% row 1: true magnetization; row 2: field along z and tilted;
% row 3: opposite tilt and a line cut through all three
L   = S.labels;
fig = figure('Color','w','Position',[30 30 1400 980]);
mag = sqrt(sum(S.mvec.^2, 3));  mm = max(mag(:));
c   = max(abs([S.Bz_zero_clean(:); S.Bz_tilt_clean(:); S.Bz_opp_clean(:)]));

% cut through the frame centre along the in-plane direction of component 1
ph = L.c1_phi;
smax = min(S.x(end)/max(abs(cosd(ph)), 1e-9), S.y(end)/max(abs(sind(ph)), 1e-9));
s  = linspace(-smax, smax, 400);
xq = s*cosd(ph);  yq = s*sind(ph);

pan = {mag/mm,           '|m| / max',                         [0 1];
       S.mvec(:,:,1)/mm, 'm_x / max',                         [-1 1];
       S.mvec(:,:,2)/mm, 'm_y / max',                         [-1 1];
       S.mvec(:,:,3)/mm, 'm_z / max',                         [-1 1];
       S.Bz_zero_clean,  'clean B_z, field along z (nT)',     [-c c];
       S.Bz_zero,        'noisy B_z, field along z (nT)',     [-c c];
       S.Bz_tilt_clean,  'clean B_z, tilted (nT)',            [-c c];
       S.Bz_tilt,        'noisy B_z, tilted (nT)',            [-c c];
       S.Bz_opp_clean,   'clean B_z, opposite tilt (nT)',     [-c c];
       S.Bz_opp,         'noisy B_z, opposite tilt (nT)',     [-c c]};
for p = 1:10
    subplot(3,4,p);
    imagesc(S.x, S.y, pan{p,1}, pan{p,3}); axis image; set(gca,'YDir','normal');
    colormap(gca, jet(256)); colorbar; title(pan{p,2}); xlabel('x (\mum)');
    if any(p == [1 5 9]), ylabel('y (\mum)'); end
    if p == 1
        hold on; contour(S.x, S.y, S.mask_input, [0.5 0.5], 'w', 'LineWidth', 1);
        plot(xq, yq, 'w--', 'LineWidth', 1);
    end
end

subplot(3,4,[11 12]);
v0 = interp2(S.x, S.y, S.Bz_zero_clean, xq, yq);
vt = interp2(S.x, S.y, S.Bz_tilt_clean, xq, yq);
vo = interp2(S.x, S.y, S.Bz_opp_clean,  xq, yq);
vm = interp2(S.x, S.y, S.mask,          xq, yq);
area(s, vm*c, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none'); hold on;   % where the sample is
plot(s, v0, 'k', s, vt, 'r', s, vo, 'b', 'LineWidth', 1.3);
grid on; xlim([s(1) s(end)]);
xlabel('position along the dashed line (\mum)'); ylabel('B_z (nT)');
legend('sample (grey)', 'field along z', 'tilted', 'opposite tilt', 'Location', 'eastoutside');
title(sprintf('clean cut along the in-plane direction of comp 1 (\\phi = %.0f\\circ)', ph));

txt = sprintf(['shape %d, %d arms, %d bubbles, frame %.1f x %.1f um, pixel %.0f nm (%d px, %.1f h scan), ' ...
               't = %.1f nm, h = %.0f nm, tip = %.0f nm, noise = %.0f%%   |   ' ...
               'comp 1: \\theta = %.0f\\circ, \\phi = %.0f\\circ (opposite: \\phi = %.0f\\circ)'], ...
               L.shape_type, L.n_arms, L.n_bubbles, L.fov_x, L.fov_y, 1e3*L.dx_out, L.n_pixels, ...
               L.scan_hours, 1e3*L.thickness, 1e3*L.h, 1e3*L.d_tip, 100*L.noise_frac, ...
               L.c1_theta, L.c1_phi, mod(L.c1_phi + 360, 360) - 180);
if L.n_comp == 2
    if L.c2_orbital, ostr = ' (orbital)'; else, ostr = ''; end
    txt = [txt sprintf('   |   comp 2: \\theta = %.0f\\circ, \\phi = %.0f\\circ, ratio %.2f%s', ...
           L.c2_theta, L.c2_phi, L.c2_ratio, ostr)];
end
if L.extra, txt = [txt sprintf('   |   extra layer %.0f nm below, ratio %.2f', 1e3*L.extra_depth, L.extra_ratio)]; end
annotation('textbox', [0 0.955 1 0.045], 'String', txt, 'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', 'FontSize', 9);
end

function sot_run_checks()
%% ---- check 1: FFT vs direct dipole sum ----
dx = 0.05; n = 240; h = 0.3;
x = ((0:n-1) - n/2 + 0.5)*dx;  [X,Y] = meshgrid(x,x);
src = (abs(X) < 1.5) & (abs(Y) < 1.0);
rng(1);
mz = 1e12*src.*(1 + 0.3*randn(n));  mx = 0.7e12*src.*(1 + 0.3*randn(n));  my = -0.4e12*src;
Bfft = sot_forward(mx, my, mz, dx, h, 0);
mu0 = 4*pi*1e-7; c = 9.2740100783e-24/1e-4;
k = find(src); xs = X(k)*1e-6; ys = Y(k)*1e-6; dA = (dx*1e-6)^2;
mxs = mx(k)*c*dA; mys = my(k)*c*dA; mzs = mz(k)*c*dA;
ip = round(n/2) + (-50:10:50);
Bdir = zeros(size(ip));
for t = 1:numel(ip)
    rx = X(n/2, ip(t))*1e-6 - xs; ry = Y(n/2, ip(t))*1e-6 - ys; rz = h*1e-6;
    r  = sqrt(rx.^2 + ry.^2 + rz.^2);
    mdr = mxs.*rx + mys.*ry + mzs.*rz;
    Bdir(t) = mu0/(4*pi)*sum(3*mdr.*rz./r.^5 - mzs./r.^3)*1e9;
end
fprintf('Check 1: FFT vs direct dipole sum, max rel. diff = %.2f%%\n', ...
        100*max(abs(Bfft(n/2, ip) - Bdir))/max(abs(Bdir)));

%% ---- check 2: paper Fig. 3d-f ----
dx = 0.025; nx = 600; ny = 400;
x = ((0:nx-1) - nx/2 + 0.5)*dx;
strip = repmat(double(abs(x) < 1.3), ny, 1) * 0.63e12;
figure('Color','w','Position',[50 50 1200 300]);
th = [0 45 -45];
for t = 1:3
    B = sot_forward(strip*sind(th(t)), 0*strip, strip*cosd(th(t)), dx, 0.2, 0);
    b = B(ny/2, :);
    [bmax, imax] = max(b); [bmin, imin] = min(b);
    fprintf('Check 2: theta_s = %+3d deg: max %+5.1f nT at x = %+5.2f um, min %+5.1f nT at x = %+5.2f um\n', ...
            th(t), bmax, x(imax), bmin, x(imin));
    subplot(1,3,t); plot(x, b, 'b', 'LineWidth', 1.5); hold on;
    plot([-1.3 -1.3], [-55 60], 'k--', [1.3 1.3], [-55 60], 'k--');
    xlim([-2.7 2.7]); ylim([-55 60]); grid on;
    xlabel('x (\mum)'); ylabel('B_z (nT)'); title(sprintf('\\theta_s = %d\\circ', th(t)));
end

%% ---- check 3: lock-in amplitude of a blinking magnet ----
a  = 0.4;                                   % transition sits 0.4 n_ac above the parked density
ph = linspace(0, 2*pi, 200001);  ph(end) = [];
on = double(sin(ph) > a);                   % magnet "on" while the wobble is past the transition
b1 = 2*mean(on .* sin(ph));                 % first harmonic = what the lock-in reports
fprintf('Check 3: simulated lock-in %.4f vs formula (2/pi)sqrt(1-a^2) = %.4f\n', b1, 2/pi*sqrt(1-a^2));

%% ---- check 4: thick-film kernel vs a stack of 40 thin sheets ----
t = 0.02;  Bthick = sot_forward(strip*sind(45), 0*strip, strip*cosd(45), dx, 0.15, 0, t);
Bstack = 0*Bthick;
for s = 1:40
    Bstack = Bstack + sot_forward(strip*sind(45)/40, 0*strip, strip*cosd(45)/40, dx, 0.15 + (s-0.5)*t/40, 0);
end
fprintf('Check 4: thick film (20 nm) vs 40 stacked sheets, max rel. diff = %.3f%%\n', ...
        100*max(abs(Bthick(:) - Bstack(:)))/max(abs(Bstack(:))));
end

function D = sot_generate_dataset(N, outFile, seed, P)
% SOT_GENERATE_DATASET  N random samples, saved to a .mat file.
%   Frames differ in size, so every image field is a cell array: D.Bz_tilt{i}, ...
if nargin < 3 || isempty(seed), seed = 0; end
if nargin < 4 || isempty(P),    P = sot_default_params(); end
rng(seed);
f = {'Bz_tilt','Bz_opp','Bz_zero','Bz_tilt_clean','Bz_opp_clean','Bz_zero_clean', ...
     'Bz_extra','mvec','mask','mask_input'};
for q = 1:numel(f), D.(f{q}) = cell(N,1); end
D.x = cell(N,1);  D.y = cell(N,1);  D.bubbles = cell(N,1);  D.params = P;
for i = 1:N
    S = sot_make_sample(P);
    if i == 1
        ln = fieldnames(S.labels);
        for q = 1:numel(ln), D.labels.(ln{q}) = zeros(N,1); end
    end
    for q = 1:numel(f),  D.(f{q}){i} = single(S.(f{q})); end
    for q = 1:numel(ln), D.labels.(ln{q})(i) = S.labels.(ln{q}); end
    D.x{i} = S.x;  D.y{i} = S.y;  D.bubbles{i} = S.bubbles;
    if mod(i, 50) == 0, fprintf('%d / %d\n', i, N); end
end
if nargin >= 2 && ~isempty(outFile)
    save(outFile, '-struct', 'D', mat_format());
    fprintf('saved %d samples to %s\n', N, outFile);
end
end


% ======================= neural network: data =======================
function G = grid_params()
% fixed physical grid the network sees: 128 x 128 pixels of 80 nm (10.24 um square)
G.n          = 128;
G.dx         = 0.08;
G.x          = ((0:G.n-1) - G.n/2 + 0.5)*G.dx;
G.nch        = 6;       % Bz, known mask, valid frame, h, tip, pixel size
G.h_err      = 0.015;   % error on the h you would "know" (um)
G.tip_err    = 0.010;   % error on the tip size you would "know" (um)
G.disp_every = 100;     % progress message every this many generated samples
end

function [X, Y] = make_training_set(N, seed, P, G)
% N simulated samples -> network inputs X (n x n x nch x N) and unit-vector labels Y (3 x N)
rng(seed);
X = zeros(G.n, G.n, G.nch, N, 'single');
Y = zeros(3, N, 'single');
[XG, YG] = meshgrid(G.x, G.x);
t0 = tic;
for i = 1:N
    S = sot_make_sample(P);
    X(:,:,:,i) = to_network_input(S, XG, YG, G);
    L = S.labels;
    Y(:,i) = [sind(L.c1_theta)*cosd(L.c1_phi); sind(L.c1_theta)*sind(L.c1_phi); cosd(L.c1_theta)];
    if mod(i, G.disp_every) == 0 || i == N
        el = toc(t0);
        fprintf('generated %d of %d samples | %.1f min elapsed, ~%.1f min left\n', ...
                i, N, el/60, el/i*(N-i)/60);
    end
end
end

function Xi = to_network_input(S, XG, YG, G)
% resample one sample onto the fixed grid and stack the input channels
L     = S.labels;
valid = double(abs(XG) <= S.x(end) + 1e-6 & abs(YG) <= S.y(end) + 1e-6);   % scanned frame
v     = valid > 0;
B     = interp2(S.x, S.y, S.Bz_tilt,    XG, YG, 'linear', 0);
M     = interp2(S.x, S.y, S.mask_input, XG, YG, 'linear', 0);
B     = B - median(B(v));                   % remove the offset
B     = B / max(std(B(v)), eps);            % amplitude does not matter for the direction
B(~v) = 0;
h     = L.h     + G.h_err*randn;            % what you would believe h is
d     = L.d_tip + G.tip_err*randn;          % what you would believe the tip size is
on    = ones(G.n, 'single');
Xi    = single(cat(3, B, M, valid, on*(h - 0.185)/0.035, on*(d - 0.195)/0.025, ...
                   on*(L.dx_out - 0.075)/0.025));
end

function [X, Y] = apply_sym(X, Y, k, mir)
% rotate the whole sample by k*90 deg, then mirror x if mir. The moment Y (3 x N) and the
% known-direction channels (7 = cos phi, 8 = sin phi, if present) transform with it.
for r = 1:k
    X = flip(permute(X, [2 1 3 4]), 2);                 % rotate the maps by +90 deg
    Y = [-Y(2,:); Y(1,:); Y(3,:)];                      % (mx,my) -> (-my,mx)
    if size(X, 3) >= 8, X(:,:,7:8,:) = cat(3, -X(:,:,8,:), X(:,:,7,:)); end
end
if mir
    X = flip(X, 2);  Y(1,:) = -Y(1,:);                  % x -> -x, mx -> -mx
    if size(X, 3) >= 8, X(:,:,7,:) = -X(:,:,7,:); end
end
end

function [X, Y] = augment(X, Y)
% random symmetry of the physics for a training batch
[X, Y] = apply_sym(X, Y, randi([0 3]), rand < 0.5);
end

function X = add_phi_channels(X, Y)
% the in-plane direction of the moment as two constant channels (cos phi, sin phi).
% Use only when it is known, e.g. in 1/2M where the spin follows the applied field.
ph = atan2(Y(2,:), Y(1,:));
c  = reshape(cos(ph), 1, 1, 1, []);  s = reshape(sin(ph), 1, 1, 1, []);
X  = cat(3, X, repmat(c, size(X,1), size(X,2)), repmat(s, size(X,1), size(X,2)));
end

function Yp = predict_dirs_tta(net, X, bs)
% prediction averaged over the 8 symmetric copies of each map (4 rotations x mirror):
% each copy's answer is transformed back before averaging
if nargin < 3, bs = 64; end
N  = size(X, 4);  Yp = zeros(3, N, 'single');
for k = 0:3
    for mir = 0:1
        Xt = apply_sym(X, zeros(3, N, 'single'), k, mir);
        Q  = predict_dirs(net, Xt, bs);
        if mir, Q(1,:) = -Q(1,:); end                   % undo the mirror
        for r = 1:k, Q = [Q(2,:); -Q(1,:); Q(3,:)]; end % undo the rotations
        Yp = Yp + Q;
    end
end
Yp = Yp ./ sqrt(sum(Yp.^2, 1));
end

% ======================= neural network: model and training =======================
function net = build_cnn(G)
% VGG-style CNN: 4 stages of (conv-BN-ReLU) x2 + pooling, global pooling, 2 dense layers
w = [32 64 128 256];
layers = imageInputLayer([G.n G.n G.nch], 'Normalization', 'none', 'Name', 'in');
for s = 1:numel(w)
    for r = 1:2
        layers = [layers
            convolution2dLayer(3, w(s), 'Padding', 'same', 'Name', sprintf('conv%d_%d', s, r))
            batchNormalizationLayer('Name', sprintf('bn%d_%d', s, r))
            reluLayer('Name', sprintf('relu%d_%d', s, r))]; %#ok<AGROW>
    end
    layers = [layers; maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool%d', s))]; %#ok<AGROW>
end
layers = [layers
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    fullyConnectedLayer(3, 'Name', 'out')];       % raw (mx,my,mz); normalized in the loss
net = dlnetwork(layerGraph(layers));
fprintf('network: %d layers, %.2f M learnable parameters\n', numel(layers), ...
        sum(cellfun(@numel, net.Learnables.Value))/1e6);
end
function net = train_cnn(net, Xtr, Ytr, Xva, Yva, T)
% Adam training with a smooth (cosine) learning-rate schedule. After every epoch the
% network is checkpointed (a crash loses at most one epoch; rerunning resumes) and the
% best network so far is kept in T.best_file. With T.use_phi the known in-plane
% direction is added as two input channels.
N  = size(Xtr, 4);  nb = floor(N/T.batch);
if T.use_phi, Xva = add_phi_channels(Xva, Yva); end
useGPU = canUseGPU();
if useGPU, fprintf('training on the GPU\n'); else, fprintf('training on the CPU (slower)\n'); end
avgG = [];  avgSqG = [];  it = 0;  ep0 = 1;  best = inf;
if exist(T.best_file, 'file'), Q = load(T.best_file, 'best'); best = Q.best; end
if exist(T.ckpt_file, 'file')
    Q = load(T.ckpt_file);
    net = Q.net;  avgG = Q.avgG;  avgSqG = Q.avgSqG;  it = Q.it;  ep0 = Q.ep + 1;
    fprintf('resuming from %s after epoch %d of %d\n', T.ckpt_file, Q.ep, T.epochs);
elseif T.check_start
    err = loss_angle(predict_dirs(net, Xva, T.batch), Yva, phi_weight(Yva, T));
    fprintf('starting network: validation median error %.1f deg (phi suppressed near 90 deg)\n', median(err));
    if median(err) < best, best = median(err);  ep = 0;  save(T.best_file, 'net', 'best', 'ep'); end
end
t0 = tic;
for ep = ep0:T.epochs
    lr   = T.lr_min + 0.5*(T.lr - T.lr_min)*(1 + cos(pi*(ep-1)/max(T.epochs-1, 1)));
    perm = randperm(N);
    for b = 1:nb
        it = it + 1;
        ii = perm((b-1)*T.batch + (1:T.batch));
        Xb = Xtr(:,:,:,ii);
        if T.use_phi, Xb = add_phi_channels(Xb, Ytr(:,ii)); end
        [Xb, Yb] = augment(Xb, Ytr(:,ii));
        if useGPU, Xb = gpuArray(Xb);  Yb = gpuArray(Yb); end
        W = phi_weight(Yb, T);                          % weight of the phi error per sample
        if useGPU, W = gpuArray(W); end
        [loss, grad, state] = dlfeval(@model_loss, net, dlarray(Xb, 'SSCB'), dlarray(Yb, 'CB'), W);
        net.State = state;
        [net, avgG, avgSqG] = adamupdate(net, grad, avgG, avgSqG, it, lr);
        if mod(b, T.disp_every) == 0 || b == nb
            fprintf('epoch %2d/%d | trained on %6d of %d samples this epoch | loss %.4f | lr %.1e | %.1f min\n', ...
                    ep, T.epochs, b*T.batch, N, double(gather(extractdata(loss))), lr, toc(t0)/60);
        end
    end
    Ypv = predict_dirs(net, Xva, T.batch);
    e3d = angle_err(Ypv, Yva);                          % plain 3D error, for information
    err = loss_angle(Ypv, Yva, phi_weight(Yva, T));     % the error the loss sees: used for 'best'
    nv  = numel(err);
    fprintf(['epoch %2d VALIDATION: within 5 deg: %d of %d (%.1f%%) | within 10 deg: %d of %d (%.1f%%) | ' ...
             'median error %.1f deg (plain 3D: %.1f deg)\n'], ep, sum(err < 5), nv, 100*mean(err < 5), ...
             sum(err < 10), nv, 100*mean(err < 10), median(err), median(e3d));
    if median(err) < best
        best = median(err);
        save(T.best_file, 'net', 'best', 'ep');
        fprintf('   -> best network so far, saved to %s\n', T.best_file);
    end
    save(T.ckpt_file, 'net', 'avgG', 'avgSqG', 'it', 'ep');
end
fprintf('training finished: best validation median error %.1f deg (network in %s)\n', best, T.best_file);
end

function [loss, grad, state] = model_loss(net, X, Y, W)
% 3D unit-vector loss with the phi part weighted by W (1 x N), see dir_loss
[Yp, state] = forward(net, X);
Yp   = Yp ./ sqrt(sum(Yp.^2, 1) + 1e-6);
loss = sum(dir_loss(Yp, Y, W), 2) / size(Y, 2);
grad = dlgradient(loss, net.Learnables);
end

function L = dir_loss(Yp, Y, W)
% per-sample loss (1 x N) between unit vectors Yp and Y (3 x N). The squared distance
% splits exactly into a theta part and a phi part:
%   |Yp - Y|^2 = 2(1 - cos dtheta) + 2 sin(theta_p) sin(theta_t) (1 - cos dphi)
% The phi part (already zero at the poles) is multiplied by W; W = 1 is the plain 3D loss.
rp   = sqrt(Yp(1,:).^2 + Yp(2,:).^2 + 1e-12);         % sin(theta) of the prediction
rt   = sqrt(Y(1,:).^2 + Y(2,:).^2);                   % sin(theta) of the truth
Lth  = 1 - Yp(3,:).*Y(3,:) - rp.*rt;                  % 1 - cos(dtheta)
Lphi = rp.*rt - Yp(1,:).*Y(1,:) - Yp(2,:).*Y(2,:);    % sin sin (1 - cos dphi)
L    = 2*(Lth + W.*Lphi);
end

function W = phi_weight(Y, T)
% weight of the phi error vs the TRUE polar angle: W = 1 - exp(-(|theta - c|/w)^p),
% T.phi_supp = [c w p]. Default [90 5 4]: W = 0 at 90 deg, 0.03 at 88, 0.63 at 85,
% 0.98 at 83, 1 below 80 deg. w = 0 switches the suppression off (plain 3D loss).
if T.phi_supp(2) <= 0, W = ones(1, size(Y, 2), 'like', Y); return; end
th = acosd(max(min(Y(3,:), 1), -1));
W  = 1 - exp(-(abs(th - T.phi_supp(1))/T.phi_supp(2)).^T.phi_supp(3));
end

function err = loss_angle(Yp, Y, W)
% the angle (deg) whose plain 3D loss equals the weighted loss: the error the training sees
err = acosd(max(min(1 - dir_loss(Yp, Y, W)/2, 1), -1));
end

function Yp = predict_dirs(net, X, bs)
% predicted unit vectors (3 x N) for inputs X, in batches
if nargin < 3, bs = 64; end
N  = size(X, 4);  Yp = zeros(3, N, 'single');  useGPU = canUseGPU();
for s = 1:bs:N
    ii = s:min(s+bs-1, N);
    Xb = X(:,:,:,ii);
    if useGPU, Xb = gpuArray(Xb); end
    Yp(:,ii) = gather(extractdata(predict(net, dlarray(Xb, 'SSCB'))));
end
Yp = Yp ./ sqrt(sum(Yp.^2, 1));
end

function err = angle_err(Yp, Y)
% angle (deg) between predicted and true directions
err = acosd(min(max(sum(Yp .* Y, 1), -1), 1));
end

function show_examples(net, X, Y, k)
% print true vs predicted angles for the first k samples
Yp = predict_dirs(net, X(:,:,:,1:k));
for i = 1:k
    tt = acosd(Y(3,i));   pt = atan2d(Y(2,i),  Y(1,i));
    tp = acosd(Yp(3,i));  pp = atan2d(Yp(2,i), Yp(1,i));
    fprintf('example %d: true theta %5.1f phi %6.1f | predicted theta %5.1f phi %6.1f | error %.1f deg\n', ...
            i, tt, pt, tp, pp, angle_err(Yp(:,i), Y(:,i)));
end
end


% ======================= benchmark: the paper's two-map method =======================
function v = dir_vec(theta, phi)
% unit vector of polar angle theta (from +z) and azimuth phi (from +x), degrees
v = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
end

function v = two_map_method(S, h, d, phi)
% The paper's procedure (Methods, "Experimental determination of isospin tilt angle"):
%  1) reconstruct m(x,y) from the field-along-z map, assuming the moment points along z
%  2) tilt it towards the KNOWN in-plane direction phi (in the paper: the applied-field azimuth)
%  3) find the tilt that best reproduces the tilted map.
% With a free amplitude, offset and plane, step 3 is one linear least-squares fit:
%   Bz_tilt ~ a*F_z + b*F_inplane + c0 + cx*x + cy*y,  tilt = atan2(b, a).
dx   = S.x(2) - S.x(1);
supp = support_from_mask(S.mask_input, dx);
m    = reconstruct_mz(double(S.Bz_zero), supp, dx, h, d);
[F1, F2] = fields_from_m(m, dx, h, d, phi);
[ny, nx] = size(S.Bz_tilt);
[XX, YY] = meshgrid(linspace(-1, 1, nx), linspace(-1, 1, ny));
c  = [F1(:) F2(:) ones(ny*nx, 1) XX(:) YY(:)] \ double(S.Bz_tilt(:));
th = atan2(c(2), c(1));                       % tilt from +z towards phi (rad)
v  = [sin(th)*cosd(phi); sin(th)*sind(phi); cos(th)];
end

function supp = support_from_mask(mask, dx)
% where m may be non-zero: the known mask, widened by 150 nm for its misregistration
r  = max(1, round(0.15/dx));
[u, w] = meshgrid(-r:r);
supp = conv2(double(mask > 0.3), double(u.^2 + w.^2 <= r^2), 'same') > 0;
end

function [Kz, Kp] = map_kernels(Ny, Nx, dx, h, d, phi)
% Fourier kernels (normalized to max 1) for a moment along z and in-plane along phi
[KX, KY] = kgrid(Ny, Nx, dx);  K = sqrt(KX.^2 + KY.^2);
T = ones(size(K));  q = K*d/2;  nz = q > 0;
T(nz) = 2*besselj(1, q(nz)) ./ q(nz);
E  = exp(-K*h) .* T;
Kz = E .* K;  sc = max(Kz(:));  Kz = Kz/sc;
Kp = E .* (-1i) .* (KX*cosd(phi) + KY*sind(phi)) / sc;
end

function m = reconstruct_mz(B, supp, dx, h, d)
% Regularized least squares, solved with conjugate gradients (CGLS):
%   min || field(m) + offset + plane - B ||^2 + lam*||grad m||^2,   m = 0 outside supp
[ny, nx] = size(B);
C.ny = ny;  C.nx = nx;  C.Ny = 2*ny;  C.Nx = 2*nx;            % zero-padded grid
C.ry = floor(ny/2) + (1:ny);  C.rx = floor(nx/2) + (1:nx);
C.Kz = map_kernels(C.Ny, C.Nx, dx, h, d, 0);
C.sidx = find(supp);  C.ns = numel(C.sidx);
[XX, YY] = meshgrid(linspace(-1, 1, nx), linspace(-1, 1, ny));
C.Bk = [ones(ny*nx, 1) XX(:) YY(:)] / sqrt(ny*nx);          % offset + plane basis
C.sl = sqrt(0.03);                                         % sqrt of the smoothness weight
B  = (B - median(B(:))) / max(std(B(:)), eps);
xv = zeros(C.ns + 3, 1);
r  = [B(:); zeros(ny*(nx-1) + (ny-1)*nx, 1)];
s  = cgls_AT(r, C);  p = s;  gam = s'*s;
for it = 1:150
    q  = cgls_A(p, C);  al = gam/max(q'*q, eps);
    xv = xv + al*p;  r = r - al*q;
    s  = cgls_AT(r, C);  gn = s'*s;
    p  = s + (gn/max(gam, eps))*p;  gam = gn;
end
m = zeros(ny, nx);  m(C.sidx) = xv(1:C.ns);
end

function out = cgls_A(xv, C)
% forward operator: [field of m + offset + plane ; smoothness terms]
Mf = zeros(C.ny, C.nx);  Mf(C.sidx) = xv(1:C.ns);
Mp = zeros(C.Ny, C.Nx);  Mp(C.ry, C.rx) = Mf;
F  = real(ifft2(C.Kz .* fft2(Mp)));
dd = F(C.ry, C.rx) + reshape(C.Bk*xv(C.ns+1:end), C.ny, C.nx);
gx = Mf(:, 2:end) - Mf(:, 1:end-1);  gy = Mf(2:end, :) - Mf(1:end-1, :);
out = [dd(:); C.sl*gx(:); C.sl*gy(:)];
end

function out = cgls_AT(r, C)
% transpose of cgls_A
n1 = C.ny*C.nx;  n2 = C.ny*(C.nx-1);
Rd = reshape(r(1:n1), C.ny, C.nx);
Gx = reshape(r(n1+1:n1+n2), C.ny, C.nx-1);  Gy = reshape(r(n1+n2+1:end), C.ny-1, C.nx);
Rp = zeros(C.Ny, C.Nx);  Rp(C.ry, C.rx) = Rd;
G  = real(ifft2(C.Kz .* fft2(Rp)));  gm = G(C.ry, C.rx);
Ar = zeros(C.ny, C.nx);
Ar(:, 1:end-1) = Ar(:, 1:end-1) - Gx;  Ar(:, 2:end) = Ar(:, 2:end) + Gx;
Ar(1:end-1, :) = Ar(1:end-1, :) - Gy;  Ar(2:end, :) = Ar(2:end, :) + Gy;
gm  = gm + C.sl*Ar;
out = [gm(C.sidx); C.Bk'*Rd(:)];
end

function [F1, F2] = fields_from_m(m, dx, h, d, phi)
% field of m pointing along z (F1) and in-plane along phi (F2), same normalization
[ny, nx] = size(m);  Ny = 2*ny;  Nx = 2*nx;
ry = floor(ny/2) + (1:ny);  rx = floor(nx/2) + (1:nx);
[Kz, Kp] = map_kernels(Ny, Nx, dx, h, d, phi);
Mp = zeros(Ny, Nx);  Mp(ry, rx) = m;  Mk = fft2(Mp);
F1 = real(ifft2(Kz .* Mk));  F1 = F1(ry, rx);
F2 = real(ifft2(Kp .* Mk));  F2 = F2(ry, rx);
end

% ======================= workflow helpers =======================
function tf = has_dl_toolbox()
% true in MATLAB with the Deep Learning Toolbox installed
tf = ~exist('OCTAVE_VERSION', 'builtin') && ~isempty(ver('nnet'));
end

function require_dl_toolbox()
if ~has_dl_toolbox()
    error('This step needs MATLAB''s Deep Learning Toolbox (not available in Octave).');
end
end

function fmt = mat_format()
% MATLAB: HDF5-based v7.3 (no 2 GB limit); Octave: v7 (it cannot write v7.3)
if exist('OCTAVE_VERSION', 'builtin'), fmt = '-v7'; else, fmt = '-v7.3'; end
end

function f = best_file_of(type)
% file holding the best network of each type
switch type
    case 'one_map',     f = 'sot_cnn_best.mat';
    case 'one_map_phi', f = 'sot_cnn_phi_best.mat';
    otherwise, error('NET_TYPE must be ''one_map'' or ''one_map_phi''.');
end
end

function T = train_settings(mode, type)
% training settings for MODE = 'train' (from scratch) or 'finetune' (gentle continuation)
T.batch      = 64;                         % samples per update
T.disp_every = 50;                         % progress line every this many batches
T.use_phi    = strcmp(type, 'one_map_phi');% feed the known in-plane direction
T.best_file  = best_file_of(type);         % best network so far (lowest validation median error)
T.ckpt_file  = strrep(T.best_file, '_best', ['_ckpt_' mode]);
T.phi_supp   = [90 5 4];                   % suppress the phi error near theta = 90 deg:
                                           % [centre width power] (deg), width 0 = off
switch mode
    case 'train',    T.epochs = 30;  T.lr = 1e-3;  T.lr_min = 1e-5;  T.check_start = false;
        case 'finetune', T.epochs = 15;  T.lr = 5e-4;  T.lr_min = 1e-6;  T.check_start = true;
end
end

function net = load_best_network(type, required)
% the best saved network of this type ([] if none and not required)
net = [];  f = best_file_of(type);
if ~has_dl_toolbox() && ~required
    fprintf('%s network: skipped (needs MATLAB''s Deep Learning Toolbox)\n', type);
    return
end
require_dl_toolbox();
if exist(f, 'file')
    Q = load(f, 'net', 'best');  net = Q.net;
    fprintf('%s network: using %s (validation median error %.1f deg)\n', type, f, Q.best);
elseif strcmp(type, 'one_map') && exist('sot_cnn.mat', 'file')
    Q = load('sot_cnn.mat', 'net');  net = Q.net;
    fprintf('one_map network: using sot_cnn.mat (saved at the end of the first training run)\n');
elseif required
    error('No saved %s network found: train it first (MODE = ''train'').', type);
else
    fprintf('%s network: not trained yet (MODE = ''train'', NET_TYPE = ''%s'')\n', type, type);
end
end

function [Xtr, Ytr, Xva, Yva] = load_training_data(files)
% all training files, concatenated (sizes are read first so memory is allocated only once)
ntr = 0;  nva = 0;
for f = 1:numel(files)
    if ~exist(files{f}, 'file')
        error('%s not found: create it first (MODE = ''make_data'').', files{f});
    end
    w = whos('-file', files{f});
    s = [w(strcmp({w.name}, 'Xtr')).size 1];  ntr = ntr + s(4);  sz = s(1:3);
    s = [w(strcmp({w.name}, 'Xva')).size 1];  nva = nva + s(4);
end
Xtr = zeros([sz ntr], 'single');  Ytr = zeros(3, ntr, 'single');
Xva = zeros([sz nva], 'single');  Yva = zeros(3, nva, 'single');
a = 0;  b = 0;
for f = 1:numel(files)
    fprintf('loading %s\n', files{f});
    Q = load(files{f}, 'Xtr', 'Ytr', 'Xva', 'Yva');
    k = size(Q.Xtr, 4);  Xtr(:,:,:,a+(1:k)) = Q.Xtr;  Ytr(:,a+(1:k)) = Q.Ytr;  a = a + k;
    k = size(Q.Xva, 4);  Xva(:,:,:,b+(1:k)) = Q.Xva;  Yva(:,b+(1:k)) = Q.Yva;  b = b + k;
    clear Q
end
fprintf('training on %d maps, validating on %d maps\n', ntr, nva);
end

function make_data_file(D, P, G)
% create one training-data file. Several can be made in parallel (e.g. on a cluster)
% with different file names and seeds, and all listed in DATA_FILES.
if exist(D.file, 'file'), fprintf('%s already exists, nothing to do\n', D.file); return; end
P.p_partial = D.p_partial;
fprintf('--- generating %s: %d training + %d validation maps, %.0f%% partial frames ---\n', ...
        D.file, D.n_train, D.n_val, 100*D.p_partial);
[Xtr, Ytr] = make_training_set(D.n_train, D.seed, P, G);
[Xva, Yva] = make_training_set(D.n_val, D.seed + 5000, P, G);
save(D.file, 'Xtr', 'Ytr', 'Xva', 'Yva', 'G', mat_format());
fprintf('saved %s\n', D.file);
end

function TS = get_test_set(file, N, seed, P, G)
% load a test set, or generate and save it
if exist(file, 'file')
    fprintf('loading %s\n', file);
    TS = load(file);
else
    TS = make_test_set(N, seed, P, G);
    save(file, '-struct', 'TS', mat_format());
end
end

function TS = make_test_set(N, seed, P, G)
% N new samples with everything both methods need: network inputs, the paper
% method's answer (from the tilted + field-along-z maps) and all sample labels
rng(seed);
TS.X = zeros(G.n, G.n, G.nch, N, 'single');
TS.Y = zeros(3, N, 'single');  TS.Ypaper = zeros(3, N, 'single');
[XG, YG] = meshgrid(G.x, G.x);
fprintf('--- generating test set (and running the paper''s two-map method on it) ---\n');
t0 = tic;
for i = 1:N
    S = sot_make_sample(P);  L = S.labels;
    TS.X(:,:,:,i)  = to_network_input(S, XG, YG, G);
    TS.Y(:,i)      = dir_vec(L.c1_theta, L.c1_phi);
    hb = L.h + G.h_err*randn;  db = L.d_tip + G.tip_err*randn;   % same knowledge as the network
    TS.Ypaper(:,i) = two_map_method(S, hb, db, L.c1_phi);
    if i == 1
        ln = fieldnames(L);
        for q = 1:numel(ln), TS.labels.(ln{q}) = zeros(N, 1); end
    end
    for q = 1:numel(ln), TS.labels.(ln{q})(i) = L.(ln{q}); end
    if mod(i, G.disp_every) == 0 || i == N
        el = toc(t0);
        fprintf('test map %d of %d done | %.1f min elapsed, ~%.1f min left\n', i, N, el/60, el/i*(N-i)/60);
    end
end
end

% ======================= error analysis =======================
function report_errors(Yps, names, Y, L)
% side-by-side comparison of several methods on the same test maps
nm = numel(Yps);  n = size(Y, 2);  tag = char('A' + (0:nm-1));
cl = @(v) max(min(v, 1), -1);
tY = acosd(cl(Y(3,:)));
E = cell(1, nm);  Pt = cell(1, nm);
for j = 1:nm
    E{j}  = angle_err(Yps{j}, Y);
    Pt{j} = abs(acosd(cl(Yps{j}(3,:))) - tY);            % polar angle theta only
end
fprintf('\n================ RESULTS on %d test maps never used in training ================\n', n);
for j = 1:nm, fprintf('  [%c] %s\n', tag(j), names{j}); end
fprintf('\n%-38s', '');  for j = 1:nm, fprintf('%11s', ['[' tag(j) ']']); end;  fprintf('\n');
row = @(lab, f) fprintf('%-38s%s\n', lab, sprintf('%11s', f{:}));
row('median error, full 3D direction', cellfun(@(e) sprintf('%.1f deg', median(e)), E, 'UniformOutput', false));
row('median error, polar angle theta only', cellfun(@(e) sprintf('%.1f deg', median(e)), Pt, 'UniformOutput', false));
row('maps within 5 deg (3D)', cellfun(@(e) sprintf('%d', sum(e < 5)), E, 'UniformOutput', false));
row('maps within 10 deg (3D)', cellfun(@(e) sprintf('%d', sum(e < 10)), E, 'UniformOutput', false));
row('90% of maps are better than', cellfun(@(e) sprintf('%.1f deg', prctile_(e, 90)), E, 'UniformOutput', false));
row('worst map', cellfun(@(e) sprintf('%.1f deg', max(e)), E, 'UniformOutput', false));

fprintf('\nmedian 3D error (deg) per group:\n');
shp = {'strip', 'trapezoid', 'polygon', 'blob'};
print_groups('noise level', E, tag, {'5-10%', '10-15%', '15-20%', '20-25%'}, ...
    bins_(L.noise_frac, [0.05 0.10 0.15 0.20 0.25]));
print_groups('polar angle theta', E, tag, {'0-30', '30-60', '60-90', '90-120', '120-150', '150-180'}, ...
    bins_(L.c1_theta, 0:30:180));
print_groups('bubbles', E, tag, {'none', '1-3', '4 or more'}, ...
    {L.n_bubbles == 0, L.n_bubbles >= 1 & L.n_bubbles <= 3, L.n_bubbles >= 4});
print_groups('extra source layer (gate)', E, tag, {'no', 'yes'}, {L.extra == 0, L.extra == 1});
print_groups('sample shape', E, tag, shp, {L.shape_type == 1, L.shape_type == 2, L.shape_type == 3, L.shape_type == 4});
print_groups('pixel size', E, tag, {'50 nm', '75 nm', '100 nm'}, ...
    {round(1e3*L.dx_out) == 50, round(1e3*L.dx_out) == 75, round(1e3*L.dx_out) == 100});
print_groups('arms', E, tag, {'no arms', 'with arms'}, {L.n_arms == 0, L.n_arms > 0});

[~, w] = sort(E{1}, 'descend');
fprintf('\n10 worst test maps of [A]:\n  %5s', 'map');
for j = 1:nm, fprintf('%8s', ['[' tag(j) ']']); end
fprintf('%7s %6s %6s %8s %5s %10s %7s\n', 'noise', 'theta', 'phi', 'bubbles', 'gate', 'shape', 'pixel');
for k = w(1:min(10, n))
    fprintf('  %5d', k);
    for j = 1:nm, fprintf('%7.1fd', E{j}(k)); end
    fprintf('%6.0f%% %6.0f %6.0f %8d %5d %10s %4.0f nm\n', 100*L.noise_frac(k), L.c1_theta(k), ...
            L.c1_phi(k), L.n_bubbles(k), L.extra(k), shp{L.shape_type(k)}, 1e3*L.dx_out(k));
end

col = {[0.85 0.1 0.1], [1 0.55 0], [0.1 0.6 0.1], [0.1 0.2 0.9], [0.5 0 0.6]};
figure('Color', 'w', 'Position', [100 100 1150 430]);
subplot(1, 2, 1);  hold on;
for j = 1:nm, plot(sort(E{j}), (1:n)/n*100, 'Color', col{mod(j-1,5)+1}, 'LineWidth', 1.5); end
grid on; xlim([0 15]); xlabel('error (deg)'); ylabel('% of test maps with a smaller error');
legend(strcat('[', cellstr(tag'), {'] '}, names(:)), 'Location', 'southeast');   % {'] '}: strcat keeps the space
title('how many maps are within a given error');
subplot(1, 2, 2);  hold on;
for j = unique([1 nm]), plot(100*L.noise_frac, E{j}, '.', 'Color', col{mod(j-1,5)+1}); end
grid on; xlabel('noise (% of peak field)'); ylabel('error (deg)'); ylim([0 15]);
legend(cellstr(strcat('[', tag(unique([1 nm]))', ']'))); title('error vs noise');
end

function print_groups(name, E, tag, labels, idx)
% median error of every method per group
fprintf('  %-26s %5s', name, 'maps');
for j = 1:numel(E), fprintf('%8s', ['[' tag(j) ']']); end
fprintf('\n');
for g = 1:numel(labels)
    k = idx{g}(:)';
    if ~any(k), continue; end
    fprintf('    %-24s %5d', labels{g}, nnz(k));
    for j = 1:numel(E), fprintf('%8.1f', median(E{j}(k))); end
    fprintf('\n');
end
end

function idx = bins_(v, edges)
% logical index of v in each bin [edges(i), edges(i+1))
idx = cell(1, numel(edges) - 1);
for i = 1:numel(edges) - 1
    idx{i} = v >= edges(i) & (v < edges(i+1) | (i == numel(edges) - 1 & v <= edges(end)));
end
end

function p = prctile_(v, q)
% percentile without the Statistics Toolbox
v = sort(v(:));  p = v(max(1, ceil(q/100*numel(v))));
end


function plot_residuals(Yps, names, Y, ttl)
% residual histograms, one row per method:
%   dtheta        = predicted - true polar angle (signed: shows bias)
%   dphi*sin(theta) = azimuth residual scaled to its real angular size (signed)
%   3D error      = angle between predicted and true direction
% each panel: median (solid red), interquartile range (shaded, dashed edges)
nm = numel(Yps);
tT = acosd(max(min(Y(3,:), 1), -1));  pT = atan2d(Y(2,:), Y(1,:));
lab   = {'\Delta\theta = predicted - true (deg)', '\Delta\phi \cdot sin\theta (deg)', '3D error (deg)'};
edges = {-15:0.5:15, -15:0.5:15, 0:0.5:20};
figure('Color', 'w', 'Name', ['residuals: ' ttl], 'Position', [60 60 1300 300*nm]);
for j = 1:nm
    Yp = Yps{j};
    tP = acosd(max(min(Yp(3,:), 1), -1));  pP = atan2d(Yp(2,:), Yp(1,:));
    dphi = mod(pP - pT + 180, 360) - 180;                 % wrapped to [-180, 180)
    R = {tP - tT, dphi .* sind(tT), angle_err(Yp, Y)};
    for c = 1:3
        subplot(nm, 3, (j-1)*3 + c);
        r = double(R{c}(:));  e = edges{c};
        cnt = histcounts_(r, e);
        bar((e(1:end-1) + e(2:end))/2, cnt, 1, 'FaceColor', [0.55 0.65 0.85], 'EdgeColor', 'none');
        hold on;
        q   = quantile_(r, [0.25 0.5 0.75]);
        top = max(cnt)*1.15 + 1;
        patch([q(1) q(3) q(3) q(1)], [0 0 top top], [1 0.6 0], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        plot([q(1) q(1)], [0 top], 'r--', [q(3) q(3)], [0 top], 'r--', 'LineWidth', 1);
        plot([q(2) q(2)], [0 top], 'r-', 'LineWidth', 2);
        xlim([e(1) e(end)]);  ylim([0 top]);  grid on;
        xlabel(lab{c});
        if c == 1, ylabel({names{j}, 'number of maps'}); end
        title(sprintf('median %.2f | IQR %.2f  [%.2f, %.2f]', q(2), q(3) - q(1), q(1), q(3)));
        nout = sum(r < e(1) | r > e(end));
        if nout > 0
            text(e(end), 0.93*top, sprintf('%d maps outside the plot  ', nout), ...
                 'HorizontalAlignment', 'right', 'FontSize', 8);
        end
    end
end
annotation('textbox', [0 0.965 1 0.035], 'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'String', [ttl '   |   solid red: median   |   shaded band, dashed red: interquartile range (middle 50% of maps)']);
end

function c = histcounts_(v, e)
% number of values in each bin [e(i), e(i+1)) (last bin closed); values outside are not counted
c = zeros(1, numel(e) - 1);
for i = 1:numel(e) - 1
    if i < numel(e) - 1, c(i) = sum(v >= e(i) & v < e(i+1));
    else,                c(i) = sum(v >= e(i) & v <= e(end)); end
end
end

function q = quantile_(v, p)
% quantiles with linear interpolation (no Statistics Toolbox needed)
v = sort(v(:));  n = numel(v);
pos = 1 + (n - 1)*p(:)';  lo = floor(pos);  hi = ceil(pos);
q = v(lo)' + (pos - lo).*(v(hi)' - v(lo)');
end