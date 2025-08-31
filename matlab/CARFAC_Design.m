% Copyright 2012 The CARFAC Authors. All Rights Reserved.
% Author: Richard F. Lyon
%
% This file is part of an implementation of Lyon's cochlear model:
% "Cascade of Asymmetric Resonators with Fast-Acting Compression"
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function CF = CARFAC_Design(n_ears, fs, ...
  CF_CAR_params, CF_AGC_params, CF_IHC_params, CF_SYN_params, ...
  CF_version_keyword)
% function CF = CARFAC_Design(n_ears, fs, ...
%   CF_CAR_params, CF_AGC_params, CF_IHC_params, CF_version_keyword)
%
% This function designs the CARFAC (Cascade of Asymmetric Resonators with
% Fast-Acting Compression); that is, it take bundles of parameters and
% computes all the filter coefficients needed to run it.  Before running,
% CARFAC_Init is needed, to build and initialize the state variables.
%
% n_ears (typically 1 or 2, can be more) is number of sound channels.
% fs is sample rate (per second)
% CF_CAR_params bundles all the pole-zero filter cascade parameters
% CF_AGC_params bundles all the automatic gain control parameters
% CF_IHC_params bundles all the inner hair cell parameters
% Indendent of how many of these are provided, if the last arg is a string
% it will be interpreted as a CF_version_keyword ('just_hwr' or 'one_cap';
% omit or 'two_cap' for default v2 two-cap IHC model; or 'do_syn')
%
% See other functions for designing and characterizing the CARFAC:
% [naps, CF] = CARFAC_Run(CF, input_waves)
% transfns = CARFAC_Transfer_Functions(CF, to_channels, from_channels)
%
% Scale is like Glasberg & Moore's ERB curve, but Greenwood's breakf.
% ERB_break_freq = 165.3;  % Not 1000/4.37, 228.8 Hz breakf of Moore.
% ERB_Q = 1000/(24.7*4.37);    % 9.2645
% Edit CF.CF_CAR_params and call CARFAC_Design again to change.
% Similarly for changing other things.
%
% All args are defaultable; for sample/default args see the code; they
% make 71 channels at default fs = 22050.
% For "modern" applications we prefer fs = 48000, which makes 84 channels.

switch nargin
  case 0
    last_arg = [];
  case 1
    last_arg = n_ears;
  case 2
    last_arg = fs;
  case 3
    last_arg = CF_CAR_params;
  case 4
    last_arg = CF_AGC_params;
  case 5
    last_arg = CF_IHC_params;
  case 6
    last_arg = CF_SYN_params;
  case 7
    last_arg = CF_version_keyword;
end
% Last arg being a keyword can be 'do_syn' or an IHC version.
if ischar(last_arg)  % string is a character array, not a string array.
  CF_version_keyword = last_arg;
  num_args = nargin - 1;
else
  CF_version_keyword = 'two_cap';  % The CARFAC v2 default
  num_args = nargin;
end
% Now num_args does not count the version_keyword.  Can be up to 6.
if num_args < 1
  n_ears = 1;  % if more than 1, make them identical channels;
  % then modify the design if necessary for different reasons
end
if num_args < 2
  fs = 48000;  % In v3, revised from 22050 to encourage better realism.
end
if num_args < 3
  CF_CAR_params = CAR_params_default;
end
if num_args < 4
  CF_AGC_params = AGC_params_default;
end
if num_args < 5  % Default IHC_params
  CF_IHC_params = IHC_params_default(CF_version_keyword);
end
if num_args < 6  % Default the SYN_params.
  CF_SYN_params = SYN_params_default(CF_IHC_params.do_syn);
end

% First count how many filter stages (PZFC/CARFAC channels):
pole_Hz = CF_CAR_params.first_pole_theta * fs / (2*pi);
n_ch = 0;
while pole_Hz > CF_CAR_params.min_pole_Hz
  n_ch = n_ch + 1;
  pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ...
    ERB_Hz(pole_Hz, CF_CAR_params.ERB_break_freq, CF_CAR_params.ERB_Q);
end
% Now we have n_ch, the number of channels, so can make the array
% and compute all the frequencies again to put into it:
pole_freqs = zeros(n_ch, 1);
pole_Hz = CF_CAR_params.first_pole_theta * fs / (2*pi);
for ch = 1:n_ch
  pole_freqs(ch) = pole_Hz;
  pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ...
    ERB_Hz(pole_Hz, CF_CAR_params.ERB_break_freq, CF_CAR_params.ERB_Q);
end
% Now we have n_ch, the number of channels, and pole_freqs array.

max_channels_per_octave = log(2) / log(pole_freqs(1)/pole_freqs(2));

% Convert to include an ear_array, each w coeffs and state...
CAR_coeffs = CARFAC_DesignFilters(CF_CAR_params, fs, pole_freqs);
AGC_coeffs = CARFAC_DesignAGC(CF_AGC_params, fs, n_ch);
IHC_coeffs = CARFAC_DesignIHC(CF_IHC_params, fs, n_ch);
SYN_coeffs = CARFAC_DesignSynapses(CF_SYN_params, fs, pole_freqs);

% Copy same designed coeffs into each ear (can do differently in the
% future, e.g. for unmatched OHC_health).
for ear = 1:n_ears
  ears(ear).CAR_coeffs = CAR_coeffs;
  ears(ear).AGC_coeffs = AGC_coeffs;
  ears(ear).IHC_coeffs = IHC_coeffs;
  ears(ear).SYN_coeffs = SYN_coeffs;
end

CF = struct( ...
  'fs', fs, ...
  'max_channels_per_octave', max_channels_per_octave, ...
  'CAR_params', CF_CAR_params, ...
  'AGC_params', CF_AGC_params, ...
  'IHC_params', CF_IHC_params, ...
  'SYN_params', CF_SYN_params, ...
  'n_ch', n_ch, ...
  'pole_freqs', pole_freqs, ...
  'ears', ears, ...
  'n_ears', n_ears, ...
  'open_loop', 0, ...
  'linear_car', 0, ...
  'do_syn', CF_SYN_params.do_syn);


%% Design the filter coeffs:
function CAR_coeffs = CARFAC_DesignFilters(CAR_params, fs, pole_freqs)

n_ch = length(pole_freqs);

% the filter design coeffs:
% scalars first:
CAR_coeffs = struct( ...
  'n_ch', n_ch, ...
  'velocity_scale', CAR_params.velocity_scale, ...
  'v_offset', CAR_params.v_offset, ...
  'ac_coeff', 2 * pi * CAR_params.ac_corner_Hz / fs ...
  );

% don't really need these zero arrays, but it's a clue to what fields
% and types are needed in other language implementations:
CAR_coeffs.r1_coeffs = zeros(n_ch, 1);
CAR_coeffs.a0_coeffs = zeros(n_ch, 1);
CAR_coeffs.c0_coeffs = zeros(n_ch, 1);
CAR_coeffs.h_coeffs = zeros(n_ch, 1);
CAR_coeffs.g0_coeffs = zeros(n_ch, 1);
CAR_coeffs.ga_coeffs = zeros(n_ch, 1);
CAR_coeffs.gb_coeffs = zeros(n_ch, 1);
CAR_coeffs.gc_coeffs = zeros(n_ch, 1);
CAR_coeffs.OHC_health = ones(n_ch, 1);  % 0 to 1 to derate OHC activity.

% zero_ratio comes in via h.  In book's circuit D, zero_ratio is 1/sqrt(a),
% and that a is here 1 / (1+f) where h = f*c.
% solve for f:  1/zero_ratio^2 = 1 / (1+f)
% zero_ratio^2 = 1+f => f = zero_ratio^2 - 1
f = CAR_params.zero_ratio^2 - 1;  % nominally 1 for half-octave

% Make pole positions, s and c coeffs, h and g coeffs, etc.,
% which mostly depend on the pole angle theta:
theta = pole_freqs .* (2 * pi / fs);
c0 = sin(theta);
a0 = cos(theta);

% different possible interpretations for min-damping r:
% r = exp(-theta * CF_CAR_params.min_zeta).
% Compress theta to give somewhat higher Q at highest thetas:
ff = CAR_params.high_f_damping_compression;  % 0 to 1; typ. 0.5
x = theta/pi;
theta = pi * (x - ff * x.^3);  % when ff is 0, this is just theta,
%                       and when ff is 1 it goes to zero at theta = pi.
max_zeta = CAR_params.max_zeta;
CAR_coeffs.r1_coeffs = (1 - theta .* max_zeta);  % "r1" for the max-damping condition

min_zeta = CAR_params.min_zeta;
if min_zeta <= 0  % Use this to do a new design strategy
  local_low_level_q = pole_freqs ./ ERB_Hz( ...
    pole_freqs, CAR_params.ERB_break_freq, CAR_params.ERB_Q);
  % Number of overlapping channels is about ERB_per_step^-1, so this:
  min_zetas = CAR_params.ERB_per_step^-0.5 ./ (2*local_low_level_q);
  min_zetas = min(min_zetas, 0.75*max_zeta);  % Keep some low CF action.
  % "r1" for the max-damping condition
  CAR_coeffs.r1_coeffs = exp(-theta .* max_zeta);
  r0_coeffs = exp(-theta .* min_zetas);  % min_damping condition.
  CAR_coeffs.zr_coeffs = r0_coeffs - CAR_coeffs.r1_coeffs;
else
  % Increase the min damping where channels are spaced out more, by pulling
  % toward ERB_Hz/pole_freqs (close to 0.1 at high f)
  min_zetas = min_zeta + 0.25*(ERB_Hz(pole_freqs, ...
    CAR_params.ERB_break_freq, CAR_params.ERB_Q) ./ pole_freqs - min_zeta);
  CAR_coeffs.r1_coeffs = (1 - theta .* max_zeta);  % "r1" for the max-damping condition
  CAR_coeffs.zr_coeffs = theta .* ...
    (max_zeta - min_zetas);  % how r relates to undamping
end

% undamped coupled-form coefficients:
CAR_coeffs.a0_coeffs = a0;
CAR_coeffs.c0_coeffs = c0;

% the zeros follow via the h_coeffs
h = c0 .* f;
CAR_coeffs.h_coeffs = h;

% Efficient approximation with g as quadratic function of undamping.
% First get g at both ends and the half-way point.
undamping = 0.0;
g0 = CARFAC_Design_Stage_g(CAR_coeffs, undamping);
undamping = 1.0;
g1 = CARFAC_Design_Stage_g(CAR_coeffs, undamping);
undamping = 0.5;
ghalf = CARFAC_Design_Stage_g(CAR_coeffs, undamping);
% Store fixed coefficients for A*undamping.^2 + B^undamping + C
CAR_coeffs.ga_coeffs = 2*(g0 + g1 - 2*ghalf);
CAR_coeffs.gb_coeffs = 4*ghalf - 3*g0 - g1;
CAR_coeffs.gc_coeffs = g0;

% Set up initial stage gains.
% Maybe re-do this at Init time?
undamping = CAR_coeffs.OHC_health;  % Typically just ones.
% Avoid running this model function at Design time; see tests.
% CAR_coeffs.g0_coeffs = CARFAC_Stage_g(CAR_coeffs, undamping);
CAR_coeffs.g0_coeffs = CARFAC_Design_Stage_g(CAR_coeffs, undamping);


%% Design the AGC coeffs:
function AGC_coeffs = CARFAC_DesignAGC(AGC_params, fs, n_ch)

% AGC1 pass is smoothing from base toward apex; AGC2 pass is back.
AGC1_scales = AGC_params.AGC1_scales;
AGC2_scales = AGC_params.AGC2_scales;
n_AGC_stages = AGC_params.n_stages;
% Add up 1, 2, 4, 8, giving 15 by default:
total_DC_gain = sum(AGC_params.AGC_stage_gain.^(0:(n_AGC_stages - 1)));
detect_scale = 1 / total_DC_gain;  % Multiply the detect inputs by this.


% 2025 new way, one struct instead of an array.
tau_fs = AGC_params.time_constants * fs;
mix_coeffs = zeros(1, n_AGC_stages);
% zero mix at stage 1; compute the others:
decim = AGC_params.decimation(1);
for stage = 2:n_AGC_stages
  decim = decim * AGC_params.decimation(stage);
  mix_coeffs(stage) = AGC_params.AGC_mix_coeff * decim / tau_fs(stage);
end
AGC_coeffs = struct( ...
  'n_AGC_stages', n_AGC_stages, ...
  'n_ch', n_ch, ...
  'decimation', AGC_params.decimation, ...
  'AGC_stage_gain', AGC_params.AGC_stage_gain, ...
  'mix_coeffs', mix_coeffs, ...
  'feedback_gains', zeros(1, n_AGC_stages), ...
  'input_gains', zeros(1, n_AGC_stages), ...
  'next_stage_gains', zeros(1, n_AGC_stages), ...
  'spatial_FIR', zeros(3, n_AGC_stages), ...
  'simpler_decimating', all(AGC_params.decimation(2:end) == 1) ...
  );
decim = 1;
for stage = 1:n_AGC_stages
  % Temporal design for the parallel of cascades:
  decim = decim * AGC_params.decimation(stage);
  ntimes = tau_fs(stage) / decim;  % Number of updates per time tau.
  AGC_epsilon = 1 - exp(-1 / ntimes);
  AGC_coeffs.feedback_gains(stage) = 1 - AGC_epsilon;
  AGC_coeffs.input_gains(stage) = AGC_epsilon / total_DC_gain;
  % The last of the next_stage_gains just multiplies zeros.
  AGC_coeffs.next_stage_gains(stage) = AGC_epsilon * AGC_params.AGC_stage_gain;
  % Spatial design using 3-point FIR smoother across channels:
  delay = (AGC2_scales(stage) - AGC1_scales(stage)) / ntimes;
  spread_sq = (AGC1_scales(stage)^2 + AGC2_scales(stage)^2) / ntimes;
  [spatial_FIR, FIR_OK] = Design_FIR_coeffs(spread_sq, delay);
  if ~FIR_OK
    error( ...
      'AGC 3-point FIR design failed; try less decimation, or less spread or delay, or longer time constant.')
  end
  % If it's feasible (not too much spread or delay requested):
  AGC_coeffs.spatial_FIR(:, stage) = spatial_FIR';
end


%% Design the AGC's 3-point FIR spatial filter:
function [FIR, OK] = Design_FIR_coeffs(delay_variance, mean_delay)
% function [FIR, OK] = Design_FIR_coeffs(delay_variance, mean_delay)
% The smoothing function is a space-domain smoothing, but it considered
% here by analogy to time-domain smoothing, which is why its potential
% off-centeredness is called a delay.  Since it's a smoothing filter, it is
% also analogous to a discrete probability distribution (a p.m.f.), with
% mean corresponding to delay and variance corresponding to squared spatial
% spread (in samples, or channels, and the square thereof, respecitively).
% Here we design a filter implementation's coefficient via the method of
% moment matching, so we get the intended delay and spread, and don't worry
% too much about the shape of the distribution, which will be some kind of
% blob not too far from Gaussian if we run several FIR iterations.

% based on solving to match mean and variance of [a, 1-a-b, b]:
a = (delay_variance + mean_delay*mean_delay - mean_delay) / 2;
b = (delay_variance + mean_delay*mean_delay + mean_delay) / 2;
FIR = [a, 1 - a - b, b];
OK = all(FIR >= [0.0, 0.25, 0.0]);


%% Design the IHC coeffs:
function IHC_coeffs = CARFAC_DesignIHC(IHC_params, fs, n_ch)

if IHC_params.just_hwr
  IHC_coeffs = struct( ...
    'n_ch', n_ch, ...
    'just_hwr', 1);
else
  if IHC_params.one_cap
    gmax = CARFAC_Detect(10);  % output conductance at a high level
    rmin = 1 / gmax;
    c = IHC_params.tau_out * gmax;
    ri = IHC_params.tau_in / c;
    % to get approx steady-state average, double rmin for 50% duty cycle
    saturation_current = 1 / (2/gmax + ri);
    % also consider the zero-signal equilibrium:
    g0 = CARFAC_Detect(0);
    r0 = 1 / g0;
    rest_current = 1 / (ri + r0);
    cap_voltage = 1 - rest_current * ri;
    IHC_coeffs = struct( ...
      'n_ch', n_ch, ...
      'just_hwr', 0, ...
      'lpf_coeff', 1 - exp(-1/(IHC_params.tau_lpf * fs)), ...
      'out_rate', rmin / (IHC_params.tau_out * fs), ...
      'in_rate', 1 / (IHC_params.tau_in * fs), ...
      'one_cap', IHC_params.one_cap, ...
      'output_gain', 1 / (saturation_current - rest_current), ...
      'rest_output', rest_current / (saturation_current - rest_current), ...
      'rest_cap', cap_voltage);
    % one-channel state for testing/verification:
    IHC_state = struct( ...
      'cap_voltage', IHC_coeffs.rest_cap, ...
      'lpf1_state', 0, ...
      'lpf2_state', 0, ...
      'ihc_accum', 0);
  else
    g1max = CARFAC_Detect(10);  % receptor conductance at high level
    r1min = 1 / g1max;
    c1 = IHC_params.tau1_out * g1max;  % capacitor for min depletion tau
    r1 = IHC_params.tau1_in / c1;  % resistance for recharge tau
    % to get approx steady-state average, double r1min for 50% duty cycle
    saturation_current1 = 1 / (2*r1min + r1);  % Approximately.
    % also consider the zero-signal equilibrium:
    g10 = CARFAC_Detect(0);
    r10 = 1/g10;
    rest_current1 = 1 / (r1 + r10);
    cap1_voltage = 1 - rest_current1 * r1;  % quiescent/initial state

    % Second cap similar, but using receptor voltage as detected signal.
    max_vrecep = r1 / (r1min + r1);  % Voltage divider from 1.
    % Identity from receptor potential to neurotransmitter conductance:
    g2max = max_vrecep;  % receptor resistance at very high level
    r2min = 1 / g2max;
    c2 = IHC_params.tau2_out * g2max;  % capacitor for min depletion tau
    r2 = IHC_params.tau2_in / c2;  % resistance for recharge tau
    % to get approx steady-state average, double r2min for 50% duty cycle
    saturation_current2 = 1 / (2 * r2min + r2);
    % also consider the zero-signal equilibrium:
    rest_vrecep = r1 * rest_current1;
    g20 = rest_vrecep;
    r20 = 1 / g20;
    rest_current2 = 1 / (r2 + r20);
    cap2_voltage = 1 - rest_current2 * r2;  % quiescent/initial state

    IHC_coeffs = struct(...
      'n_ch', n_ch, ...
      'just_hwr', 0, ...
      'lpf_coeff', 1 - exp(-1/(IHC_params.tau_lpf * fs)), ...
      'out1_rate', r1min / (IHC_params.tau1_out * fs), ...
      'in1_rate', 1 / (IHC_params.tau1_in * fs), ...
      'out2_rate', r2min / (IHC_params.tau2_out * fs), ...
      'in2_rate', 1 / (IHC_params.tau2_in * fs), ...
      'one_cap', IHC_params.one_cap, ...
      'output_gain', 1 / (saturation_current2 - rest_current2), ...
      'rest_output', rest_current2 / (saturation_current2 - rest_current2), ...
      'rest_cap2', cap2_voltage, ...
      'rest_cap1', cap1_voltage);
    % one-channel state for testing/verification:
    IHC_state = struct( ...
      'cap1_voltage', IHC_coeffs.rest_cap1, ...
      'cap2_voltage', IHC_coeffs.rest_cap2, ...
      'lpf1_state', 0, ...
      'lpf2_state', 0, ...
      'ihc_accum', 0);
  end
end


%% Design the SYN coeffs:
function SYN_coeffs = CARFAC_DesignSynapses(SYN_params, fs, pole_freqs)

if ~SYN_params.do_syn
  SYN_coeffs = [];  % Just return empty if we're not doing SYN.
  return
end

n_ch = length(pole_freqs);
n_classes = SYN_params.n_classes;

v_widths = SYN_params.v_width * ones(1, n_classes);

% Do some design.  First, gains to get sat_rate when sigmoid is 1, which
% involves the reservoir steady-state solution.
% Most of these are not per-channel, but could be expanded that way
% later if desired.

% Mean sat prob of spike per sample per neuron, likely same for all
% classes.
% Use names 1 for sat and 0 for spont in some of these.
p1 = SYN_params.sat_rates / fs;
p0 = SYN_params.spont_rates / fs;

w1 = SYN_params.sat_reservoir;
q1 = 1 - w1;
% Assume the sigmoid is switching between 0 and 1 at 50% duty cycle, so
% normalized mean value is 0.5 at saturation.
s1 = 0.5;
r1 = s1*w1;
% solve q1 = a1*r1 for gain coeff a1:
a1 = q1 ./ r1;
% solve p1 = a2*r1 for gain coeff a2:
a2 = p1 ./ r1;

% Now work out how to get the desired spont.
r0 = p0 ./ a2;
q0 = r0 .* a1;
w0 = 1 - q0;
s0 = r0 ./ w0;
% Solve for (negative) sigmoid midpoint offset that gets s0 right.
offsets = log((1 - s0)./s0);

spont_p = a2 .* w0 .* s0;  % should match p0; check it; yes it does.

agc_weights = fs * SYN_params.agc_weights;
spont_sub = (SYN_params.healthy_n_fibers .* spont_p) * agc_weights';

% Copy stuff needed at run time into coeffs.
SYN_coeffs = struct( ...
  'n_ch', n_ch, ...
  'n_classes', n_classes, ...
  'n_fibers', ones(n_ch,1) * SYN_params.healthy_n_fibers, ...
  'v_widths', v_widths, ...
  'v_halfs', offsets .* v_widths, ...  % Same units as v_recep and v_widths.
  'a1', a1, ...  % Feedback gain
  'a2', a2, ...  % Output gain
  'agc_weights', agc_weights, ... % For making a nap out to agc in.
  'spont_p', spont_p, ... % used only to init the output LPF
  'spont_sub', spont_sub, ...
  'res_lpf_inits', q0, ...
  'res_coeff', 1 - exp(-1/(SYN_params.reservoir_tau * fs)), ...
  'lpf_coeff', 1 - exp(-1/(SYN_params.tau_lpf * fs)));
