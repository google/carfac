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
  fs = 22050;  % Keeping this poor default in v3 to encourage users to always specify.
end

if num_args < 3
  CF_CAR_params = struct( ...
    'velocity_scale', 0.1, ...  % for the velocity nonlinearity
    'v_offset', 0.04, ...  % offset gives a quadratic part
    'min_zeta', 0.10, ... % minimum damping factor in mid-freq channels
    'max_zeta', 0.35, ... % maximum damping factor in mid-freq channels
    'first_pole_theta', 0.85*pi, ...
    'zero_ratio', sqrt(2), ... % how far zero is above pole
    'high_f_damping_compression', 0.5, ... % 0 to 1 to compress zeta
    'ERB_per_step', 0.5, ... % assume G&M's ERB formula
    'min_pole_Hz', 30, ...
    'ERB_break_freq', 165.3, ...  % 165.3 is Greenwood map's break freq.
    'ERB_Q', 1000/(24.7*4.37), ...  % Glasberg and Moore's high-cf ratio
    'ac_corner_Hz', 20 ...    % AC couple at 20 Hz corner
    );
end

if num_args < 4
  CF_AGC_params = struct( ...
    'n_stages', 4, ...
    'time_constants', 0.002 * 4.^(0:3), ...
    'AGC_stage_gain', 2, ...  % gain from each stage to next slower stage
    'decimation', [8, 2, 2, 2], ...  % how often to update the AGC states
    'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   % in units of channels
    'AGC2_scales', 1.65 * sqrt(2).^(0:3), ... % spread more toward base
    'AGC_mix_coeff', 0.5);
end

one_cap = 0;         % bool; 1 for Allen model, 0 for new default two-cap
just_hwr = 0;        % bool; 0 for normal/fancy IHC; 1 for HWR
do_syn = 0;
switch CF_version_keyword
  case 'just_hwr'
    just_hwr = 1;        % bool; 0 for normal/fancy IHC; 1 for HWR
  case 'one_cap'
    one_cap = 1;         % bool; 1 for Allen model, as text states we use
  case 'do_syn';
    do_syn = 1;
  case 'two_cap'
    % nothing to do; accept the v2 default, two-cap IHC, no SYN.
  otherwise
    error('unknown IHC_keyword in CARFAC_Design')
end

if num_args < 5  % Default IHC_params
  CF_IHC_params = struct( ...
    'just_hwr', just_hwr, ...  % not just a simple HWR
    'one_cap', one_cap, ...    % bool; 0 for new two-cap hack
    'do_syn', do_syn, ...      % bool; 1 for v3 synapse feature
    'tau_lpf', 0.000080, ...   % 80 microseconds smoothing twice
    'tau_out', 0.0005, ...     % depletion tau is pretty fast
    'tau_in', 0.010, ...       % recovery tau is slower
    'tau1_out', 0.000500, ...  % depletion tau is fast 500 us
    'tau1_in', 0.000200, ...   % recovery tau is very fast 200 us
    'tau2_out', 0.001, ...     % depletion tau is pretty fast 1 ms
    'tau2_in', 0.010);         % recovery tau is slower 10 ms
end

if num_args < 6  % Default the SYN_params.
  n_classes = 3;  % Default.  Modify params and redesign to change.
  % Parameters could generally have columns if channel-dependent.
  CF_SYN_params = struct( ...
    'do_syn', do_syn, ...  % This may just turn it off completely.
    'fs', fs, ...
    'n_classes', n_classes, ...
    'tau_lpf', 0.000080, ...
    'out_rate', 0.02, ...  % Depletion can be quick (few ms).
    'recovery', 0.0001);  % Recovery tau about 1000 to 10,000 sample times?
end

% first figure out how many filter stages (PZFC/CARFAC channels):
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
  'do_syn', do_syn);


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


%% the AGC design coeffs:
function AGC_coeffs = CARFAC_DesignAGC(AGC_params, fs, n_ch)

n_AGC_stages = AGC_params.n_stages;

% AGC1 pass is smoothing from base toward apex;
% AGC2 pass is back, which is done first now (in double exp. version)
AGC1_scales = AGC_params.AGC1_scales;
AGC2_scales = AGC_params.AGC2_scales;

decim = 1;

total_DC_gain = 0;

%%
% Convert to vector of AGC coeffs
AGC_coeffs = struct([]);
for stage = 1:n_AGC_stages
  AGC_coeffs(stage).n_ch = n_ch;
  AGC_coeffs(stage).n_AGC_stages = n_AGC_stages;
  AGC_coeffs(stage).AGC_stage_gain = AGC_params.AGC_stage_gain;
  tau = AGC_params.time_constants(stage);  % time constant in seconds

  new_way = 1;  % To try it out...
  if new_way
    % Instead of starting with decimation ratios, start with 3-tap FIR
    % and 1 iteration, and find decimation ratios that work.
    % decide on target spread (variance) and delay (mean) of impulse
    % response as a distribution to be convolved ntimes:
    % TODO (dicklyon): specify spread and delay instead of scales???
    n_taps = 3;
    n_iterations = 1;
    stage_decim = AGC_params.decimation(stage);
    FIR_OK = 0;
    while ~FIR_OK
      try_decim = decim * stage_decim;  % net decim through this stage.
      ntimes = tau * (fs / try_decim);
      delay = (AGC2_scales(stage) - AGC1_scales(stage)) / ntimes;
      spread_sq = (AGC1_scales(stage)^2 + AGC2_scales(stage)^2) / ntimes;

      [AGC_spatial_FIR, FIR_OK] = Design_FIR_coeffs( ...
        n_taps, spread_sq, delay, n_iterations);
      if ~FIR_OK
        stage_decim = stage_decim - 1;
        if stage_decim < 1
          error('AGC design failed.')
        end
      end
    end
    if stage_decim < 2
      disp('Warning:  No decimation, inefficient AGC design.')
    end
    decim = decim * stage_decim;  % Overall decimation through this stage.
    % Here we should have valid FIR filter and decim for the stage.
    AGC_coeffs(stage).AGC_epsilon = 1 - exp(-decim / (tau * fs));
    AGC_coeffs(stage).decimation = stage_decim;
    AGC_coeffs(stage).AGC_spatial_iterations = n_iterations;
    AGC_coeffs(stage).AGC_spatial_FIR = AGC_spatial_FIR;
    AGC_coeffs(stage).AGC_spatial_n_taps = n_taps;
  else
    AGC_coeffs(stage).decimation = AGC_params.decimation(stage);
    decim = decim * AGC_params.decimation(stage);  % net decim to this stage
    % epsilon is how much new input to take at each update step:
    AGC_coeffs(stage).AGC_epsilon = 1 - exp(-decim / (tau * fs));

    % effective number of smoothings in a time constant:
    ntimes = tau * (fs / decim);  % typically 5 to 50

    % decide on target spread (variance) and delay (mean) of impulse
    % response as a distribution to be convolved ntimes:
    % TODO (dicklyon): specify spread and delay instead of scales???
    delay = (AGC2_scales(stage) - AGC1_scales(stage)) / ntimes;
    spread_sq = (AGC1_scales(stage)^2 + AGC2_scales(stage)^2) / ntimes;

    % get pole positions to better match intended spread and delay of
    % [[geometric distribution]] in each direction (see wikipedia)
    u = 1 + 1 / spread_sq;  % these are based on off-line algebra hacking.
    p = u - sqrt(u^2 - 1);  % pole that would give spread if used twice.
    dp = delay * (1 - 2*p +p^2)/2;
    polez1 = p - dp;
    polez2 = p + dp;
    AGC_coeffs(stage).AGC_polez1 = polez1;
    AGC_coeffs(stage).AGC_polez2 = polez2;

    % try a 3- or 5-tap FIR as an alternative to the double exponential:
    n_taps = 0;
    FIR_OK = 0;
    n_iterations = 1;
    while ~FIR_OK
      switch n_taps
        case 0
          % first attempt a 3-point FIR to apply once:
          n_taps = 3;
        case 3
          % second time through, go wider but stick to 1 iteration
          n_taps = 5;
        case 5
          % apply FIR multiple times instead of going wider:
          n_iterations = n_iterations + 1;
          if n_iterations > 16
            error('Too many n_iterations in CARFAC_DesignAGC');
          end
        otherwise
          % to do other n_taps would need changes in CARFAC_Spatial_Smooth
          % and in Design_FIR_coeffs
          error('Bad n_taps in CARFAC_DesignAGC');
      end
      [AGC_spatial_FIR, FIR_OK] = Design_FIR_coeffs( ...
        n_taps, spread_sq, delay, n_iterations);
    end
    % when FIR_OK, store the resulting FIR design in coeffs:
    AGC_coeffs(stage).AGC_spatial_iterations = n_iterations;
    AGC_coeffs(stage).AGC_spatial_FIR = AGC_spatial_FIR;
    AGC_coeffs(stage).AGC_spatial_n_taps = n_taps;
  end
  % accumulate DC gains from all the stages, accounting for stage_gain:
  total_DC_gain = total_DC_gain + AGC_params.AGC_stage_gain^(stage-1);

  % TODO (dicklyon) -- is this the best binaural mixing plan?
  if stage == 1
    AGC_coeffs(stage).AGC_mix_coeffs = 0;
  else
    AGC_coeffs(stage).AGC_mix_coeffs = AGC_params.AGC_mix_coeff / ...
      (tau * (fs / decim));
  end
end

% adjust stage 1 detect_scale to be the reciprocal DC gain of the AGC filters:
AGC_coeffs(1).detect_scale = 1 / total_DC_gain;


%%
function [FIR, OK] = Design_FIR_coeffs(n_taps, delay_variance, ...
  mean_delay, n_iter)
% function [FIR, OK] = Design_FIR_coeffs(n_taps, delay_variance, ...
%   mean_delay, n_iter)
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

% reduce mean and variance of smoothing distribution by n_iterations:
mean_delay = mean_delay / n_iter;
delay_variance = delay_variance / n_iter;
switch n_taps
  case 3
    % based on solving to match mean and variance of [a, 1-a-b, b]:
    a = (delay_variance + mean_delay*mean_delay - mean_delay) / 2;
    b = (delay_variance + mean_delay*mean_delay + mean_delay) / 2;
    FIR = [a, 1 - a - b, b];
    OK = FIR(2) >= 0.25;
  case 5
    % based on solving to match [a/2, a/2, 1-a-b, b/2, b/2]:
    a = ((delay_variance + mean_delay*mean_delay)*2/5 - mean_delay*2/3) / 2;
    b = ((delay_variance + mean_delay*mean_delay)*2/5 + mean_delay*2/3) / 2;
    % first and last coeffs are implicitly duplicated to make 5-point FIR:
    FIR = [a/2, 1 - a - b, b/2];
    OK = FIR(2) >= 0.15;
  otherwise
    error('Bad n_taps in AGC_spatial_FIR');
end


%% the IHC design coeffs:
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

%% the SYN design coeffs:
function SYN_coeffs = CARFAC_DesignSynapses(SYN_params, fs, pole_freqs)

if ~SYN_params.do_syn
  SYN_coeffs = [];  % Just return empty if we're not doing SYN.
  return
end

n_ch = length(pole_freqs);
n_cl = SYN_params.n_classes;
col_n_ch_ones = ones(n_ch, 1);

v_width = 0.05;
v_widths = v_width * ones(1, n_cl);
max_rate = 2500;  % % Instantaneous max at onset, per Kiang figure 5.18.
max_rates = max_rate * ones(1, n_cl);
switch n_cl  % Just enough to test that both 2 and 3 can work.
  case 2
    offsets = [3, 6];
    agc_weights_col = 0.8 * (fs/1000) * [1; 1];
    res_inits = [0.2, 0.6];
    rest_output = 0.8 * (fs/1000) * 0.016;  % Subject off to get agc_in near 0 in quiet.
    n_fibers = [50, 60]
    v_half = v_widths .* [3, 6;]
  case 3
    offsets = [3, 5, 7];
    agc_weights_col = 0.8 * (fs/1000) * [1; 1; 1];
    res_inits = [0.13, 0.55, 0.9]
    rest_output = (fs/1000) * 0.016;
    n_fibers = [50, 35, 25];
    v_half = v_widths .* [3, 5, 7];
  otherwise
    error('unimplemented n_classes in in SYN_params in CARFAC_DesignSynapses');
end


% Copy stuff from params to coeffs and design a few things.
SYN_coeffs = struct( ...
  'n_ch', n_ch, ...
  'n_classes', n_cl, ...
  'max_probs', max_rates / SYN_params.fs, ...
  'n_fibers', col_n_ch_ones * n_fibers, ...  % Synaptopathy comes in here; channelize it, too.
  'v_width', v_width, ...
  'v_half', col_n_ch_ones * v_half, ...  % Same units as v_width and v_recep.
  'out_rate', 0.1, ...  % Depletion can be quick (few ms).
  'recovery', 1e-3, ...  % Recovery tau about 1000 sample times.  Or 10X this?
  'agc_weights_col', agc_weights_col/2, ... % try to make syn_out resemble ihc_out to go to agc_in.
  'rest_output', rest_output, ...
  'res_inits', res_inits, ...
  'lpf_coeff', 1 - exp(-1/(SYN_params.tau_lpf * fs)));

