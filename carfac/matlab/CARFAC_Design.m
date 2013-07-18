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
  CF_CAR_params, CF_AGC_params, CF_IHC_params)
% function CF = CARFAC_Design(n_ears, fs, ...
%   CF_CAR_params, CF_AGC_params, CF_IHC_params)
%
% This function designs the CARFAC (Cascade of Asymmetric Resonators with
% Fast-Acting Compression); that is, it take bundles of parameters and
% computes all the filter coefficients needed to run it.
%
% fs is sample rate (per second)
% CF_CAR_params bundles all the pole-zero filter cascade parameters
% CF_AGC_params bundles all the automatic gain control parameters
% CF_IHC_params bundles all the inner hair cell parameters
%
% See other functions for designing and characterizing the CARFAC:
% [naps, CF] = CARFAC_Run(CF, input_waves)
% transfns = CARFAC_Transfer_Functions(CF, to_channels, from_channels)
%
% Defaults to Glasberg & Moore's ERB curve:
% ERB_break_freq = 1000/4.37;  % 228.833
% ERB_Q = 1000/(24.7*4.37);    % 9.2645
%
% All args are defaultable; for sample/default args see the code; they
% make 96 channels at default fs = 22050, 114 channels at 44100.

if nargin < 1
  n_ears = 1;  % if more than 1, make them identical channels;
  % then modify the design if necessary for different reasons
end

if nargin < 2
  fs = 22050;
end

if nargin < 3
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
    'ERB_break_freq', 165.3, ...  % Greenwood map's break freq.
    'ERB_Q', 1000/(24.7*4.37));  % Glasberg and Moore's high-cf ratio
end

if nargin < 4
  CF_AGC_params = struct( ...
    'n_stages', 4, ...
    'time_constants', 0.002 * 4.^(0:3), ...
    'AGC_stage_gain', 2, ...  % gain from each stage to next slower stage
    'decimation', [8, 2, 2, 2], ...  % how often to update the AGC states
    'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   % in units of channels
    'AGC2_scales', 1.65 * sqrt(2).^(0:3), ... % spread more toward base
    'AGC_mix_coeff', 0.5);
end

if nargin < 5
  % HACK: these constant control the defaults
  one_cap = 1;         % bool; 1 for Allen model, as text states we use
  just_hwr = 0;        % book; 0 for normal/fancy IHC; 1 for HWR
  if just_hwr
    CF_IHC_params = struct('just_hwr', 1, ...  % just a simple HWR
        'ac_corner_Hz', 20);
  else
    if one_cap
      CF_IHC_params = struct( ...
        'just_hwr', just_hwr, ...        % not just a simple HWR
        'one_cap', one_cap, ...   % bool; 0 for new two-cap hack
        'tau_lpf', 0.000080, ...  % 80 microseconds smoothing twice
        'tau_out', 0.0005, ...    % depletion tau is pretty fast
        'tau_in', 0.010, ...        % recovery tau is slower
        'ac_corner_Hz', 20);
    else
      CF_IHC_params = struct( ...
        'just_hwr', just_hwr, ...        % not just a simple HWR
        'one_cap', one_cap, ...   % bool; 0 for new two-cap hack
        'tau_lpf', 0.000080, ...  % 80 microseconds smoothing twice
        'tau1_out', 0.010, ...    % depletion tau is pretty fast
        'tau1_in', 0.020, ...     % recovery tau is slower
        'tau2_out', 0.0025, ...   % depletion tau is pretty fast
        'tau2_in', 0.005, ...        % recovery tau is slower
        'ac_corner_Hz', 20);
    end
  end
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

% Copy same designed coeffs into each ear (can do differently in the
% future).
for ear = 1:n_ears
  ears(ear).CAR_coeffs = CAR_coeffs;
  ears(ear).AGC_coeffs = AGC_coeffs;
  ears(ear).IHC_coeffs = IHC_coeffs;
end

CF = struct( ...
  'fs', fs, ...
  'max_channels_per_octave', max_channels_per_octave, ...
  'CAR_params', CF_CAR_params, ...
  'AGC_params', CF_AGC_params, ...
  'IHC_params', CF_IHC_params, ...
  'n_ch', n_ch, ...
  'pole_freqs', pole_freqs, ...
  'ears', ears, ...
  'n_ears', n_ears );



%% Design the filter coeffs:
function CAR_coeffs = CARFAC_DesignFilters(CAR_params, fs, pole_freqs)

n_ch = length(pole_freqs);

% the filter design coeffs:
% scalars first:
CAR_coeffs = struct( ...
  'n_ch', n_ch, ...
  'velocity_scale', CAR_params.velocity_scale, ...
  'v_offset', CAR_params.v_offset ...
  );

% don't really need these zero arrays, but it's a clue to what fields
% and types are need in ohter language implementations:
CAR_coeffs.r1_coeffs = zeros(n_ch, 1);
CAR_coeffs.a0_coeffs = zeros(n_ch, 1);
CAR_coeffs.c0_coeffs = zeros(n_ch, 1);
CAR_coeffs.h_coeffs = zeros(n_ch, 1);
CAR_coeffs.g0_coeffs = zeros(n_ch, 1);

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

zr_coeffs = pi * (x - ff * x.^3);  % when ff is 0, this is just theta,
%                       and when ff is 1 it goes to zero at theta = pi.
max_zeta = CAR_params.max_zeta;
CAR_coeffs.r1_coeffs = (1 - zr_coeffs .* max_zeta);  % "r1" for the max-damping condition

min_zeta = CAR_params.min_zeta;
% Increase the min damping where channels are spaced out more, by pulling 
% 25% of the way toward ERB_Hz/pole_freqs (close to 0.1 at high f)
min_zetas = min_zeta + 0.25*(ERB_Hz(pole_freqs, ...
  CAR_params.ERB_break_freq, CAR_params.ERB_Q) ./ pole_freqs - min_zeta);
CAR_coeffs.zr_coeffs = zr_coeffs .* ...
  (max_zeta - min_zetas);  % how r relates to undamping

% undamped coupled-form coefficients:
CAR_coeffs.a0_coeffs = a0;
CAR_coeffs.c0_coeffs = c0;

% the zeros follow via the h_coeffs
h = c0 .* f;
CAR_coeffs.h_coeffs = h;

% for unity gain at min damping, radius r; only used in CARFAC_Init:
relative_undamping = ones(n_ch, 1);  % max undamping to start
% this function needs to take CAR_coeffs even if we haven't finished
% constucting it by putting in the g0_coeffs:
CAR_coeffs.g0_coeffs = CARFAC_Stage_g(CAR_coeffs, relative_undamping);


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

  AGC_coeffs(stage).decimation = AGC_params.decimation(stage);
  tau = AGC_params.time_constants(stage);  % time constant in seconds
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
    OK = FIR(2) >= 0.2;
  case 5
    % based on solving to match [a/2, a/2, 1-a-b, b/2, b/2]:
    a = ((delay_variance + mean_delay*mean_delay)*2/5 - mean_delay*2/3) / 2;
    b = ((delay_variance + mean_delay*mean_delay)*2/5 + mean_delay*2/3) / 2;
    % first and last coeffs are implicitly duplicated to make 5-point FIR:
    FIR = [a/2, 1 - a - b, b/2];
    OK = FIR(2) >= 0.1;
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
    ro = 1 / CARFAC_Detect(10);  % output resistance at a very high level
    c = IHC_params.tau_out / ro;
    ri = IHC_params.tau_in / c;
    % to get steady-state average, double ro for 50% duty cycle
    saturation_output = 1 / (2*ro + ri);
    % also consider the zero-signal equilibrium:
    r0 = 1 / CARFAC_Detect(0);
    current = 1 / (ri + r0);
    cap_voltage = 1 - current * ri;
    IHC_coeffs = struct( ...
      'n_ch', n_ch, ...
      'just_hwr', 0, ...
      'lpf_coeff', 1 - exp(-1/(IHC_params.tau_lpf * fs)), ...
      'out_rate', ro / (IHC_params.tau_out * fs), ...
      'in_rate', 1 / (IHC_params.tau_in * fs), ...
      'one_cap', IHC_params.one_cap, ...
      'output_gain', 1/ (saturation_output - current), ...
      'rest_output', current / (saturation_output - current), ...
      'rest_cap', cap_voltage);
    % one-channel state for testing/verification:
    IHC_state = struct( ...
      'cap_voltage', IHC_coeffs.rest_cap, ...
      'lpf1_state', 0, ...
      'lpf2_state', 0, ...
      'ihc_accum', 0);
  else
    ro = 1 / CARFAC_Detect(10);  % output resistance at a very high level
    c2 = IHC_params.tau2_out / ro;
    r2 = IHC_params.tau2_in / c2;
    c1 = IHC_params.tau1_out / r2;
    r1 = IHC_params.tau1_in / c1;
    % to get steady-state average, double ro for 50% duty cycle
    saturation_output = 1 / (2*ro + r2 + r1);
    % also consider the zero-signal equilibrium:
    r0 = 1 / CARFAC_Detect(0);
    current = 1 / (r1 + r2 + r0);
    cap1_voltage = 1 - current * r1;
    cap2_voltage = cap1_voltage - current * r2;
    IHC_coeffs = struct(...
      'n_ch', n_ch, ...
      'just_hwr', 0, ...
      'lpf_coeff', 1 - exp(-1/(IHC_params.tau_lpf * fs)), ...
      'out1_rate', 1 / (IHC_params.tau1_out * fs), ...
      'in1_rate', 1 / (IHC_params.tau1_in * fs), ...
      'out2_rate', ro / (IHC_params.tau2_out * fs), ...
      'in2_rate', 1 / (IHC_params.tau2_in * fs), ...
      'one_cap', IHC_params.one_cap, ...
      'output_gain', 1/ (saturation_output - current), ...
      'rest_output', current / (saturation_output - current), ...
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
% one more late addition that applies to all cases:
IHC_coeffs.ac_coeff = 2 * pi * IHC_params.ac_corner_Hz / fs;

%%
% default design result, running this function with no args, should look
% like this, before CARFAC_Init puts state storage into it:
%

% CF = CARFAC_Design
% CAR_params = CF.CAR_params
% AGC_params = CF.AGC_params
% IHC_params = CF.IHC_params
% CAR_coeffs = CF.ears(1).CAR_coeffs
% AGC_coeffs = CF.ears(1).AGC_coeffs
% AGC_coeffs(1)
% AGC_coeffs(2)
% AGC_coeffs(3)
% AGC_coeffs(4)
% IHC_coeffs = CF.ears(1).IHC_coeffs

% CF = 
%                          fs: 22050
%     max_channels_per_octave: 12.2709
%                  CAR_params: [1x1 struct]
%                  AGC_params: [1x1 struct]
%                  IHC_params: [1x1 struct]
%                        n_ch: 71
%                  pole_freqs: [71x1 double]
%                        ears: [1x1 struct]
%                      n_ears: 1
% CAR_params = 
%                 velocity_scale: 0.1000
%                       v_offset: 0.0400
%                       min_zeta: 0.1000
%                       max_zeta: 0.3500
%               first_pole_theta: 2.6704
%                     zero_ratio: 1.4142
%     high_f_damping_compression: 0.5000
%                   ERB_per_step: 0.5000
%                    min_pole_Hz: 30
%                 ERB_break_freq: 165.3000
%                          ERB_Q: 9.2645
% AGC_params = 
%           n_stages: 4
%     time_constants: [0.0020 0.0080 0.0320 0.1280]
%     AGC_stage_gain: 2
%         decimation: [8 2 2 2]
%        AGC1_scales: [1 1.4142 2.0000 2.8284]
%        AGC2_scales: [1.6500 2.3335 3.3000 4.6669]
%      AGC_mix_coeff: 0.5000
% IHC_params = 
%         just_hwr: 0
%          one_cap: 1
%          tau_lpf: 8.0000e-05
%          tau_out: 5.0000e-04
%           tau_in: 0.0100
%     ac_corner_Hz: 20
% CAR_coeffs = 
%               n_ch: 71
%     velocity_scale: 0.1000
%           v_offset: 0.0400
%          r1_coeffs: [71x1 double]
%          a0_coeffs: [71x1 double]
%          c0_coeffs: [71x1 double]
%           h_coeffs: [71x1 double]
%          g0_coeffs: [71x1 double]
%          zr_coeffs: [71x1 double]
% AGC_coeffs = 
% 1x4 struct array with fields:
%     n_ch
%     n_AGC_stages
%     AGC_stage_gain
%     decimation
%     AGC_epsilon
%     AGC_polez1
%     AGC_polez2
%     AGC_spatial_iterations
%     AGC_spatial_FIR
%     AGC_spatial_n_taps
%     AGC_mix_coeffs
%     detect_scale
% ans = 
%                       n_ch: 71
%               n_AGC_stages: 4
%             AGC_stage_gain: 2
%                 decimation: 8
%                AGC_epsilon: 0.1659
%                 AGC_polez1: 0.1737
%                 AGC_polez2: 0.2472
%     AGC_spatial_iterations: 1
%            AGC_spatial_FIR: [0.2856 0.3108 0.4036]
%         AGC_spatial_n_taps: 3
%             AGC_mix_coeffs: 0
%               detect_scale: 0.0667
% ans = 
%                       n_ch: 71
%               n_AGC_stages: 4
%             AGC_stage_gain: 2
%                 decimation: 2
%                AGC_epsilon: 0.0867
%                 AGC_polez1: 0.1845
%                 AGC_polez2: 0.2365
%     AGC_spatial_iterations: 1
%            AGC_spatial_FIR: [0.2994 0.3178 0.3828]
%         AGC_spatial_n_taps: 3
%             AGC_mix_coeffs: 0.0454
%               detect_scale: []
% ans = 
%                       n_ch: 71
%               n_AGC_stages: 4
%             AGC_stage_gain: 2
%                 decimation: 2
%                AGC_epsilon: 0.0443
%                 AGC_polez1: 0.1921
%                 AGC_polez2: 0.2288
%     AGC_spatial_iterations: 1
%            AGC_spatial_FIR: [0.3099 0.3212 0.3689]
%         AGC_spatial_n_taps: 3
%             AGC_mix_coeffs: 0.0227
%               detect_scale: []
% ans = 
%                       n_ch: 71
%               n_AGC_stages: 4
%             AGC_stage_gain: 2
%                 decimation: 2
%                AGC_epsilon: 0.0224
%                 AGC_polez1: 0.1975
%                 AGC_polez2: 0.2235
%     AGC_spatial_iterations: 1
%            AGC_spatial_FIR: [0.3177 0.3230 0.3594]
%         AGC_spatial_n_taps: 3
%             AGC_mix_coeffs: 0.0113
%               detect_scale: []
% IHC_coeffs = 
%            n_ch: 71
%        just_hwr: 0
%       lpf_coeff: 0.4327
%        out_rate: 0.0996
%         in_rate: 0.0045
%         one_cap: 1
%     output_gain: 49.3584
%     rest_output: 1.0426
%        rest_cap: 0.5360
%        ac_coeff: 0.0057

