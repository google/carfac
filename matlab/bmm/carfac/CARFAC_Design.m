% Copyright 2012, Google, Inc.
% Author: Richard F. Lyon
%
% This Matlab file is part of an implementation of Lyon's cochlear model:
% "Cascade of Asymmetric Resonators with Fast-Acting Compression"
% to supplement Lyon's upcoming book "Human and Machine Hearing"
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

function CF = CARFAC_Design(fs, CF_filter_params, ...
  CF_AGC_params, ERB_break_freq, ERB_Q, CF_IHC_params)
% function CF = CARFAC_Design(fs, CF_filter_params, ...
%   CF_AGC_params, ERB_break_freq, ERB_Q, CF_IHC_params)
%
% This function designs the CARFAC (Cascade of Asymmetric Resonators with
% Fast-Acting Compression); that is, it take bundles of parameters and
% computes all the filter coefficients needed to run it.
%
% fs is sample rate (per second)
% CF_filter_params bundles all the pole-zero filter cascade parameters
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

if nargin < 6
  % HACK: these constant control the defaults
  one_cap = 0;         % bool; 0 for new two-cap hack
  just_hwr = 0;        % book; 0 for normal/fancy IHC; 1 for HWR
  if just_hwr
    CF_IHC_params = struct('just_hwr', 1);  % just a simple HWR
  else
    if one_cap
      CF_IHC_params = struct( ...
        'just_hwr', just_hwr, ...        % not just a simple HWR
        'one_cap', one_cap, ...   % bool; 0 for new two-cap hack
        'tau_lpf', 0.000080, ...  % 80 microseconds smoothing twice
        'tau_out', 0.0005, ...    % depletion tau is pretty fast
        'tau_in', 0.010 );        % recovery tau is slower
    else
      CF_IHC_params = struct( ...
        'just_hwr', just_hwr, ...        % not just a simple HWR
        'one_cap', one_cap, ...   % bool; 0 for new two-cap hack
        'tau_lpf', 0.000080, ...  % 80 microseconds smoothing twice
        'tau1_out', 0.020, ...    % depletion tau is pretty fast
        'tau1_in', 0.020, ...     % recovery tau is slower
        'tau2_out', 0.005, ...   % depletion tau is pretty fast
        'tau2_in', 0.005 );        % recovery tau is slower
    end
  end
end

if nargin < 5
  %	Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  % ERB = 24.7 * (1 + 4.37 * CF_Hz / 1000);
  ERB_Q = 1000/(24.7*4.37);  % 9.2645
  if nargin < 4
    ERB_break_freq = 1000/4.37;  % 228.833
  end
end

if nargin < 3
  CF_AGC_params = struct( ...
    'n_stages', 4, ...
    'time_constants', [1, 4, 16, 64]*0.002, ...
    'AGC_stage_gain', 2, ...  % gain from each stage to next slower stage
    'decimation', [8, 2, 2, 2], ...  % how often to update the AGC states
    'AGC1_scales', [1, 2, 4, 6]*1, ...   % in units of channels
    'AGC2_scales', [1, 2, 4, 6]*1.5, ... % spread more toward base
    'detect_scale', 0.15, ...  % the desired damping range
    'AGC_mix_coeff', 0.5);
end

if nargin < 2
  CF_filter_params = struct( ...
    'velocity_scale', 0.2, ...  % for the "cubic" velocity nonlinearity
    'v_offset', 0.01, ...  % offset gives a quadratic part
    'v2_corner', 0.2, ...  % corner for essential nonlin
    'v_damp_max', 0.01, ... % damping delta damping from velocity nonlin
    'min_zeta', 0.12, ...
    'first_pole_theta', 0.78*pi, ...
    'zero_ratio', sqrt(2), ...
    'ERB_per_step', 0.3333, ... % assume G&M's ERB formula
    'min_pole_Hz', 40 );
end

if nargin < 1
  fs = 22050;
end

% first figure out how many filter stages (PZFC/CARFAC channels):
pole_Hz = CF_filter_params.first_pole_theta * fs / (2*pi);
n_ch = 0;
while pole_Hz > CF_filter_params.min_pole_Hz
  n_ch = n_ch + 1;
  pole_Hz = pole_Hz - CF_filter_params.ERB_per_step * ...
    ERB_Hz(pole_Hz, ERB_break_freq, ERB_Q);
end
% Now we have n_ch, the number of channels, so can make the array
% and compute all the frequencies again to put into it:
pole_freqs = zeros(n_ch, 1);
pole_Hz = CF_filter_params.first_pole_theta * fs / (2*pi);
for ch = 1:n_ch
  pole_freqs(ch) = pole_Hz;
  pole_Hz = pole_Hz - CF_filter_params.ERB_per_step * ...
    ERB_Hz(pole_Hz, ERB_break_freq, ERB_Q);
end
% now we have n_ch, the number of channels, and pole_freqs array

CF = struct( ...
  'fs', fs, ...
  'filter_params', CF_filter_params, ...
  'AGC_params', CF_AGC_params, ...
  'IHC_params', CF_IHC_params, ...
  'n_ch', n_ch, ...
  'pole_freqs', pole_freqs, ...
  'filter_coeffs', CARFAC_DesignFilters(CF_filter_params, fs, pole_freqs), ...
  'AGC_coeffs', CARFAC_DesignAGC(CF_AGC_params, fs), ...
  'IHC_coeffs', CARFAC_DesignIHC(CF_IHC_params, fs), ...
  'n_mics', 0 );

% adjust the AGC_coeffs to account for IHC saturation level to get right
% damping change as specified in CF.AGC_params.detect_scale
CF.AGC_coeffs.detect_scale = CF.AGC_params.detect_scale / ...
  (CF.IHC_coeffs.saturation_output * CF.AGC_coeffs.AGC_gain);

%% Design the filter coeffs:
function filter_coeffs = CARFAC_DesignFilters(filter_params, fs, pole_freqs)

n_ch = length(pole_freqs);

% the filter design coeffs:

filter_coeffs = struct('velocity_scale', filter_params.velocity_scale, ...
  'v_offset', filter_params.v_offset, ...
  'v2_corner', filter_params.v2_corner, ...
  'v_damp_max', filter_params.v_damp_max ...
  );

filter_coeffs.r_coeffs = zeros(n_ch, 1);
filter_coeffs.a_coeffs = zeros(n_ch, 1);
filter_coeffs.c_coeffs = zeros(n_ch, 1);
filter_coeffs.h_coeffs = zeros(n_ch, 1);
filter_coeffs.g_coeffs = zeros(n_ch, 1);

% zero_ratio comes in via h.  In book's circuit D, zero_ratio is 1/sqrt(a),
% and that a is here 1 / (1+f) where h = f*c.
% solve for f:  1/zero_ratio^2 = 1 / (1+f)
% zero_ratio^2 = 1+f => f = zero_ratio^2 - 1
f = filter_params.zero_ratio^2 - 1;  % nominally 1 for half-octave

% Make pole positions, s and c coeffs, h and g coeffs, etc.,
% which mostly depend on the pole angle theta:
theta = pole_freqs .* (2 * pi / fs);

% different possible interpretations for min-damping r:
% r = exp(-theta * CF_filter_params.min_zeta).
% Using sin gives somewhat higher Q at highest thetas.
r = (1 - sin(theta) * filter_params.min_zeta);
filter_coeffs.r_coeffs = r;

% undamped coupled-form coefficients:
filter_coeffs.a_coeffs = cos(theta);
filter_coeffs.c_coeffs = sin(theta);

% the zeros follow via the h_coeffs
h = sin(theta) .* f;
filter_coeffs.h_coeffs = h;

% unity gain at min damping, radius r:
filter_coeffs.g_coeffs = (1 - 2*r.*cos(theta) + r.^2) ./ ...
  (1 - 2*r .* cos(theta) + h .* r .* sin(theta) + r.^2);


%% the AGC design coeffs:
function AGC_coeffs = CARFAC_DesignAGC(AGC_params, fs)

AGC_coeffs = struct('AGC_stage_gain', AGC_params.AGC_stage_gain);

% AGC1 pass is smoothing from base toward apex;
% AGC2 pass is back, which is done first now
AGC1_scales = AGC_params.AGC1_scales;
AGC2_scales = AGC_params.AGC2_scales;

n_AGC_stages = AGC_params.n_stages;
AGC_coeffs.AGC_epsilon = zeros(1, n_AGC_stages);  % the 1/(tau*fs) roughly
decim = 1;
AGC_coeffs.decimation = AGC_params.decimation;

total_DC_gain = 0;
for stage = 1:n_AGC_stages
  tau = AGC_params.time_constants(stage);  % time constant in seconds
  decim = decim * AGC_params.decimation(stage);  % net decim to this stage
  % epsilon is how much new input to take at each update step:
  AGC_coeffs.AGC_epsilon(stage) = 1 - exp(-decim / (tau * fs));
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
  AGC_coeffs.AGC_polez1(stage) = polez1;
  AGC_coeffs.AGC_polez2(stage) = polez2;
  
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
  AGC_coeffs.AGC_spatial_iterations(stage) = n_iterations;
  AGC_coeffs.AGC_spatial_FIR(:,stage) = AGC_spatial_FIR;
  AGC_coeffs.AGC_n_taps(stage) = n_taps;
  
  % accumulate DC gains from all the stages, accounting for stage_gain:
  total_DC_gain = total_DC_gain + AGC_params.AGC_stage_gain^(stage-1);
  
  % TODO (dicklyon) -- is this the best binaural mixing plan?
  if stage == 1
    AGC_coeffs.AGC_mix_coeffs(stage) = 0;
  else
    AGC_coeffs.AGC_mix_coeffs(stage) = AGC_params.AGC_mix_coeff / ...
      (tau * (fs / decim));
  end
end

AGC_coeffs.AGC_gain = total_DC_gain;

% % print some results
% AGC_coeffs
% AGC_spatial_FIR = AGC_coeffs.AGC_spatial_FIR


%%
function [FIR, OK] = Design_FIR_coeffs(n_taps, var, mn, n_iter)
% function [FIR, OK] = Design_FIR_coeffs(n_taps, spread_sq, delay, n_iter)

% reduce mean and variance of smoothing distribution by n_iterations:
mn = mn / n_iter;
var = var / n_iter;
switch n_taps
  case 3
    % based on solving to match mean and variance of [a, 1-a-b, b]:
    a = (var + mn*mn - mn) / 2;
    b = (var + mn*mn + mn) / 2;
    FIR = [a, 1 - a - b, b];
    OK = FIR(2) >= 0.2;
  case 5
    % based on solving to match [a/2, a/2, 1-a-b, b/2, b/2]:
    a = ((var + mn*mn)*2/5 - mn*2/3) / 2;
    b = ((var + mn*mn)*2/5 + mn*2/3) / 2;
    % first and last coeffs are implicitly duplicated to make 5-point FIR:
    FIR = [a/2, 1 - a - b, b/2];
    OK = FIR(2) >= 0.1;
  otherwise
    error('Bad n_taps in AGC_spatial_FIR');
end


%% the IHC design coeffs:
function IHC_coeffs = CARFAC_DesignIHC(IHC_params, fs)

if IHC_params.just_hwr
  IHC_coeffs = struct('just_hwr', 1);
  IHC_coeffs.saturation_output = 10;  % HACK: assume some max out
else
  if IHC_params.one_cap
    IHC_coeffs = struct(...
      'just_hwr', 0, ...
      'lpf_coeff', 1 - exp(-1/(IHC_params.tau_lpf * fs)), ...
      'out_rate', 1 / (IHC_params.tau_out * fs), ...
      'in_rate', 1 / (IHC_params.tau_in * fs), ...
      'one_cap', IHC_params.one_cap);
  else
    IHC_coeffs = struct(...
      'just_hwr', 0, ...
      'lpf_coeff', 1 - exp(-1/(IHC_params.tau_lpf * fs)), ...
      'out1_rate', 1 / (IHC_params.tau1_out * fs), ...
      'in1_rate', 1 / (IHC_params.tau1_in * fs), ...
      'out2_rate', 1 / (IHC_params.tau2_out * fs), ...
      'in2_rate', 1 / (IHC_params.tau2_in * fs), ...
      'one_cap', IHC_params.one_cap);
  end
  
  % run one channel to convergence to get rest state:
  IHC_coeffs.rest_output = 0;
  IHC_state = struct( ...
    'cap_voltage', 0, ...
    'cap1_voltage', 0, ...
    'cap2_voltage', 0, ...
    'lpf1_state', 0, ...
    'lpf2_state', 0, ...
    'ihc_accum', 0);
  
  IHC_in = 0;
  for k = 1:30000
    [IHC_out, IHC_state] = CARFAC_IHCStep(IHC_in, IHC_coeffs, IHC_state);
  end
  
  IHC_coeffs.rest_output = IHC_out;
  IHC_coeffs.rest_cap = IHC_state.cap_voltage;
  IHC_coeffs.rest_cap1 = IHC_state.cap1_voltage;
  IHC_coeffs.rest_cap2 = IHC_state.cap2_voltage;
  
  LARGE = 2;
  IHC_in = LARGE;  % "Large" saturating input to IHC; make it alternate
  for k = 1:30000
    [IHC_out, IHC_state] = CARFAC_IHCStep(IHC_in, IHC_coeffs, IHC_state);
    prev_IHC_out = IHC_out;
    IHC_in = -IHC_in;
  end
  
  IHC_coeffs.saturation_output = (IHC_out + prev_IHC_out) / 2;
end

%%
% default design result, running this function with no args, should look
% like this, before CARFAC_Init puts state storage into it:
%
%
% CF = CARFAC_Design
% CF.filter_params
% CF.AGC_params
% CF.filter_coeffs
% CF.AGC_coeffs
% CF.IHC_coeffs
%
% CF =
%                fs: 22050
%     filter_params: [1x1 struct]
%        AGC_params: [1x1 struct]
%        IHC_params: [1x1 struct]
%              n_ch: 96
%        pole_freqs: [96x1 double]
%     filter_coeffs: [1x1 struct]
%        AGC_coeffs: [1x1 struct]
%        IHC_coeffs: [1x1 struct]
%            n_mics: 0
% ans =
%       velocity_scale: 0.2000
%             v_offset: 0.0100
%            v2_corner: 0.2000
%           v_damp_max: 0.0100
%             min_zeta: 0.1200
%     first_pole_theta: 2.4504
%           zero_ratio: 1.4142
%         ERB_per_step: 0.3333
%          min_pole_Hz: 40
% ans =
%           n_stages: 4
%     time_constants: [0.0020 0.0080 0.0320 0.1280]
%     AGC_stage_gain: 2
%         decimation: [8 2 2 2]
%        AGC1_scales: [1 2 4 8]
%        AGC2_scales: [1.5000 3 6 12]
%       detect_scale: 0.1500
%      AGC_mix_coeff: 0.3500
% ans =
%     velocity_scale: 0.2000
%           v_offset: 0.0100
%          v2_corner: 0.2000
%         v_damp_max: 0.0100
%           r_coeffs: [96x1 double]
%           a_coeffs: [96x1 double]
%           c_coeffs: [96x1 double]
%           h_coeffs: [96x1 double]
%           g_coeffs: [96x1 double]
% ans =
%             AGC_stage_gain: 2
%                AGC_epsilon: [0.1659 0.0867 0.0443 0.0224]
%                 decimation: [8 2 2 2]
%     AGC_spatial_iterations: [1 1 2 3]
%            AGC_spatial_FIR: [3x4 double]
%                 AGC_n_taps: [3 5 5 5]
%             AGC_mix_coeffs: [0 0.0317 0.0159 0.0079]
%                   AGC_gain: 15
%               detect_scale: 0.0664
% ans =
%              just_hwr: 0
%             lpf_coeff: 0.4327
%             out1_rate: 0.0023
%              in1_rate: 0.0023
%             out2_rate: 0.0091
%              in2_rate: 0.0091
%               one_cap: 0
%           rest_output: 0.0365
%              rest_cap: 0
%             rest_cap1: 0.9635
%             rest_cap2: 0.9269
%     saturation_output: 0.1507


