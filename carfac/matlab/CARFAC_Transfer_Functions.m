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

function [complex_transfns_freqs, ...
  stage_numerators, stage_denominators, group_delays] = ...
  CARFAC_Transfer_Functions(CF, freqs, to_channels, from_channels)
% function [complex_transfns_freqs, ...
%   stage_numerators, stage_denominators, group_delays] = ...
%   CARFAC_Transfer_Functions(CF, freqs, to_channels, from_channels)
%
% Return transfer functions as polynomials in z (nums & denoms);
% And evaluate them at freqs if it's given, to selected output,
%   optionally from selected starting points (from 0, input, by default).
%   complex_transfns_freqs has a row of complex gains per to_channel.

% always start with the rational functions, whether we want to return
% them or not; this defaults to ear 1 only:
[stage_numerators, stage_denominators] = CARFAC_Rational_Functions(CF);

if nargin >= 2
  % Evaluate at the provided list of frequencies.
  if ~isrow(freqs)
    if iscolumn(freqs)
      freqs = freqs';
    else
      error('Bad freqs_row in CARFAC_Transfer_Functions');
    end
  end
  if any(freqs < 0)
    error('Negatives in freqs_row in CARFAC_Transfer_Functions');
  end
  z_row = exp((i * 2 * pi / CF.fs) * freqs);  % z = exp(sT)
  gains = Rational_Eval(stage_numerators, stage_denominators, z_row);
  
  % Now multiply gains from input to output places; use logs?
  log_gains = log(gains);
  cum_log_gains = cumsum(log_gains);  % accum across cascaded stages
  
  % And figure out which cascade products we want:
  n_ch = CF.n_ch;
  if nargin < 3
    to_channels = 1:n_ch;
  end
  if isempty(to_channels) || any(to_channels < 1 | to_channels > n_ch)
    error('Bad to_channels in CARFAC_Transfer_Functions');
  end
  if nargin < 4 || isempty(from_channels)
    from_channels = 0;  % tranfuns from input, called channel 0.
  end
  if length(from_channels) == 1
    from_channels = from_channels * ones(1,length(to_channels));
  end
  % Default to cum gain of 1 (log is 0), from input channel 0:
  from_cum = zeros(length(to_channels), length(z_row));
  not_input = from_channels > 0;
  from_cum(not_input, :) = cum_log_gains(from_channels(not_input), :);
  log_transfns = cum_log_gains(to_channels, :) - from_cum;
  complex_transfns_freqs = exp(log_transfns);
  
  if nargout >= 4
    phases = imag(log_gains);  % no wrapping problem on single stages
    cum_phases = cumsum(phases);  % so no wrapping here either
    group_delays = -diff(cum_phases')';  % diff across frequencies
    group_delays = group_delays ./ (2*pi*repmat(diff(freqs), n_ch, 1));
  end
else
  % If no freqs are provided, do nothing but return the stage info above:
  complex_transfns_freqs = [];
end



function gains = Rational_Eval(numerators, denominators, z_row)
% function gains = Rational_Eval(numerators, denominators, z_row)
% Evaluate rational function at row of z values.

zz = [z_row .* z_row; z_row; ones(size(z_row))];
% dot product of each poly row with each [z2; z; 1] col:
gains = (numerators * zz) ./ (denominators * zz);



function [stage_numerators, stage_denominators] = ...
  CARFAC_Rational_Functions(CF, ear)
% function [stage_z_numerators, stage_z_denominators] = ...
%   CARFAC_Rational_Functions(CF, ear)
% Return transfer functions of all stages as rational functions.

if nargin < 2
  ear = 1;
end

n_ch = CF.n_ch;
coeffs = CF.ears(ear).CAR_coeffs;

a0 = coeffs.a0_coeffs;
c0 = coeffs.c0_coeffs;
zr = coeffs.zr_coeffs;

% get r, adapted if we have state:
r1 =  coeffs.r1_coeffs;  % max-damping condition
if isfield(CF.ears(ear), 'CAR_state')
  state = CF.ears(ear).CAR_state;
  zB = state.zB_memory; % current delta-r from undamping
  r = r1 + zB;
else
  zB = 0;  % HIGH-level linear condition by default
end

relative_undamping = zB ./ zr;
g = CARFAC_Stage_g(coeffs, relative_undamping);
a = a0 .* r;
c = c0 .* r;
r2 = r .* r;
h = coeffs.h_coeffs;

stage_denominators = [ones(n_ch, 1), -2 * a, r2];
stage_numerators = [g .* ones(n_ch, 1), g .* (-2 * a + h .* c), g .* r2];


%% example
% CF = CARFAC_Design
% f = (0:100).^2;  % frequencies to 10 kHz, unequally spaced
% to_ch = 10:10:96;  % selected output channels
% from_ch = to_ch - 10;  % test the inclusion of 0 in from list
% tf = CARFAC_Transfer_Functions(CF, f, to_ch, from_ch);
% figure
% plot(f, 20*log(abs(tf)')/log(10));  % dB vs lin. freq for 10 taps

