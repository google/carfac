% Copyright 2012 The CARFAC Authors. All Rights Reserved.
% Author Richard F. Lyon
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

function [naps, CF] = CARFAC_Run_Linear(CF, input_waves, relative_undamping)
% function [naps, CF] = CARFAC_Run_Linear(CF, input_waves, relative_undamping)
%
% This function runs the CARFAC; that is, filters a 1 or more channel
% sound input to make one or more neural activity patterns (naps);
% however, unlike CARFAC_Run, it forces it to be linear, and gives a
% linear (not detected) output.

% only saving one of these, really:
velocity_scale = CF.ears(1).CAR_coeffs.velocity_scale;
for ear = 1:CF.n_ears
  % make it effectively linear for now
  CF.ears(ear).CAR_coeffs.velocity_scale = 0;
end

[n_samp, n_ears] = size(input_waves);
n_ch = CF.n_ch;

if nargin < 3
  relative_undamping = 1;  % default to min-damping condition
end

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end

for ear = 1:CF.n_ears
  coeffs = CF.ears(ear).CAR_coeffs;
  % Set the state of damping, and prevent interpolation from there:
  CF.ears(ear).CAR_state.zB_memory(:) = coeffs.zr_coeffs .* relative_undamping;  % interpolator state
  CF.ears(ear).CAR_state.dzB_memory(:) = 0;  % interpolator slope
  CF.ears(ear).CAR_state.g_memory = CARFAC_Stage_g(coeffs, relative_undamping);
  CF.ears(ear).CAR_state.dg_memory(:) = 0;  % interpolator slope
end

naps = zeros(n_samp, n_ch, n_ears);

for k = 1:n_samp
  % at each time step, possibly handle multiple channels
  for ear = 1:n_ears
    [filters_out, CF.ears(ear).CAR_state] = CARFAC_CAR_Step( ...
      input_waves(k, ear), CF.ears(ear).CAR_coeffs, CF.ears(ear).CAR_state);
    naps(k, :, ear) = filters_out;  % linear
  end
  % skip IHC and AGC updates
end

for ear = 1:CF.n_ears
  CF.ears(ear).CAR_coeffs.velocity_scale = velocity_scale;
end

