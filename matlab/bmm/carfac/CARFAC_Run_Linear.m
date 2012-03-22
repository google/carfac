% Copyright 2012, Google, Inc.
% Author Richard F. Lyon
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

function [naps, CF] = CARFAC_Run_Linear(CF, input_waves, extra_damping)
% function [naps, CF] = CARFAC_Run_Linear(CF, input_waves, extra_damping)
%
% This function runs the CARFAC; that is, filters a 1 or more channel
% sound input to make one or more neural activity patterns (naps);
% however, unlike CARFAC_Run, it forces it to be linear, and gives a
% linear (not detected) output.

saved_v_damp_max = CF.CAR_coeffs.v_damp_max;
CF.CAR_coeffs.v_damp_max = 0.00;  % make it linear for now

[n_samp, n_ears] = size(input_waves);
n_ch = CF.n_ch;

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end

for ear = 1:CF.n_ears
  % Set the state of damping, and prevent interpolation from there:
  CF.CAR_state(ear).zB_memory(:) = extra_damping;  % interpolator state
  CF.CAR_state(ear).dzB_memory(:) = 0;  % interpolator slope
  CF.CAR_state(ear).g_memory = CARFAC_Stage_g( ...
    CF.CAR_coeffs(ear), extra_damping);
  CF.CAR_state(ear).dg_memory(:) = 0;  % interpolator slope
end

naps = zeros(n_samp, n_ch, n_ears);

for k = 1:n_samp
  % at each time step, possibly handle multiple channels
  for ear = 1:n_ears
    [filters_out, CF.CAR_state(ear)] = CARFAC_CAR_Step( ...
      input_waves(k, ear), CF.CAR_coeffs, CF.CAR_state(ear));
    naps(k, :, ear) = filters_out;  % linear
  end
  % skip IHC and AGC updates
end

CF.CAR_coeffs.v_damp_max = saved_v_damp_max;

