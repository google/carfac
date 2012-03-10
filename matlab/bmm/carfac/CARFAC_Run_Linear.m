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

saved_v_damp_max = CF.filter_coeffs.v_damp_max;
CF.filter_coeffs.v_damp_max = 0.00;  % make it linear for now

[n_samp, n_mics] = size(input_waves);
n_ch = CF.n_ch;

if n_mics ~= CF.n_mics
  error('bad number of input_waves channels passed to CARFAC_Run')
end

for mic = 1:CF.n_mics
  % for the state of the AGC interpolator:
  CF.filter_state(mic).zB_memory(:) = extra_damping;  % interpolator state
  CF.filter_state(mic).dzB_memory(:) = 0;  % interpolator slope
end

naps = zeros(n_samp, n_ch, n_mics);

for k = 1:n_samp
  % at each time step, possibly handle multiple channels
  for mic = 1:n_mics
    [filters_out, CF.filter_state(mic)] = CARFAC_FilterStep( ...
      input_waves(k, mic), CF.filter_coeffs, CF.filter_state(mic));
    naps(k, :, mic) = filters_out;  % linear
  end
  % skip IHC and AGC updates
end

CF.filter_coeffs.v_damp_max = saved_v_damp_max;

