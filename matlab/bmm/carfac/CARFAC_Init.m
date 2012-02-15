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

function CF_struct = CARFAC_Init(CF_struct, n_mics)
% function CF_struct = CARFAC_Init(CF_struct, n_mics)
%
% Initialize state for n_mics channels (default 1).
% This allocates and zeros all the state vector storage in the CF_struct.

% TODO (dicklyon): Review whether storing state in the same struct as
% the design is a good thing, or whether we want another
% level of object.  I like fewer structs and class types.

if nargin < 2
  n_mics = 1;  % monaural
end

% % this is probably what I'd do in the C++ version:
% if CF_struct.n_mics ~= n_mics;
%   % free the state and make new number of channels
%   % make a struct arrray, one element per mic channel, numbered:
%   for k = 1:n_mics
%     CF_struct.state(k)  = struct('mic_number', k);
%   end
% end
% But this code doesn't work because I don't understand struct arrays.

% For now I don't ever free anything if n_mics is reduced;
% so be sure to respect n_mics, not the size of the state struct array.

AGC_time_constants = CF_struct.AGC_params.time_constants;
n_AGC_stages = length(AGC_time_constants);

CF_struct.n_mics = n_mics;
CF_struct.k_mod_decim = 0;  % time index phase, cumulative over segments

% This zeroing grows the struct array as needed:
for mic = 1:n_mics
  CF_struct.filter_state(mic).z1_memory = zeros(CF_struct.n_ch, 1);
  CF_struct.filter_state(mic).z2_memory = zeros(CF_struct.n_ch, 1);
  % cubic loop
  CF_struct.filter_state(mic).zA_memory = zeros(CF_struct.n_ch, 1);
  % AGC interp
  CF_struct.filter_state(mic).zB_memory = zeros(CF_struct.n_ch, 1);
  % AGC incr
  CF_struct.filter_state(mic).dzB_memory = zeros(CF_struct.n_ch, 1);
  CF_struct.filter_state(mic).zY_memory = zeros(CF_struct.n_ch, 1);
  CF_struct.filter_state(mic).detect_accum = zeros(CF_struct.n_ch, 1);
  % AGC loop filters' state:
  % HACK init
  CF_struct.AGC_state(mic).AGC_memory = zeros(CF_struct.n_ch, n_AGC_stages);
  CF_struct.AGC_state(mic).AGC_sum = zeros(CF_struct.n_ch, 1);
  % IHC state:
  if CF_struct.IHC_coeffs.just_hwr
    CF_struct.IHC_state(mic).ihc_accum = zeros(CF_struct.n_ch, 1);
  else
    CF_struct.IHC_state(mic).cap_voltage = ...
      CF_struct.IHC_coeffs(mic).rest_cap * ones(CF_struct.n_ch, 1);
    CF_struct.IHC_state(mic).cap1_voltage = ...
      CF_struct.IHC_coeffs(mic).rest_cap1 * ones(CF_struct.n_ch, 1);
    CF_struct.IHC_state(mic).cap2_voltage = ...
      CF_struct.IHC_coeffs(mic).rest_cap2 * ones(CF_struct.n_ch, 1);
    CF_struct.IHC_state(mic).lpf1_state = ...
      CF_struct.IHC_coeffs(mic).rest_output * zeros(CF_struct.n_ch, 1);
    CF_struct.IHC_state(mic).lpf2_state = ...
      CF_struct.IHC_coeffs(mic).rest_output * zeros(CF_struct.n_ch, 1);
    CF_struct.IHC_state(mic).ihc_accum = zeros(CF_struct.n_ch, 1);
  end
end


