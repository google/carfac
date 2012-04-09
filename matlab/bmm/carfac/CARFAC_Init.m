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

function CF_struct = CARFAC_Init(CF_struct, n_ears)
% function CF_struct = CARFAC_Init(CF_struct, n_ears)
%
% Initialize state for n_ears channels (default 1).
% This allocates and zeros all the state vector storage in the CF_struct.

% TODO (dicklyon): Review whether storing state in the same struct as
% the design is a good thing, or whether we want another
% level of object.  I like fewer structs and class types.

if nargin < 2
  n_ears = 1;  % monaural default
end

% % this is probably what I'd do in the C++ version:
% if CF_struct.n_ears ~= n_ears;
%   % free the state and make new number of channels
%   % make a struct arrray, one element per ear channel, numbered:
%   for k = 1:n_ears
%     CF_struct.state(k)  = struct('ear_number', k);
%   end
% end
% But this code doesn't work because I don't understand struct arrays.

% For now I don't ever free anything if n_ears is reduced;
% so be sure to respect n_ears, not the size of the state struct array.

CF_struct.n_ears = n_ears;

% These inits grow the struct arrays as needed:
for ear = 1:n_ears
  % for now there's only one coeffs, not one per ear
  CF_struct.CAR_state(ear) = CAR_Init_State(CF_struct.CAR_coeffs);
  CF_struct.IHC_state(ear) = IHC_Init_State(CF_struct.IHC_coeffs);
  CF_struct.AGC_state(ear) = AGC_Init_State(CF_struct.AGC_coeffs);
end


function state = CAR_Init_State(coeffs)
n_ch = coeffs.n_ch;
state = struct( ...
  'z1_memory', zeros(n_ch, 1), ...
  'z2_memory', zeros(n_ch, 1), ...
  'zA_memory', zeros(n_ch, 1), ...
  'zB_memory', zeros(n_ch, 1), ...
  'dzB_memory', zeros(n_ch, 1), ...
  'zY_memory', zeros(n_ch, 1), ...
  'g_memory', coeffs.g0_coeffs, ...
  'dg_memory', zeros(n_ch, 1) ...
  );


function state = AGC_Init_State(coeffs)
n_ch = coeffs.n_ch;
n_AGC_stages = coeffs.n_AGC_stages;
state = struct( ...
  'AGC_memory', zeros(n_ch, n_AGC_stages), ...
  'input_accum', zeros(n_ch, n_AGC_stages), ...
  'decim_phase', zeros(n_AGC_stages, 1) ... % integer decimator phase
  );


function state = IHC_Init_State(coeffs)
n_ch = coeffs.n_ch;
if coeffs.just_hwr
  state = struct('ihc_accum', zeros(n_ch, 1));
else
  if coeffs.one_cap
    state = struct( ...
      'ihc_accum', zeros(n_ch, 1), ...
      'cap_voltage', coeffs.rest_cap * ones(n_ch, 1), ...
      'lpf1_state', coeffs.rest_output * ones(n_ch, 1), ...
      'lpf2_state', coeffs.rest_output * ones(n_ch, 1) ...
      );
  else
    state = struct( ...
      'ihc_accum', zeros(n_ch, 1), ...
      'cap1_voltage', coeffs.rest_cap1 * ones(n_ch, 1), ...
      'cap2_voltage', coeffs.rest_cap2* ones(n_ch, 1), ...
      'lpf1_state', coeffs.rest_output * ones(n_ch, 1), ...
      'lpf2_state', coeffs.rest_output * ones(n_ch, 1) ...
      );
  end
end



