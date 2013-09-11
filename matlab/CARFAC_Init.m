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

function CF = CARFAC_Init(CF)
% function CF = CARFAC_Init(CF)
%
% Initialize state for one or more ears of CF.
% This allocates and zeros all the state vector storage in the CF struct.

n_ears = CF.n_ears;

for ear = 1:n_ears
  % for now there's only one coeffs, not one per ear
  CF.ears(ear).CAR_state = CAR_Init_State(CF.ears(ear).CAR_coeffs);
  CF.ears(ear).IHC_state = IHC_Init_State(CF.ears(ear).IHC_coeffs);
  CF.ears(ear).AGC_state = AGC_Init_State(CF.ears(ear).AGC_coeffs);
end


function state = CAR_Init_State(coeffs)
n_ch = coeffs.n_ch;
state = struct( ...
  'z1_memory', zeros(n_ch, 1), ...
  'z2_memory', zeros(n_ch, 1), ...
  'zA_memory', zeros(n_ch, 1), ...
  'zB_memory', coeffs.zr_coeffs, ...
  'dzB_memory', zeros(n_ch, 1), ...
  'zY_memory', zeros(n_ch, 1), ...
  'g_memory', coeffs.g0_coeffs, ...
  'dg_memory', zeros(n_ch, 1) ...
  );


function state = AGC_Init_State(coeffs)
n_ch = coeffs(1).n_ch;
n_AGC_stages = coeffs(1).n_AGC_stages;
state = struct([]);
for stage = 1:n_AGC_stages
  % Initialize state recursively...
  state(stage).AGC_memory = zeros(n_ch, 1);
  state(stage).input_accum = zeros(n_ch, 1);
  state(stage).decim_phase = 0;  % integer decimator phase
end


function state = IHC_Init_State(coeffs)
n_ch = coeffs.n_ch;
if coeffs.just_hwr
  state = struct('ihc_accum', zeros(n_ch, 1), ...
                 'ac_coupler', zeros(n_ch, 1));
else
  if coeffs.one_cap
    state = struct( ...
      'ihc_accum', zeros(n_ch, 1), ...
      'cap_voltage', coeffs.rest_cap * ones(n_ch, 1), ...
      'lpf1_state', coeffs.rest_output * ones(n_ch, 1), ...
      'lpf2_state', coeffs.rest_output * ones(n_ch, 1), ...
      'ac_coupler', zeros(n_ch, 1) ...
      );
  else
    state = struct( ...
      'ihc_accum', zeros(n_ch, 1), ...
      'cap1_voltage', coeffs.rest_cap1 * ones(n_ch, 1), ...
      'cap2_voltage', coeffs.rest_cap2* ones(n_ch, 1), ...
      'lpf1_state', coeffs.rest_output * ones(n_ch, 1), ...
      'lpf2_state', coeffs.rest_output * ones(n_ch, 1), ...
      'ac_coupler', zeros(n_ch, 1) ...
      );
  end
end



