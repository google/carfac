% // clang-format off
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
% The synapse's resting point (spont_p, res_lpf_inits and spont_sub) is
% first solved from its current coeffs by SYN_Solve_Rest, so after changing
% v_halfs, n_fibers or agc_weights (e.g. to model synaptopathy) call this
% again and the synapse starts at its new rest, with output 0 in quiet.

n_ears = CF.n_ears;

for ear = 1:n_ears
  % for now there's only one coeffs, not one per ear
  CF.ears(ear).CAR_state = CAR_Init_State(CF.ears(ear).CAR_coeffs);
  CF.ears(ear).IHC_state = IHC_Init_State(CF.ears(ear).IHC_coeffs);
  CF.ears(ear).AGC_state = AGC_Init_State(CF.ears(ear).AGC_coeffs);
  if CF.do_syn
     CF.ears(ear).SYN_coeffs = SYN_Solve_Rest(CF.ears(ear).SYN_coeffs);
     CF.ears(ear).SYN_state = SYN_Init_State(CF.ears(ear).SYN_coeffs);
  end
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
  'dg_memory', zeros(n_ch, 1), ...
  'ac_coupler', zeros(n_ch, 1) ...
  );


function state = AGC_Init_State(coeffs)
% 2025 new way, one struct instead of array of them.
state = struct( ...
  'AGC_memory', zeros(coeffs.n_ch, coeffs.n_AGC_stages), ...
   'decim_phase', zeros(1, coeffs.n_AGC_stages));  % small ints
if coeffs.simpler_decimating  % One decimation factor vs per stage.
  state.input_accum = zeros(coeffs.n_ch, 1);
else
  state.input_accum = zeros(coeffs.n_ch, coeffs.n_AGC_stages);
end


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
      'cap2_voltage', coeffs.rest_cap2 * ones(n_ch, 1), ...
      'lpf1_state', coeffs.rest_output * ones(n_ch, 1) ...
      );
  end
end

function coeffs = SYN_Solve_Rest(coeffs)
% Solve the synapse's resting point from its current coeffs, i.e. the
% steady state of CARFAC_SYN_Step with v_recep at 0: q = a1*r and
% r = (1 - q)*s, so q = a1*s / (1 + a1*s).  Each is per channel and class,
% whether v_halfs is per class or per channel and class.
n_ch = coeffs.n_ch;
s = 1 ./ (1 + exp(coeffs.v_halfs ./ coeffs.v_widths));
q = coeffs.a1 .* s ./ (1 + coeffs.a1 .* s);
coeffs.res_lpf_inits = ones(n_ch, 1) .* q;  % Resting reservoir.
coeffs.spont_p = ones(n_ch, 1) .* (coeffs.a2 .* (1 - q) .* s);  % Firing prob.
coeffs.spont_sub = (coeffs.n_fibers .* coeffs.spont_p) * coeffs.agc_weights';

function state = SYN_Init_State(coeffs)
n_ch = coeffs.n_ch;
state = struct( ...
  'reservoirs', ones(n_ch, 1) .* coeffs.res_lpf_inits, ...  % 0 full, 1 empty.
  'lpf_state', ones(n_ch, 1) .* coeffs.spont_p);
