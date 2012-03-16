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

function [ihc_out, state] = CARFAC_IHC_Step(filters_out, coeffs, state);
% function [ihc_out, state] = CARFAC_IHC_Step(filters_out, coeffs, state);
%
% One sample-time update of inner-hair-cell (IHC) model, including the
% detection nonlinearity and one or two capacitor state variables.

just_hwr = coeffs.just_hwr;

if just_hwr
  ihc_out = max(0, filters_out);
  state.ihc_accum = state.ihc_accum + ihc_out;
else
  one_cap = coeffs.one_cap;

  detect = CARFAC_Detect(filters_out);  % detect with HWR or so

  if one_cap
    ihc_out = detect .* state.cap_voltage;
    state.cap_voltage = state.cap_voltage - ihc_out .* coeffs.out_rate + ...
      (1 - state.cap_voltage) .* coeffs.in_rate;
  else
    % change to 2-cap version more like Meddis's:
    ihc_out = detect .* state.cap2_voltage;
    state.cap1_voltage = state.cap1_voltage - ...
      (state.cap1_voltage - state.cap2_voltage) .* coeffs.out1_rate + ...
      (1 - state.cap1_voltage) .* coeffs.in1_rate;

    state.cap2_voltage = state.cap2_voltage - ihc_out .* coeffs.out2_rate + ...
      (state.cap1_voltage - state.cap2_voltage) .* coeffs.in2_rate;
  end

  % smooth it twice with LPF:

  state.lpf1_state = state.lpf1_state + coeffs.lpf_coeff * ...
    (ihc_out - state.lpf1_state);

  state.lpf2_state = state.lpf2_state + coeffs.lpf_coeff * ...
    (state.lpf1_state - state.lpf2_state);

  ihc_out = state.lpf2_state - coeffs.rest_output;

  state.ihc_accum = state.ihc_accum + max(0, ihc_out);
end
