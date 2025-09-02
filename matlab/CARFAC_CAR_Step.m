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

function [car_out, state] = CARFAC_CAR_Step(x_in, CAR_coeffs, state)
% function [zY, state] = CARFAC_CAR_Step(x_in, CAR_coeffs, state)
%
% One sample-time update step for the filter part of the CARFAC.

% Most of the update is parallel; maybe ripple inputs at the end.

% do the DOHC stuff:
g = state.g_memory + state.dg_memory;  % interp g
zB = state.zB_memory + state.dzB_memory; % AGC interpolation state
% update the nonlinear function of "velocity", and zA (delay of z2):
zA = state.zA_memory;
v = state.z2_memory - zA;
if CAR_coeffs.linear
  nlf = 1;
else
  % nlf = CARFAC_OHC_NLF(v .* widen, CAR_coeffs);  % widen v with feedback
  nlf = CARFAC_OHC_NLF(v, CAR_coeffs);
end
% zB * nfl is "undamping" delta r:
r = CAR_coeffs.r1_coeffs + zB .* nlf;
zA = state.z2_memory;

% now reduce state by r and rotate with the fixed cos/sin coeffs:
z1 = r .* (CAR_coeffs.a0_coeffs .* state.z1_memory - ...
  CAR_coeffs.c0_coeffs .* state.z2_memory);
% z1 = z1 + inputs;
z2 = r .* (CAR_coeffs.c0_coeffs .* state.z1_memory + ...
  CAR_coeffs.a0_coeffs .* state.z2_memory);

if isfield(CAR_coeffs, 'use_delay_buffer') && CAR_coeffs.use_delay_buffer;
  % To avoid the sequential ripple, use zY as delay per stage.
  % Optional fully-parallel update uses a delay per stage.
  zY = state.zY_memory;
  zY(2:end) = zY(1:(end-1));  % Propagate delayed last outputs zy
  zY(1) = x_in;  % fill in new input
  z1 = z1 + zY;  % add new stage inputs to z1 states
  zY = g .* (CAR_coeffs.h_coeffs .* z2 + zY);  % Outputs from z2
else
  zY = CAR_coeffs.h_coeffs .* z2;  % partial output
  % Ripple input-output path, instead of parallel, to avoid delay...
  % this is the only part that doesn't get computed "in parallel":
  in_out = x_in;
  for ch = 1:length(zY)
    % could do this here, or later in parallel:
    z1(ch) = z1(ch) + in_out;
    % ripple, saving final channel outputs in zY
    in_out = g(ch) * (in_out + zY(ch));
    zY(ch) = in_out;
  end
end

% put new state back in place of old
% (z1 is a genuine temp; the others can update by reference in C)
state.z1_memory = z1;
state.z2_memory = z2;
state.zA_memory = zA;
state.zB_memory = zB;
state.zY_memory = zY;
state.g_memory = g;

% AC couple the filters_out, with 20 Hz corner (previously part of IHC)
ac_diff = zY - state.ac_coupler;
state.ac_coupler = state.ac_coupler + CAR_coeffs.ac_coeff * ac_diff;

car_out = ac_diff;

