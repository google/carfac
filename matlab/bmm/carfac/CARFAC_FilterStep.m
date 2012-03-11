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

function [zY, state] = CARFAC_FilterStep(x_in, filter_coeffs, state)
% function [zY, state] = CARFAC_FilterStep(x_in, filter_coeffs, state)
%
% One sample-time update step for the filter part of the CARFAC.

% Most of the update is parallel; finally we ripple inputs at the end.

% Local nonlinearity zA and AGC feedback zB reduce pole radius:
zA = state.zA_memory;
zB = state.zB_memory + state.dzB_memory; % AGC interpolation
r1 = filter_coeffs.r1_coeffs;
g = state.g_memory + state.dg_memory;  % interp g
v_offset  = filter_coeffs.v_offset;
v2_corner = filter_coeffs.v2_corner;
v_damp_max = filter_coeffs.v_damp_max;

% zB and zA are "extra damping", and multiply zr (compressed theta):
r = r1 - filter_coeffs.zr_coeffs .* (zA + zB); 

% now reduce state by r and rotate with the fixed cos/sin coeffs:
z1 = r .* (filter_coeffs.a0_coeffs .* state.z1_memory - ...
  filter_coeffs.c0_coeffs .* state.z2_memory);
% z1 = z1 + inputs;
z2 = r .* (filter_coeffs.c0_coeffs .* state.z1_memory + ...
  filter_coeffs.a0_coeffs .* state.z2_memory);

% update the "velocity" for cubic nonlinearity, into zA:
zA = (((state.z2_memory - z2) .* filter_coeffs.velocity_scale) + ...
  v_offset) .^ 2;
% soft saturation to make it more like an "essential" nonlinearity:
zA = v_damp_max * zA ./ (v2_corner + zA);

zY = filter_coeffs.h_coeffs .* z2;  % partial output

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

% put new state back in place of old
% (z1 and z2 are genuine temps; the others can update by reference in C)
state.z1_memory = z1;
state.z2_memory = z2;
state.zA_memory = zA;
state.zB_memory = zB;
state.zY_memory = zY;
state.g_memory = g;

% accum the straight hwr version in case someone wants it:
hwr_detect = max(0, zY);  % detect with HWR
state.detect_accum = state.detect_accum + hwr_detect;

