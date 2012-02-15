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

% Use each stage previous Y as input to next:
inputs = [x_in; state.zY_memory(1:(end-1))];

% Local nonlinearity zA and AGC feedback zB reduce pole radius:
zA = state.zA_memory;
zB = state.zB_memory + state.dzB_memory; % AGC interpolation
r0 = filter_coeffs.r_coeffs;
r = r0 - filter_coeffs.c_coeffs .* (zA + zB);

% now reduce state by r and rotate with the fixed cos/sin coeffs:
z1 = r .* (filter_coeffs.a_coeffs .* state.z1_memory - ...
  filter_coeffs.c_coeffs .* state.z2_memory);
z1 = z1 + inputs;
z2 = r .* (filter_coeffs.c_coeffs .* state.z1_memory + ...
  filter_coeffs.a_coeffs .* state.z2_memory);

% update the "velocity" for cubic nonlinearity, into zA:
zA = (((state.z2_memory - z2) .* filter_coeffs.velocity_scale) - 0.2) .^ 2;

velocity_damp_max = 1/16;
zA = velocity_damp_max * zA ./ (1 + zA);  % soft max at velocity_damp_max

% Get outputs from inputs and new z2 values:
zY = filter_coeffs.g_coeffs .* (inputs + filter_coeffs.h_coeffs .* z2);

% put new state back in place of old
state.z1_memory = z1;
state.z2_memory = z2;
state.zA_memory = zA;
state.zB_memory = zB;
state.zY_memory = zY;

% accum the straight hwr version for the sake of AGC range:
hwr_detect = max(0, zY);  % detect with HWR
state.detect_accum = state.detect_accum + hwr_detect;

