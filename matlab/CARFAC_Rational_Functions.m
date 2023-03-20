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

function [stage_numerators, stage_denominators] = ...
  CARFAC_Rational_Functions(CF, ear)
% function [stage_z_numerators, stage_z_denominators] = ...
%   CARFAC_Rational_Functions(CF, ear)
% Return transfer functions of all stages as rational functions.

if nargin < 2
  ear = 1;
end

n_ch = CF.n_ch;
coeffs = CF.ears(ear).CAR_coeffs;

a0 = coeffs.a0_coeffs;
c0 = coeffs.c0_coeffs;
zr = coeffs.zr_coeffs;

% get r, adapted if we have state:
r1 =  coeffs.r1_coeffs;  % max-damping condition
if isfield(CF.ears(ear), 'CAR_state')
  state = CF.ears(ear).CAR_state;
  zB = state.zB_memory; % current delta-r from undamping
  r = r1 + zB;
else
  zB = 0;  % HIGH-level linear condition by default
  r = r1;
end

relative_undamping = zB ./ zr;
g = CARFAC_Stage_g(coeffs, relative_undamping);
a = a0 .* r;
c = c0 .* r;
r2 = r .* r;
h = coeffs.h_coeffs;

stage_denominators = [ones(n_ch, 1), -2 * a, r2];
stage_numerators = [g .* ones(n_ch, 1), g .* (-2 * a + h .* c), g .* r2];

