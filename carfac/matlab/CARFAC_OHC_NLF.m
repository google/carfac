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

function nlf = CARFAC_OHC_NLF(velocities, CAR_coeffs)
% function nlf = CARFAC_OHC_NLF(velocities, CAR_coeffs)
% start with a quadratic nonlinear function, and limit it via a
% rational function; make the result go to zero at high
% absolute velocities, so it will do nothing there.

nlf = 1 ./ (1 + ...
  (velocities * CAR_coeffs.velocity_scale + CAR_coeffs.v_offset) .^ 2 );

