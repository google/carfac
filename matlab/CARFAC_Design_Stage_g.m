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

function ideal_g = CARFAC_Design_Stage_g(CAR_coeffs, undamping)
% function ideal_g = CARFAC_Stage_g(CAR_coeffs, undamping)
% Return the stage gain g needed to get unity gain at DC
% See also CARFAC_Stage_g, simplified approximation used at run time,
% based on quadratic coefficient computed at Design time.

r1 = CAR_coeffs.r1_coeffs;  % at max damping
a0 = CAR_coeffs.a0_coeffs;
c0 = CAR_coeffs.c0_coeffs;
h  = CAR_coeffs.h_coeffs;
zr = CAR_coeffs.zr_coeffs;
r  = r1 + zr .* undamping;  % r at specified damping
n  = 1 - 2*r.*a0 + r.^2;
d  = 1 - 2*r.*a0 + h.*r.*c0 + r.^2;
ideal_g = n ./ d;
