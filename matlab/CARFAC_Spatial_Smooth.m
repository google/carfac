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

function stage_state = CARFAC_Spatial_Smooth(coeffs, stage_state)
% function AGC_state = CARFAC_Spatial_Smooth( ...
%   n_taps, n_iterations, FIR_coeffs, AGC_state)

n_iterations = coeffs.AGC_spatial_iterations;

use_FIR = n_iterations >= 0;

if use_FIR
  FIR_coeffs = coeffs.AGC_spatial_FIR;
  switch coeffs.AGC_spatial_n_taps
    case 3
      for iter = 1:n_iterations
        stage_state = ...
          FIR_coeffs(1) * stage_state([1, 1:(end-1)], :) + ...
          FIR_coeffs(2) * stage_state + ...
          FIR_coeffs(3) * stage_state([2:end, end], :);
      end
    case 5  % 5-tap smoother duplicates first and last coeffs:
      for iter = 1:n_iterations
        stage_state = ...
          FIR_coeffs(1) * (stage_state([1, 2, 1:(end-2)], :) + ...
          stage_state([1, 1:(end-1)], :)) + ...
          FIR_coeffs(2) *  stage_state + ...
          FIR_coeffs(3) * (stage_state([2:end, end], :) + ...
          stage_state([3:end, end, end-1], :));
      end
    otherwise
      error('Bad AGC_spatial_n_taps in CARFAC_Spatial_Smooth');
  end
else
  % use IIR method, back-and-forth first-order smoothers:
  stage_state = SmoothDoubleExponential(stage_state, ...
    coeffs.AGC_polez1(stage), coeffs.AGC_polez2(stage));
end
