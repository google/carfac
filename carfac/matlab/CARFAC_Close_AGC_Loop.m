% Copyright 2012 The CARFAC Authors. All Rights Reserved.
% Author Richard F. Lyon
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

function CF = CARFAC_Close_AGC_Loop(CF)
% function CF = CARFAC_Close_AGC_Loop(CF)

% fastest decimated rate determines interp needed:
decim1 = CF.AGC_params.decimation(1);

for ear = 1:CF.n_ears
  undamping = 1 - CF.ears(ear).AGC_state(1).AGC_memory; % stage 1 result
  % Update the target stage gain for the new damping:
  new_g = CARFAC_Stage_g(CF.ears(ear).CAR_coeffs, undamping);
  % set the deltas needed to get to the new damping:
  CF.ears(ear).CAR_state.dzB_memory = ...
    (CF.ears(ear).CAR_coeffs.zr_coeffs .* undamping - ...
    CF.ears(ear).CAR_state.zB_memory) / decim1;
  CF.ears(ear).CAR_state.dg_memory = ...
    (new_g - CF.ears(ear).CAR_state.g_memory) / decim1;
end
