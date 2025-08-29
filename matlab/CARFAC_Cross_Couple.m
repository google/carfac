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

function ears = CARFAC_Cross_Couple(ears);

n_ears = length(ears);
if n_ears > 1
  n_stages = ears(1).AGC_coeffs.n_AGC_stages;
  % now cross-ear mix the stages that updated (leading stages at phase 0):
  for stage = 1:n_stages
    if ears(1).AGC_coeffs.non_decimating || ...
        (ears(1).AGC_state.decim_phase(stage) == 0)
      mix_coeff = ears(1).AGC_coeffs.mix_coeffs(stage);
      if mix_coeff > 0  % Non-critical optimization, typically for stage 1.
        this_stage_sum = 0;
        % sum up over the ears and get their mean:
        for ear = 1:n_ears
          stage_state = ears(ear).AGC_state.AGC_memory(:, stage);
          this_stage_sum = this_stage_sum + stage_state;
        end
        this_stage_mean = this_stage_sum / n_ears;
        % now move them all toward the mean:
        for ear = 1:n_ears
          stage_state = ears(ear).AGC_state.AGC_memory(:, stage);
          ears(ear).AGC_state.AGC_memory(:, stage) = ...
            stage_state + mix_coeff * (this_stage_mean - stage_state);
        end
      end
    else
      break  % Don't look at later stages if this one not updated.
    end
  end
end
