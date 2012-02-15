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

function state = CARFAC_AGCStep(AGC_coeffs, avg_detects, state)
% function state = CARFAC_AGCStep(AGC_coeffs, avg_detects, state)
%
% one time step (at decimated low AGC rate) of the AGC state update

n_AGC_stages = length(AGC_coeffs.AGC_epsilon);
n_mics = length(state);
n_ch = size(state(1).AGC_sum, 1);  % number of channels

optimize_for_mono = n_mics == 1;  % mono optimization
if ~optimize_for_mono
  stage_sum = zeros(n_ch, 1);
end

for stage = 1:n_AGC_stages
  if ~optimize_for_mono  % skip if mono
    if stage > 1
      prev_stage_mean = stage_sum / n_mics;
    end
    stage_sum(:) = 0;  % sum accumulating over mics at this stage
  end
  epsilon = AGC_coeffs.AGC_epsilon(stage);  % for this stage's LPF pole
  polez1 = AGC_coeffs.AGC1_polez(stage);
  polez2 = AGC_coeffs.AGC2_polez(stage);
  for mic = 1:n_mics
    if stage == 1
      AGC_in = AGC_coeffs.detect_scale * avg_detects(:,mic);
      AGC_in = max(0, AGC_in);  % don't let neg inputs in
    else
      % prev. stage mixed with prev_stage_sum
      if optimize_for_mono
        % Mono optimization ignores AGC_mix_coeff,
        % assuming all(prev_stage_mean == AGC_memory(:, stage - 1));
        % but we also don't even allocate or compute the sum or mean.
        AGC_in = AGC_coeffs.AGC_stage_gain * ...
          state(mic).AGC_memory(:, stage - 1);
      else
        AGC_in = AGC_coeffs.AGC_stage_gain * ...
          (AGC_coeffs.AGC_mix_coeff * prev_stage_mean + ...
            (1 - AGC_coeffs.AGC_mix_coeff) * ...
              state(mic).AGC_memory(:, stage - 1));
      end
    end
    AGC_stage = state(mic).AGC_memory(:, stage);
    % first-order recursive smooting filter update:
    AGC_stage = AGC_stage + epsilon * (AGC_in - AGC_stage);

    % spatially spread it; using diffusion coeffs like in smooth1d
    AGC_stage = SmoothDoubleExponential(AGC_stage, polez1, polez2);

    state(mic).AGC_memory(:, stage) = AGC_stage;
    if stage == 1
      state(mic).sum_AGC = AGC_stage;
    else
      state(mic).sum_AGC = state(mic).sum_AGC + AGC_stage;
    end
    if ~optimize_for_mono
      stage_sum = stage_sum + AGC_stage;
    end
  end
end


