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

function [state, updated] = CARFAC_AGCStep(AGC_coeffs, detects, state)
% function [state, updated] = CARFAC_AGCStep(AGC_coeffs, detects, state)
%
% one time step (at decimated low AGC rate) of the AGC state update

n_mics = length(state);
[n_ch, n_AGC_stages] = size(state(1).AGC_memory);  % number of channels

optimize_for_mono = n_mics == 1;  % mono optimization

stage = 1;
ins = AGC_coeffs.detect_scale * detects;
[state, updated] = CARFAC_AGC_Recurse(AGC_coeffs, ins, n_AGC_stages, ...
  n_mics, n_ch, optimize_for_mono, stage, state);





function [state, updated] = CARFAC_AGC_Recurse(coeffs, ins, n_stages, ...
  n_mics, n_ch, mono, stage, state)
% function [state, updated = CARFAC_AGC_Recurse(coeffs, ins, n_stages, ...
%   n_mics, n_ch, mono, stage, state)

decim = coeffs.decimation(stage);  % decim phase for this stage
decim_phase = mod(state(1).decim_phase(stage) + 1, decim);
state(1).decim_phase(stage) = decim_phase;

% accumulate input for this stage from detect or previous stage:
for mic = 1:n_mics
  state(mic).input_accum(:, stage) = ...
    state(mic).input_accum(:, stage) + ins(:, mic);
end

% nothing else to do if it's not the right decim_phase
if decim_phase == 0
  % do lots of work, at decimated rate
  
  % decimated inputs for this stage, and to be decimated more for next:
  for mic = 1:n_mics
    ins(:,mic) = state(mic).input_accum(:, stage) / decim;
    state(mic).input_accum(:, stage) = 0;  % reset accumulator
  end
  
  if stage < n_stages  % recurse to evaluate next stage(s)
    state = CARFAC_AGC_Recurse(coeffs, ins, n_stages, ...
      n_mics, n_ch, mono, stage+1, state);
  end
  
  epsilon = coeffs.AGC_epsilon(stage);  % for this stage's LPF pole
  stage_gain = coeffs.AGC_stage_gain;
  
  for mic = 1:n_mics
    AGC_in = ins(:,mic);  % the newly decimated input for this mic
    
%     AGC_in = max(0, AGC_in);  % don't let neg inputs in
    
    %  add the latest output (state) of next stage...
    if stage < n_stages
      AGC_in = AGC_in + stage_gain * state(mic).AGC_memory(:, stage+1);
    end
    
    AGC_stage_state = state(mic).AGC_memory(:, stage);
    % first-order recursive smoothing filter update, in time:
    AGC_stage_state = AGC_stage_state + ...
                        epsilon * (AGC_in - AGC_stage_state);
    % spatial smooth:
    AGC_stage_state = ...
                  CARFAC_Spatial_Smooth(coeffs, stage, AGC_stage_state);
    % and store the state back (in C++, do it all in place?)
    state(mic).AGC_memory(:, stage) = AGC_stage_state;
    
    if ~mono
      if mic == 1
        this_stage_sum =  AGC_stage_state;
      else
        this_stage_sum = this_stage_sum + AGC_stage_state;
      end
    end
  end
  if ~mono
    mix_coeff = coeffs.AGC_mix_coeffs(stage);
    if mix_coeff > 0
      this_stage_mean = this_stage_sum / n_mics;
      for mic = 1:n_mics
        state(mic).AGC_memory(:, stage) = ...
          state(mic).AGC_memory(:, stage) + ...
            mix_coeff * ...
              (this_stage_mean - state(mic).AGC_memory(:, stage));
      end
    end
  end
  updated = 1;  % bool to say we have new state
else
  updated = 0;
end
