% Copyright 2013 The CARFAC Authors. All Rights Reserved.
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

function [state, updated] = CARFAC_AGC_Step(detects, coeffs, state)
% function [state, updated] = CARFAC_AGC_Step(detects, coeffs, state)
%
% one time step of the AGC state update; decimates internally

stage = 1;
AGC_in = coeffs(1).detect_scale * detects;
[state, updated] = CARFAC_AGC_Recurse(coeffs, AGC_in, stage, state);


function [state, updated] = CARFAC_AGC_Recurse(coeffs, AGC_in, ...
  stage, state)
% function [state, updated] = CARFAC_AGC_Recurse(coeffs, AGC_in, ...
%   stage, state)

% decim factor for this stage, relative to input or prev. stage:
decim = coeffs(stage).decimation;
% decim phase of this stage (do work on phase 0 only):
decim_phase = mod(state(stage).decim_phase + 1, decim);
state(stage).decim_phase = decim_phase;

% accumulate input for this stage from detect or previous stage:
state(stage).input_accum = state(stage).input_accum + AGC_in;

% nothing else to do if it's not the right decim_phase
if decim_phase == 0
  % do lots of work, at decimated rate.
  % decimated inputs for this stage, and to be decimated more for next:
  AGC_in = state(stage).input_accum / decim;
  state(stage).input_accum(:) = 0;  % reset accumulator

  if stage < coeffs(1).n_AGC_stages
    state = CARFAC_AGC_Recurse(coeffs, AGC_in, stage+1, state);
    % and add its output to this stage input, whether it updated or not:
    AGC_in = AGC_in + ...
      coeffs(stage).AGC_stage_gain * state(stage + 1).AGC_memory;
  end

  AGC_stage_state = state(stage).AGC_memory;
  % first-order recursive smoothing filter update, in time:
  AGC_stage_state = AGC_stage_state + ...
    coeffs(stage).AGC_epsilon * (AGC_in - AGC_stage_state);
  % spatial smooth:
  AGC_stage_state = ...
    CARFAC_Spatial_Smooth(coeffs(stage), AGC_stage_state);
  % and store the state back (in C++, do it all in place?)
  state(stage).AGC_memory = AGC_stage_state;

  updated = 1;  % bool to say we have new state
else
  updated = 0;
end
