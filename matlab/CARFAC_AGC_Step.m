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
% one time step of the AGC state update; maybe decimates internally.
% AGC input is detects; detect_scale is already in the coefficients.

if coeffs.simpler_decimating
  % Simpler way:  Just use one decimation factor, which can be 1 or
  % any positive integer, and update all stages in parallel; no more
  % recursion and no more special-casing for non-decimating.
  decim = coeffs.decimation(1);  % Later decimation values are all 1.
  % decim phase (do work on phase 0 only):
  decim_phase = mod(state.decim_phase(1) + 1, decim);
  state.decim_phase(1) = decim_phase;  % only phase(1) is used.

  % Accumulate input for AGC from detects:
  state.input_accum(:) = state.input_accum(:) + detects;

  % nothing else to do if it's not the right decim_phase
  if decim_phase == 0
    AGC_in = state.input_accum(:, 1) / decim;
    state.input_accum(:, 1) = 0;  % reset accumulator

    % 2025 new decimation-free structure, intended to be GPU friendly.
    % state is a column per stage, to be updated in parallel.
    % First the time-update step with detects as input;
    % new state is a linear function of old state and inputs, and we allow
    % a sample of delay per stage, not using each new stage state until the
    % the next time step, to make it parallelize efficiently and to make
    % the delay a bit closer to what happens in the decimating version.
    state.AGC_memory = ...  % Broadcasting gain rows across channels
      coeffs.feedback_gains .* state.AGC_memory + ...
      coeffs.input_gains .* AGC_in + ...
      coeffs.next_stage_gains .* ...
      [state.AGC_memory(:, 2:end), zeros(coeffs.n_ch, 1)];
    % Then the 3-point FIR spatial smooth.
    % Just mix in some left and right neighbors like in book figure 19.6.
    % The .* broadcasts the row of coeffs across channels (rows); all
    % stages are done in parallel.
    state.AGC_memory = ...  % Mix in neighbors, replicating edges.
      coeffs.spatial_FIR(1, :) .* state.AGC_memory([1, 1:(end-1)], :) + ...
      coeffs.spatial_FIR(2, :) .* state.AGC_memory(:, :) + ...
      coeffs.spatial_FIR(3, :) .* state.AGC_memory([2:end, end], :);
    updated = 1;  % Updates on every call.
  else
    updated = 0;
  end
else
  % Old method keeps track of things using the call stack, and uses
  % updates as soon as possible to minimize added delay.
  stage = 1;
  [state, updated] = CARFAC_AGC_Recurse(coeffs, detects, stage, state);
end


function [state, updated] = CARFAC_AGC_Recurse(coeffs, AGC_in, ...
  stage, state)
% function [state, updated] = CARFAC_AGC_Recurse(coeffs, AGC_in, ...
%   stage, state)

% decim factor for this stage, relative to input or prev. stage:
decim = coeffs.decimation(stage);
% decim phase of this stage (do work on phase 0 only):
decim_phase = mod(state.decim_phase(stage) + 1, decim);
state.decim_phase(stage) = decim_phase;

% Accumulate input for this stage from detect or from previous stage:
state.input_accum(:, stage) = state.input_accum(:, stage) + AGC_in;

% nothing else to do if it's not the right decim_phase
if decim_phase == 0
  % do lots of work, at decimated rate.
  % decimated inputs for this stage, and to be decimated more for next:
  AGC_in = state.input_accum(:, stage) / decim;
  state.input_accum(:, stage) = 0;  % reset accumulator

  AGC_stage_state = state.AGC_memory(:, stage);
  % First-order recursive smoothing filter update, in time:
  AGC_stage_state = ...
    coeffs.feedback_gains(stage) * AGC_stage_state + ...
    coeffs.input_gains(stage) * AGC_in;
  if stage < coeffs.n_AGC_stages
    % Add in state from next stage, if exists, whether it updates or not.
    % This call maybe updates stage+1 and later ones, doesn't touch this
    % stage, whose state we're playing with in AGC_stage_state.
    state = CARFAC_AGC_Recurse(coeffs, AGC_in, stage+1, state);  % RECURSE
    % The coefficient for adding in the next stage state is premultiplied
    % by the AGC_epsilon.
    AGC_stage_state = AGC_stage_state + ...
      coeffs.next_stage_gains(stage) * state.AGC_memory(:, stage+1);
  end
  % spatial smooth:
  FIR_coeffs = coeffs.spatial_FIR(:, stage);
  AGC_stage_state = ...
    FIR_coeffs(1) * AGC_stage_state([1, 1:(end-1)], :) + ...
    FIR_coeffs(2) * AGC_stage_state + ...
    FIR_coeffs(3) * AGC_stage_state([2:end, end], :);
  % Finally store the state back (could do it all in-place).
  state.AGC_memory(:, stage) = AGC_stage_state;

  updated = 1;  % bool to say we have new state; only used from stage 1.
else
  updated = 0;
end
