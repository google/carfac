% // clang-format off
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

function [naps, CF, BM, seg_ohc, seg_agc, firings_all] = ...
  CARFAC_Run_Segment(CF, input_waves)
% function [naps, CF, BM, seg_ohc, seg_agc, firings_all] = ...
%   CARFAC_Run_Segment(CF, input_waves)
%
% This function runs the CARFAC; that is, filters a 1 or more channel
% sound input segment to make one or more neural activity patterns (naps);
% it can be called multiple times for successive segments of any length,
% as long as the returned CF with modified state is passed back in each
% time.
%
% input_waves is a column vector if there's just one audio channel;
% more generally, it has a row per time sample, a column per audio channel.
%
% naps has a row per time sample, a column per filterbank channel, and
% a layer per audio channel if more than 1.
% BM is basilar membrane motion (filter outputs before detection).
%
% the input_waves are assumed to be sampled at the same rate as the
% CARFAC is designed for; a resampling may be needed before calling this.
%
% The function works as an outer iteration on time, updating all the
% filters and AGC states concurrently, so that the different channels can
% interact easily.  The inner loops are over filterbank channels, and
% this level should be kept efficient.
%
% seg_ohc seg_agc are optional extra outputs useful for seeing what the
% ohc nonlinearity and agc are doing; both in terms of extra damping.

if nargout > 2
  do_BM = 1;
else
  do_BM = 0;
end

[n_samp, n_ears] = size(input_waves);

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end

if ~isfield(CF, 'open_loop')  % Find open_loop in CF or default it.
  CF.open_loop = 0;
end

if ~isfield(CF, 'linear_car')  % Find linear in CF or default it.
  CF.linear_car = 0;
end

if ~isfield(CF, 'use_delay_buffer')  % To let CAR be fully parallel.
  CF.use_delay_buffer = 0;
end

n_ch = CF.n_ch;
naps = zeros(n_samp, n_ch, n_ears);  % allocate space for result
if do_BM
  BM = zeros(n_samp, n_ch, n_ears);
  seg_ohc = zeros(n_samp, n_ch, n_ears);
  seg_agc = zeros(n_samp, n_ch, n_ears);
  if CF.do_syn
    firings_all = zeros(n_samp, n_ch, CF.SYN_params.n_classes, n_ears);
  else
    firings_all = [];  % In case someone asked for it when it's not used.
  end
end

% A 2022 addition to make open-loop running behave.  In open_loop mode, these
% coefficients are set, per AGC filter outputs, when we CARFAC_Close_AGC_Loop
% on AGC filter output updates.  They drive the zB and g coefficients to the
% intended value by the next update, but if open_loop they would just keep
% going, extrapolating.  The point of open_loop mode is to stop them moving,
% so we need to make sure these deltas are zeroed in case the mode is switched
% from closed to open, as in some tests to evaluate the transfer functions
% before and after adapting to a signal.
if CF.open_loop
  % The interpolators may be running if it was previously run closed loop.
  for ear = 1:CF.n_ears
    CF.ears(ear).CAR_state.dzB_memory = 0;  % To stop intepolating zB.
    CF.ears(ear).CAR_state.dg_memory = 0;  % To stop intepolating g.
  end
end

% Apply control coeffs to where they are needed.
for ear = 1:CF.n_ears
  CF.ears(ear).CAR_coeffs.linear = CF.linear_car;  % Skip OHC nonlinearity.
  CF.ears(ear).CAR_coeffs.use_delay_buffer = CF.use_delay_buffer;
end

detects = zeros(n_ch, n_ears);
for k = 1:n_samp
  % at each time step, possibly handle multiple channels
  for ear = 1:n_ears

    % This would be cleaner if we could just get and use a reference to
    % CF.ears(ear), but Matlab doesn't work that way...

    [car_out, CF.ears(ear).CAR_state] = CARFAC_CAR_Step( ...
      input_waves(k, ear), CF.ears(ear).CAR_coeffs, CF.ears(ear).CAR_state);

    % update IHC state & output on every time step, too
    [ihc_out, CF.ears(ear).IHC_state, v_recep] = CARFAC_IHC_Step( ...
      car_out, CF.ears(ear).IHC_coeffs, CF.ears(ear).IHC_state);

    if CF.do_syn
      % Use v_recep from IHC_Step to
      % update the SYNapse state and get firings and new nap.
      [syn_out, firings, CF.ears(ear).SYN_state] = CARFAC_SYN_Step( ...
        v_recep, CF.ears(ear).SYN_coeffs, CF.ears(ear).SYN_state);
      % Use sum over syn_outs classes, appropriately scaled, as nap to agc.
      % firings always positive, unless v2 ihc_out.
      % syn_out can go a little negative; should be zero at rest.
      nap = syn_out;
      % Maybe still should add a way to return firings (of the classes).
      firings_all(k, :, :, ear) = firings;
    else
      % v2, ihc_out already has rest_output subtracted.
      nap = ihc_out;  % If no SYN, ihc_out goes to nap and agc as in v2.
    end

    % Use nap to run the AGC update step, maybe decimating internally.
    [CF.ears(ear).AGC_state, AGC_updated] = CARFAC_AGC_Step( ...
      nap, CF.ears(ear).AGC_coeffs, CF.ears(ear).AGC_state);

    % save some output data:
    naps(k, :, ear) = nap;  % output to neural activity pattern
    if do_BM
      BM(k, :, ear) = car_out;
      state = CF.ears(ear).CAR_state;
      seg_ohc(k, :, ear) = state.zA_memory;
      %  seg_agc(k, :, ear) = state.zB_memory;
      % Better thing to return, easier to interpret AGC net (stage 1) state:
      % seg_agc(k, :, ear) = CF.ears(ear).AGC_state(1).AGC_memory;
      seg_agc(k, :, ear) = CF.ears(ear).AGC_state.AGC_memory(:, 1);
    end
  end

  % connect the feedback from AGC_state to CAR_state when it updates;
  % all ears together here due to mixing across them:
  if AGC_updated
    if n_ears > 1
      % do multi-aural cross-coupling:
      CF.ears = CARFAC_Cross_Couple(CF.ears);
    end
    if ~CF.open_loop
      CF = CARFAC_Close_AGC_Loop(CF);  % Starts the interpolation of zB and g.
    end
  end
end
