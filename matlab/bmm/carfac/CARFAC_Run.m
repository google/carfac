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

function [naps, CF, decim_naps] = CARFAC_Run ...
  (CF, input_waves, AGC_plot_fig_num)
% function [naps, CF, CF.cum_k, decim_naps] = CARFAC_Run ...
%    (CF, input_waves, CF.cum_k, AGC_plot_fig_num)
% This function runs the CARFAC; that is, filters a 1 or more channel
% sound input to make one or more neural activity patterns (naps).
%
% The CF struct holds the filterbank design and state; if you want to
% break the input up into segments, you need to use the updated CF
% to keep the state between segments.
%
% input_waves is a column vector if there's just one audio channel;
% more generally, it has a row per time sample, a column per audio channel.
%
% naps has a row per time sample, a column per filterbank channel, and
% a layer per audio channel if more than 1.
% decim_naps is like naps but time-decimated by the int CF.decimation.
%
% the input_waves are assumed to be sampled at the same rate as the
% CARFAC is designed for; a resampling may be needed before calling this.
%
% The function works as an outer iteration on time, updating all the
% filters and AGC states concurrently, so that the different channels can
% interact easily.  The inner loops are over filterbank channels, and
% this level should be kept efficient.
%
% See other functions for designing and characterizing the CARFAC:
% CF = CARFAC_Design(fs, CF_filter_params, CF_AGC_params, n_mics)
% transfns = CARFAC_Transfer_Functions(CF, to_chans, from_chans)

[n_samp, n_mics] = size(input_waves);
n_ch = CF.n_ch;

if nargin < 3
  AGC_plot_fig_num = 0;
end

if n_mics ~= CF.n_mics
  error('bad number of input_waves channels passed to CARFAC_Run')
end

% pull coeffs out of struct first, into local vars for convenience
decim = CF.AGC_params.decimation;

naps = zeros(n_samp, n_ch, n_mics);
if nargout > 2
  % make decimated detect output:
  decim_naps = zeros(ceil(n_samp/decim), CF.n_ch, CF.n_mics);
else
  decim_naps = [];
end

decim_k = 0;

sum_abs_response = 0;

for k = 1:n_samp
  CF.k_mod_decim = mod(CF.k_mod_decim + 1, decim);  % global time phase
  % at each time step, possibly handle multiple channels
  for mic = 1:n_mics
    [filters_out, CF.filter_state(mic)] = CARFAC_FilterStep( ...
      input_waves(k, mic), CF.filter_coeffs, CF.filter_state(mic));

    % update IHC state & output on every time step, too
    [ihc_out, CF.IHC_state(mic)] = CARFAC_IHCStep( ...
      filters_out, CF.IHC_coeffs, CF.IHC_state(mic));

%     sum_abs_response = sum_abs_response + abs(filters_out);

    naps(k, :, mic) = ihc_out;  % output to neural activity pattern
  end

  % conditionally update all the AGC stages and channels now:
  if CF.k_mod_decim == 0
    % just for the plotting option:
    decim_k = decim_k + 1;   % index of decimated signal for display
    if ~isempty(decim_naps)
      for mic = 1:n_mics
        % this is HWR out of filters, not IHCs
        avg_detect = CF.filter_state(mic).detect_accum / decim;
        % This HACK is the IHC version:
        avg_detect = CF.IHC_state(mic).ihc_accum / decim;  % for cochleagram
        decim_naps(decim_k, :, mic) = avg_detect;  % for cochleagram
%         decim_naps(decim_k, :, mic) = sum_abs_response / decim;  % HACK for mechanical out ABS
%         sum_abs_response(:) = 0;
      end
    end

    % get the avg_detects to connect filter_state to AGC_state:
    avg_detects = zeros(n_ch, n_mics);
    for mic = 1:n_mics
%       % mechanical response from filter output through HWR as AGC in:
%       avg_detects(:, mic) = CF.filter_state(mic).detect_accum / decim;
      CF.filter_state(mic).detect_accum(:) = 0;  % zero the detect accumulator
      % New HACK, IHC output relative to rest as input to AGC:
      avg_detects(:, mic) = CF.IHC_state(mic).ihc_accum / decim;
      CF.IHC_state(mic).ihc_accum(:) = 0;  % zero the detect accumulator
    end

    % run the AGC update step:
    CF.AGC_state = CARFAC_AGCStep(CF.AGC_coeffs, avg_detects, CF.AGC_state);

    % connect the feedback from AGC_state to filter_state:
    for mic = 1:n_mics
      new_damping = CF.AGC_state(mic).sum_AGC;
%       max_damping = 0.15;  % HACK
%       new_damping = min(new_damping, max_damping);
      % set the delta needed to get to new_damping:
      CF.filter_state(mic).dzB_memory = ...
        (new_damping - CF.filter_state(mic).zB_memory) ...
          / decim;
    end

    if AGC_plot_fig_num
      figure(AGC_plot_fig_num); hold off
      maxsum = 0;
      for mic = 1:n_mics
        plot(CF.AGC_state(mic).AGC_memory)
        agcsum = sum(CF.AGC_state(mic).AGC_memory, 2);
        maxsum(mic) = max(maxsum, max(agcsum));
        hold on
        plot(agcsum, 'k-')
      end
      axis([0, CF.n_ch, 0, max(0.001, max(maxsum))]);
      drawnow
    end
  end
end

