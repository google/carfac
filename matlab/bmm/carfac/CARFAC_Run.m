% Copyright 2012, Google, Inc.
% Author Richard F. Lyon
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
% function [naps, CF, decim_naps] = CARFAC_Run ...
%   (CF, input_waves, AGC_plot_fig_num)
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

% fastest decimated rate determines some interp needed:
decim1 = CF.AGC_params.decimation(1);

naps = zeros(n_samp, n_ch, n_mics);
decim_k = 0;
k_NAP_decim = 0;
NAP_decim = 8;
if nargout > 2
  % make decimated detect output:
  decim_naps = zeros(ceil(n_samp/NAP_decim), CF.n_ch, CF.n_mics);
else
  decim_naps = [];
end


k_AGC = 0;
AGC_plot_decim = 16;  % how often to plot AGC state; TODO: use segments


detects = zeros(n_ch, n_mics);
for k = 1:n_samp
  CF.k_mod_decim = mod(CF.k_mod_decim + 1, decim1);  % global time phase
  k_NAP_decim = mod(k_NAP_decim + 1, NAP_decim);  % phase of decimated nap
  % at each time step, possibly handle multiple channels
  for mic = 1:n_mics
    [filters_out, CF.filter_state(mic)] = CARFAC_FilterStep( ...
      input_waves(k, mic), CF.filter_coeffs, CF.filter_state(mic));
    
    % update IHC state & output on every time step, too
    [ihc_out, CF.IHC_state(mic)] = CARFAC_IHCStep( ...
      filters_out, CF.IHC_coeffs, CF.IHC_state(mic));
    
    detects(:, mic) = ihc_out;  % for input to AGC, and out to SAI
    
    naps(k, :, mic) = ihc_out;  % output to neural activity pattern
    
  end
  if ~isempty(decim_naps) && (k_NAP_decim == 0)
    decim_k = decim_k + 1;   % index of decimated NAP
    for mic = 1:n_mics
      decim_naps(decim_k, :, mic) = CF.IHC_state(mic).ihc_accum / ...
        NAP_decim;  % for cochleagram
      CF.IHC_state(mic).ihc_accum = zeros(n_ch,1);
    end
  end
  % run the AGC update step, taking input from IHC_state, decimating
  % internally, all mics at once due to mixing across them:
  [CF.AGC_state, updated] = ...
    CARFAC_AGCStep(CF.AGC_coeffs, detects, CF.AGC_state);
  
  % connect the feedback from AGC_state to filter_state when it updates
  if updated
    for mic = 1:n_mics
      new_damping = CF.AGC_state(mic).AGC_memory(:, 1);  % stage 1 result
      % set the delta needed to get to new_damping:
      % TODO: update this to use da and dc instead of dr maybe?
      CF.filter_state(mic).dzB_memory = ...
        (new_damping - CF.filter_state(mic).zB_memory) ...
        / decim1;
    end
  end
  
  k_AGC = mod(k_AGC + 1, AGC_plot_decim);
  if AGC_plot_fig_num && k_AGC == 0
    figure(AGC_plot_fig_num); hold off; clf
    set(gca, 'Position', [.25, .25, .5, .5])
    
    maxsum = 0;
    for mic = 1:n_mics
      plot(CF.AGC_state(mic).AGC_memory(:, 1), 'k-', 'LineWidth', 1)
      maxes(mic) = max(CF.AGC_state(mic).AGC_memory(:));
      hold on
      for stage = 1:3;
        plot(2^(stage-1) * (CF.AGC_state(mic).AGC_memory(:, stage) - ...
          2 * CF.AGC_state(mic).AGC_memory(:, stage+1)));
      end
      stage = 4;
      plot(2^(stage-1) * CF.AGC_state(mic).AGC_memory(:, stage));
    end
    axis([0, CF.n_ch+1, -0.01, max(maxes) + 0.01]);
    drawnow
  end
  
end

