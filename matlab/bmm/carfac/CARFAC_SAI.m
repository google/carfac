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

function [CF, sai] = CARFAC_SAI(CF, k, n_mics, naps, sai)
% function sai = CARFAC_SAI(CF_struct, n_mics, naps, sai)
%
% Calculate the Stabilized Auditory Image from naps
%

  threshold_alpha = CF.sai_params.threshold_alpha;
  threshold_jump = CF.sai_params.threshold_jump_factor;
  threshold_offset = CF.sai_params.threshold_jump_offset;

  sai2 = reshape(sai,CF.sai_params.sai_width * CF.n_ch,n_mics);
  naps2 = reshape(naps,CF.n_samp * CF.n_ch,n_mics);

  for mic = 1:n_mics
    data = naps(k, :, mic)';
    above_threshold = (CF.sai_state(mic).lastdata > ...
                       CF.sai_state(mic).thresholds) & ...
                      (CF.sai_state(mic).lastdata > data);
    CF.sai_state(mic).thresholds(above_threshold) = ...
        data(above_threshold) * threshold_jump + threshold_offset;
    CF.sai_state(mic).thresholds(~above_threshold) = ...
        CF.sai_state(mic).thresholds(~above_threshold) * threshold_alpha;
    CF.sai_state(mic).lastdata = data;

    % Update SAI image with strobe data.
    othermic = 3 - mic;

    % Channels that are above the threhsold
    above_ch = find(above_threshold);

    % If we are above the threshold, set the trigger index and reset the
    % sai_index
    CF.sai_state(mic).trigger_index(above_ch) = k;
    CF.sai_state(mic).sai_index(above_ch) = 1;

    % Copy the right data from the nap to the sai
    chans = (1:CF.n_ch)';
    fromindices = CF.sai_state(mic).trigger_index() + (chans - 1) * CF.n_samp;
    toindices = min((CF.sai_state(mic).sai_index() + (chans - 1) * ...
                     CF.sai_params.sai_width), ...
                     CF.sai_params.sai_width * CF.n_ch);
    sai2(toindices,mic) = naps2(fromindices,othermic);

    CF.sai_state(mic).trigger_index(:) = CF.sai_state(mic).trigger_index(:) + 1;
    CF.sai_state(mic).sai_index(:) = CF.sai_state(mic).sai_index(:) + 1;

  end

  sai = reshape(sai2,CF.sai_params.sai_width,CF.n_ch,n_mics);
  naps = reshape(naps2,CF.n_samp, CF.n_ch,n_mics);

