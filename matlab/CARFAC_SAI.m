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

function [sai_frame, sai_state, naps] = CARFAC_SAI(naps, k, sai_state, SAI_params)
% function sai = CARFAC_SAI(naps, k, sai_state, SAI_params)
%
% ...work in progress...
%
% Calculate the Stabilized Auditory Image from naps; 
% I think this is a binaural SAI by Steven Ness
%
% k seems to be a time index; it's an incremental update of the images...
% but this doesn't sound like a proper incremental approach...
%

[n_samp, n_ch, n_ears] = size(naps);

if nargin < 4
  SAI_params = struct( ...
    'frame_jump', 200, ...
    'sai_width', 500, ...
    'threshold_alpha', 0.99, ...
    'threshold_jump_factor', 1.2, ...
    'threshold_jump_offset', 0.1};
end

threshold_alpha = SAI_params.threshold_alpha;
threshold_jump = SAI_params.threshold_jump_factor;
threshold_offset = SAI_params.threshold_jump_offset;

sai2 = reshape(sai_state.sai, SAI_params.sai_width * n_ch, n_ears);
naps2 = reshape(naps, n_samp * n_ch, n_ears);

for ear = 1:n_ears
  data = naps(k, :, ear)';
  above_threshold = (sai_state(ear).lastdata > ...
    sai_state(ear).thresholds) & ...
    (sai_state(ear).lastdata > data);
  sai_state(ear).thresholds(above_threshold) = ...
    data(above_threshold) * threshold_jump + threshold_offset;
  sai_state(ear).thresholds(~above_threshold) = ...
    sai_state(ear).thresholds(~above_threshold) * threshold_alpha;
  sai_state(ear).lastdata = data;
  
  % Update SAI image with strobe data.
  otherear = 3 - ear;
  
  % Channels that are above the threhsold
  above_ch = find(above_threshold);
  
  % If we are above the threshold, set the trigger index and reset the
  % sai_index
  sai_state(ear).trigger_index(above_ch) = k;
  sai_state(ear).sai_index(above_ch) = 1;
  
  % Copy the right data from the nap to the sai
  chans = (1:n_ch)';
  fromindices = sai_state(ear).trigger_index() + (chans - 1) * n_samp;
  toindices = min((sai_state(ear).sai_index() + (chans - 1) * sai_params.sai_width), sai_params.sai_width * n_ch);
  sai2(toindices,ear) = naps2(fromindices, otherear);
  
  sai_state(ear).trigger_index(:) = sai_state(ear).trigger_index(:) + 1;
  sai_state(ear).sai_index(:) = sai_state(ear).sai_index(:) + 1;
end


sai_frame = reshape(sai2,sai_params.sai_width,n_ch,n_ears);
sai_state.sai = sai;  % probably this is not exactly what we want to store as state...


