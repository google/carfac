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

function [frame_rate, num_frames] = SAI_RunLayered(CF, input_waves)
% function [CF, SAI_movie] = CARFAC_RunLayered(CF, input_waves)
% This function runs the CARFAC and generates an SAI movie, dumped as PNG
% files for now.
%
% Computes a "layered" SAI composed of images computed at several
% time scales.
%
% Layer 1 is not decimated from the 22050 rate; subsequent layers have
% smoothing and 2X decimation each.  All layers get composited together
% into movie frames.

n_ch = CF.n_ch;
[n_samp, n_ears] = size(input_waves);
if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end
fs = CF.fs;

seglen = round(fs / 30);  % Pick about 30 fps
frame_rate = fs / seglen;
n_segs = ceil(n_samp / seglen);


% Design the composite log-lag SAI using these parameters and defaults.
n_layers = 15
width_per_layer = 36;
[layer_array, total_width, lags] = ...
  SAI_DesignLayers(n_layers, width_per_layer, seglen);

% Find where in the lag curve corresponds to the piano black keys:
pitches = fs ./ lags;
key_indices = [];
df = log(2)/width_per_layer;
for f = [BlackKeyFrequencies, 8, 4, 2, 1-df, 1, 1+df, 0.5, 0.25, 0.125, ...
    -2000, -1000, -500, -250, -125];  % Augment with beat.
  [dist, index] = min((f - pitches).^2);
  key_indices = [key_indices, index];
end
piano = zeros(1, total_width);
piano(key_indices) = 1;
piano = [piano; piano; piano];


% Make the composite SAI image array.
composite_frame = zeros(n_ch, total_width);

% Make the history buffers in the layers_array:
for layer = 1:n_layers
  layer_array(layer).nap_buffer = zeros(layer_array(layer).buffer_width, n_ch);
  layer_array(layer).nap_fraction = 0;  % leftover fraction to shift in.
  % The SAI frame is transposed to be image-like.
  layer_array(layer).frame = zeros(n_ch, layer_array(layer).frame_width);
end

n_marginal_rows = 100;
marginals = [];
average_composite = 0;

future_lags = layer_array(1).future_lags;
% marginals_frame = zeros(total_width - future_lags + 2*n_ch, total_width);
marginals_frame = zeros(n_ch, total_width);

for seg_num = 1:n_segs
  % seg_range is the range of input sample indices for this segment
  if seg_num == n_segs
    % The last segment may be short of seglen, but do it anyway:
    seg_range = (seglen*(seg_num - 1) + 1):n_samp;
  else
    seg_range = seglen*(seg_num - 1) + (1:seglen);
  end
  [seg_naps, CF] = CARFAC_Run_Segment(CF, input_waves(seg_range, :));
  
  seg_naps = max(0, seg_naps);  % Rectify
  
  if seg_num == n_segs  % pad out the last result
    seg_naps = [seg_naps; zeros(seglen - size(seg_naps,1), size(seg_naps, 2))];
  end
 
  % Shift new data into some or all of the layer buffers:
  layer_array = SAI_UpdateBuffers(layer_array, seg_naps, seg_num);

  for layer = n_layers:-1:1  % Stabilize and blend from coarse to fine
    update_interval = layer_array(layer).update_interval;
    if 0 == mod(seg_num, update_interval)
      layer_array(layer) = SAI_StabilizeLayer(layer_array(layer));
      composite_frame = SAI_BlendFrameIntoComposite( ...
        layer_array(layer), composite_frame);
    end
  end
  
  average_composite = average_composite + ...
    0.01 * (composite_frame - average_composite);
 
  if isempty(marginals)
    marginals = zeros(n_marginal_rows, total_width);
  end
  for row = n_marginal_rows:-1:11
    % smooth from row above (lower number)
    marginals(row, :) = marginals(row, :) + ...
      2^((10 - row)/8) * (1.01*marginals(row - 1, :) - marginals(row, :));
  end
  lag_marginal = mean(composite_frame, 1);  % means max out near 1 or 2
  lag_marginal = lag_marginal - 0.75*smooth1d(lag_marginal, 30)';
  
  freq_marginal = mean(layer_array(1).nap_buffer);
  % emphasize local peaks:
  freq_marginal = freq_marginal - 0.5*smooth1d(freq_marginal, 5)';
  
  
%   marginals_frame = [marginals_frame(:, 2:end), ...
%     [lag_marginal(1:(end - future_lags)), freq_marginal(ceil((1:(2*end))/2))]'];
  marginals_frame = [marginals_frame(:, 2:end), freq_marginal(1:end)'];
  
  for row = 10:-1:1
    marginals(row, :) = lag_marginal - (10 - row) / 40;
  end
    
  if 0 == mod(seg_num, update_interval) || seg_num == 1
    coc_gram = layer_array(end).nap_buffer';
    [n_ch, n_width] = size(composite_frame);
    coc_gram = [coc_gram, zeros(n_ch, n_width - size(coc_gram, 2))];
    coc_gram = coc_gram(:, (end-total_width+1):end);
  end
  
  display_frame = [ ...  % coc_gram; ...
    4 * marginals_frame; ...
    composite_frame(ceil((1:(2*end))/2), :); ...
    piano; ...
    10*max(0,marginals)];
  
  cmap = jet;
  cmap = 1 - gray;  % jet
  figure(10)
  image(32*display_frame);
  colormap(cmap);

  drawnow
  imwrite(32*display_frame, cmap, sprintf('frames/frame%05d.png', seg_num));
end

num_frames = seg_num;

return


function frequencies = BlackKeyFrequencies
black_indices = [];
for index = 0:87
  if any(mod(index, 12) == [1 4 6 9 11])
    black_indices = [black_indices, index];
  end
end
frequencies = 27.5 * 2.^(black_indices / 12);


