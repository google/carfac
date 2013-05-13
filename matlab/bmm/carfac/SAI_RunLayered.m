% Copyright 2013, Google, Inc.
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

function [frame_rate, num_frames] = SAI_RunLayered(CF, input_waves)
% function [CF, SAI_movie] = CARFAC_Run_Layered_SAI(CF, input_waves)
% This function runs the CARFAC and generates an SAI movie, dumped as PNG
% files for now.

% Layer 1 is not decimated from the 22050 rate; subsequent layers have
% smoothing and 2X decimation each.  All layers get composited togehter
% into movie frames.

n_ch = CF.n_ch;
[n_samp, n_ears] = size(input_waves);
if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end
fs = CF.fs;

% Design the composite log-lag SAI using these parameters and defaults.
n_layers = 10;
width_per_layer = 40;
[layer_array, total_width] = SAI_DesignLayers(n_layers, width_per_layer);

% Make the composite SAI image array.
composite_frame = zeros(n_ch, total_width);

seglen = round(fs / 30);  % Pick about 60 fps
frame_rate = fs / seglen;
n_segs = ceil(n_samp / seglen);

% Make the history buffers in the layers_array:
for layer = 1:n_layers
  layer_array(layer).nap_buffer = zeros(layer_array(layer).buffer_width, n_ch);
  layer_array(layer).nap_fraction = 0;  % leftover fraction to shift in.
  % The SAI frame is transposed to be image-like.
  layer_array(layer).frame = zeros(n_ch, layer_array(layer).frame_width);
end

n_marginal_rows = 34;
marginals = [];

average_frame = 0;
for seg_num = 1:n_segs
  % k_range is the range of input sample indices for this segment
  if seg_num == n_segs
    % The last segment may be short of seglen, but do it anyway:
    k_range = (seglen*(seg_num - 1) + 1):n_samp;
  else
    k_range = seglen*(seg_num - 1) + (1:seglen);
  end
  % Process a segment to get a slice of decim_naps, and plot AGC state:
  [seg_naps, CF] = CARFAC_Run_Segment(CF, input_waves(k_range, :));
  
  seg_naps = max(0, seg_naps);  % Rectify
  
  if seg_num == n_segs  % pad out the last result
    seg_naps = [seg_naps; zeros(seglen - size(seg_naps,1), size(seg_naps, 2))];
  end
 
  % Shift new data into some or all of the layer buffers:
  layer_array = SAI_UpdateBuffers(layer_array, seg_naps, seg_num);


  for layer = n_layers:-1:1  % blend from coarse to fine
    update_interval = layer_array(layer).update_interval;
    if 0 == mod(seg_num, update_interval)
      layer_array(layer) = SAI_StabilizeLayer(layer_array(layer));
      new_frame = layer_array(layer).frame;
      composite_frame = SAI_BlendFrameIntoComposite( ...
        layer_array(layer), composite_frame);
    end
  end
 
  if isempty(marginals)
    composite_width = size(composite_frame, 2);
    marginals = zeros(n_marginal_rows, composite_width);
  end
  for row = n_marginal_rows:-1:11
    % smooth from row above (lower number)
    marginals(row, :) = marginals(row, :) + ...
      2^((10 - row)/2) * (1.06*marginals(row - 1, :) - marginals(row, :));
  end
  lag_marginal = mean(composite_frame, 1);  % means max out near 1 or 2
  for row = 10:-1:1
    marginals(row, :) = (lag_marginal - smooth1d(lag_marginal, 30)') - ...
      (10 - row) / 40;
  end
    
  if 0 == mod(seg_num, update_interval) || seg_num == 1
    coc_gram = layer_array(end).nap_buffer';
    [n_ch, n_width] = size(composite_frame);
    coc_gram = [coc_gram, zeros(n_ch, n_width - size(coc_gram, 2))];
  end
  
  display_frame = [coc_gram; ...
    composite_frame(floor(1:0.5:end), :); 20*max(0,marginals)];
  
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




