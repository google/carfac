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

seglen = round(fs * 0.020);  % Pick about 20 ms segments
frame_rate = fs / seglen;
n_segs = ceil(n_samp / seglen);

% Make the history buffers in the layers_array:
for layer = 1:n_layers
  layer_array(layer).nap_buffer = zeros(layer_array(layer).buffer_width, n_ch);
  layer_array(layer).nap_fraction = 0;  % leftover fraction to shift in.
end

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
      nap_buffer = real(layer_array(layer).nap_buffer);
      n_buffer_times = size(nap_buffer, 1);
      width = layer_array(layer).frame_width;  % To render linear SAI to.
      new_frame = zeros(n_ch, width);
      
      % Make the window to use for all the channels at this layer.
      layer_factor = 1.5;
      window_size = layer_array(layer).window_width;
      after_samples = layer_array(layer).future_lags;

      window_range = (1:window_size) + ...
        (n_buffer_times - window_size) - after_samples;
      window = sin((1:window_size)' * pi / window_size);
      % This should not go negative!
      offset_range = (1:width) + ...
        (n_buffer_times - width - window_size);
      % CHECK
      if any(offset_range < 0)
        error;
      end
      
      % smooth across channels; more in later layers
      smoothed_buffer =  smooth1d(nap_buffer', 0.25*(layer - 2))';
      
      % For each buffer column (channel), pick a trigger and align into SAI_frame
      for ch = 1:n_ch
        smooth_wave = smoothed_buffer(:, ch);  % for the trigger
        
        [peak_val, trigger_time] = max(smooth_wave(window_range) .* window);
        nap_wave = nap_buffer(:, ch);  % for the waveform
        if peak_val <= 0  % just use window center instead
          [peak_val, trigger_time] = max(window);
        end
        if layer == n_layers  % mark the trigger points to display as imaginary.
          layer_array(layer).nap_buffer(trigger_time + window_range(1) - 1, ch) = ...
            layer_array(layer).nap_buffer(trigger_time + window_range(1) - 1, ch) + 1i;
        end
        new_frame(ch, :) = nap_wave(trigger_time + offset_range)';
      end
      composite_frame = SAI_BlendFrameIntoComposite(new_frame, ...
        layer_array(layer), composite_frame);
    end
  end
  
  
  lag_marginal = mean(composite_frame, 1);  % means max out near 1 or 2
  frame_bottom = zeros(size(composite_frame));  % will end up being 1/3
  n_bottom_rows = size(frame_bottom, 1);
  for height = 1:n_bottom_rows
    big_ones = lag_marginal > 1*height/n_bottom_rows;
    frame_bottom(n_bottom_rows - height + 1, big_ones) = 2;  % 2 for black
  end
    
  if 0 == mod(seg_num, update_interval) || seg_num == 1
    coc_gram = layer_array(end).nap_buffer';
    [n_ch, n_width] = size(composite_frame);
    coc_gram = [coc_gram, zeros(n_ch, n_width - size(coc_gram, 2))];
    trigger_gram = 2 * (imag(coc_gram) ~= 0);
    coc_gram = real(coc_gram);
  end
  
  display_frame = [coc_gram; trigger_gram; ...
    composite_frame(floor(1:0.5:end), :); frame_bottom];
  
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




