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

function [layer_array, total_width, lag_period_samples] = SAI_DesignLayers( ...
  n_layers, width_per_layer, seglen)
% function [layer_array, total_width] = SAI_DesignLayers( ...
%   n_layers, width_per_layer)
%
% The layer_array is a struct array containing an entry for each layer
% in a layer of power-of-2 decimated pieces of SAI that get composited
% into a log-lag SAI.
% Each struct has the following fields:
%  .width - number of pixels occupied in the final composite SAI,
%     not counting the overlap into pixels counted for other layers.
%  .target_indices - column indices in the final composite SAI,
%     counting the overlap region(s).
%  .lag_curve - for each point in the final composite SAI, the float index
%     in the layer's buffer to interp from.
%  .alpha - the blending coefficent, mostly 1, tapering toward 0 in the overlap
%     region(s).
% Layer 1 has no overlap to it right, and layer n_layers has none to its
% left, but sizes of the target_indices, lag_curve, and alpha vectors are
% otherwise width + left_overlap + right_overlap.  The total width of the
% final composite SAI is the sum of the widths.
% Other fields could be added to hold state, such as history buffers for
% each layer, or those could go in state struct array...

% Elevate these to a param struct?
if nargin < 1
  n_layers = 12;
end
if nargin < 2
  width_per_layer = 24;  % resolution "half life" in space; half-semitones
end
future_lags = 0 * width_per_layer;
width_extra_last_layer = 0 * width_per_layer;
left_overlap = 10;
right_overlap = 10;
first_window_width = seglen;  % Less would be a problem.
min_window_width = 1*width_per_layer;  % or somewhere on that order
window_exponent = 1.4;
alpha_max = 1; 

width_first_layer = future_lags + 2 * width_per_layer;

% Start with NAP_samples_per_SAI_sample, declining to 1 from here:
max_samples_per = 2^(n_layers - 1);
% Construct the overall lag-warping function:
NAP_samples_per_SAI_sample = [ ...
  max_samples_per * ones(1, width_extra_last_layer), ...
  max_samples_per * ...
    2 .^ (-(1:(width_per_layer * (n_layers - 1))) / width_per_layer), ...
  ones(1, width_first_layer), ];  % w/o future for now.

lag_period_samples = cumsum(NAP_samples_per_SAI_sample(end:-1:1));
lag_period_samples = lag_period_samples(end:-1:1);  % Put it back in order.
lag_period_samples = lag_period_samples - lag_period_samples(end - future_lags);

% Each layer needs a lag_warp for a portion of that, divided by
% 2^(layer-1), where the portion includes some overlap into its neighbors
% with higher layer numbers on left, lower on right.

% Layer 1, rightmost, representing recent, current and near-future (negative
% lag) relative to trigger time, has 1 NAP sample per SAI sample.  Other
% layers map more than one NAP sample into 1 SAI sample.  Layer 2 is
% computed as 2X decimated, 2 NAP samples per SAI sample, but then gets 
% interpolated to between 1 and 2 (and outside that range in the overlap
% regions) to connect up smoothly.  Each layer is another 2X decimated.
% The last layer limits out at 1 (representing 2^(n_layers) SAI samples)
% at the width_extra_last_layer SAI samples that extend to the far past.

layer_array = [];  % to hold a struct array
for layer = 1:n_layers
  layer_array(layer).width = width_per_layer;
  layer_array(layer).left_overlap = left_overlap;
  layer_array(layer).right_overlap = right_overlap;
  layer_array(layer).future_lags = 0;
  % Layer decimation factors:  1 1 1 1 2 2 2 4 4 4 8 ...
  layer_array(layer).update_interval = max(1, 2 ^ floor((layer - 2) / 3));
end
% Patch up the exceptions.
layer_array(1).width = width_first_layer;
layer_array(end).width = layer_array(end).width + width_extra_last_layer;
layer_array(1).right_overlap = 0;
layer_array(end).left_overlap = 0;
layer_array(1).future_lags = future_lags;

% For each layer, working backwards, from left, find the locations they
% they render into in the final SAI.
offset = 0;
for layer = n_layers:-1:1
  width = layer_array(layer).width;
  left = layer_array(layer).left_overlap;
  right = layer_array(layer).right_overlap;
  
  % Size of the vectors needed.
  n_final_lags = left + width + right;
  layer_array(layer).n_final_lags = n_final_lags;
  
  % Integer indices into the final composite SAI for this layer.
  target_indices = ((1 - left):(width + right)) + offset;
  layer_array(layer).target_indices = target_indices;
    
  % Make a blending coefficient alpha, ramped in the overlap zone.
  alpha = ones(1, n_final_lags);
  alpha(1:left) = alpha(1:left) .* (1:left)/(left + 1);
  alpha(end + 1 - (1:right)) = ...
    alpha(end + 1 - (1:right)) .* (1:right)/(right + 1);
  layer_array(layer).alpha = alpha * alpha_max;
  
  offset = offset + width;  % total width from left through this layer.
  
  % Smooth across channels a little before picking triggers:
  layer_array(layer).channel_smoothing_scale = 0.25*(layer-1);
end
total_width = offset;  % Return size of SAI this will make.

% for each layer, fill in its lag-resampling function for interp1:
for layer = 1:n_layers
  width = layer_array(layer).width;
  left = layer_array(layer).left_overlap;
  right = layer_array(layer).right_overlap;
  
  % Still need to adjust this to make lags match at edges:
  target_indices = layer_array(layer).target_indices;
  samples_per = NAP_samples_per_SAI_sample(target_indices);
  % Accumulate lag backwards from the zero-lag point, convert to units of
  % samples in the current layer.
  lag_curve = (cumsum(samples_per(end:-1:1))) / 2^(layer-1);
  lag_curve = lag_curve(end:-1:1);  % Turn it back to corrent order.
  % Now adjust it to match the zero-lag point or a lag-point from
  % previous layer, and reverse it back into place.
  if layer == 1
    lag_adjust = lag_curve(end) - 0;
  else
    % Align right edge to previous layer's left edge, adjusting for 2X
    % scaling factor difference.
    lag_adjust = lag_curve(end - right) - last_left_lag / 2;
  end
  lag_curve = lag_curve - lag_adjust;
  % lag_curve is now offsets from right end of layer's frame.
  layer_array(layer).lag_curve = lag_curve;
  % Specify number of point to generate in pre-warp frame.
  layer_array(layer).frame_width = ceil(1 + lag_curve(1));
  if layer < n_layers  % to avoid the left = 0 unused end case.
    % A point to align next layer to.
    last_left_lag = lag_curve(left) - layer_array(layer).future_lags;  
  end
  
  % Specify a good window width (in history buffer, for picking triggers) 
  % in samples for this layer, exponentially approaching minimum.
  layer_array(layer).window_width = round(min_window_width + ...
    first_window_width / window_exponent^(layer - 1));
  
  % Say about how long the history buffer needs to be to shift any trigger
  % location in the range of the window to a fixed location.  Assume
  % using two window placements overlapped 50%.
  n_triggers = 2;
  layer_array(layer).n_window_pos = n_triggers;
  layer_array(layer).buffer_width = layer_array(layer).frame_width + ...
    floor((1 + (n_triggers - 1)/2) * layer_array(layer).window_width);
  % Make sure it's big enough for next layer to shift in what it wants.
  n_shift = ceil(seglen / (2.0^(layer - 1)));
  if layer_array(layer).buffer_width < 6 + n_shift;
    layer_array(layer).buffer_width = 6 + n_shift;
  end
end

return

