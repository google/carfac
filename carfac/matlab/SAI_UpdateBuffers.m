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

function layer_array = SAI_UpdateBuffers(layer_array, seg_naps, seg_num)
% function layer_array = SAI_UpdateBuffers(layer_array, seg_naps, seg_num)
%
% Input/Output: layer_array contains all the coefficients and state for
% the layer of different time scales of SAI;
% we might want to separate these as in CARFAC.
%
% seg_naps is a new segmeent of NAP from the CAR-FAC to shift into the
% first layer.  Each subsequent layer gets input off the input end of the
% previous layer, with smoothing and decimation.
%
% The segment index seg_num is used to control sub-sampled updates of
% the larger-scale layers.

n_layers = length(layer_array);
[seg_len, n_nap_ch] = size(seg_naps);

% Array of what to shift in to first or next layer.
new_chunk = seg_naps;

gain = 1.05;  % gain from layer to layer; could be layer dependent.

%% 
% Decimate using a 2-3-4-filter and partial differencing emphasize onsets:
kernel = filter([1 1]/2, 1, filter([1 1 1]/3, 1, [1 1 1 1 0 0 0 0]/4));
% kernel = kernel + 2*diff([0, kernel]);
% figure(1)
% plot(kernel)

%% 
for layer = 1:n_layers
  [n_lags, n_ch] = size(layer_array(layer).nap_buffer);
  if (n_nap_ch ~= n_ch)
    error('Wrong number of channels in nap_buffer.');
  end
  
  interval = layer_array(layer).update_interval;
  if (0 == mod(seg_num, interval))
    % Account for 2X decimation and infrequent updates; find number of time
    % points to shift in.  Tolerate slip of a fraction of a sample.
    n_shift = seg_len * interval / (2.0^(layer - 1));
    if layer > 1
      % Add the leftover fraction before floor.
      n_shift = n_shift + layer_array(layer).nap_fraction;
      layer_array(layer).nap_fraction = n_shift - floor(n_shift);
      n_shift = floor(n_shift);
      % Grab new stuff from new end (big time indices) of previous layer.
      % Take twice as many times as we need, + 5, for decimation, and do
      % smoothing to get new points.
      new_chunk = ...
        layer_array(layer - 1).nap_buffer((end - 2*n_shift - 4):end, :);
      new_chunk = filter(kernel, 1, new_chunk);
      % new_chunk = gain * new_chunk(7:2:end, :);
      % try a little extra smoothing:
      new_chunk = gain * (new_chunk(7:2:end, :) + new_chunk(6:2:(end-1), :))/2;
      
    end
    % Put new stuff in at latest time indices.
    layer_array(layer).nap_buffer = ...
      [layer_array(layer).nap_buffer((1 + n_shift):end, :); ...
      new_chunk];  % this should fit just right if we have n_shift new times.
  end
end

return

