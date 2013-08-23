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

function layer_struct = SAI_StabilizeLayer(layer_struct)
% Pick trigger points in buffer, shift rows to offset_from_end,
% and blend into frame

frame = layer_struct.frame;

nap_buffer = real(layer_struct.nap_buffer);
n_buffer_times = size(nap_buffer, 1);
[n_ch, width] = size(frame);

% Make the window to use for all the channels at this layer.
window_width = layer_struct.window_width;
n_window_pos = layer_struct.n_window_pos;
% Windows are always (approx) 50% overlapped:
window_hop = window_width / 2;

window = sin((1:window_width)' * pi / window_width);
window_start = (n_buffer_times - window_width) - ...
    floor((n_window_pos - 1) * window_hop);
window_range = (1:window_width) + window_start - layer_struct.future_lags;
% This should not go negative!
offset_range = (1:width) + window_start - width;
% CHECK
if any(offset_range < 0)
  error;
end

% smooth across channels; more in later layers
smoothed_buffer = smooth1d(nap_buffer', layer_struct.channel_smoothing_scale)';

% For each buffer column (channel), pick a trigger and align into SAI_frame
for ch = 1:n_ch
  smooth_wave = smoothed_buffer(:, ch);  % for the trigger

  % Do several window positions and triggers
  for w = 1:n_window_pos
    % move the window to later and go again
    current_window_offset = floor((w - 1) * window_hop);
    [peak_val, trigger_time] = ...
        max(smooth_wave(window_range + current_window_offset) .* window);
    nap_wave = nap_buffer(:, ch);  % for the waveform
    if peak_val <= 0  % just use window center instead
      [peak_val, trigger_time] = max(window);
    end
    trigger_time = trigger_time + current_window_offset;
    alpha = (0.025 + peak_val) / (0.5 + peak_val);  % alpha 0.05 to near 1.0
    frame(ch, :) = alpha * nap_wave(trigger_time + offset_range)' + ...
      (1 - alpha) * frame(ch, :);
  end
end

layer_struct.frame = frame;
