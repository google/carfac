function layer_struct = SAI_StabilizeLayer(layer_struct)
% Pick trigger points in buffer, shift rows to offset_from_end,
% and blend into frame

frame = layer_struct.frame;

nap_buffer = real(layer_struct.nap_buffer);
n_buffer_times = size(nap_buffer, 1);
[n_ch, width] = size(frame);

% Make the window to use for all the channels at this layer.
window_size = layer_struct.window_width;
n_window_pos = layer_struct.n_window_pos;
% Windows are always (approx) 50% overlapped:
d_win = window_size / 2;

after_samples = layer_struct.future_lags;

window_range = (1:window_size) + ...
  (n_buffer_times - window_size) - after_samples - ...
  floor((n_window_pos - 1) * d_win);
window = sin((1:window_size)' * pi / window_size);
% This should not go negative!
offset_range = (1:width) + ...
  (n_buffer_times - width - window_size) - floor((n_window_pos - 1) * d_win);
% CHECK
if any(offset_range < 0)
  error;
end

% smooth across channels; more in later layers
smoothed_buffer =  smooth1d(nap_buffer', layer_struct.channel_smoothing_scale)';

% For each buffer column (channel), pick a trigger and align into SAI_frame
for ch = 1:n_ch
  smooth_wave = smoothed_buffer(:, ch);  % for the trigger
  
  % Do several window positions and triggers
  for w = 1:n_window_pos
    % move the window to later and go aggain
    [peak_val, trigger_time] = max(smooth_wave(window_range + ...
      floor((w - 1) * d_win)) .* window);
    nap_wave = nap_buffer(:, ch);  % for the waveform
    if peak_val <= 0  % just use window center instead
      [peak_val, trigger_time] = max(window);
    end
    % TODO(dicklyon):  alpha blend here.
    trigger_time = trigger_time + floor((w - 1) * d_win);
    alpha = (0.025 + peak_val) / (0.5 + peak_val);  % alpha 0.05 to near 1.0
    frame(ch, :) = alpha * nap_wave(trigger_time + offset_range)' + ...
      (1 - alpha) * frame(ch, :);
  end
end

layer_struct.frame = frame;
