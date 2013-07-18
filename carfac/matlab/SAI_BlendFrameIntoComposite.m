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

function composite_frame = SAI_BlendFrameIntoComposite( ...
  layer_struct, composite_frame)

new_frame = layer_struct.frame;
n_ch = size(new_frame, 1);

if layer_struct.right_overlap == 0  % A layer 1 hack only.
  for row = 1:n_ch
    % Taper new_frame down near zero lag for a nicer result...
    taper_size = round(6 + 60*(row/n_ch)^2);  %  hack
    zero_pos = layer_struct.frame_width - layer_struct.future_lags;
    taper = [-taper_size:min(taper_size, layer_struct.future_lags)];
    col_range = zero_pos + taper;
    taper = (0.4 + 0.6*abs(taper) / taper_size) .^ 2;
    taper(taper == 0) = 0.5;
    new_frame(row, col_range) = new_frame(row, col_range) .* taper;
  end
end

alpha = layer_struct.alpha;
lag_curve = layer_struct.lag_curve;
target_columns = layer_struct.target_indices;

frame_width = size(new_frame, 2);

% Lags are measured from 0 at the right.
stretched_frame = interp1(new_frame', frame_width - lag_curve)';
alpha = repmat(alpha, size(new_frame, 1), 1);
composite_frame(:, target_columns) = ...
  (1 - alpha) .* composite_frame(:, target_columns) + ...
  alpha .* stretched_frame;
