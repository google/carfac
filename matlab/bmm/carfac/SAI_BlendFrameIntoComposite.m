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

function composite_frame = SAI_BlendFrameIntoComposite(new_frame, ...
  layer_struct, composite_frame)

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
