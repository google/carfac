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

function sai_struct = SAI_Run_Segment(sai_struct, input)
% function sai_frame = SAI_Run_Segment(sai_struct, input)
% Compute a single SAI frame from the given multichannel input signal.

% Store state used between successive calls to this function in sai_struct.
if ~isfield(sai_struct, 'nap_buffer')
  % Make the history buffer.
  buffer_width = sai_struct.width + ...
      floor((1 + (sai_struct.n_window_pos - 1)/2) * sai_struct.window_width);
  n_ch = size(input, 2);
  sai_struct.nap_buffer = zeros(buffer_width, n_ch);
end
if ~isfield(sai_struct, 'frame')
  % The SAI frame is transposed to be image-like.
  sai_struct.frame = zeros(n_ch, sai_struct.width);
end

if size(input, 1) < sai_struct.window_width  % pad out the last result
  input = [input; ...
           zeros(sai_struct.window_width - size(input, 1), size(input, 2))];
end
size(input)
sai_struct.window_width

assert(size(input, 1) == sai_struct.window_width)

% Shift new data into the buffer.
num_shift = size(input, 1);
sai_struct.nap_buffer = [sai_struct.nap_buffer((1 + num_shift):end,:); ...
                         input];

sai_struct = SAI_StabilizeLayer(sai_struct);
  
return
