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

function [frame_rate, num_frames] = SAI_Run(CF, sai_struct, input_waves)
% function [frame_rate, num_frames] = SAI_Run(CF, sai_struct, input_waves)
% This function runs the CARFAC and display an SAI movie.

n_ch = CF.n_ch;
[n_samp, n_ears] = size(input_waves);
if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end
fs = CF.fs;

seglen = round(fs / 30);  % Pick about 30 fps
frame_rate = fs / seglen;
n_segs = ceil(n_samp / seglen);

% State stored in sai_struct.
% Make the history buffer.
buffer_width = sai_struct.width + ...
    floor((1 + (sai_struct.n_window_pos - 1)/2) * sai_struct.window_width);
sai_struct.nap_buffer = zeros(buffer_width, n_ch);
% The SAI frame is transposed to be image-like.
sai_struct.frame = zeros(n_ch, sai_struct.width);

for seg_num = 1:n_segs
  % seg_range is the range of input sample indices for this segment
  if seg_num == n_segs
    % The last segment may be short of seglen, but do it anyway:
    seg_range = (seglen*(seg_num - 1) + 1):n_samp;
  else
    seg_range = seglen*(seg_num - 1) + (1:seglen);
  end
  % NOTE: seg_naps might have multiple channels.
  [seg_naps, CF] = CARFAC_Run_Segment(CF, input_waves(seg_range, :));

  % Rectify.
  % NOTE: This might not be necessary.
  seg_naps = max(0, seg_naps);  

  sai_struct = SAI_Run_Segment(sai_struct, seg_naps);
  
  cmap = 1 - gray;  % jet
  figure(10)
  imagesc(32 * sai_struct.frame);
  colormap(cmap);
  colorbar

  drawnow
%  imwrite(32*display_frame, cmap, sprintf('frames/frame%05d.png', seg_num));
end

num_frames = seg_num;

return
