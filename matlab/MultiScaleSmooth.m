% Copyright 2012 The CARFAC Authors. All Rights Reserved.
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

function MultiScaleSmooth(waves, n_scales)
% function MultiScaleSmooth(waves, n_scales)
%
% Let's take columns as waveforms, and smooth them to different scales;
% these inputs can be carfac NAPs, for example, and the peaks of the
% smoothed versions can be used as trigger events, even tracking back
% to less-smoothed versions.
% And we'll deciamte 2:1 at every other smoothing.
%
% Until we decide what we want, we'll just plot things, one plot per scale.

fig_offset1 = 10;

if nargin < 2
  n_scales = 20;
end

smoothed = waves;

for scale_no = 1:n_scales

  if mod(scale_no, 2) == 1
    newsmoothed = filter([1, 1]/2, 1, smoothed);
    diffsmoothed = max(0, smoothed - newsmoothed);
    smoothed = newsmoothed;
  else
    newsmoothed = filter([1, 2, 1]/4, 1, smoothed);
    diffsmoothed = max(0, smoothed - newsmoothed);
    smoothed = newsmoothed(1:2:end, :);
  end

  figure(scale_no + fig_offset1)
  imagesc(squeeze(smoothed(:,:,1))')

  drawnow
  pause(1)

end

