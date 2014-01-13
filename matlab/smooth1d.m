% Copyright 2010 The CARFAC Authors. All Rights Reserved.
% Author Richard F. Lyon
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

function cols = smooth1d(cols, scale)
% smooth1d - Smooth the columns of the input.
%
% output = smooth1d(input, smoothing_factor)
%
% Smooth the columns of input using a one-pole smoothing filter, using the
% provided smoothing factor.
%
% TODO(dross, dicklyon): make this code satisfy the Google Matlab style.

[nr, nc, nl] = size(cols);
if nr == 1
    if nl == 1
        cols = cols';
        [nr, nc, nl] = size(cols);
    else
        disp('error in shape passed to smooth1d')
    end
end

if scale==0
    polez = 0; % no smoothing at all
    return;
else
    % this coefficient matches the curvature at DC of a discrete Gaussian:
    t = scale^2;
    polez = 1 + 1/t - sqrt((1+1/t)^2 - 1); % equiv. to the one below
    % polez = 1 + 1/t - sqrt((2/t) + 1/t^2); % equiv. to the one above
    % polez is Z position of real pole
end

[x, state] = filter(1-polez, [1, -polez], cols);
cols = cols(end:-1:1,:);
[cols, state] = filter(1-polez, [1, -polez], cols, state);
cols = cols(end:-1:1,:);
[cols, state] = filter(1-polez, [1, -polez], cols, state);

return
