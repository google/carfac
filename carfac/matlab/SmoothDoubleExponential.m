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

function signal_vecs = SmoothDoubleExponential(signal_vecs, ...
  polez1, polez2, fast_matlab_way)
% function signal_vecs = SmoothDoubleExponential(signal_vecs, ...
%   polez1, polez2, fast_matlab_way)
%
% Smooth the input column vectors in signal_vecs using forward
% and backwards one-pole smoothing filters, backwards first, with
% approximately reflecting edge conditions.
%
% It will be done with Matlab's filter function if "fast_matlab_way"
% is nonzero or defaulted; use 0 to test the algorithm for how to do it
% in sequential c code.

if nargin < 4
  fast_matlab_way = 1;
  % can also use the slow way with explicit loop like we'll do in C++
end

if fast_matlab_way
  [junk, Z_state] = filter(1-polez1, [1, -polez1], ...
    signal_vecs((end-10):end, :));  % initialize state from 10 points
  [signal_vecs(end:-1:1), Z_state] = filter(1-polez2, [1, -polez2], ...
    signal_vecs(end:-1:1), Z_state*polez2/polez1);
  signal_vecs = filter(1-polez1, [1, -polez1], signal_vecs, ...
    Z_state*polez1/polez2);
else
  npts = size(signal_vecs, 1);
  state = zeros(size(signal_vecs, 2));
  for index = npts-10:npts
    input = signal_vecs(index, :);
    state = state + (1 - polez1) * (input - state);
  end
  % smooth backward with polez2, starting with state from above:
  for index = npts:-1:1
    input = signal_vecs(index, :);
    state = state + (1 - polez2) * (input - state);
    signal_vecs(index, :) = state;
  end
  % smooth forward with polez1, starting with state from above:
  for index = 1:npts
    input = signal_vecs(index, :);
    state = state + (1 - polez1) * (input - state);
    signal_vecs(index, :) = state;
  end
end

