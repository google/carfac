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

function ERB = ERB_Hz(CF_Hz, ERB_break_freq, ERB_Q)
% function ERB = ERB_Hz(CF_Hz, ERB_break_freq, ERB_Q)
%
% Auditory filter nominal Equivalent Rectangular Bandwidth
%	Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
% ERB = 24.7 * (1 + 4.37 * CF_Hz / 1000);

if nargin < 3
  ERB_Q = 1000/(24.7*4.37);  % 9.2645
  if nargin < 2
    ERB_break_freq = 1000/4.37;  % 228.833
  end
end

ERB = (ERB_break_freq + CF_Hz) / ERB_Q;
