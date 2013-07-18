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

function conductance = CARFAC_Detect(x_in)
% function conductance = CARFAC_detect(x_in)
% An IHC-like sigmoidal detection nonlinearity for the CARFAC.
% Resulting conductance is in about [0...1.3405]


a = 0.175;   % offset of low-end tail into neg x territory
% this parameter is adjusted for the book, to make the 20% DC
% response threshold at 0.1

set = x_in > -a;
z = x_in(set) + a;

% zero is the final answer for many points:
conductance = zeros(size(x_in));
conductance(set) = z.^3 ./ (z.^3 + z.^2 + 0.1);


%% other things I tried:
%
% % zero is the final answer for many points:
% conductance = zeros(size(x_in));
%
% order = 4;  % 3 is a little cheaper; 4 has continuous second deriv.
%
% % thresholds and terms involving just a, b, s are scalar ops; x are vectors
%
% switch order
%   case 3
%     a = 0.15;  % offset of low-end tail into neg x territory
%     b = 1; % 0.44;   % width of poly segment
%     slope = 0.7;
%
%     threshold1 = -a;
%     threshold2 = b - a;
%
%     set2 = x_in > threshold2;
%     set1 = x_in > threshold1 & ~set2;
%
%     s = slope/(2*b - 3/2*b^2);  % factor to make slope at breakpoint
%     t = s * (b^2 - (b^3) / 2);
%
%     x = x_in(set1) - threshold1;
%     conductance(set1) = s * x .* (x - x .* x / 2);  % x.^2 - 0.5x.^3
%
%     x = x_in(set2) - threshold2;
%     conductance(set2) = t + slope * x ./ (1 + x);
%
%
%   case 4
%     a = 0.24;  % offset of low-end tail into neg x territory
%     b = 0.57;   % width of poly segment; 0.5 to end at zero curvature,
%     a = 0.18;  % offset of low-end tail into neg x territory
%     b = 0.57;   % width of poly segment; 0.5 to end at zero curvature,
%     % 0.57 to approx. match curvature of the upper segment.
%     threshold1 = -a;
%     threshold2 = b - a;
%
%
%     set2 = x_in > threshold2;
%     set1 = x_in > threshold1 & ~set2;
%
%     s = 1/(3*b^2 - 4*b^3);  % factor to make slope 1 at breakpoint
%     t = s * (b^3 - b^4);
%
%     x = x_in(set1) - threshold1;
%     conductance(set1) = s * x .* x .* (x - x .* x);  % x.^3 - x.^4
%
%     x = x_in(set2) - threshold2;
%     conductance(set2) = t + x ./ (1 + x);
%
% end
%
