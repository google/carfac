% Copyright 2012 Google Inc. All Rights Reserved.
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

% Test/demo hacking for carfac matlab stuff, two-channel (binaural) case

clear variables

agc_plot_fig_num = 1;

tic

file_signal = wavread('../test_data/binaural_test.wav');
file_signal = file_signal(9000+(1:15000));  % trim for a faster test

itd_offset = 22;  % about 1 ms
test_signal = [file_signal((itd_offset+1):end), ...
               file_signal(1:(end-itd_offset))] / 10;
             
CF_struct = CARFAC_Design;  % default design

% Run stereo test:
n_ears = 2
CF_struct = CARFAC_Init(CF_struct, n_ears);
  
[CF_struct, nap_decim, nap] = CARFAC_Run(CF_struct, test_signal, agc_plot_fig_num);

% Display results for 2 ears:
for ear = 1:n_ears
  smooth_nap = nap_decim(:, :, ear);
  figure(ear + n_ears)  % Makes figures 3 and 4
  image(63 * ((smooth_nap)' .^ 0.5))

  colormap(1 - gray);
end

toc

% Show resulting data, even though M-Lint complains:
CF_struct
CF_struct.CAR_state
CF_struct.AGC_state
min_max = [min(nap(:)), max(nap(:))]
min_max_decim = [min(nap_decim(:)), max(nap_decim(:))]

