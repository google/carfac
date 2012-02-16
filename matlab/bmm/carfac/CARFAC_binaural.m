% Copyright 2012, Google, Inc.
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

% Test/demo hacking for carfac matlab stuff, two-channel (binaural) case

clear variables

agc_plot_fig_num = 1;

tic

file_signal = wavread('plan.wav');
file_signal = file_signal(9000+(1:15000));  % trim for a faster test

itd_offset = 22;  % about 1 ms
test_signal = [file_signal((itd_offset+1):end), ...
               file_signal(1:(end-itd_offset))] / 10;
             
CF_struct = CARFAC_Design;  % default design

% Run mono, then stereo test:
n_mics = 2
CF_struct = CARFAC_Init(CF_struct, n_mics);
  
[nap, CF_struct, nap_decim] = CARFAC_Run(CF_struct, test_signal, agc_plot_fig_num);

% Display results for 1 or 2 mics:
for mic = 1:n_mics
  smooth_nap = nap_decim(:, :, mic);
  figure(mic + n_mics)  % Makes figures 2, 3, and 4
  image(63 * ((smooth_nap)' .^ 0.5))
    
  colormap(1 - gray);
end

toc

% Show resulting data, even though M-Lint complains:
CF_struct
CF_struct.filter_state
CF_struct.AGC_state
min_max = [min(nap(:)), max(nap(:))]
min_max_decim = [min(nap_decim(:)), max(nap_decim(:))]

