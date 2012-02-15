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

% Test/demo hacking for audio/hearing/filterbanks/carfac/matlab/ stuff

clear variables

agc_plot_fig_num = 1;

tic

%test_signal = wavread('pbav4p-22050.wav');
test_signal = wavread('plan-twochannel-1ms.wav')
%test_signal = wavread('binaural.wav')
%test_signal = test_signal(20000:2);  % trim for a faster test

CF_struct = CARFAC_Design;  % default design
cum_k = 0;  % not doing segments, so just count from 0

% Run mono, then stereo test:
n_mics = 2
CF_struct = CARFAC_Init(CF_struct, n_mics);

[nap, CF_struct, cum_k, nap_decim] = ...
    CARFAC_Run(CF_struct, test_signal, cum_k, agc_plot_fig_num);

% Display results for 1 or 2 mics:
for mic = 1:n_mics
  smooth_nap = nap_decim(:, :, mic);
  figure(mic + n_mics)  % Makes figures 2, 3, and 4
  image(63 * ((smooth_nap)' .^ 0.5))

  colormap(1 - gray);
end

toc

% Show resulting data, even though M-Lint complains:
cum_k
CF_struct
CF_struct.filter_state
CF_struct.AGC_state
min_max = [min(nap(:)), max(nap(:))]
min_max_decim = [min(nap_decim(:)), max(nap_decim(:))]
