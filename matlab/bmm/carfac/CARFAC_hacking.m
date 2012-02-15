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

%% Test/demo hacking for CARFAC Matlab stuff:

clear variables

%%
file_signal = wavread('plan.wav');

% file_signal = file_signal(9000+(1:10000));  % trim for a faster test
file_signal = file_signal(9300+(1:5000));  % trim for a faster test

% repeat with negated signal to compare responses:
file_signal = [file_signal; -file_signal];

% make a long test signal by repeating at different levels:
test_signal = [];
for dB = -60:20:40  % -80:20:60
  test_signal = [test_signal; file_signal * 10^(dB/20)];
end

%%
CF_struct = CARFAC_Design;  % default design

%% Run mono, then stereo test:

agc_plot_fig_num = 6;

for n_mics = 1  % :2
  CF_struct = CARFAC_Init(CF_struct, n_mics);

  [nap, CF_struct, nap_decim] = CARFAC_Run(CF_struct, ...
    test_signal, agc_plot_fig_num);

%   nap = deskew(nap);  % deskew doesn't make much difference

  MultiScaleSmooth(nap_decim, 10);

%   nap_decim = nap;
%   nap_decim = smooth1d(nap_decim, 1);
%   nap_decim = nap_decim(1:8:size(nap_decim, 1), :);

  % Display results for 1 or 2 mics:
  for mic = 1:n_mics
    smooth_nap = nap_decim(:, :, mic);
    if n_mics == 1
      mono_max = max(smooth_nap(:));
    end
    figure(3 + mic + n_mics)  % Makes figures 5, ...
    image(63 * ((max(0, smooth_nap)/mono_max)' .^ 0.5))
    title('smooth nap from nap decim')
    colormap(1 - gray);
  end

  % Show resulting data, even though M-Lint complains:
  CF_struct
  CF_struct.k_mod_decim
  CF_struct.filter_state
  CF_struct.AGC_state
  min_max = [min(nap(:)), max(nap(:))]
  min_max_decim = [min(nap_decim(:)), max(nap_decim(:))]

  % For the 2-channel pass, add a silent second channel:
  test_signal = [test_signal, zeros(size(test_signal))];
end

% Expected result:  Figure 3 looks like figure 2, a tiny bit darker.
% and figure 4 is empty (all zero)
