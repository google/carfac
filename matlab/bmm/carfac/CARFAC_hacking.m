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
use_plan_file = 1;
if use_plan_file
  
  file_signal = wavread('plan.wav');
  %   file_signal = file_signal(8100+(1:20000));  % trim for a faster test
  file_signal = file_signal(10000+(1:10000));  % trim for a faster test
  
else
  flist = [1000];
  alist = [1];
  flist = 1000;
  alist = 1;
  sine_signal = 0;
  times = (0:19999)' / 22050;
  for fno = 1:length(flist)
    sine_signal = sine_signal + alist(fno)*sin(flist(fno)*2*pi*times);
  end
  growth_power = 0;  % use 0 for flat, 4 or more for near exponential
  file_signal = 1.0 * (sine_signal .* (times/max(times)).^growth_power);
end

% repeat with negated signal to compare responses:
% file_signal = [file_signal; -file_signal];

% make a long test signal by repeating at different levels:
do_distortion_figs = 0;  % use 1 for distortion figure hack....
if do_distortion_figs
  dB_list = -40:20:0
else
  dB_list = -80:20:60
end
% dB = dB_list(1);
% test_signal =  10^(dB/20)* file_signal(1:4000) % lead-in [];
test_signal = [];
for dB =  dB_list
  test_signal = [test_signal; file_signal * 10^(dB/20)];
end


%% Run mono, then stereo test:

agc_plot_fig_num = 6;

for n_ears = 1    %  1:2
  
  CF_struct = CARFAC_Design(n_ears);  % default design
  
  if n_ears == 2
    % For the 2-channel pass, add a silent second channel:
    test_signal = [test_signal, zeros(size(test_signal))];
  end
  
  CF_struct = CARFAC_Init(CF_struct);
  
  [CF_struct, nap_decim, nap, BM, ohc, agc] = CARFAC_Run(CF_struct, test_signal, ...
    agc_plot_fig_num);
  
  %   nap = deskew(nap);  % deskew doesn't make much difference
  
  %   dB_BM = 10/log(10) * log(filter(1, [1, -0.995], BM(:, 20:50, :).^2));
  dB_BM = 10/log(10) * log(filter(1, [1, -0.995], BM(:, :, :).^2));
  
  % only ear 1:
  MultiScaleSmooth(dB_BM(5000:200:end, :, 1), 1);
  
  % Display results for 1 or 2 ears:
  for ear = 1:n_ears
    smooth_nap = nap_decim(:, :, ear);
    if n_ears == 1
      mono_max = max(smooth_nap(:));
    end
    figure(3 + ear + n_ears)  % Makes figures 5, ...
    image(63 * ((max(0, smooth_nap)/mono_max)' .^ 0.5))
    title('smooth nap from nap decim')
    colormap(1 - gray);
  end
  
  % Show resulting data, even though M-Lint complains:
  CF_struct
  CF_struct.ears(1).CAR_state
  CF_struct.ears(1).AGC_state
  min_max_decim = [min(nap_decim(:)), max(nap_decim(:))]
  
  %%
  if do_distortion_figs
    channels = [38, 39, 40];
    times = 0000 + (3200:3599);
    smoothed_ohc = ohc;
    epsi = 0.4;
    %   smoothed_ohc = filter(epsi, [1, -(1-epsi)], smoothed_ohc);
    %   smoothed_ohc = filter(epsi, [1, -(1-epsi)], smoothed_ohc);
    
    figure(101); hold off
    plot(smoothed_ohc(times, channels))
    hold on
    plot(smoothed_ohc(times, channels(2)), 'k-', 'LineWidth', 1.5)
    title('OHC')
    
    figure(105); hold off
    plot(agc(times, channels))
    hold on
    plot(agc(times, channels(2)), 'k-', 'LineWidth', 1.5)
    title('AGC')
    
    figure(103); hold off
    plot(BM(times, channels))
    hold on
    plot(BM(times, channels(2)), 'k-', 'LineWidth', 1.5)
    title('BM')
    
    extra_damping = smoothed_ohc + agc;
    figure(102); hold off
    plot(extra_damping(times, channels))
    hold on
    plot(extra_damping(times, channels(2)), 'k-', 'LineWidth', 1.5)
    title('extra damping')
    
    distortion = -(extra_damping - smooth1d(extra_damping, 10)) .* BM;
    distortion = filter(epsi, [1, -(1-epsi)], distortion);
    distortion = filter(epsi, [1, -(1-epsi)], distortion);
    %   distortion = filter(epsi, [1, -(1-epsi)], distortion);
    figure(104); hold off
    plot(distortion(times, channels))
    hold on
    plot(distortion(times, channels(2)), 'k-', 'LineWidth', 1.5)
    title('distortion')
    
  end
  %%
end

% Expected result:  Figure 3 looks like figure 2, a tiny bit darker.
% and figure 4 is empty (all zero)
