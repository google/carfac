% Copyright 2012 The CARFAC Authors. All Rights Reserved.
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

function status = CARFAC_Test(do_plots)
% CARFAC_TEST returns status = 0 if all tests pass; nonzero if fail,
% and prints messages about tests and failures.  Argument do_plots is
% optional, defaults to 1 (as when executing the file as a script).
% Run CARFAC_Test(0) to suppress plotting.

if nargin < 1,
  do_plots = 1;
  close ALL  % So plots don't accumulate.
end  % Produce plots by default.

% Run tests, and see if any fail (have nonzero status):
status = 0;  % 0 for OK so far; 1 for test fail; 2 for error.
status = status | test_CAR_freq_response(do_plots);
status = status | test_IHC1(do_plots);  % one_cap, v1
status = status | test_IHC2(do_plots);  % two_cap, v2
status = status | test_IHC3(do_plots);  % do_syn, v3
status = status | test_AGC_steady_state(do_plots);
status = status | test_AGC_steady_state_non_decimating(do_plots);
status = status | test_stage_g_calculation(do_plots);
status = status | test_whole_carfac1(do_plots);
status = status | test_whole_carfac2(do_plots);
status = status | test_whole_carfac3(do_plots);
status = status | test_whole_carfac1_non_decimating(do_plots);
status = status | test_whole_carfac2_non_decimating(do_plots);
status = status | test_whole_carfac3_non_decimating(do_plots);
status = status | test_delay_buffer(do_plots);
status = status | test_OHC_health(do_plots);
status = status | test_multiaural_silent_channel(do_plots);
status = status | test_multiaural_silent_channel_non_decimating(do_plots);
status = status | test_multiaural_carfac(do_plots);
status = status | test_spike_rates(do_plots);
report_status(status, 'CARFAC_Test', 1)
return


function status = test_CAR_freq_response(do_plots)
% Test: Make sure that the CAR frequency response looks right.

status = 0;

CF = CARFAC_Design(1, 22050);  % 1 ear; use old test fs.
CF = CARFAC_Init(CF);
n_points = 2^14;
fft_len = n_points * 2;  % Lots of zero padding for good freq. resolution.
impulse_responses = zeros(fft_len, CF.n_ch);
ear = CF.ears(1);
ear.CAR_coeffs.linear = 1;
impulse = 1;
for i = 1:n_points
  [car_out, state] = CARFAC_CAR_Step( ...
    impulse, ear.CAR_coeffs, ear.CAR_state);
  ear.CAR_state = state;
  impulse = 0;
  impulse_responses(i, :) = car_out';
end
complex_spectra = fft(impulse_responses);
db_spectra = 20 * log10(abs(complex_spectra) + 1e-50);

if do_plots
  figure
  db_spectra(db_spectra < -20) = -20;
  imagesc(db_spectra(1:(n_points+1), :))
  set(gca,'YDir','normal')
  colorbar()
  xlabel('Channel Number')
  ylabel('Frequency bin')
  drawnow
end

expected = [ ...
  [10, 5604, 39.34, 705.6, 7.9];
  [20, 3245, 55.61, 429.8, 7.6];
  [30, 1809, 60.46, 248.1, 7.3];
  [40, 965, 59.18, 138.7, 7.0];
  [50, 477, 52.81, 74.8, 6.4];
  [60, 195, 38.98, 37.5, 5.2];
  [70, 32, 7.95, 14.6, 2.2];
  ];

% Test: check overall frequency response of a cascade of CAR filters.
% Match Figure 16.6 of Lyon's book
spectrum_freqs = (0:n_points)' * CF.fs / fft_len;
for i = 1:size(expected, 1)
  channel = expected(i, 1) + 1;  % Channel number was 0 based from Python.
  correct_cf = expected(i, 2);
  correct_gain = expected(i, 3);
  correct_bw = expected(i, 4);
  correct_q = expected(i, 5);
  cf_amp_bw = find_peak_response(spectrum_freqs, ...
    db_spectra(:, channel), 3);  % 3 dB width
  cf = round(cf_amp_bw(1));  % Zero decimals on this one.
  % Round to 1 or 2 decimal places, require exact match then.
  % 2 Decimal places for gain.
  gain = round(cf_amp_bw(2), 2);
  bw = round(cf_amp_bw(3), 1);
  q = round(cf_amp_bw(1) / cf_amp_bw(3), 1);
  fprintf(1, ...
    ['%d: cf is %.1f Hz, peak gain is %.1f dB,' ...
    ' 3 dB bandwidth is %.1f Hz (Q = %.1f).\n'], ...
    channel, cf, gain, bw, q);
  if cf ~= correct_cf
    status = 1;
    fprintf(1, 'Mismatch cf %f should be %f.\n', cf, correct_cf)
  end
  if gain ~= correct_gain
    status = 1;
    fprintf(1, 'Mismatch gain %f should be %f.\n', gain, correct_gain)
  end
  if bw ~= correct_bw
    status = 1;
    fprintf(1, 'Mismatch bw %f should be %f.\n', bw, correct_bw)
  end
  if q ~= correct_q
    status = 1;
    fprintf(1, 'Mismatch q %f should be %f.\n', q, correct_q)
  end
end
report_status(status, 'test_CAR_freq_response')
return


function [blip_maxes, blip_ac] = run_IHC(test_freq, version, do_plots)
fs = 40000;
sampling_interval = 1 / fs;
tmax = 0.28;
t = (0:sampling_interval:(tmax - sampling_interval/2))';
% test signal, ramping square wave gating a sinusoid:
omega0 = 2 * pi * 25;  % for making a 25 Hz square wave envelope
present = (1 - sign(sin(omega0 * t + pi / 2 + 1e-6))) / 2;
stim_num = present .* floor((t / 0.04));
% start at 0.09, 6 db below mean response threshold
amplitude = 0.09 * 2.^stim_num;
omega = 2 * pi * test_freq;
quad_sin = present .* sin(omega * t);
quad_cos = present .* cos(omega * t);
x_in = quad_sin .* amplitude;

CF = CARFAC_Design(1, fs, version);
CF = CARFAC_Init(CF);

neuro_output = zeros(size(x_in));
ihc_state = CF.ears(1).IHC_state;
if CF.do_syn
  syn_state = CF.ears(1).SYN_state;
end
for k = 1:length(x_in)
  [ihc_out, ihc_state, v_recep] = CARFAC_IHC_Step( ...
    x_in(k), CF.ears(1).IHC_coeffs, ihc_state);
  if CF.do_syn  % ignore ihc_out and use receptor_potential.
    [syn_out, firings, syn_state] = CARFAC_SYN_Step( ...
      v_recep, CF.ears(1).SYN_coeffs, syn_state);
    % This can go a little negative; should be zero at rest.
    neuro_output(k) = syn_out(1);
    class_firings(k, :) = firings(1, :);
  else  % ignore receptor_potential and use ihc_out.
    neuro_output(k) = ihc_out(1);
  end
end

if do_plots
  figure
  plot(t, neuro_output)
  xlabel('Seconds')
  title(sprintf('IHC Response for tone blips at %d Hz', test_freq))
  drawnow
  if CF.do_syn
    figure
    plot(t, class_firings)
  end
end
blip_maxes = [];
blip_ac = [];
for blip_num = 1:6
  blip = neuro_output .* (stim_num == blip_num);
  blip_max = max(blip);
  carrier_power = sum(blip .* quad_sin)^2 + sum(blip .* quad_cos)^2;
  carrier_rms = sqrt(carrier_power);
  fprintf(1, 'Blip %d: Max of %f, AC rms is %f\n', ...
    blip_num, blip_max, carrier_rms);
  blip_maxes(end+1) = blip_max;
  blip_ac(end+1) = sqrt(carrier_power);
end
return


function status = test_IHC1(do_plots)
% Test: Make sure that IHC (inner hair cell) runs as expected.

status = 0;

test_freqs = [300, 3000];
for k = 1:length(test_freqs)
  switch test_freqs(k)
    case 300
      expected_results = [ ...
        [2.752913, 721.001685];
        [4.815015, 969.505412];
        [7.062418, 1147.285676];
        [9.138118, 1239.521055];
        [10.969522, 1277.061337];
        [12.516468, 1285.880084];
        ];
    case 3000
      expected_results = [ ...
        [1.417657, 234.098558];
        [2.804747, 316.717957];
        [4.802444, 376.787575];
        [7.030791, 408.011707];
        [9.063014, 420.602740];
        [10.634581, 423.674628];
        ];
    otherwise
      fprintf(1, 'No test_results for %f Hz in test_IHC.\n', ...
        test_freqs(k));
  end
  [blip_maxes, blip_ac] = run_IHC(test_freqs(k), 'one_cap', do_plots);
  num_blips = length(blip_maxes);
  if num_blips ~= size(expected_results, 1)
    fprintf(1, ...
      'Unmatched num_blips %d and expected_results rows %d in test_IHC.\n',...
      num_blips, size(expected_results, 1));
    status = 2;
  else
    expected_maxes = expected_results(:, 1)';
    expected_acs = expected_results(:, 2)';

    fprintf(1, 'Golden data for Matlab and Python test_IHC:\n');
    fprintf(1, '        [%f, %f];\n', [blip_maxes; blip_ac])

    for i = 1:num_blips
      if abs(expected_maxes(i) - blip_maxes(i)) > expected_maxes(i)/1e6
        status = 1;
        fprintf(1, ...
          'test_IHC fails with i = %d, expected_max = %f, blip_max = %f\n', ...
          i, expected_maxes(i), blip_maxes(i))
      end
      if abs(expected_acs(i) - blip_ac(i)) > expected_acs(i)/1e6
        status = 1;
        fprintf(1, ...
          'test_IHC fails with i = %d, expected_ac = %f, blip_ac = %f\n', ...
          i, expected_acs(i), blip_ac(i))
      end
    end
  end
end
report_status(status, 'test_IHC')
return


function status = test_IHC2(do_plots)
% Test: Make sure that IHC (inner hair cell) runs as expected.
% Two-cap version with receptor potential; slightly different blips.

status = 0;

test_freqs = [300, 3000];
for k = 1:length(test_freqs)
  switch test_freqs(k)
    case 300
      expected_results = [ ...
        [2.026682, 544.901381];
        [3.533259, 756.736631];
        [5.108579, 923.142282];
        [6.423783, 1017.472318];
        [7.454677, 1059.407644];
        [8.231247, 1071.902335];
        ];
    case 3000
      expected_results = [ ...
        [0.698303, 93.388172];
        [1.520033, 131.832247];
        [2.660770, 163.287206];
        [3.872406, 182.022912];
        [4.909175, 191.225206];
        [5.666469, 194.912279];
        ];
    otherwise
      fprintf(1, 'No test_results for %f Hz in test_IHC.\n', ...
        test_freqs(k));
  end
  [blip_maxes, blip_ac] = run_IHC(test_freqs(k), 'two_cap', do_plots);
  num_blips = length(blip_maxes);
  if num_blips ~= size(expected_results, 1)
    fprintf(1, ...
      'Unmatched num_blips %d and expected_results rows %d in test_IHC2.\n',...
      num_blips, size(expected_results, 1));
    status = 2;
  else
    expected_maxes = expected_results(:, 1)';
    expected_acs = expected_results(:, 2)';

    fprintf(1, 'Golden data for Matlab and Python test_IHC2:\n');
    fprintf(1, '        [%f, %f];\n', [blip_maxes; blip_ac])

    for i = 1:num_blips
      if abs(expected_maxes(i) - blip_maxes(i)) > expected_maxes(i)/1e6
        status = 1;
        fprintf(1, ...
          'test_IHC2 fails with i = %d, expected_max = %f, blip_max = %f\n', ...
          i, expected_maxes(i), blip_maxes(i))
      end
      if abs(expected_acs(i) - blip_ac(i)) > expected_acs(i)/1e6
        status = 1;
        fprintf(1, ...
          'test_IHC2 fails with i = %d, expected_ac = %f, blip_ac = %f\n', ...
          i, expected_acs(i), blip_ac(i))
      end
    end
  end
end
report_status(status, 'test_IHC2')
return


function status = test_IHC3(do_plots)
% Test: Make sure that IHC (inner hair cell) runs as expected.
% Two-cap version with receptor potential; slightly different blips.

status = 0;

test_freqs = [300, 3000];
for k = 1:length(test_freqs)
  switch test_freqs(k)
    case 300
      expected_results = [ ...
        [1.055837, 184.180863];
        [3.409906, 483.204136];
        [6.167359, 837.629296];
        [7.096430, 956.279101];
        [7.103324, 927.060415];
        [7.123434, 895.574871];
        ];
    case 3000
      expected_results = [ ...
        [0.167683, 24.929044];
        [0.620939, 74.045598];
        [1.894064, 175.367201];
        [3.541070, 269.322147];
        [4.899921, 303.684269];
        [5.572545, 278.428744];
        ];
    otherwise
      fprintf(1, 'No test_results for %f Hz in test_IHC.\n', ...
        test_freqs(k));
  end
  [blip_maxes, blip_ac] = run_IHC(test_freqs(k), 'do_syn', do_plots);
  num_blips = length(blip_maxes);
  if num_blips ~= size(expected_results, 1)
    fprintf(1, ...
      'Unmatched num_blips %d and expected_results rows %d in test_IHC3.\n',...
      num_blips, size(expected_results, 1));
    status = 2;
  else
    expected_maxes = expected_results(:, 1)';
    expected_acs = expected_results(:, 2)';

    fprintf(1, 'Golden data for Matlab and Python test_IHC3:\n');
    fprintf(1, '        [%f, %f];\n', [blip_maxes; blip_ac])

    for i = 1:num_blips
      % Lowered precision from 1e6 to 0.5e6 as the values are lower,
      % and don't quite quite enough digits printed in the default format.
      if abs(expected_maxes(i) - blip_maxes(i)) > expected_maxes(i)/0.5e6
        status = 1;
        fprintf(1, ...
          'test_IHC3 fails with i = %d, expected_max = %f, blip_max = %f\n', ...
          i, expected_maxes(i), blip_maxes(i))
      end
      if abs(expected_acs(i) - blip_ac(i)) > expected_acs(i)/0.5e6
        status = 1;
        fprintf(1, ...
          'test_IHC3 fails with i = %d, expected_ac = %f, blip_ac = %f\n', ...
          i, expected_acs(i), blip_ac(i))
      end
    end
  end
end
report_status(status, 'test_IHC3')
return


function status = test_AGC_steady_state(do_plots)
% Test: Make sure that the AGC adapts to an appropriate steady state,
% like figure 19.7
status = test_AGC_steady_state_core(do_plots, 0);
return


function status = test_AGC_steady_state_non_decimating(do_plots)
% Test: Make sure that the AGC adapts to an appropriate steady state,
% like figure 19.7
status = test_AGC_steady_state_core(do_plots, 1);
return

function status = test_AGC_steady_state_core(do_plots, non_decimating)
% Test: Make sure 2025 non-decimating changes is "close enough" to same.

status = 0;

if non_decimating
  CAR_params = CAR_params_default;
  AGC_params = AGC_params_default;
  AGC_params.decimation = [1, 1, 1, 1];  % Override default.
  CF = CARFAC_Design(1, 22050, CAR_params, AGC_params);
else
  CF = CARFAC_Design(1, 22050);  % With default [8, 2, 2, 2] decimation.
end
CF = CARFAC_Init(CF);
agc_input = zeros(CF.n_ch, 1);
test_channel = 40;
n_points = 16384;
num_stages = CF.AGC_params.n_stages;  % 4
decim = CF.ears.AGC_coeffs.decimation(1);  % 8
agc_response = zeros(num_stages, floor(n_points / decim), CF.n_ch);
num_outputs = 0;
for i = 1:n_points
  agc_input(test_channel) = 100;  % Leave other channels at 0 input.
  [agc_state, agc_updated] = CARFAC_AGC_Step(agc_input, ...
    CF.ears(1).AGC_coeffs, CF.ears(1).AGC_state);
  CF.ears(1).AGC_state = agc_state;
  if agc_updated  % Every 8 samples.
    num_outputs = num_outputs + 1;
    for stage = 1:num_stages
      agc_response(stage, num_outputs, :) = agc_state.AGC_memory(:, stage)';
    end
  end
end

% Test: Plot spatial response to match Figure 19.7
if do_plots
  figure
  hold on
  plot(squeeze(agc_response(:, end, :))')
  title('Steady state spatial responses of the stages')
  drawnow
end

if CF.ears(1).AGC_coeffs.non_decimating
  % 2025 non-decimating way is higher/sharper near the peak.
  expected_ch_amp_bws = [ ...
    [39.680614, 9.160676, 8.470642];
    [39.760187, 4.531205, 7.819946];
    [39.828831, 2.117139, 6.951839];
    [39.896631, 0.833373, 5.552264];
    ];
else
  expected_ch_amp_bws = [ ...
    [39.033166, 8.359763, 9.598703];
    [39.201534, 4.083376, 9.019020];
    [39.374404, 1.878256, 8.219043];
    [39.565957, 0.712351, 6.994498];
    ];
end

if num_stages ~= size(expected_ch_amp_bws, 1)
  fprintf(1, ...
    'Unmatched num_stages %d and expected_ch_amp_bws rows %d in test_IHC.\n',...
    num_stages, size(expected_ch_amp_bws, 1));
  status = 2;
else
  ch_amp_bws = [];
  for i = 1:num_stages
    % Find_peak_response wants to find the width at a fixed level (3 dB)
    % below the peak.  We call this function twice: the first time with
    % a simple estimate of the max; then a second time with a threshold
    % that is 50% down from the interpolated peak amplitude to get width.
    state_response = squeeze(agc_response(i, end, :))';
    amp = max(state_response);
    ch_amp_bw = find_peak_response(1:CF.n_ch, state_response, amp/2);
    amp = ch_amp_bw(2);
    ch_amp_bw = find_peak_response(1:CF.n_ch, state_response, amp/2);
    ch_amp_bws = [ch_amp_bws ; ch_amp_bw];  % Collect for printing Golden.

    fprintf(1, ...
      'AGC Stage %d: Peak at channel %f, value is %f, fwhm %f.\n',...
      [i, ch_amp_bw]);

    expected_ch = expected_ch_amp_bws(i, 1);
    if abs(ch_amp_bw(1) - expected_ch) > expected_ch / 1e5
      status = 1;
      fprintf(1, 'Peak channel location %f does not match expected %f.\n', ...
        ch_amp_bw(1), expected_ch);
    end
    expected_amp = expected_ch_amp_bws(i, 2);
    if abs(ch_amp_bw(2) - expected_amp) > expected_amp / 1e5
      status = 1;
      fprintf(1, 'Peak amplitude %f does not match expected %f.\n', ...
        ch_amp_bw(2), expected_amp);
    end
    expected_bw = expected_ch_amp_bws(i, 3);
    if abs(ch_amp_bw(3) - expected_bw) > expected_bw / 1e5
      status = 1;
      fprintf(1, 'Peak bandwidth %f does not match expected %f.\n', ...
        ch_amp_bw(3), expected_bw);
    end
  end
  fprintf(1, 'Golden data for Matlab test_AGC_steady_state:\n');
  fprintf(1, '  [%f, %f, %f];\n', ch_amp_bws')
  fprintf(1, 'Golden data for Python test_agc_steady_state:\n');
  fprintf(1, '        %d: [%f, %f, %f],\n', ...
    [(0:(num_stages-1))', ch_amp_bws]')
end
report_status(status, 'test_AGC_steady_state')
return


function status = test_stage_g_calculation(do_plots)
% Make sure the quadratic stage_g calculation agrees with the ratio of
% polynomias from the book
status = 0;

fs = 22050;
CF = CARFAC_Design(1, fs);
% CF = CARFAC_Init(CF);

if do_plots
  figure; clf
end
for undamping = 0:0.1:1  % including the "training" points 0, 0.5, 1.
  ideal_g = CARFAC_Design_Stage_g(CF.ears(1).CAR_coeffs, undamping);
  stage_g = CARFAC_Stage_g(CF.ears(1).CAR_coeffs, undamping);
  if do_plots
    plot(ideal_g, 'g-')
    hold on
    plot(stage_g, 'r.')
    plot(1000*abs(ideal_g - stage_g), 'm+')
    drawnow
    title('test stage g calculation')
    xlabel('channel')
    ylabel('Stage gains and 1000*abs(error)')
  end
  % One part per thousand gain error is less than 0.01 dB.
  if any(abs(ideal_g - stage_g)/ideal_g > 1e-3)
    status = 1;
  end
end

report_status(status, 'test_stage_g_calculation')
return


function status = test_whole_carfac1(do_plots)
status = test_whole_carfac(do_plots, 'one_cap', 0);
report_status(status, 'test_whole_carfac1')
return


function status = test_whole_carfac2(do_plots)
status = test_whole_carfac(do_plots, 'two_cap', 0);
report_status(status, 'test_whole_carfac2')
return


function status = test_whole_carfac3(do_plots)
status = test_whole_carfac(do_plots, 'do_syn', 0);
report_status(status, 'test_whole_carfac3')
return


function status = test_whole_carfac1_non_decimating(do_plots)
status = test_whole_carfac(do_plots, 'one_cap', 1);
report_status(status, 'test_whole_carfac4')
return


function status = test_whole_carfac2_non_decimating(do_plots)
status = test_whole_carfac(do_plots, 'two_cap', 1);
report_status(status, 'test_whole_carfac5')
return


function status = test_whole_carfac3_non_decimating(do_plots)
status = test_whole_carfac(do_plots, 'do_syn', 1);
report_status(status, 'test_whole_carfac6')
return


function status = test_whole_carfac(do_plots, version_string, non_decimating)
% Test: Make sure that the AGC adapts to a tone. 
% Test with open-loop impulse response.

status = 0;

fs = 22050;
fp = 1000;  % Probe tone
t = (0:(1/fs):(2 - 1/fs))';  % Sample times for 2s of tone
amplitude = 0.1;
sinusoid = amplitude * sin(2 * pi * t * fp);

impulse_dur = 0.5;  % 0.25 is about enough; this is conservative.
impulse = zeros(round(impulse_dur*fs), 1);  % For short impulse wave.
impulse(1) = 1e-4;  % Small amplitude impulse to keep it pretty linear

if non_decimating
  CAR_params = CAR_params_default;
  AGC_params = AGC_params_default;
  AGC_params.decimation = [1, 1, 1, 1];  % Override default.
  CF = CARFAC_Design(1, 22050, CAR_params, AGC_params, version_string);
else
  CF = CARFAC_Design(1, 22050, version_string);  % With default decimation.
end
CF = CARFAC_Init(CF);

CF.open_loop = 1;  % For measuring impulse response.
CF.linear_car = 1;  % For measuring impulse response.
[~, CF, bm_initial] = CARFAC_Run_Segment(CF, impulse);

CF.open_loop = 0;  % To let CF adapt to signal.
CF.linear_car = 0;  % Normal mode.
[nap, CF, bm_sine] = CARFAC_Run_Segment(CF, sinusoid);

% Capture AGC state response at end, for analysis later.
num_stages = CF.AGC_params.n_stages;  % 4
agc_response = zeros(num_stages, CF.n_ch);
for stage = 1:num_stages
  agc_response(stage, :) = CF.ears(1).AGC_state.AGC_memory(:, stage);
end

CF.open_loop = 1;  % For measuring impulse response.
CF.linear_car = 1;  % For measuring impulse response.
[~, CF] = CARFAC_Run_Segment(CF, 0*impulse);  % To let ringing die out.
[~, CF, bm_final] = CARFAC_Run_Segment(CF, impulse);

% Now compare impulse responses bm_initial and bm_final.

fft_len = 2048;  % Because 1024 is too sensitive to delay and such.
num_bins = fft_len/2 + 1;
freqs = (fs / fft_len) * (0:num_bins-1)';
initial_freq_response = 20*log10(abs(fft(bm_initial(1:fft_len, :))));
final_freq_response   = 20*log10(abs(fft(bm_final(1:fft_len, :))));
initial_freq_response = initial_freq_response(1:num_bins, :);
final_freq_response   = final_freq_response(1:num_bins, :);

if do_plots
  % Match Figure 19.9(right) of Lyon's book
  figure; clf
  semilogx(freqs, initial_freq_response, ':')
  hold on
  semilogx(freqs, final_freq_response, '-')
  ylabel('dB')
  xlabel('Frequency')
  title('Initial (dotted) vs. Adapted at 1kHz (solid) Frequency Response')
  axis([0, max(freqs), -100, -15])
  %   savefig('/tmp/whole_carfac_response.png')
  drawnow
end

initial_resps = [];  % To collect peak [cf, amplitude, bw] per channel.
final_resps = [];
for ch = 1:CF.n_ch
  initial_resps = [initial_resps; ...
    find_peak_response(freqs, initial_freq_response(:, ch), 3)];
  final_resps = [final_resps; ...
    find_peak_response(freqs, final_freq_response(:, ch), 3)];
end

if do_plots
  figure; clf('reset')
  plot(1:CF.n_ch, initial_resps(:,2), ':')
  hold on
  plot(1:CF.n_ch, final_resps(:,2))
  xlabel('Ear Channel #')
  ylabel('dB')
  title('NP: Initial (dotted) vs. Adapted (solid) Peak Gain')
  %.   savefig('/tmp/whole_carfac_peak_gain.png')
  drawnow
end

% Test for change in peak gain after adaptation.
% Golden data table of frequency, channel, peak frequency, delta:
switch version_string
  case 'one_cap'
    % Before moving ac coupling into CAR, peaks gains a little different:
    %   125, 65,   118.944255,     0.186261
    %   250, 59,   239.771898,     0.910003
    %   500, 50,   514.606412,     7.243568
    %   1000, 39,  1099.433179,    31.608529
    %   2000, 29,  2038.873929,    27.242882
    %   4000, 17,  4058.881505,    13.865787
    %   8000,  3,  8289.882476,     3.574972
    results = [
      125, 65,      119.007,        0.264
      250, 59,      239.791,        0.986
      500, 50,      514.613,        7.309
      1000, 39,     1099.436,       31.644
      2000, 29,     2038.875,       27.214
      4000, 17,     4058.882,       13.823
      8000,  3,     8289.883,        3.565
      ];
  case 'two_cap'
    results = [
      125, 65,      119.007,        0.258
      250, 59,      239.791,        0.963
      500, 50,      514.613,        7.224
      1000, 39,     1099.436,       31.373
      2000, 29,     2038.875,       26.244
      4000, 17,     4058.882,       12.726
      8000,  3,     8289.883,        3.212
      ];
  case 'do_syn'
    results = [
      125, 65,      119.007,        0.238
      250, 59,      239.791,        0.942
      500, 50,      514.613,        7.249
      1000, 40,     1030.546,       30.843
      2000, 29,     2038.875,       22.514
      4000, 17,     4058.882,        7.691
      8000,  4,     7925.624,        1.935
      ];
end

% Print data blocks that can be used to update golden test data.
test_cfs = 125 * 2.^(0:6);
% Print the golden data table for the above test center frequencies.
fprintf(1, 'Golden data for Matlab:\n');
for j = 1:length(test_cfs)
  cf = test_cfs(j);
  c = find_closest_channel(final_resps(:,1), cf);
  dB_change = initial_resps(c, 2) - final_resps(c, 2);
  fprintf(1, '  %d, %2d, %12.3f, %12.3f\n', [cf, c, ...
    initial_resps(c, 1), dB_change]);
end
% Print the golden data table for the Python test, 0-based channel index.
fprintf(1, 'Golden data for Python:\n')
for j = 1:length(test_cfs)
  result = results(j, :);
  cf = test_cfs(j);
  c = find_closest_channel(final_resps(:,1), cf);
  dB_change = initial_resps(c, 2) - final_resps(c, 2);
  fprintf(1, '        %d: [%d, %.3f, %.3f],\n', [cf, c - 1, ...
    initial_resps(c, 1), dB_change]);
end

for j = 1:size(results, 1)
  result = results(j, :);
  cf = result(1);
  expected_c = result(2);  % Which will be one more than in the Python.
  expected_cf = result(3);
  expected_change = result(4);
  c = find_closest_channel(final_resps(:,1), cf);
  dB_change = initial_resps(expected_c, 2) - final_resps(expected_c, 2);
  fprintf(1, ...
    'Channel %d has CF of %6f and an adaptation change of %f dB\n', ...
    expected_c, initial_resps(expected_c, 1), dB_change);

  if c ~= expected_c
    status = 1;
    fprintf(1, 'c = %d should equal expected_c %d\n', c, expected_c);
  end

  if non_decimating
    tolerance = expected_cf / 1000;
  else
    tolerance = expected_cf / 10000;
  end
  if abs(initial_resps(expected_c, 1) - expected_cf) > tolerance
    status = 1;
    fprintf(1, ...
      'initial_resps(c, 0) = %8.3f not close to expected_cf %8.3f\n', ...
      initial_resps(c, 1), expected_cf);
  end
  if non_decimating
    tolerance = 0.02 + expected_change/30;
  else
    tolerance = 0.01;  % dB
  end
  if abs(dB_change - expected_change) > tolerance
    status = 1;
    fprintf(1, 'dB_change = %6.3f not close to expected_change %6.3f\n', ...
      dB_change, expected_change);
  end
end

if do_plots  % Plot final AGC state
  figure
  plot(agc_response')
  title('Steady state spatial responses of the stages')
  axis([0, CF.n_ch + 1, 0, 1])
  drawnow
end
return


function status = test_delay_buffer(do_plots)
% Test: Verify simple delay of linear impulse response.

status = 0;

fs = 22050;

impulse_dur = 0.1;  % Short impulse.
impulse = zeros(round(impulse_dur*fs), 1);
impulse(1) = 1e-4;

CF = CARFAC_Design(1, fs);

CF = CARFAC_Init(CF);
CF.open_loop = 1;  % For measuring impulse response.
CF.linear_car = 1;  % For measuring impulse response.
CF.use_delay_buffer = 0;  % No delay per stage.
[~, CF, bm_initial] = CARFAC_Run_Segment(CF, impulse);

CF = CARFAC_Init(CF);
CF.use_delay_buffer = 1;  % Add a delay per stage.
[~, CF, bm_delayed] = CARFAC_Run_Segment(CF, impulse);

max_max_rel_error = 0;
for ch = 1:CF.n_ch
  impresp = bm_initial(1:(end-ch), ch);
  delayed = bm_delayed(ch:end-1, ch);
  max_abs = max(abs(impresp));
  max_abs_error = max(abs(impresp - delayed));
  max_rel = max_abs_error / max_abs;
  max_max_rel_error = max(max_max_rel_error, max_rel);
  if max_rel > 1e-6
    status = 1;
    fprintf(1, 'Channel %d delayed max_rel %f\n', ch, max_rel)
  end
end
fprintf(1, 'Delay linear max_max_rel_error = %f\n', max_max_rel_error);

% Try normal nonlinear operation and see how different it is:
CF = CARFAC_Init(CF);
CF.open_loop = 0;  % Let the AGC work.
CF.linear_car = 0;  % Let the OHC NLF work.
CF.use_delay_buffer = 0;  % No delay per stage.
[~, CF, bm_initial] = CARFAC_Run_Segment(CF, impulse);

CF = CARFAC_Init(CF);  % Re-Init to reset AGC state to zero.
CF.use_delay_buffer = 1;  % Add a delay per stage.
[~, CF, bm_delayed] = CARFAC_Run_Segment(CF, impulse);

max_max_rel_error = 0;
for ch = 1:CF.n_ch
  impresp = bm_initial(1:(end-ch), ch);
  delayed = bm_delayed(ch:end-1, ch);
  max_abs = max(abs(impresp));
  max_abs_error = max(abs(impresp - delayed));
  max_rel = max_abs_error / max_abs;
  max_max_rel_error = max(max_max_rel_error, max_rel);
  if max_rel > 0.025  % Needs more tolerance to pass, now it's nonlinear.
    status = 1;
    fprintf(1, 'Channel %d delayed max_rel %f\n', ch, max_rel)
  end
end
fprintf(1, 'Delay nonlinear max_max_rel_error = %f\n', max_max_rel_error);

report_status(status, 'test_delay_buffer')
return


function status = test_multiaural_silent_channel(do_plots)
status = test_multiaural_silent_core(do_plots, 0);


function status = test_multiaural_silent_channel_non_decimating(do_plots)
status = test_multiaural_silent_core(do_plots, 1);


function status = test_multiaural_silent_core(do_plots, non_decimating)
% Test multiaural functionality with 2 ears. Runs a 50ms sample of a pair of
% C Major chords, and tests a binaural carfac, with 1 silent ear against
% a simple monoaural carfac with only the chords.
%
% Tests that:
% 1. The ratio of the BM in total is within an expected ratio [1, 1.25]
% 2. Checks a golden set against a precise set of ratios for these chords
% The latter is to ensure identical behavior in python.
status = 0;
fs = 22050;
t = (0:(1/fs):(0.05 - 1/fs))';  % 50ms
amplitude = 1e-3;  % -70 dBFS, around 30-40 dB SPL

% c major chord of c-e-g at 523.25, 659.25 and 783.99
% and 32.7, 41.2 and 49
freqs = [523.25 659.25 783.99 32.7 41.2 49];
c_chord = amplitude * sum(sin(2 * pi * t * freqs), 2);
binaural_audio = [c_chord, zeros(size(t))];

version_string = 'one_cap';  % Legacy test.
if non_decimating
  CAR_params = CAR_params_default;
  AGC_params = AGC_params_default;
  AGC_params.decimation = [1, 1, 1, 1];  % Override default.
  CF = CARFAC_Design(2, 22050, CAR_params, AGC_params, version_string);
  MONO_CF = CARFAC_Design(1, fs, CAR_params, AGC_params, version_string);
else
  CF = CARFAC_Design(2, 22050, version_string);
  MONO_CF = CARFAC_Design(1, fs, version_string);
end

CF = CARFAC_Init(CF);
MONO_CF = CARFAC_Init(MONO_CF);
[naps, CF, bm_baseline] = CARFAC_Run_Segment(CF, binaural_audio);
[mono_naps, MONO_CF, mono_bm_baseline] = CARFAC_Run_Segment(MONO_CF, c_chord);
good_ear_bm = bm_baseline(:, :, 1);
rms_good_ear = rms(good_ear_bm);
rms_mono = rms(mono_bm_baseline);
tf_ratio = rms_good_ear ./ rms_mono;

if ~non_decimating
  % Data from non_decimating case is not used as golden.
  fprintf(1, 'for python, tf_ratio = [');
  for ch = 1:size(tf_ratio')
    fprintf(1, ' %.4f,', tf_ratio(ch));
    if mod(ch, 5) == 0
      fprintf(1, '\n');
    end
  end
  fprintf(1, ']\nfor matlab, expected_tf_ratio = [');
  for ch = 1:size(tf_ratio')
    fprintf(1, '%.4f, ', tf_ratio(ch));
    if mod(ch, 5) == 0
      fprintf(1, '...\n');
    end
  end
  fprintf(1, '];\n');
end
expected_tf_ratio = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ...
  1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ...
  1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ...
  1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ...
  1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ...
  1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ...
  1.0000, 1.0000, 1.0001, 1.0001, 1.0001, ...
  1.0002, 1.0004, 1.0007, 1.0018, 1.0050, ...
  1.0133, 1.0290, 1.0463, 1.0562, 1.0552, ...
  1.0505, 1.0497, 1.0417, 1.0426, 1.0417, ...
  1.0320, 1.0110, 1.0093, 1.0124, 1.0065, ...
  1.0132, 1.0379, 1.0530, 1.0503, 1.0477, ...
  1.0556, 1.0659, 1.0739, 1.0745, 1.0762, ...
  1.0597, 1.0200, 1.0151, 1.0138, 1.0129, ...
  1.0182, ];
if non_decimating
  % Later channels are dominated by aliasing effects, which are different
  % with the 2025 non-decimating version, so ignore the tiny responses
  % there.
  tf_ratio = tf_ratio(1:52);
  expected_tf_ratio = expected_tf_ratio(1:52);
end
max_error = max(abs(expected_tf_ratio - tf_ratio));
if max_error > 1e-3
  status = 1
  fprintf(1, 'Expected TF Ratio is not within 1e-3 of TF Ratio\n');
end
if any(tf_ratio < 0.999) | any(tf_ratio > 1.25)
  status = 1;
  fprintf(1, 'bm ratio is expected to be between 1 and 1.2 for noise\n');
end
report_status(status, 'test_multiaural_silent_channel_carfac');
return


function status = test_multiaural_carfac(do_plots)
% Tests that in binaural carfac, providing identical noise to both ears
% gives identical nap output at end.
status = 0;
fs = 22050;
t = (0:(1/fs):(1 - 1/fs))';  % Sample times for 1s of noise
amplitude = 1e-4;  % -80 dBFS, around 20 or 30 dB SPL
noise = amplitude * randn(size(t));
binaural_noise = [noise noise];
CF = CARFAC_Design(2, fs, 'one_cap');  % Legacy
CF = CARFAC_Init(CF);
[naps, CF, bm_baseline] = CARFAC_Run_Segment(CF, binaural_noise);
ear_one_naps = naps(:, :, 1);
ear_two_naps = naps(:, :, 2);
max_error =  max(abs(ear_one_naps - ear_two_naps));
if max_error > 1e-6
  status = 1;
  fprintf(1, 'Failed to have both ears equal in binaural');
end
report_status(status, 'test_multiaural_carfac');
return


function status = test_OHC_health(do_plots)
% Test: Verify frequency dependent reduced gain with reduced health.

status = 0;

fs = 22050;
t = (0:(1/fs):(1 - 1/fs))';  % Sample times for 1s of noise
amplitude = 1e-4;  % -80 dBFS, around 20 or 30 dB SPL
noise = amplitude * randn(size(t));

CF = CARFAC_Design(1, fs, 'one_cap');  % Legacy
CF = CARFAC_Init(CF);
[~, CF, bm_baseline] = CARFAC_Run_Segment(CF, noise);

half_ch = floor(CF.n_ch/2);
ch = 1:half_ch;
CF.ears(1).CAR_coeffs.OHC_health(ch) = ...
  CF.ears(1).CAR_coeffs.OHC_health(ch) * 0.5;
CF = CARFAC_Init(CF);
[~, CF, bm_less_healthy] = CARFAC_Run_Segment(CF, noise);

rms_baseline = rms(bm_baseline);
rms_less_healthy = rms(bm_less_healthy);
tf_ratio = rms_less_healthy ./ rms_baseline;

if do_plots
  figure
  plot(tf_ratio)
  xlabel('channel number')
  ylabel('tf_ratio')
  title('unhealthy hf OHC noise transfer function ratio')
  drawnow
end

% Expect tf_ratio low in early channels, close to 1 later.
if any(tf_ratio(10:half_ch) > 0.11)  % 0.1 works, but seed dependent.
  status = 1;
  fprintf(1, 'tf_ratio too high in early channels in test_OHC_health\n')
end
if any(tf_ratio(half_ch+6:end-2) < 0.35)
  status = 1;
  fprintf(1, 'tf_ratio too low in later channels in test_OHC_health\n')
end
report_status(status, 'test_OHC_health')
return


function cf_amp_bw = find_peak_response(freqs, db_gains, bw_level)
% Returns center frequency, amplitude at this point, and the 3dB width."""
[~, peak_bin] = max(db_gains);
[peak_frac, amplitude] = quadratic_peak_interpolation( ...
  db_gains(peak_bin - 1), db_gains(peak_bin), db_gains(peak_bin + 1));
peak_loc = peak_bin + peak_frac;
if peak_frac < 0
  cf = (1 + peak_frac)*freqs(peak_bin) - ...
    peak_frac*freqs(peak_bin - 1);
else
  cf = (1 - peak_frac)*freqs(peak_bin) + ...
    peak_frac*freqs(peak_bin + 1);
end
% cf = linear_interp(freqs, peak_bin + peak_frac)
freqs_3db = find_zero_crossings(freqs, db_gains - amplitude + bw_level);
if length(freqs_3db) >= 2
  bw = freqs_3db(2) - freqs_3db(1);
else
  bw = 0;
end
cf_amp_bw = [cf, amplitude, bw];
return


% Formula from:
% https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
function [location, amplitude] = quadratic_peak_interpolation(...
  alpha, beta, gamma)
location = 1 / 2 * (alpha - gamma) / (alpha - 2 * beta + gamma);
amplitude = beta - 1.0 / 4 * (alpha - gamma) * location;
return


function index = find_closest_channel(values, desired)
[~, index] = min((values - desired).^2);
return


function zclist = find_zero_crossings(x, y)
locs = find(y(2:end) .* y(1:end-1) < 0, 2);
a = y(locs);
b = y(locs+1);
frac = -a ./ (b - a);
zclist = x(locs) .* (1 - frac) + x(locs + 1) .* frac;
return


function status = test_spike_rates(do_plots)
% Test: Assure the 3 class rates versus level look good.

status = 0;
fs = 22050;
fp = 1000;  % Probe tone frequency
duration = 0.25;
dbstep = 10;   % 10 is good
dbfs = -104:dbstep:6;  % 0 to 110 dB SPL

t = (0:(1/fs):(duration - 1/fs))';  % Sample times for short duration
sinusoid = sin(2 * pi * t * fp);
signal = [];
time = [];
t_start = 0;
for db = dbfs  % Levels spanning a huge range
  amplitude = sqrt(2) * 10.^(db/20);
  signal = [signal; amplitude*sinusoid];
  time = [time; t + t_start];
  t_start = t_start + duration;
end

CF = CARFAC_Design(1, fs, 'do_syn');  % v3 3-class synapse model
CF = CARFAC_Init(CF);
[nap, CF, bm, ohc, agc, firings] = CARFAC_Run_Segment(CF, signal);  % nap has 3 columns of firings

if do_plots
  %%
  chan = find(CF.pole_freqs * 1.06 < fp, 1); % probably best channel
  chan_firings = squeeze(firings(:, chan, :, 1));  % Just one channel, 3 class columns.
  healthy_n_fibers = CF.SYN_params.healthy_n_fibers;
  rates = chan_firings ./ healthy_n_fibers;

  figure();
  plot(time, chan_firings);
  title('Instantaneous rates of 3 fiber-group classes')
  xlabel('time in seconds, with 10 dB steps from -100 to 0 dB FS')
  ylabel('firings per sample')
  for db = dbfs + 104
    text(duration * (db/dbstep + 0.4), 12, num2str(db))
  end

  figure();
  plot(time, fs*smooth1d(rates, fs*0.005)) % Per fiber
  title('Mean rates of 3 fiber classes')
  xlabel('time in seconds, with 10 dB steps from 0 to 110 dB SPL rms')
  ylabel('firings per second per fiber')
  for db = dbfs + 104
    text(duration * (db/dbstep + 0.4), 100, num2str(db))
  end
  octave_basal_chan = find(CF.pole_freqs * 1.06 < fp*2, 1);
  half_octave_basal_chan = find(CF.pole_freqs * 1.06 < fp*sqrt(2), 1);
  best_chan = find(CF.pole_freqs * 1.06 < fp, 1);
  half_octave_apical_chan = find(CF.pole_freqs * 1.06 < fp/sqrt(2), 1);
  channels = [octave_basal_chan, half_octave_basal_chan, best_chan, ...
    half_octave_apical_chan];
  figure()
  plot(time(1:8:end), agc(1:8:end, channels))
  text(2.55, 0.15, [num2str(channels(4)), ': apical 0.5'])
  text(2.55, 0.5, [num2str(channels(3)), ': best'])
  text(2.58, 0.74, [num2str(channels(2)), ': basal 0.5'])
  text(2.45, 0.93, [num2str(channels(1)), ': basal 1'])
  for db = dbfs + 104
    text(duration * (db/dbstep + 0.4), 0.4, num2str(db))
  end
end

report_status(status, 'test_spike_rates')
return


function report_status(status, name, extra)
if nargin < 3, extra = 0; end
if extra
  if status
    disp(['FAIL ' name '; at least one test failed.'])
  else
    disp(['PASS ' name '; all tests passed.'])
  end
else
  if status
    disp(['FAIL ' name])
    if status > 1
      disp('(status > 1 => error in test or expected results size)')
    end
  else
    disp(['PASS ' name])
  end
end
return
