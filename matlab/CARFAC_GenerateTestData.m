% Copyright 2013 The CARFAC Authors. All Rights Reserved.
% Author: Alex Brandmeyer
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

function CARFAC_GenerateTestData()
% function GenerateTestData()
% This function generates a set of text files in the AIMC repository that
% can be used to compare the output of the C++ version of CARFAC with that
% of the Matlab version.
%
% Naming convention for files containing audio samples for file test_name.wav:
%   test_name-audio.txt
% Each line contains a space-separated list of samples from each channel.
%
% Naming convention for files containing CARFAC/SAI outputs:
%   test_name-{matlab,cpp}-signal_name(optional_channel_number).txt
% Each line contains a space-separated list of elements from a single row.

% This designates a subdirectory of the C++ CARFAC folder to store the
% test data.
test_data_dir = '../test_data/';

sai_struct = struct('width', 500, ...
                    'future_lags', 250, ...
                    'n_window_pos', 2, ...
                    'channel_smoothing_scale', 0);


test_name = 'binaural_test';
samples_to_read = [9000, 9903];  % Trim for a faster test.
signal = wavread([test_data_dir test_name '.wav'], samples_to_read);
assert(size(signal, 2) == 1, 'Expected mono signal.');
% Construct a binaural signal by delaying the signal between the ears.
itd_offset = 22;  % about 1 ms
signal = [signal((itd_offset+1):end), signal(1:(end-itd_offset))] / 10;
n_ears = size(signal, 2);
CF_struct = CARFAC_Design(n_ears);
WriteTestData(test_data_dir, 'binaural_test', signal, CF_struct, sai_struct);

test_name = 'long_test';
samples_to_read = [80001, 82000];  % Trim for a faster test.
[signal, fs] = wavread([test_data_dir test_name '.wav'], samples_to_read);
assert(size(signal, 2) == 2, 'Expected stereo signal.');
n_ears = size(signal, 2);
CF_struct = CARFAC_Design(n_ears, fs);
WriteTestData(test_data_dir, 'long_test', signal, CF_struct, sai_struct);

% We explicitly initialize parameter structures using defaults in order to
% generate test data for evaluation of precision issues in C++ version.

CF_CAR_params = struct( ...
    'velocity_scale', 0.1, ...  % for the velocity nonlinearity
    'v_offset', 0.04, ...  % offset gives a quadratic part
    'min_zeta', 0.10, ... % minimum damping factor in mid-freq channels
    'max_zeta', 0.35, ... % maximum damping factor in mid-freq channels
    'first_pole_theta', 0.85*pi, ...
    'zero_ratio', sqrt(2), ... % how far zero is above pole
    'high_f_damping_compression', 0.5, ... % 0 to 1 to compress zeta
    'ERB_per_step', 0.5, ... % assume G&M's ERB formula
    'min_pole_Hz', 30, ...
    'ERB_break_freq', 165.3, ...  % Greenwood map's break freq.
    'ERB_Q', 1000/(24.7*4.37));

CF_AGC_params = struct( ...
    'n_stages', 4, ...
    'time_constants', 0.002 * 4.^(0:3), ...
    'AGC_stage_gain', 2, ...  % gain from each stage to next slower stage
    'decimation', [8, 2, 2, 2], ...  % how often to update the AGC states
    'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   % in units of channels
    'AGC2_scales', 1.65 * sqrt(2).^(0:3), ... % spread more toward base
    'AGC_mix_coeff', 0.5);

one_cap = 1;         % bool; 1 for Allen model, as text states we use
just_hwr = 0;        % book; 0 for normal/fancy IHC; 1 for HWR
if just_hwr
    CF_IHC_params = struct('just_hwr', 1, ...  % just a simple HWR
        'ac_corner_Hz', 20);
else
    if one_cap
        CF_IHC_params = struct( ...
            'just_hwr', just_hwr, ...        % not just a simple HWR
            'one_cap', one_cap, ...   % bool; 0 for new two-cap hack
            'tau_lpf', 0.000080, ...  % 80 microseconds smoothing twice
            'tau_out', 0.0005, ...    % depletion tau is pretty fast
            'tau_in', 0.010, ...        % recovery tau is slower
            'ac_corner_Hz', 20);
    else
        CF_IHC_params = struct( ...
            'just_hwr', just_hwr, ...        % not just a simple HWR
            'one_cap', one_cap, ...   % bool; 0 for new two-cap hack
            'tau_lpf', 0.000080, ...  % 80 microseconds smoothing twice
            'tau1_out', 0.010, ...    % depletion tau is pretty fast
            'tau1_in', 0.020, ...     % recovery tau is slower
            'tau2_out', 0.0025, ...   % depletion tau is pretty fast
            'tau2_in', 0.005, ...        % recovery tau is slower
            'ac_corner_Hz', 20);
    end
end


% Now we setup tests in which various parameters for the individual steps
% are modified in order to determine where the precision changes occur.

% First we test the removal of the AGC stage using the open_loop option.
open_loop = 1;
test_name = 'agc_test';
samples_to_read = [80001, 82000];  % Trim for a faster test.
[signal, fs] = wavread([test_data_dir 'long_test' '.wav'], samples_to_read);
assert(size(signal, 2) == 2, 'Expected stereo signal.');
n_ears = size(signal, 2);
CF_struct = CARFAC_Design(n_ears, fs, CF_CAR_params, CF_AGC_params, ...
            CF_IHC_params);
WriteTestData(test_data_dir, 'agc_test', signal, CF_struct, sai_struct, open_loop);

% The next test verifies that the 'just_hwr' option in the IHC stage works.
open_loop = 0;
test_name = 'ihc_just_hwr_test';
CF_IHC_params.just_hwr = 1;
samples_to_read = [80001, 82000];  % Trim for a faster test.
[signal, fs] = wavread([test_data_dir 'long_test' '.wav'], samples_to_read);
assert(size(signal, 2) == 2, 'Expected stereo signal.');
n_ears = size(signal, 2);
CF_struct = CARFAC_Design(n_ears, fs, CF_CAR_params, CF_AGC_params, ...
            CF_IHC_params);
WriteTestData(test_data_dir, 'ihc_just_hwr_test', signal, CF_struct, sai_struct, open_loop);

function WriteTestData(test_data_dir, test_name, signal, CF_struct, sai_struct, open_loop)
% function WriteTestData(test_data_dir, test_name, signal, CF_struct, sai_struct, open_loop)
% 
% Helper function to run CARFAC and SAI over the given signal and
% save the results to text files in test_data_dir.

% The following section generates data for the binaural test of the C++
% version of CARFAC.
if nargin < 6
    open_loop = 0;
end

filename_prefix = [test_data_dir test_name];

WriteMatrixToFile([filename_prefix '-audio.txt'], signal);

CF_struct = CARFAC_Init(CF_struct);
[CF_struct, nap_decim, nap, bm, ohc, agc] = CARFAC_Run(CF_struct, signal, 0, open_loop);

% Store the data for each ear of each output signal in a separate file.
for ear = 1:CF_struct.n_ears
  WriteMatrixToFile([filename_prefix '-matlab-nap' num2str(ear) '.txt'], ...
                    nap(:,:,ear));
  WriteMatrixToFile([filename_prefix '-matlab-bm' num2str(ear) '.txt'], ...
                    bm(:,:,ear));
end

ear = 1;
sai_struct.window_width = size(signal, 1);
sai_struct = SAI_Run_Segment(sai_struct, nap(:,:,ear));
WriteMatrixToFile([filename_prefix '-matlab-sai' num2str(ear) '.txt'], ...
                  sai_struct.frame);


function WriteMatrixToFile(filename, matrix)
precision_level = 9;
dlmwrite(filename, matrix, 'precision', precision_level, 'delimiter', ' ');
