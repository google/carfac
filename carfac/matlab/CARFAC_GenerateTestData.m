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
                    'window_width', 2000, ...
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
20

function WriteTestData(test_data_dir, test_name, signal, CF_struct, sai_struct)
% The following section generates data for the binaural test of the C++
% version of CARFAC.
filename_prefix = [test_data_dir test_name];

WriteMatrixToFile([filename_prefix '-audio.txt'], signal);

CF_struct = CARFAC_Init(CF_struct);         
[CF_struct, nap_decim, nap, bm, ohc, agc] = CARFAC_Run(CF_struct, signal);
 
% Store the data for each ear of each output signal in a separate file.
for ear = 1:CF_struct.n_ears
  WriteMatrixToFile([filename_prefix '-matlab-nap' num2str(ear) '.txt'], ...
                    nap(:,:,ear));
  WriteMatrixToFile([filename_prefix '-matlab-bm' num2str(ear) '.txt'], ...
                    bm(:,:,ear));
end

ear = 1;
sai_struct = SAI_Run_Segment(sai_struct, nap(:,:,ear));
WriteMatrixToFile([filename_prefix '-matlab-sai' num2str(ear) '.txt'], ...
                  sai_struct.frame);


function WriteMatrixToFile(filename, matrix)
precision_level = 9;
dlmwrite(filename, matrix, 'precision', precision_level, 'delimiter', ' ');
