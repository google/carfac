% Author: Alex Brandmeyer
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

function CARFAC_GenerateTestData()
% function GenerateTestData()
% This function generates a set of text files in the AIMC repository that
% can be used to compare the output of the C++ version of CARFAC with that
% of the Matlab version.

% This designates a subdirectory of the C++ CARFAC folder to store the
% test data
data_dir = '../../../carfac/test_data/';
precision_level = 9;

% The following section generates data for the binaural test of the C++
% version of CARFAC.
n_ears = 2;
 
file_signal = wavread('plan.wav');
file_signal = file_signal(9000:9903);  % trim for a faster test
 
itd_offset = 22;  % about 1 ms
test_signal = [file_signal((itd_offset+1):end), ...
               file_signal(1:(end-itd_offset))] / 10;
           
filename = 'file_signal_binaural_test.txt'
data = test_signal;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
           
CF_struct = CARFAC_Design(n_ears);
CF_struct = CARFAC_Init(CF_struct);         
[CF_struct, nap_decim, nap, bm, ohc, agc] = CARFAC_Run(CF_struct, test_signal);
 
%Store the data for each each as individual 2d text data.
nap1 = nap(:,:,1);
nap2 = nap(:,:,2);
bm1 = bm(:,:,1);
bm2 = bm(:,:,2);
 
filename = 'binaural_test_nap1.txt'
data = nap1;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
 
filename = 'binaural_test_bm1.txt'
data = bm1;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
 
filename = 'binaural_test_nap2.txt'
data = nap2;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
 
filename = 'binaural_test_bm2.txt'
data = bm2;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');


% Longer audio segment test

n_ears = 2;
start = 80001;
n_timepoints = 2000;
 
[test_signal, fs] = wavread([data_dir 'Anka_SLTS.wav']);
test_signal = test_signal(start:start+n_timepoints-1,:);
size(test_signal)
 
filename = 'file_signal_long_test.txt'
data = test_signal;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
           
CF_struct = CARFAC_Design(n_ears, fs);
CF_struct = CARFAC_Init(CF_struct);         
[CF_struct, nap_decim, nap, bm, ohc, agc] = CARFAC_Run(CF_struct, test_signal);
 
%Store the data for each each as individual 2d text data.
nap1 = nap(:,:,1);
nap2 = nap(:,:,2);
bm1 = bm(:,:,1);
bm2 = bm(:,:,2);
 
filename = 'long_test_nap1.txt'
data = nap1;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
 
filename = 'long_test_bm1.txt'
data = bm1;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
 
filename = 'long_test_nap2.txt'
data = nap2;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
 
filename = 'long_test_bm2.txt'
data = bm2;
dlmwrite([data_dir filename],data,'precision', precision_level,'delimiter',' ');
