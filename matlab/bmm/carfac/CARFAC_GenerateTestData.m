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

% These are some basic settings for the CARFAC design and test data.
n_ears = 1;
wav_fn = 'plan.wav';

seg_len = 441;
start_index = 13000;
end_index = start_index + seg_len - 1;
file_signal = wavread(wav_fn);
file_signal = file_signal(start_index:end_index, 1);

filename = 'file_signal_monaural_test.txt'
data = file_signal;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% This designs and initializes the CARFAC.
CF_struct = CARFAC_Design(n_ears);
CF_struct = CARFAC_Init(CF_struct);



% Now we go through the coefficient structures and store each set in a text
% file.

% r1 coeffs
filename = 'r1_coeffs.txt'
data = CF_struct.ears.CAR_coeffs.r1_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% a0 coeffs
filename = 'a0_coeffs.txt'
data = CF_struct.ears.CAR_coeffs.a0_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% c0 coeffs
filename = 'c0_coeffs.txt'
data = CF_struct.ears.CAR_coeffs.c0_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% h coeffs
filename = 'h_coeffs.txt'
data = CF_struct.ears.CAR_coeffs.h_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% g0 coeffs
filename = 'g0_coeffs.txt'
data = CF_struct.ears.CAR_coeffs.g0_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% zr coeffs
filename = 'zr_coeffs.txt'
data = CF_struct.ears.CAR_coeffs.zr_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% Now we store the IHC coefficients in an output vector.

coeff = CF_struct.ears.IHC_coeffs;
ihc_coeffs = [coeff.just_hwr; coeff.lpf_coeff; coeff.out_rate; ...
    coeff.in_rate; coeff.one_cap; coeff.output_gain; ...
    coeff.rest_output; coeff.rest_cap; coeff.ac_coeff];

filename = 'ihc_coeffs.txt'
data = ihc_coeffs;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

% For each of the AGC Stages, we store a text file containing all of the
% coefficients.

agc_coeffs = CF_struct.ears.AGC_coeffs

for stage = 1:4
    data = zeros(14,1);
    data(1) = agc_coeffs(stage).n_ch;
    data(2) = agc_coeffs(stage).n_AGC_stages;
    data(3) = agc_coeffs(stage).AGC_stage_gain;
    data(4) = agc_coeffs(stage).decimation;
    data(5) = agc_coeffs(stage).AGC_epsilon;
    data(6) = agc_coeffs(stage).AGC_polez1;
    data(7) = agc_coeffs(stage).AGC_polez2;
    data(8) = agc_coeffs(stage).AGC_spatial_iterations;
    data(9) = agc_coeffs(stage).AGC_spatial_FIR(1);
    data(10) = agc_coeffs(stage).AGC_spatial_FIR(2);
    data(11) = agc_coeffs(stage).AGC_spatial_FIR(3);
    data(12) = agc_coeffs(stage).AGC_spatial_n_taps;
    data(13) = agc_coeffs(stage).AGC_mix_coeffs;
    data(14) = agc_coeffs(1).detect_scale;
    filename = ['agc_coeffs_' int2str(stage) '.txt']
    dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');
end

% This section of code runs the single segment of data which was selected
% and stores the nap, bm, ohc and agc outputs of the CARFAC.

[CF_struct, nap_decim, nap, bm, ohc, agc] = CARFAC_Run(CF_struct, file_signal);

filename = 'monaural_test_nap.txt'
data = nap;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

filename = 'monaural_test_bm.txt'
data = bm;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

filename = 'monaural_test_ohc.txt'
data = ohc;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');

filename = 'monaural_test_agc.txt'
data = agc;
dlmwrite([data_dir filename],data,'precision', 12,'delimiter',' ');