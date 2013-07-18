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

function CARFAC_Compare_CPP_Test_Data()
data_dir = '../../../carfac/test_data/';
filename = 'long_test_nap1.txt';
matlab_nap1 = dlmread([data_dir filename]);
filename = 'cpp_nap_output_1_long_test.txt';
cpp_nap1 = dlmread([data_dir filename]);
filename = 'long_test_nap2.txt';
matlab_nap2 = dlmread([data_dir filename]);
filename = 'cpp_nap_output_2_long_test.txt';
cpp_nap2 = dlmread([data_dir filename]);

factor = 10;
figure(1)
subplot(2,2,1)
image(matlab_nap1' * factor);
title('Long Test: Matlab NAP, Ear 1');
ylabel('Channel');
xlabel('Sample Index');
subplot(2,2,2)
image(matlab_nap2' * factor);
title('Long Test: Matlab NAP, Ear 2')
ylabel('Channel');
xlabel('Sample Index');
subplot(2,2,3)
image(cpp_nap1' * factor);
title('Long Test: C++ NAP, Ear 1')
ylabel('Channel');
xlabel('Sample Index');
subplot(2,2,4)
image(cpp_nap2' * factor);
title('Long Test: C++ NAP, Ear 2')
ylabel('Channel');
xlabel('Sample Index');

filename = 'binaural_test_nap1.txt';
matlab_nap1 = dlmread([data_dir filename]);
filename = 'cpp_nap_output_1_binaural_test.txt';
cpp_nap1 = dlmread([data_dir filename]);
filename = 'binaural_test_nap2.txt';
matlab_nap2 = dlmread([data_dir filename]);
filename = 'cpp_nap_output_2_binaural_test.txt';
cpp_nap2 = dlmread([data_dir filename]);

factor = 10;
figure(2)
subplot(2,2,1)
image(matlab_nap1' * factor);
title('Binaural Test: Matlab NAP, Ear 1');
ylabel('Channel');
xlabel('Sample Index');
subplot(2,2,2)
image(matlab_nap2' * factor);
title('Binaural Test: Matlab NAP, Ear 2')
ylabel('Channel');
xlabel('Sample Index');
subplot(2,2,3)
image(cpp_nap1' * factor);
title('Binaural Test: C++ NAP, Ear 1')
ylabel('Channel');
xlabel('Sample Index');
subplot(2,2,4)
image(cpp_nap2' * factor);
title('Binaural Test: C++ NAP, Ear 2')
ylabel('Channel');
xlabel('Sample Index');
