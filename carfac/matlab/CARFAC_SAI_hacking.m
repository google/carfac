% Copyright 2013 The CARFAC Authors. All Rights Reserved.
% Author: Richard F. Lyon
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

%% Test/demo hacking for CARFAC_SAI Matlab stuff:

clear variables

system('mkdir frames');

%%

dB_list = -40; %  -60:20:0

wav_fn = '../test_data/binaural_test.wav';

if ~exist(['./', wav_fn], 'file')
  error('wav file not found')
end

wav_fn
[file_signal, fs] = wavread(wav_fn);

% if fs == 44100
%   file_signal = (file_signal(1:2:end-1, :) + file_signal(2:2:end, :)) / 2;
%   fs = fs / 2;
% end
% 
% if fs ~= 22050
%   error('unexpected sample rate')
% end

file_signal = file_signal(:, 1);  % mono
file_signal = [file_signal; zeros(fs, 1)];  % pad with a second of silence


% make a long test signal by repeating at different levels:
test_signal = [];
for dB =  dB_list
  test_signal = [test_signal; file_signal * 10^(dB/20)];
end

%%
CF_struct = CARFAC_Design(1, fs);  % default design

CF_struct = CARFAC_Init(CF_struct);

[frame_rate, num_frames] = SAI_RunLayered(CF_struct, test_signal);

%%
png_name_pattern = 'frames/frame%05d.png';
MakeMovieFromPngsAndWav(round(frame_rate), png_name_pattern, ...
  wav_fn, ['CARFAC_SAI_movie_', wav_fn(1:end-4), '.mpg'])

%%
system('rm -r frames');


