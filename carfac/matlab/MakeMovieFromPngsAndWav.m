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

function MakeMovieFromPngsAndWav(frame_rate, png_name_pattern, ...
  wav_filename, out_filename)

system(['rm "', out_filename, '"']);

if ~exist(wav_filename, 'file')
  error('wave file is missing', wav_filename)
end

ffmpeg_command = ['/opt/local/bin/ffmpeg' ...
  ' -r ' num2str(frame_rate) ...
  ' -i ' png_name_pattern ...
  ' -i "' wav_filename ...
  '" -b:v 1024k' ...
  ' "' out_filename '"'];

system(ffmpeg_command);
