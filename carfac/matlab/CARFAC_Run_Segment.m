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

function [naps, CF, BM, seg_ohc, seg_agc] = CARFAC_Run_Segment(...
  CF, input_waves, open_loop)
% function [naps, CF, BM, seg_ohc, seg_agc] = CARFAC_Run_Segment(...
%   CF, input_waves, open_loop)
% 
% This function runs the CARFAC; that is, filters a 1 or more channel
% sound input segment to make one or more neural activity patterns (naps);
% it can be called multiple times for successive segments of any length,
% as long as the returned CF with modified state is passed back in each
% time.
%
% input_waves is a column vector if there's just one audio channel;
% more generally, it has a row per time sample, a column per audio channel.
%
% naps has a row per time sample, a column per filterbank channel, and
% a layer per audio channel if more than 1.
% BM is basilar membrane motion (filter outputs before detection).
%
% the input_waves are assumed to be sampled at the same rate as the
% CARFAC is designed for; a resampling may be needed before calling this.
%
% The function works as an outer iteration on time, updating all the
% filters and AGC states concurrently, so that the different channels can
% interact easily.  The inner loops are over filterbank channels, and
% this level should be kept efficient.
%
% seg_ohc seg_agc are optional extra outputs useful for seeing what the
% ohc nonlinearity and agc are doing; both in terms of extra damping.

if nargin < 3
  open_loop = 0;
end

if nargout > 2
  do_BM = 1;
else
  do_BM = 0;
end

[n_samp, n_ears] = size(input_waves);

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end

n_ch = CF.n_ch;
naps = zeros(n_samp, n_ch, n_ears);  % allocate space for result
if do_BM
  BM = zeros(n_samp, n_ch, n_ears);
  seg_ohc = zeros(n_samp, n_ch, n_ears);
  seg_agc = zeros(n_samp, n_ch, n_ears);
end

detects = zeros(n_ch, n_ears);
for k = 1:n_samp
  % at each time step, possibly handle multiple channels
  for ear = 1:n_ears
    
    % This would be cleaner if we could just get and use a reference to
    % CF.ears(ear), but Matlab doesn't work that way...
    
    [car_out, CF.ears(ear).CAR_state] = CARFAC_CAR_Step( ...
      input_waves(k, ear), CF.ears(ear).CAR_coeffs, CF.ears(ear).CAR_state);
    
    % update IHC state & output on every time step, too
    [ihc_out, CF.ears(ear).IHC_state] = CARFAC_IHC_Step( ...
      car_out, CF.ears(ear).IHC_coeffs, CF.ears(ear).IHC_state);
    
    % run the AGC update step, decimating internally,
    [CF.ears(ear).AGC_state, updated] = CARFAC_AGC_Step( ...
      ihc_out, CF.ears(ear).AGC_coeffs, CF.ears(ear).AGC_state);
    
    % save some output data:
    naps(k, :, ear) = ihc_out;  % output to neural activity pattern
    if do_BM
      BM(k, :, ear) = car_out;
      state = CF.ears(ear).CAR_state;
      seg_ohc(k, :, ear) = state.zA_memory;
      seg_agc(k, :, ear) = state.zB_memory;;
    end
  end
  
  % connect the feedback from AGC_state to CAR_state when it updates;
  % all ears together here due to mixing across them:
  if updated 
    if n_ears > 1
      % do multi-aural cross-coupling:
      CF.ears = CARFAC_Cross_Couple(CF.ears);
    end
    if ~open_loop
      CF = CARFAC_Close_AGC_Loop(CF);
    end
  end
end

