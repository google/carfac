% Copyright 2012, Google, Inc.
% Author Richard F. Lyon
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

function [CF, decim_naps, naps, BM] = CARFAC_Run ...
  (CF, input_waves, AGC_plot_fig_num)
% function [CF, decim_naps, naps] = CARFAC_Run ...
%   (CF, input_waves, AGC_plot_fig_num)
% This function runs the CARFAC; that is, filters a 1 or more channel
% sound input to make one or more neural activity patterns (naps).
%
% The CF struct holds the filterbank design and state; if you want to
% break the input up into segments, you need to use the updated CF
% to keep the state between segments.
%
% input_waves is a column vector if there's just one audio channel;
% more generally, it has a row per time sample, a column per audio channel.
%
% naps has a row per time sample, a column per filterbank channel, and
% a layer per audio channel if more than 1.
% decim_naps is like naps but time-decimated by the int CF.decimation.
%
% the input_waves are assumed to be sampled at the same rate as the
% CARFAC is designed for; a resampling may be needed before calling this.
%
% The function works as an outer iteration on time, updating all the
% filters and AGC states concurrently, so that the different channels can
% interact easily.  The inner loops are over filterbank channels, and
% this level should be kept efficient.

[n_samp, n_ears] = size(input_waves);
n_ch = CF.n_ch;

if nargin < 3
  AGC_plot_fig_num = 0;
end

if nargout > 3
  BM = zeros(n_samp, n_ch, n_ears);
else
  BM = [];
end

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end


naps = zeros(n_samp, n_ch, n_ears);

seglen = 256;
n_segs = ceil(n_samp / seglen);

if nargout > 1
  % make decimated detect output:
  decim_naps = zeros(n_segs, CF.n_ch, CF.n_ears);
else
  decim_naps = [];
end

if nargout > 2
  % make decimated detect output:
  naps = zeros(n_samp, CF.n_ch, CF.n_ears);
else
  naps = [];
end

for seg_num = 1:n_segs
  if seg_num == n_segs
    % The last segement may be short of seglen, but do it anyway:
    k_range = (seglen*(seg_num - 1) + 1):n_samp;
  else
    k_range = seglen*(seg_num - 1) + (1:seglen);
  end
  % Process a segment to get a slice of decim_naps, and plot AGC state:
  if ~isempty(BM)
    [seg_naps, CF, seg_BM] = CARFAC_Run_Segment(CF, input_waves(k_range, :));
  else
    [seg_naps, CF] = CARFAC_Run_Segment(CF, input_waves(k_range, :));
  end
  
  if ~isempty(BM)
    for ear = 1:n_ears
      % Accumulate segment BM to make full BM
      BM(k_range, :, ear) = seg_BM(:, :, ear);
    end
  end
  
  if ~isempty(naps)
    for ear = 1:n_ears
      % Accumulate segment naps to make full naps
      naps(k_range, :, ear) = seg_naps(:, :, ear);
    end
  end
  
  if ~isempty(decim_naps)
    for ear = 1:n_ears
      decim_naps(seg_num, :, ear) = CF.ears(ear).IHC_state.ihc_accum / seglen;
      CF.ears(ear).IHC_state.ihc_accum = zeros(n_ch,1);
    end
  end
  
  if AGC_plot_fig_num
    figure(AGC_plot_fig_num); hold off; clf
    for ear = 1:n_ears
      maxes(ear) = max(CF.ears(ear).AGC_state.AGC_memory(:));
      hold on
      for stage = 1:4;
        plot(2^(stage-1) * CF.ears(ear).AGC_state.AGC_memory(:, stage));
      end
    end
    axis([0, CF.n_ch+1, 0.0, max(maxes) * 1.01 + 0.002]);
    drawnow
  end

end



