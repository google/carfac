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

function [CF, decim_naps, naps, BM, ohc, agc] = CARFAC_Run ...
  (CF, input_waves, AGC_plot_fig_num, open_loop)
% function [CF, decim_naps, naps, BM, ohc, agc] = CARFAC_Run ...
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
% ohc and agc are optional extra outputs for diagnosing internals.

[n_samp, n_ears] = size(input_waves);
n_ch = CF.n_ch;

if nargin < 4
  open_loop = 0;
end

if nargin < 3
  AGC_plot_fig_num = 0;
end

if nargout > 3
  BM = zeros(n_samp, n_ch, n_ears);
else
  BM = [];
end

if nargout > 4
  ohc = zeros(n_samp, n_ch, n_ears);
else
  ohc = [];
end

if nargout > 5
  agc = zeros(n_samp, n_ch, n_ears);
else
  agc = [];
end

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end


naps = zeros(n_samp, n_ch, n_ears);

seglen = 441;  % anything should work; this is 20 ms at default fs
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
    % ask for everything in this case, for laziness:
    [seg_naps, CF, seg_BM, seg_ohc, seg_agc] = CARFAC_Run_Segment(CF, input_waves(k_range, :), open_loop);
  else
    [seg_naps, CF] = CARFAC_Run_Segment(CF, input_waves(k_range, :), open_loop);
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
  
  if ~isempty(ohc)
    for ear = 1:n_ears
      % Accumulate segment naps to make full naps
      ohc(k_range, :, ear) = seg_ohc(:, :, ear);
    end
  end
  
  if ~isempty(agc)
    for ear = 1:n_ears
      % Accumulate segment naps to make full naps
      agc(k_range, :, ear) = seg_agc(:, :, ear);
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
    maxmax = 0;
    for ear = 1:n_ears
      hold on
      for stage = 1:4;
        stage_response = 2^(stage-1) * CF.ears(ear).AGC_state(stage).AGC_memory;
        plot(stage_response);
        maxmax = max(maxmax, max(stage_response));
      end
    end
    axis([0, CF.n_ch+1, 0.0, maxmax * 1.01 + 0.002]);
    drawnow
  end

end



