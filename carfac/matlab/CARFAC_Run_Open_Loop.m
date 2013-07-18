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

function [CF, decim_naps, naps] = CARFAC_Run_Open_Loop ...
  (CF, input_waves, AGC_plot_fig_num)
% function [CF, decim_naps, naps] = CARFAC_Run_Open_Loop ...
%   (CF, input_waves, AGC_plot_fig_num)
%
% Freeze the damping by disabling AGC feedback, and run so we can
% see what the filters and AGC do in that frozen state.  And zap the
% stage gain in the AGC so we can see the state filters without combining
% them.

[n_samp, n_ears] = size(input_waves);
n_ch = CF.n_ch;

if nargin < 3
  AGC_plot_fig_num = 0;
end

if n_ears ~= CF.n_ears
  error('bad number of input_waves channels passed to CARFAC_Run')
end


naps = zeros(n_samp, n_ch, n_ears);

seglen = 16;
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

% zero the deltas:
for ear = 1:CF.n_ears
  CF.CAR_state(ear).dzB_memory = 0;
  CF.CAR_state(ear).dg_memory = 0;
end
open_loop = 1;
CF.AGC_coeffs.AGC_stage_gain = 0;  % HACK to see the stages separately

smoothed_state = 0;

for seg_num = 1:n_segs
  if seg_num == n_segs
    % The last segement may be short of seglen, but do it anyway:
    k_range = (seglen*(seg_num - 1) + 1):n_samp;
  else
    k_range = seglen*(seg_num - 1) + (1:seglen);
  end
  % Process a segment to get a slice of decim_naps, and plot AGC state:
  [seg_naps, CF] = CARFAC_Run_Segment(CF, input_waves(k_range, :), ...
    open_loop);
  
  if ~isempty(naps)
    for ear = 1:n_ears
      % Accumulate segment naps to make full naps
      naps(k_range, :, ear) = seg_naps(:, :, ear);
    end
  end
  
  if ~isempty(decim_naps)
    for ear = 1:n_ears
      decim_naps(seg_num, :, ear) = CF.IHC_state(ear).ihc_accum / seglen;
      CF.IHC_state(ear).ihc_accum = zeros(n_ch,1);
    end
  end
  
  if AGC_plot_fig_num
    figure(AGC_plot_fig_num); hold off; clf
    set(gca, 'Position', [.25, .25, .5, .5])
    smoothed_state = (3*smoothed_state + CF.AGC_state(1).AGC_memory) / 4;
    for ear = 1
      total_state = 0;
      for stage = 1:4;
        weighted_state = smoothed_state(:, stage) * 2^(stage-1);
        plot(weighted_state, 'k-', 'LineWidth', 0.4);
        hold on
        total_state = total_state + weighted_state;
      end
      maxes(ear) = max(total_state);
      plot(total_state, 'k-', 'LineWidth', 1.1)
    end
    
    axis([0, CF.n_ch+1, 0.0, max(maxes) * 1.01 + 0.002]);
    drawnow
  end
  
end



