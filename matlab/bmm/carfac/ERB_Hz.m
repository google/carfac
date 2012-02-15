function ERB = ERB_Hz(CF_Hz, ERB_break_freq, ERB_Q)
% function ERB = ERB_Hz(CF_Hz, ERB_break_freq, ERB_Q)
%
% Auditory filter nominal Equivalent Rectangular Bandwidth
%	Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
% ERB = 24.7 * (1 + 4.37 * CF_Hz / 1000);

if nargin < 3
  ERB_Q = 1000/(24.7*4.37);  % 9.2645
  if nargin < 2
    ERB_break_freq = 1000/4.37;  % 228.833
  end
end

ERB = (ERB_break_freq + CF_Hz) / ERB_Q;
