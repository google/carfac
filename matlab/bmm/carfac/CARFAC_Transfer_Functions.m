function [complex_transfns_freqs, ...
  stage_numerators, stage_denominators] = ...
  CARFAC_Transfer_Functions(CF, freqs, to_channels, from_channels)
% function [complex_transfns_freqs, ...
%   stage_z_numerators, stage_z_denominators] = ...
%   CARFAC_Transfer_Functions(CF, freqs, to_channels, from_channels)
% Return transfer functions as polynomials in z (nums & denoms);
% And evaluate them at freqs if it's given, to selected output,
%   optionally from selected starting points (from 0, input, by default).
%   complex_transfns_freqs has a row of complex gains per to_channel.

% always start with the rational functions, whether we want to return
% them or not:
[stage_numerators, stage_denominators] = CARFAC_Rational_Functions(CF);

if nargin >= 2
  % Evaluate at the provided list of frequencies.
  if ~isrow(freqs)
    if iscolumn(freqs)
      freqs = freqs';
    else
      error('Bad freqs_row in CARFAC_Transfer_Functions');
    end
  end
  if any(freqs < 0)
    error('Negatives in freqs_row in CARFAC_Transfer_Functions');
  end
  z_row = exp((i * 2 * pi / CF.fs) * freqs);  % z = exp(sT)
  gains = Rational_Eval(stage_numerators, stage_denominators, z_row);
  
  % Now multiply gains from input to output places; use logs?
  log_gains = log(gains);
  cum_log_gains = cumsum(log_gains);
  
  % And figure out which cascade products we want:
  n_ch = CF.n_ch;
  if nargin < 3
    to_channels = 1:n_ch;
  end
  if isempty(to_channels) || any(to_channels < 1 | to_channels > n_ch)
    error('Bad to_channels in CARFAC_Transfer_Functions');
  end
  if nargin < 4 || isempty(from_channels)
    from_channels = 0;  % tranfuns from input, called channel 0.
  end
  if length(from_channels) == 1
    from_channels = from_channels * ones(length(to_channels));
  end
  % Default to cum gain of 1 (log is 0), from input channel 0:
  from_cum = zeros(length(to_channels), length(z_row));
  not_input = from_channels > 0;
  from_cum(not_input, :) = cum_log_gains(from_channels(not_input), :);
  log_transfns = cum_log_gains(to_channels, :) - from_cum;
  complex_transfns_freqs = exp(log_transfns);
else
  % If no freqs are provided, do nothing but return the stage info above:
  complex_transfns_freqs = [];
end



function gains = Rational_Eval(numerators, denominators, z_row)
% function gains = Rational_Eval(numerators, denominators, z_row)
% Evaluate rational function at row of z values.

zz = [z_row .* z_row; z_row; ones(size(z_row))];
% dot product of each poly row with each [z2; z; 1] col:
gains = (numerators * zz) ./ (denominators * zz);


function [stage_numerators, stage_denominators] = ...
  CARFAC_Rational_Functions(CF, chans)
% function [stage_z_numerators, stage_z_denominators] = ...
%   CARFAC_Rational_Functions(CF, chans)
% Return transfer functions of all stages as rational functions.

if nargin < 2
  n_ch = CF.n_ch;
  chans = 1:n_ch;
else
  n_ch = length(chans);
end
coeffs = CF.filter_coeffs;
r = coeffs.r_coeffs(chans);
a = coeffs.a_coeffs(chans) .* r;
c = coeffs.c_coeffs(chans) .* r;
r2 = r .* r;
h = coeffs.h_coeffs(chans);
g = coeffs.g_coeffs(chans);
stage_denominators = [ones(n_ch, 1), -2 * a, r2];
stage_numerators = [g .* ones(n_ch, 1), g .* (-2 * a + h .* c), g .* r2];


%% example
% CF = CARFAC_Design
% f = (0:100).^2;  % frequencies to 10 kHz, unequally spaced
% to_ch = 10:10:96;  % selected output channels
% from_ch = to_ch - 10;  % test the inclusion of 0 in from list
% tf = CARFAC_Transfer_Functions(CF, f, to_ch, from_ch);
% figure
% plot(f, 20*log(abs(tf)')/log(10));  % dB vs lin. freq for 10 taps

