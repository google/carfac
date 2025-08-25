function IHC_params = IHC_params_default(CF_version_keyword)
if nargin < 1
  CF_version_keyword = 'two_cap';
end
one_cap = 0;         % bool; 1 for Allen model, 0 for new default two-cap
just_hwr = 0;        % bool; 0 for normal/fancy IHC; 1 for HWR
do_syn = 0;
switch CF_version_keyword
  case 'just_hwr'
    just_hwr = 1;        % bool; 0 for normal/fancy IHC; 1 for HWR
  case 'one_cap'
    one_cap = 1;         % bool; 1 for Allen model, as text states we use
  case 'do_syn';
    do_syn = 1;
  case 'two_cap'
    % nothing to do; accept the v2 default, two-cap IHC, no SYN.
  otherwise
    error('unknown IHC_keyword in CARFAC_Design')
end
IHC_params = struct( ...
  'just_hwr', just_hwr, ...  % not just a simple HWR
  'one_cap', one_cap, ...    % bool; 0 for new two-cap hack
  'do_syn', do_syn, ...      % bool; 1 for v3 synapse feature
  'tau_lpf', 0.000080, ...   % 80 microseconds smoothing twice
  'tau_out', 0.0005, ...     % depletion tau is pretty fast
  'tau_in', 0.010, ...       % recovery tau is slower
  'tau1_out', 0.000500, ...  % depletion tau is fast 500 us
  'tau1_in', 0.000200, ...   % recovery tau is very fast 200 us
  'tau2_out', 0.001, ...     % depletion tau is pretty fast 1 ms
  'tau2_in', 0.010);         % recovery tau is slower 10 ms
