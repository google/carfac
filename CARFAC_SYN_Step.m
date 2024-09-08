function [syn_out, firings, state] = CARFAC_SYN_Step(v_recep, coeffs, state)
% Drive multiple synapse classes with receptor potential from IHC,
% returning instantaneous spike rates per class, for a group of neurons
% associated with the CF channel, including reductions due to synaptopathy.

if nargin < 1
  error('SYN_step needs at least a v_recep input.')
end

if nargin < 3  % Temporarily do the setup defaulting here; design it elsewhere later.
  n_ones = ones(1, 3);
  v_width = 0.05;
  % Parameters could generally have columns if channel-dependent.
  max_rate = 2500;  % % Instantaneous max at onset, per Kiang figure 5.18.
  fs = 48000;
  coeffs = struct( ...
    'fs', fs, ...
    'n_classes', length(n_ones), ...
    'max_probs', (max_rate / fs) * n_ones, ...  
    'n_fibers', [50, 35, 25], ...  % Synaptopathy comes in here; channelize it, too.
    'v_width', v_width * n_ones, ...  % Try to match IHC out units.
    'v_half', v_width .* [3, 5, 7], ...  % Same units as v_width and v_recep.
    'out_rate', 0.02, ...  % Depletion can be quick (few ms).
    'recovery', 1e-4);  % Recovery tau about 10,000 sample times.
end

if nargin < 2  % Init the state, sized according to the coeffs.
  n_ch = 1;
  % "Full" reservoir state is 1.0, empty 0.0.
  state = struct( ...
    'reservoirs', ones(n_ch, coeffs.n_classes));
end

% This release_rates is relative to a max of 1, usually way lower.
release_rates = state.reservoirs ./ ...
  (1 + exp(-(v_recep - coeffs.v_half)./coeffs.v_width));

% Smooth once with LPF (receptor potential was already smooth):
state.lpf_state = state.lpf_state + coeffs.lpf_coeff * ...
  (release_rates - state.lpf_state);
release_rates = state.lpf_state;

% Deplete reservoirs some, independent of synaptopathy numbers (we assume).
state.reservoirs = state.reservoirs - coeffs.out_rate .* release_rates;
% Refill reservoirs a little toward 1.
state.reservoirs = state.reservoirs + coeffs.recovery .* (1 - state.reservoirs);

% Includes number of effective neurons per channel here, and interval T,
% so the rates (instantaneous action potentials per second) can be huge.
firings = coeffs.n_fibers .* coeffs.max_probs .* release_rates;

% Make an output that resembles ihc_out, to go the agc_in.
syn_out = firings * coeffs.agc_weights_col - coeffs.rest_output;




