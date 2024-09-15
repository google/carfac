function [syn_out, firings, state] = CARFAC_SYN_Step(v_recep, coeffs, state)
% Drive multiple synapse classes with receptor potential from IHC,
% returning instantaneous spike rates per class, for a group of neurons
% associated with the CF channel, including reductions due to synaptopathy.

% This sigmoidal release_rates is relative to a max of 1, usually way lower.
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

% Make an output that resembles ihc_out, to go to agc_in.
syn_out = firings * coeffs.agc_weights - coeffs.rest_output;
