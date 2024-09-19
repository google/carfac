function [syn_out, firings, state] = CARFAC_SYN_Step(v_recep, coeffs, state)
% Drive multiple synapse classes with receptor potential from IHC,
% returning instantaneous spike rates per class, for a group of neurons
% associated with the CF channel, including reductions due to synaptopathy.

% Normalized offset position into neurotransmitter release sigmoid.
x = (v_recep - coeffs.v_halfs) ./ coeffs.v_widths ;

% This sigmoidal release_rates is relative to a max of 1, usually way lower.
s = 1 ./ (1 + exp(-x));
release_rates = (1 - state.reservoirs) .* s;  % z = w*s

% Smooth once with LPF (receptor potential was already smooth):
state.lpf_state = state.lpf_state + coeffs.lpf_coeff * ...
  (coeffs.a2 .* release_rates - state.lpf_state);  % this is firing probs.
firing_probs = state.lpf_state;  % Poisson rate per neuron per sample.
% Include number of effective neurons per channel here, and interval T;
% so the rates (instantaneous action potentials per second) can be huge.
firings = coeffs.n_fibers .* firing_probs;

% Feedback, to update reservoir state q for next time.
state.reservoirs = state.reservoirs + coeffs.res_coeff .* ...
  (coeffs.a1 .* release_rates - state.reservoirs);

% Make an output that resembles ihc_out, to go to agc_in.
syn_out = (coeffs.n_fibers .* (firing_probs - coeffs.spont_p)) * coeffs.agc_weights;
