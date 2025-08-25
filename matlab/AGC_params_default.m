function AGC_params = AGC_params_default
AGC_params = struct( ...
  'n_stages', 4, ...
  'time_constants', 0.002 * 4.^(0:3), ... % 2, 8, 32, 128 ms
  'AGC_stage_gain', 2, ...  % gain from each stage to next slower stage
  'decimation', [8, 2, 2, 2], ...  % how often to update the AGC states
  'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   % in units of channels
  'AGC2_scales', 1.65 * sqrt(2).^(0:3), ... % spread more toward base
  'AGC_mix_coeff', 0.5);
