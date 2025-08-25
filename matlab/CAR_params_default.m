function CAR_params = CAR_params_default
CAR_params = struct( ...
  'velocity_scale', 0.1, ...  % for the velocity nonlinearity
  'v_offset', 0.04, ...  % offset gives a quadratic part
  'min_zeta', 0.10, ...  % minimum damping factor in mid-freq channels
  'max_zeta', 0.35, ...  % maximum damping factor in mid-freq channels
  'first_pole_theta', 0.85*pi, ...
  'zero_ratio', sqrt(2), ...  % how far zero is above pole
  'high_f_damping_compression', 0.5, ...  % 0 to 1 to compress zeta
  'ERB_per_step', 0.5, ...  % assume G&M's ERB formula
  'min_pole_Hz', 30, ...
  'ERB_break_freq', 165.3, ...  % 165.3 is Greenwood map's break freq.
  'ERB_Q', 1000/(24.7*4.37), ...  % Glasberg and Moore's high-cf ratio
  'ac_corner_Hz', 20, ...  % AC couple at 20 Hz corner
  'use_delay_buffer', 0 ...  % Default to true starting in v3.
  );
