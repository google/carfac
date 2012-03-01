function stage_state = CARFAC_Spatial_Smooth(coeffs, stage, stage_state)
% function AGC_state = CARFAC_Spatial_Smooth( ...
%   n_taps, n_iterations, FIR_coeffs, AGC_state)

n_iterations = coeffs.AGC_spatial_iterations(stage);

use_FIR = n_iterations < 4;  % or whatever condition we want to try

if use_FIR
  FIR_coeffs = coeffs.AGC_spatial_FIR(:,stage);
  switch coeffs.AGC_n_taps(stage)
    case 3
      for iter = 1:n_iterations
        stage_state = ...
          FIR_coeffs(1) * stage_state([1, 1:(end-1)], :) + ...
          FIR_coeffs(2) * stage_state + ...
          FIR_coeffs(3) * stage_state([2:end, end], :);
      end
    case 5  % 5-tap smoother duplicates first and last coeffs:
      for iter = 1:n_iterations
        stage_state = ...
          FIR_coeffs(1) * (stage_state([1, 1, 1:(end-2)], :) + ...
          stage_state([1, 1:(end-1)], :)) + ...
          FIR_coeffs(2) *  stage_state + ...
          FIR_coeffs(3) * (stage_state([2:end, end], :) + ...
          stage_state([3:end, end, end], :));
      end
    otherwise
      error('Bad n_taps in CARFAC_Spatial_Smooth');
  end
else
  % use IIR method, back-and-forth firt-order smoothers:
  stage_state = SmoothDoubleExponential(stage_state, ...
    coeffs.AGC_polez1(stage), coeffs.AGC_polez2(stage));
end
