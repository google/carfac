function nlf = CARFAC_OHC_NLF(velocities, CAR_coeffs)

nlf = ((velocities .* CAR_coeffs.velocity_scale) + ...
  CAR_coeffs.v_offset) .^ 2;
% soft saturation to make it more like an "essential" nonlinearity:
nlf = CAR_coeffs.v_damp_max * nlf ./ (CAR_coeffs.v2_corner + nlf);

