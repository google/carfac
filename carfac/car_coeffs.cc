//
//  car_coeffs.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/18/13.
//  Copyright (c) 2013 Alex Brandmeyer. All rights reserved.
//

#include "car_coeffs.h"

void CARCoeffs::Design(const CARParams& car_params, const FPType fs,
                        const FloatArray& pole_freqs) {
  // TODO (alexbrandmeyer): verify acceptability of/documentation level needed
  // for use of short intermediate variable names.
  n_ch_ = pole_freqs.size();
  velocity_scale_ = car_params.velocity_scale_;
  v_offset_ = car_params.v_offset_;
  r1_coeffs_.resize(n_ch_);
  a0_coeffs_.resize(n_ch_);
  c0_coeffs_.resize(n_ch_);
  h_coeffs_.resize(n_ch_);
  g0_coeffs_.resize(n_ch_);
  FPType f = car_params.zero_ratio_ * car_params.zero_ratio_ - 1.0;
  FloatArray theta = pole_freqs * ((2.0 * PI) / fs);
  c0_coeffs_ = theta.sin();
  a0_coeffs_ = theta.cos();
  FPType ff = car_params.high_f_damping_compression_;
  FloatArray x = theta / PI;
  zr_coeffs_ = PI * (x - (ff * (x*x*x)));
  FPType max_zeta = car_params.max_zeta_;
  FPType min_zeta = car_params.min_zeta_;
  r1_coeffs_ = (1.0 - (zr_coeffs_ * max_zeta));
  FloatArray erb_freqs(n_ch_);
  for (int ch=0; ch < n_ch_; ++ch) {
    erb_freqs(ch) = ERBHz(pole_freqs(ch), car_params.erb_break_freq_,
                          car_params.erb_q_);
  }
  FloatArray min_zetas = min_zeta + (0.25 * ((erb_freqs / pole_freqs) -
                                             min_zeta));
  zr_coeffs_ *= max_zeta - min_zetas;
  h_coeffs_ = c0_coeffs_ * f;
  FloatArray relative_undamping = FloatArray::Ones(n_ch_);
  FloatArray r = r1_coeffs_ + (zr_coeffs_ * relative_undamping);
  g0_coeffs_ = (1.0 - (2.0 * r * a0_coeffs_) + (r*r)) /
                                        (1 - (2 * r * a0_coeffs_) +
                                        (h_coeffs_ * r * c0_coeffs_) + (r*r));
}