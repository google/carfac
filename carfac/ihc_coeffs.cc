//
//  ihc_coeffs.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/18/13.
//  Copyright (c) 2013 Alex Brandmeyer. All rights reserved.
//

#include "ihc_coeffs.h"

void IHCCoeffs::Design(const IHCParams& ihc_params, const FPType fs) {
  // TODO (alexbrandmeyer): verify acceptability of/documentation level needed
  // for use of short intermediate variable names.
  if (ihc_params.just_hwr_) {
    just_hwr_ = ihc_params.just_hwr_;
  } else {
    // This section calculates conductance values using two pre-defined scalars.
    FloatArray x(1);
    FPType conduct_at_10, conduct_at_0;
    x(0) = 10.0;
    x = CARFACDetect(x);
    conduct_at_10 = x(0);
    x(0) = 0.0;
    x = CARFACDetect(x);
    conduct_at_0 = x(0);
    if (ihc_params.one_cap_) {
      FPType ro = 1 / conduct_at_10 ;
      FPType c = ihc_params.tau1_out_ / ro;
      FPType ri = ihc_params.tau1_in_ / c;
      FPType saturation_output = 1 / ((2 * ro) + ri);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (ri + r0);
      cap1_voltage_ = 1 - (current * ri);
      just_hwr_ = false;
      lpf_coeff_ = 1 - exp( -1 / (ihc_params.tau_lpf_ * fs));
      out1_rate_ = ro / (ihc_params.tau1_out_ * fs);
      in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      one_cap_ = ihc_params.one_cap_;
      output_gain_ = 1 / (saturation_output - current);
      rest_output_ = current / (saturation_output - current);
      rest_cap1_ = cap1_voltage_;
    } else {
      FPType ro = 1 / conduct_at_10;
      FPType c2 = ihc_params.tau2_out_ / ro;
      FPType r2 = ihc_params.tau2_in_ / c2;
      FPType c1 = ihc_params.tau1_out_ / r2;
      FPType r1 = ihc_params.tau1_in_ / c1;
      FPType saturation_output = 1 / (2 * ro + r2 + r1);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (r1 + r2 + r0);
      cap1_voltage_ = 1 - (current * r1);
      cap2_voltage_ = cap1_voltage_ - (current * r2);
      just_hwr_ = false;
      lpf_coeff_ = 1 - exp(-1 / (ihc_params.tau_lpf_ * fs));
      out1_rate_ = 1 / (ihc_params.tau1_out_ * fs);
      in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      out2_rate_ = ro / (ihc_params.tau2_out_ * fs);
      in2_rate_ = 1 / (ihc_params.tau2_in_ * fs);
      one_cap_ = ihc_params.one_cap_;
      output_gain_ = 1 / (saturation_output - current);
      rest_output_ = current / (saturation_output - current);
      rest_cap1_ = cap1_voltage_;
      rest_cap2_ = cap2_voltage_;
    }
  }
  ac_coeff_ = 2 * PI * ihc_params.ac_corner_hz_ / fs;
}