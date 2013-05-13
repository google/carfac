//
//  car_coeffs.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/10/13.
//
// This C++ file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
// to supplement Lyon's upcoming book "Human and Machine Hearing"
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "car_coeffs.h"

//This method has been created for debugging purposes and depends on <iostream>.
//Could possibly be removed in the final version to reduce dependencies. 
void CARCoeffs::OutputCoeffs(){
  std::cout << "CARCoeffs Values" << std::endl;
  std::cout << "****************" << std::endl;
  std::cout << "n_ch_ = " << n_ch_ << std::endl;
  std::cout << "velocity_scale_ = " << velocity_scale_ << std::endl;
  std::cout << "v_offset_ = " << v_offset_ << std::endl;
  std::cout << "r1_coeffs_ = " << r1_coeffs_ << std::endl;
  std::cout << "a0_coeffs_ = " << a0_coeffs_ << std::endl;
  std::cout << "c0_coeffs_ = " << c0_coeffs_ << std::endl;
  std::cout << "h_coeffs_ = " << h_coeffs_ << std::endl;
  std::cout << "g0_coeffs_ = " << g0_coeffs_ << std::endl;
  std::cout << "zr_coeffs_ = " << zr_coeffs_ << std::endl << std::endl;
}

void CARCoeffs::DesignFilters(CARParams car_params, long fs,
                              FloatArray pole_freqs[]){
  
  n_ch_ = int(pole_freqs->size());
  velocity_scale_ = car_params.velocity_scale_;
  v_offset_ = car_params.v_offset_;
  FloatArray p_freqs = *pole_freqs;
  r1_coeffs_.resize(n_ch_);
  a0_coeffs_.resize(n_ch_);
  c0_coeffs_.resize(n_ch_);
  h_coeffs_.resize(n_ch_);
  g0_coeffs_.resize(n_ch_);
  FPType f = car_params.zero_ratio_ * car_params.zero_ratio_ - 1;
  FloatArray theta = p_freqs * ((2 * PI) / fs);
  c0_coeffs_ = sin(theta);
  a0_coeffs_ = cos(theta);
  FPType ff = car_params.high_f_damping_compression_;
  FloatArray x = theta / PI;
  zr_coeffs_ = PI * (x - (ff * (x * x * x)));//change to exponet
  FPType max_zeta = car_params.max_zeta_;
  FPType min_zeta = car_params.min_zeta_;
  r1_coeffs_ = (1 - (zr_coeffs_ * max_zeta));
  FPType curfreq;
  FloatArray erb_freqs(n_ch_);
  for (int ch=0; ch < n_ch_; ch++){
    curfreq = p_freqs(ch);
    erb_freqs(ch) = ERBHz(curfreq, car_params.erb_break_freq_,
                           car_params.erb_q_);
  }
  FloatArray min_zetas = min_zeta + (0.25 * ((erb_freqs / p_freqs) - min_zeta));
  zr_coeffs_ = zr_coeffs_ * (max_zeta - min_zetas);
  h_coeffs_ = c0_coeffs_ * f;
  FloatArray relative_undamping = FloatArray::Ones(n_ch_);
  FloatArray r = r1_coeffs_ + (zr_coeffs_ * relative_undamping);
  g0_coeffs_ = (1 - (2 * r * a0_coeffs_) + (r * r)) /
            (1 - (2 * r * a0_coeffs_) + (h_coeffs_ * r * c0_coeffs_) + (r * r));
}