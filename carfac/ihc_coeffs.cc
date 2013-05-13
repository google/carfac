//
//  ihc_coeffs.cc
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

#include "ihc_coeffs.h"

//This method has been created for debugging purposes and depends on <iostream>.
//Could possibly be removed in the final version to reduce dependencies.
void IHCCoeffs::OutputCoeffs(){
  std::cout << "IHCCoeffs Values" << std::endl;
  std::cout << "****************" << std::endl;
  std::cout << "n_ch_ = " << n_ch_ << std::endl;
  std::cout << "just_hwr_ = " << just_hwr_ << std::endl;
  std::cout << "one_cap_ = " << one_cap_ << std::endl;
  std::cout << "lpf_coeff_ = " << lpf_coeff_ << std::endl;
  std::cout << "out1_rate_ = " << out1_rate_ << std::endl;
  std::cout << "in1_rate_ = " << in1_rate_ << std::endl;
  std::cout << "out2_rate_ = " << out2_rate_ << std::endl;
  std::cout << "in2_rate_ = " << in2_rate_ << std::endl;
  std::cout << "output_gain_ = " << output_gain_ << std::endl;
  std::cout << "rest_output_ = " << rest_output_ << std::endl;
  std::cout << "rest_cap1_ = " << rest_cap1_ << std::endl;
  std::cout << "rest_cap2_ = " << rest_cap2_ << std::endl;
  std::cout << "ac_coeff_ = " << ac_coeff_ << std::endl << std::endl;
}

void IHCCoeffs::DesignIHC(IHCParams ihc_params, long fs, int n_ch){
  if (ihc_params.just_hwr_){
    n_ch_ = n_ch;
    just_hwr_ = ihc_params.just_hwr_;
  } else {
    if (ihc_params.one_cap_){
      ro_ = 1 / CARFACDetect(10);
      c_ = ihc_params.tau1_out_ / ro_;
      ri_ = ihc_params.tau1_in_ / c_;
      saturation_output_ = 1 / ((2 * ro_) + ri_);
      r0_ = 1 / CARFACDetect(0);
      current_ = 1 / (ri_ + r0_);
      cap1_voltage_ = 1 - (current_ * ri_);
      
      n_ch_ = n_ch;
      just_hwr_ = false;
      lpf_coeff_ = 1 - exp( -1 / (ihc_params.tau_lpf_ * fs));
      out1_rate_ = ro_ / (ihc_params.tau1_out_ * fs);
      in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      one_cap_ = ihc_params.one_cap_;
      output_gain_ = 1 / (saturation_output_ - current_);
      rest_output_ = current_ / (saturation_output_ - current_);
      rest_cap1_ = cap1_voltage_;

      
    } else {
      ro_ = 1 / CARFACDetect(10);
      c2_ = ihc_params.tau2_out_ / ro_;
      r2_ = ihc_params.tau2_in_ / c2_;
      c1_ = ihc_params.tau1_out_ / r2_;
      r1_ = ihc_params.tau1_in_ / c1_;
      saturation_output_ = 1 / (2 * ro_ + r2_ + r1_);
      r0_ = 1 / CARFACDetect(0);
      current_ = 1 / (r1_ + r2_ + r0_);
      cap1_voltage_ = 1 - (current_ * r1_);
      cap2_voltage_ = cap1_voltage_ - (current_ * r2_);
      
      n_ch_ = n_ch;
      just_hwr_ = false;
      lpf_coeff_ = 1 - exp(-1 / (ihc_params.tau_lpf_ * fs));
      out1_rate_ = 1 / (ihc_params.tau1_out_ * fs);
      in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      out2_rate_ = ro_ / (ihc_params.tau2_out_ * fs);
      in2_rate_ = 1 / (ihc_params.tau2_in_ * fs);
      one_cap_ = ihc_params.one_cap_;
      output_gain_ = 1 / (saturation_output_ - current_);
      rest_output_ = current_ / (saturation_output_ - current_);
      rest_cap1_ = cap1_voltage_;
      rest_cap2_ = cap2_voltage_;
    }
  }
  ac_coeff_ = 2 * PI * ihc_params.ac_corner_hz_ / fs;
}
