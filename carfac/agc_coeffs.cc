//
//  agc_coeffs.cc
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

#include "agc_coeffs.h"

//This method has been created for debugging purposes and depends on <iostream>.
//Could possibly be removed in the final version to reduce dependencies.
void AGCCoeffs::OutputCoeffs(){
  std::cout << "AGCCoeffs Values" << std::endl;
  std::cout << "****************" << std::endl;
  std::cout << "n_ch_ = " << n_ch_ << std::endl;
  std::cout << "n_agc_stages_ = " << n_agc_stages_ << std::endl;
  std::cout << "agc_stage_gain_ = " << agc_stage_gain_ << std::endl;
  std::cout << "agc_epsilon_ = " << agc_epsilon_ << std::endl;
  std::cout << "decimation_ = " << decimation_ << std::endl;
  std::cout << "agc_pole_z1_ = " << agc_pole_z1_ << std::endl;
  std::cout << "agc_pole_z2_ = " << agc_pole_z2_ << std::endl;
  std::cout << "agc_spatial_iterations_ = " << agc_spatial_iterations_
            << std::endl;
  std::cout << "agc_spatial_fir_ = " << agc_spatial_fir_ << std::endl;
  std::cout << "agc_spatial_n_taps_ = " << agc_spatial_n_taps_ << std::endl;
  std::cout << "agc_mix_coeffs_ = " << agc_mix_coeffs_ << std::endl;
  std::cout << "agc1_scales_ = " << agc1_scales_ << std::endl;
  std::cout << "agc2_scales_ = " << agc2_scales_ << std::endl;
  std::cout << "time_constants_ = " << time_constants_
            << std::endl << std::endl;
}

void AGCCoeffs::DesignAGC(AGCParams agc_params, long fs, int n_ch){
  n_ch_ = n_ch;
  n_agc_stages_ = agc_params.n_stages_;
  agc_stage_gain_ = agc_params.agc_stage_gain_;
  agc1_scales_ = agc_params.agc1_scales_;
  agc2_scales_ = agc_params.agc2_scales_;
  time_constants_ = agc_params.time_constants_;
  agc_epsilon_.resize(n_agc_stages_);
  agc_pole_z1_.resize(n_agc_stages_);
  agc_pole_z2_.resize(n_agc_stages_);
  agc_spatial_iterations_.resize(n_agc_stages_);
  agc_spatial_n_taps_.resize(n_agc_stages_);
  agc_spatial_fir_.resize(3,n_agc_stages_);
  agc_mix_coeffs_.resize(n_agc_stages_);
  mix_coeff_ = agc_params.agc_mix_coeff_;
  fir_.resize(3);
  decim_ = 1;
  decimation_ = agc_params.decimation_;
  total_dc_gain_ = 0;
  
  for (int stage=0; stage < n_agc_stages_; stage++){
    tau_ = time_constants_(stage);
    decim_ = decim_ * decimation_(stage);
    agc_epsilon_(stage) = 1 - exp((-1 * decim_) / (tau_ * fs));
    n_times_ = tau_ * (fs / decim_);
    delay_ = (agc2_scales_(stage) - agc1_scales_(stage)) / n_times_;
    spread_sq_ = (pow(agc1_scales_(stage),2) + pow(agc2_scales_(stage),2)) /
    n_times_;
    u_ = 1 + (1 / spread_sq_);
    p_ = u_ - sqrt(pow(u_,2) - 1);
    dp_ = delay_ * (1 - (2 * p_) + pow(p_,2)) / 2;
    pole_z1_ = p_ - dp_;
    pole_z2_ = p_ + dp_;
    agc_pole_z1_(stage) = pole_z1_;
    agc_pole_z2_(stage) = pole_z2_;
    n_taps_ = 0;
    fir_ok_ = 0;
    n_iterations_ = 1;
    //initialize FIR coeffs settings
    try {
      while (! fir_ok_){
        switch (n_taps_){
          case 0:
            n_taps_ = 3;
            break;
          case 3:
            n_taps_ = 5;
            break;
          case 5:
            n_iterations_ ++;
            if (n_iterations_ > 16){
              throw 10; //too many iterations
            }
            break;
          default:
            throw 20; //bad n_taps
        }
        //Design FIR Coeffs
        FPType var = spread_sq_ / n_iterations_;
        FPType mn = delay_ / n_iterations_;
        switch (n_taps_){
          case 3:
            a = (var + pow(mn,2) - mn) / 2;
            b = (var + pow(mn,2) + mn) / 2;
            fir_ << a, 1 - a - b, b;
            if (fir_(2) >= 0.2) {
              fir_ok_ = true;
            } else {
              fir_ok_ = false;
            }
            break;
          case 5:
            a = (((var + pow(mn,2)) * 2/5) - (mn * 2/3)) / 2;
            b = (((var + pow(mn,2)) * 2/5) + (mn * 2/3)) / 2;
            fir_ << a/2, 1 - a - b, b/2;
            if (fir_(2) >= 0.1) {
              fir_ok_ = true;
            } else {
              fir_ok_ = false;
            }
            break;
          default:
            throw 30; //bad n_taps in FIR design
        }
      }
    }
    catch (int e) {
      switch (e) {
        case 10:
          std::cout << "ERROR: Too many n_iterations in agc_coeffs.DesignAGC"
          << std::endl;
          break;
        case 20:
          std::cout << "ERROR: Bad n_taps in agc_coeffs.DesignAGC" << std::endl;
          break;
        case 30:
          std::cout << "ERROR: Bad n_taps in agc_coeffs.DesignAGC/FIR"
          << std::endl;
          break;
        default:
          std::cout << "ERROR: unknown error in agc_coeffs.DesignAGC"
          << std::endl;
      }
    }
    //assign output of filter design
    agc_spatial_iterations_(stage) = n_iterations_;
    agc_spatial_n_taps_(stage) = n_taps_;
    agc_spatial_fir_(0,stage) = fir_(0);
    agc_spatial_fir_(1,stage) = fir_(1);
    agc_spatial_fir_(2,stage) = fir_(2);
    total_dc_gain_ = total_dc_gain_ + pow(agc_stage_gain_,(stage));
    if (stage == 0) {
      agc_mix_coeffs_(stage) = 0;
    } else {
      agc_mix_coeffs_(stage) = mix_coeff_ / (tau_ * (fs /decim_));
    }
  }
  agc_gain_ = total_dc_gain_;
  detect_scale_ = 1 / total_dc_gain_;
}
