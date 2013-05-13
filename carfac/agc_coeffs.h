//
//  agc_coeffs.h
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

#ifndef CARFAC_Open_Source_C__Library_AGCCoeffs_h
#define CARFAC_Open_Source_C__Library_AGCCoeffs_h

#include "agc_params.h"

class AGCCoeffs {
public:
  int n_ch_;
  int n_agc_stages_;
  int agc_stage_gain_;
  FloatArray agc_epsilon_; //FloatArray
  FloatArray decimation_; //FloatArray
  FloatArray agc_pole_z1_; //FloatArray
  FloatArray agc_pole_z2_; //FloatArray
  FloatArray agc_spatial_iterations_; //FloatArray
  FloatArray2d agc_spatial_fir_; //2-d FloatArray
  FloatArray agc_spatial_n_taps_; //FloatArray
  FloatArray agc_mix_coeffs_; //FloatArray
  FPType agc_gain_;
  FPType detect_scale_;
  
  FloatArray agc1_scales_;
  FloatArray agc2_scales_;
  FloatArray time_constants_;
  FPType tau_;
  FPType decim_;
  FPType n_times_;
  FPType delay_;
  FPType spread_sq_;
  FPType u_;
  FPType p_;
  FPType dp_;
  FPType pole_z1_;
  FPType pole_z2_;
  int n_taps_;
  bool fir_ok_;
  int n_iterations_;
  FPType total_dc_gain_;
  FPType a, b;
  FPType mix_coeff_;
  FloatArray fir_;

  
  void OutputCoeffs(); //Method to view coeffs, could go in final version
  void DesignAGC(AGCParams agc_params, long fs, int n_ch);
};


#endif
