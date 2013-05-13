//
//  car_state.cc
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

#include "car_state.h"

void CARState::InitCARState(CARCoeffs car_coeffs){
  n_ch_ = car_coeffs.n_ch_;
  z1_memory_ = FloatArray::Zero(n_ch_);
  z2_memory_ = FloatArray::Zero(n_ch_);
  za_memory_ = FloatArray::Zero(n_ch_);
  zb_memory_ = car_coeffs.zr_coeffs_;
  dzb_memory_ = FloatArray::Zero(n_ch_);
  zy_memory_ = FloatArray::Zero(n_ch_);
  g_memory_ = car_coeffs.g0_coeffs_;
  dg_memory_ = FloatArray::Zero(n_ch_);
  
  
}