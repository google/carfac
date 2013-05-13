//
//  ihc_state.cc
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

#include "ihc_state.h"

void IHCState::InitIHCState(IHCCoeffs ihc_coeffs){
  n_ch_ = ihc_coeffs.n_ch_;
  ihc_accum_ = FloatArray::Zero(n_ch_);
  if (! ihc_coeffs.just_hwr_){
    ac_coupler_ = FloatArray::Zero(n_ch_);
    lpf1_state_ = ihc_coeffs.rest_output_ * FloatArray::Ones(n_ch_);
    lpf2_state_ = ihc_coeffs.rest_output_ * FloatArray::Ones(n_ch_);
    if (ihc_coeffs.one_cap_){
      cap1_voltage_ = ihc_coeffs.rest_cap1_ * FloatArray::Ones(n_ch_);
    } else {
      cap1_voltage_ = ihc_coeffs.rest_cap1_ * FloatArray::Ones(n_ch_);
      cap2_voltage_ = ihc_coeffs.rest_cap2_ * FloatArray::Ones(n_ch_);
    }
  }
}