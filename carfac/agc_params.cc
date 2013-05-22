//
//  agc_params.cc
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

#include "agc_params.h"

// The default constructor for AGCParams initializes with the settings from
// Lyon's book 'Human and Machine Hearing'
AGCParams::AGCParams() {
  n_stages_ = 4;
  agc_stage_gain_ = 2.0;
  std::vector<FPType> base_values = {1.0, 1.4, 2.0, 2.8};
  FPType agc1_factor = 1.0;
  FPType agc2_factor = 1.65;
  time_constants_.resize(n_stages_);
  agc1_scales_.resize(n_stages_);
  agc2_scales_.resize(n_stages_);
  for (int i = 0; i < n_stages_; ++i) {
    time_constants_[i] = pow(4, i) * 0.002;
    // TODO (alexbrandmeyer): check with Dick Lyon about best way to initialize.
    // Tests on AGC values fail the equality test with Matlab when using the
    // geometric method for initializing the AGC scales.
    // agc1_scales_[i] = agc1_factor * pow(2.0, i/2.0);
    // agc2_scales_[i] = agc2_factor * pow(2.0, i/2.0);
    agc1_scales_[i] = agc1_factor * base_values[i];
    agc2_scales_[i] = agc2_factor * base_values[i];
  }
  decimation_ = {8, 2, 2, 2};
  agc_mix_coeff_ = 0.5;
}