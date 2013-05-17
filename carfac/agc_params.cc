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
  agc_stage_gain_ = 2;
  FPType agc2_factor = 1.65;
  std::vector<FPType> stage_values = {1.0, 1.4, 2.0, 2.8};
  time_constants_.resize(n_stages_);
  agc1_scales_.resize(n_stages_);
  agc2_scales_.resize(n_stages_);
  for (int i = 0; i < n_stages_; ++i) {
    time_constants_(i) = pow(4, i) * 0.002;
    agc1_scales_(i) = stage_values.at(i);
    agc2_scales_(i) = stage_values.at(i) * agc2_factor;
  }
  decimation_ = {8, 2, 2, 2};
  agc_mix_coeff_ = 0.5;
}

// The overloaded constructor allows for use of different AGC parameters.
AGCParams::AGCParams(int n_stages, FPType agc_stage_gain, FPType agc_mix_coeff,
                    FloatArray time_constants, std::vector<int> decimation,
                    FloatArray agc1_scales, FloatArray agc2_scales) {
  n_stages_ = n_stages;
  agc_stage_gain_ = agc_stage_gain;
  agc_mix_coeff_ = agc_mix_coeff;
  time_constants_ = time_constants;
  decimation_ = decimation;
  agc1_scales_ = agc1_scales;
  agc2_scales_ = agc2_scales;
}