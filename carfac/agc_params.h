//
//  agc_params.h
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

#ifndef CARFAC_AGC_PARAMS_H
#define CARFAC_AGC_PARAMS_H

#include <vector>
#include "carfac_common.h"

struct AGCParams {
  AGCParams() {
    n_stages = 4;
    agc_stage_gain = 2.0;
    time_constants.resize(n_stages);
    agc1_scales.resize(n_stages);
    agc2_scales.resize(n_stages);
    agc1_scales[0] = 1.0;
    agc2_scales[0] = 1.65;
    time_constants[0] = 0.002;
    for (int i = 1; i < n_stages; ++i) {
      agc1_scales[i] = agc1_scales[i - 1] * sqrt(2.0);
      agc2_scales[i] = agc2_scales[i - 1] * sqrt(2.0);
      time_constants[i] = time_constants[i - 1] * 4.0;
    }
    decimation = {8, 2, 2, 2};
    agc_mix_coeff = 0.5;
  }
  int n_stages;
  FPType agc_stage_gain;
  FPType agc_mix_coeff;
  std::vector<FPType> time_constants;
  std::vector<int> decimation;
  std::vector<FPType> agc1_scales;
  std::vector<FPType> agc2_scales;
};

#endif  // CARFAC_AGC_PARAMS_H