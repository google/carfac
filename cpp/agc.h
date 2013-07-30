// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Alex Brandmeyer
//
// This file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
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

#ifndef CARFAC_AGC_H
#define CARFAC_AGC_H

#include <vector>

#include "common.h"

// Automatic gain control (AGC) parameters, which are used to design the AGC
// filters.
struct AGCParams {
  AGCParams() {
    num_stages = 4;
    agc_stage_gain = 2.0;
    time_constants.resize(num_stages);
    agc1_scales.resize(num_stages);
    agc2_scales.resize(num_stages);
    agc1_scales[0] = 1.0;
    agc2_scales[0] = 1.65;
    time_constants[0] = 0.002;
    for (int i = 1; i < num_stages; ++i) {
      agc1_scales[i] = agc1_scales[i - 1] * sqrt(2.0);
      agc2_scales[i] = agc2_scales[i - 1] * sqrt(2.0);
      time_constants[i] = time_constants[i - 1] * 4.0;
    }
    decimation = {8, 2, 2, 2};
    agc_mix_coeff = 0.5;
  }
  int num_stages;
  FPType agc_stage_gain;
  FPType agc_mix_coeff;
  std::vector<FPType> time_constants;
  std::vector<int> decimation;
  std::vector<FPType> agc1_scales;
  std::vector<FPType> agc2_scales;
};

// Automatic gain control filter coefficients, which are derived from a set of
// AGCParams.
struct AGCCoeffs {
  FPType agc_stage_gain;
  FPType agc_epsilon;
  int decimation;
  FPType agc_pole_z1;
  FPType agc_pole_z2;
  int agc_spatial_iterations;
  FPType agc_spatial_fir_left;
  FPType agc_spatial_fir_mid;
  FPType agc_spatial_fir_right;
  int agc_spatial_n_taps;
  FPType agc_mix_coeffs;
  FPType agc_gain;
  FPType detect_scale;
  FPType decim;
};

// Automatic gain control filter state.
struct AGCState {
  ArrayX agc_memory;
  ArrayX input_accum;
  int decim_phase;
};

#endif  // CARFAC_AGC_H
