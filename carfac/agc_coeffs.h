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

#ifndef CARFAC_AGC_COEFFS_H
#define CARFAC_AGC_COEFFS_H

#include "carfac_common.h"

struct AGCCoeffs {
  int n_agc_stages;
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

#endif  // CARFAC_AGC_COEFFS_H