//
//  ihc_coeffs.h
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

#ifndef CARFAC_IHC_COEFFS_H
#define CARFAC_IHC_COEFFS_H

#include "carfac_common.h"

struct IHCCoeffs {
  bool just_half_wave_rectify;
  bool one_capacitor;
  FPType lpf_coeff;
  FPType out1_rate;
  FPType in1_rate;
  FPType out2_rate;
  FPType in2_rate;
  FPType output_gain;
  FPType rest_output;
  FPType rest_cap1;
  FPType rest_cap2;
  FPType ac_coeff;
  FPType cap1_voltage;
  FPType cap2_voltage;
};

#endif  // CARFAC_IHC_COEFFS_H