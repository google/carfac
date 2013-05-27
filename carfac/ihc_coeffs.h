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

#ifndef CARFAC_Open_Source_C__Library_IHCCoeffs_h
#define CARFAC_Open_Source_C__Library_IHCCoeffs_h

#include "ihc_params.h"

struct IHCCoeffs {
  bool just_half_wave_rectify_;
  bool one_capacitor_;
  FPType lpf_coeff_;
  FPType out1_rate_;
  FPType in1_rate_;
  FPType out2_rate_;
  FPType in2_rate_;
  FPType output_gain_;
  FPType rest_output_;
  FPType rest_cap1_;
  FPType rest_cap2_;
  FPType ac_coeff_;
  FPType cap1_voltage_;
  FPType cap2_voltage_;
};

#endif