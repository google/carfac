//
//  ear.h
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

#ifndef CARFAC_Open_Source_C__Library_Ear_h
#define CARFAC_Open_Source_C__Library_Ear_h

#include "car_state.h"
#include "ihc_state.h"
#include "agc_state.h"


class Ear {
public:
  int n_ch_;
  FPType max_channels_per_octave_;
  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  CARCoeffs car_coeffs_;
  IHCCoeffs ihc_coeffs_;
  AGCCoeffs agc_coeffs_;
  CARState car_state_;
  IHCState ihc_state_;
  AGCState agc_state_;
  
  void InitEar(long fs, CARParams car_p, IHCParams ihc_p, AGCParams agc_p);
  FloatArray CARStep(FPType input);
  FloatArray OHC_NLF(FloatArray velocities);
  FloatArray IHCStep(FloatArray car_out);
  bool AGCStep(FloatArray ihc_out);
  bool AGCRecurse(int stage, FloatArray agc_in);
  FloatArray AGCSpatialSmooth(int stage, FloatArray stage_state);
  FloatArray AGCSmoothDoubleExponential(FloatArray stage_state);
};

#endif
