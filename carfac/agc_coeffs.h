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

#ifndef CARFAC_Open_Source_C__Library_AGCCoeffs_h
#define CARFAC_Open_Source_C__Library_AGCCoeffs_h

#include "agc_params.h"

// TODO (alexbrandmeyer): check that struct is ok given possible non-triviality
// of the 'Design' method. A change to class would require the addition of
// accessor functions.
struct AGCCoeffs {
  void Design(const AGCParams& agc_params, const int stage, const FPType fs,
              const FPType previous_stage_gain, FPType decim);
  int n_agc_stages_;
  FPType agc_stage_gain_;
  FPType agc_epsilon_;
  int decimation_;
  FPType agc_pole_z1_;
  FPType agc_pole_z2_;
  int agc_spatial_iterations_;
  std::vector<FPType> agc_spatial_fir_;
  int agc_spatial_n_taps_;
  FPType agc_mix_coeffs_;
  FPType agc_gain_;
  FPType detect_scale_;
  FPType decim_;
};

#endif