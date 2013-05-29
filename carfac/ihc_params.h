//
//  ihc_params.h
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

#ifndef CARFAC_IHC_PARAMS_H
#define CARFAC_IHC_PARAMS_H

#include "carfac_common.h"

struct IHCParams {
  IHCParams() {
    just_half_wave_rectify = false;
    one_capacitor = true;
    tau_lpf = 0.000080;
    tau1_out = 0.0005;
    tau1_in = 0.010;
    tau2_out = 0.0025;
    tau2_in = 0.005;
    ac_corner_hz = 20.0;
  };
  bool just_half_wave_rectify;
  bool one_capacitor;
  FPType tau_lpf;
  FPType tau1_out;
  FPType tau1_in;
  FPType tau2_out;
  FPType tau2_in;
  FPType ac_corner_hz;
};

#endif  // CARFAC_IHC_PARAMS_H