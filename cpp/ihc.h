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

#ifndef CARFAC_IHC_H
#define CARFAC_IHC_H

#include "common.h"

// Inner hair cell (IHC) parameters, which are used to design the IHC filters.
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
  }

  bool just_half_wave_rectify;
  bool one_capacitor;
  FPType tau_lpf;
  FPType tau1_out;
  FPType tau1_in;
  FPType tau2_out;
  FPType tau2_in;
  FPType ac_corner_hz;
};

// Inner hair cell filter coefficients, which are derived from a set of
// IHCParams.
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

// Inner hair cell filter state.
struct IHCState {
  ArrayX ihc_out;
  ArrayX cap1_voltage;
  ArrayX cap2_voltage;
  ArrayX lpf1_state;
  ArrayX lpf2_state;
  ArrayX ac_coupler;
};

#endif  // CARFAC_IHC_H
