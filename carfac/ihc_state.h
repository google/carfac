//
//  ihc_state.h
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

#ifndef CARFAC_IHC_STATE_H
#define CARFAC_IHC_STATE_H

#include "carfac_common.h"

struct IHCState {
  FloatArray ihc_out;
  FloatArray ihc_accum;
  FloatArray cap1_voltage;
  FloatArray cap2_voltage;
  FloatArray lpf1_state;
  FloatArray lpf2_state;
  FloatArray ac_coupler;
};

#endif  // CARFAC_IHC_STATE_H