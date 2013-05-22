//
//  ihc_params.cc
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

#include "ihc_params.h"

// The default constructor for IHCParams initializes with the settings from
// Lyon's book 'Human and Machine Hearing'
IHCParams::IHCParams() {
  just_hwr_ = false;
  one_cap_ = true;
  tau_lpf_ = 0.000080;
  tau1_out_ = 0.0005;
  tau1_in_ = 0.010;
  tau2_out_ = 0.0025;
  tau2_in_ = 0.005;
  ac_corner_hz_ = 20.0;
}