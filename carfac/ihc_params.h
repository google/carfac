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

#ifndef CARFAC_Open_Source_C__Library_IHCParams_h
#define CARFAC_Open_Source_C__Library_IHCParams_h

#include "carfac_common.h"

struct IHCParams {
  IHCParams();
  IHCParams(bool jh, bool oc, FPType tlpf, FPType t1out, FPType t1in,
                 FPType t2out, FPType t2in, FPType acchz);
  bool just_hwr_;
  bool one_cap_;
  FPType tau_lpf_;
  FPType tau1_out_;
  FPType tau1_in_;
  FPType tau2_out_;
  FPType tau2_in_;
  FPType ac_corner_hz_;
};

#endif